require 'torch'
require 'nn'


local layer, parent = torch.class('nn.GRIDGRU', 'nn.Module')

--[[
Adapted from Grid LSTM : http://arxiv.org/abs/1507.01526
--]]

function layer:__init(input_dim, hidden_dim)
  parent.__init(self)

  local D, H = input_dim, hidden_dim
  self.input_dim, self.hidden_dim = D, H

  self.weight = torch.Tensor(D + H, 3 * H + 3 * D)
  self.gradWeight = torch.Tensor(D + H, 3 * H + 3 * D):zero()
  self.bias = torch.Tensor(3 * H + 3 * D)
  self.gradBias = torch.Tensor(3 * H + 3 * D):zero()
  --self.weightd = torch.Tensor(D + H, 3 * D)
  --self.gradWeightd = torch.Tensor(D + H, 3 * D):zero()
  --self.biasd = torch.Tensor(3 * D)
  --self.gradBiasd = torch.Tensor(3 * D):zero()
  self:reset()

  self.cell = torch.Tensor()    -- This will be (N, T, H)
  self.gates = torch.Tensor()   -- This will be (N, T, 3H)
  self.gatesd = torch.Tensor()   -- This will be (N, T, 3H)
  self.buffer1 = torch.Tensor() -- This will be (N, H)
  self.buffer2 = torch.Tensor() -- This will be (N, H)
  self.buffer3 = torch.Tensor() -- This will be (H,)
  self.grad_a_buffer = torch.Tensor() -- This will be (N, 3H)
  self.buffer1d = torch.Tensor() -- This will be (N, D)
  self.buffer2d = torch.Tensor() -- This will be (N, D)
  self.buffer3d= torch.Tensor() -- This will be (D,)
  self.grad_a_bufferd = torch.Tensor() -- This will be (N, 3D)
  self.h0 = torch.Tensor()
  self.remember_states = false
  self.grad_h0 = torch.Tensor()
  self.grad_x = torch.Tensor()
  self.gradInput = {self.grad_h0, self.grad_x}
end


function layer:reset(std)
  if not std then
    std = 1.0 / math.sqrt(self.hidden_dim + self.input_dim)
  end
  --self.bias:zero()
  self.bias:normal(0,std) 
  self.weight:normal(0, std)
  return self
end


function layer:resetStates()
  self.h0 = self.h0.new()
end


local function check_dims(x, dims)
  assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    assert(x:size(i) == d)
  end
end


function layer:_unpack_input(input)
  local h0, x = nil, nil
  
  if torch.type(input) == 'table' and #input == 2 then
    h0, x = unpack(input)
  elseif torch.isTensor(input) then
    x = input
  else
    assert(false, 'invalid input')
  end
  return h0, x
end


function layer:_get_sizes(input, gradOutput)
  local h0, x = self:_unpack_input(input)
  local N, T = x:size(1), x:size(2)
  local H, D = self.hidden_dim, self.input_dim
  check_dims(x, {N, T, D})
  if h0 then
    check_dims(h0, {N, H})
  end
  
  if gradOutput then
    check_dims(gradOutput, {N, T, D})
  end
  return N, T, D, H
end


--[[
Input:
- h0: Initial hidden state, (N, H)
- x: Input sequence, (N, T, D)

Output:
- h: Sequence of hidden states, (N, T, D)
--]]


function layer:updateOutput(input)
  local h0, x = self:_unpack_input(input)
  local N, T, D, H = self:_get_sizes(input)

  self._return_grad_h0 = (h0 ~= nil)
  
  if not h0 then
    h0 = self.h0
    if h0:nElement() == 0 or not self.remember_states then
      h0:resize(N, H):zero()
    elseif self.remember_states then
      local prev_N, prev_T = self.output:size(1), self.output:size(2)
      assert(prev_N == N, 'batch sizes must be the same to remember states')
      h0:copy(self.cell[{{}, prev_T}])
    end
  end

  local bias_expand = self.bias:view(1, 3 * H + 3 * D):expand(N, 3 * H + 3 * D)
  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]
  local bias_expandt = bias_expand[{{},{1, 3 * H}}]
  local Wxt = Wx[{{},{1, 3 * H}}]
  local Wht = Wh[{{},{1, 3 * H}}]
  local bias_expandd = bias_expand[{{},{3 * H + 1, 3 * H + 3 * D}}]
  local Wxd = Wx[{{},{3 * H + 1, 3 * H + 3 * D}}]
  local Whd = Wh[{{},{3 * H + 1, 3 * H + 3 * D}}]

  local h, ht = self.output, self.cell
  h:resize(N, T, D):zero()
  ht:resize(N, T, H):zero()
  local prev_ht = h0
  self.gates:resize(N, T, 3 * H):zero()
  self.gatesd:resize(N, T, 3 * D):zero()
  for t = 1, T do
    local cur_x = x[{{}, t}]
    local next_ht = ht[{{}, t}]
    local cur_gates = self.gates[{{}, t}]
    cur_gates:addmm(bias_expandt, cur_x, Wxt)
    cur_gates[{{}, {1, 2 * H}}]:addmm(prev_ht, Wht[{{}, {1, 2 * H}}])
    cur_gates[{{}, {1, 2 * H}}]:sigmoid()
    
    local u = cur_gates[{{}, {1, H}}] --update gate : u = sig(Wx * x + Wh * prev_h + b)
    local r = cur_gates[{{}, {H + 1, 2 * H}}] --reset gate : r = sig(Wx * x + Wh * prev_h + b)
    next_ht:cmul(r, prev_ht) --temporary buffer : r . prev_h
    cur_gates[{{}, {2 * H + 1, 3 * H}}]:addmm(next_ht, Wht[{{}, {2 * H + 1, 3 * H}}]) -- hc += Wh * r . prev_h
    local hc = cur_gates[{{}, {2 * H + 1, 3 * H}}]:tanh() --hidden candidate : hc = tanh(Wx * x + Wh * r . prev_h + b)
    next_ht:addcmul(prev_ht,-1, u, prev_ht)
    next_ht:addcmul(u,hc)  --next_h = (1-u) . prev_h + u . hc   
    prev_ht = next_ht
    
    local next_h = h[{{}, t}]
    local cur_gatesd = self.gatesd[{{}, t}]
    cur_gatesd:addmm(bias_expandd, prev_ht, Whd)
    cur_gatesd[{{}, {1, 2 * D}}]:addmm(cur_x, Wxd[{{}, {1, 2 * D}}])
    cur_gatesd[{{}, {1, 2 * D}}]:sigmoid()
    
    local ud = cur_gatesd[{{}, {1, D}}] --update gate : u = sig(Wx * x + Wh * prev_h + b)
    local rd = cur_gatesd[{{}, {D + 1, 2 * D}}] --reset gate : r = sig(Wx * x + Wh * prev_h + b)
    next_h:cmul(rd, cur_x) --temporary buffer : r . x
    cur_gatesd[{{}, {2 * D + 1, 3 * D}}]:addmm(next_h, Wxd[{{}, {2 * D + 1, 3 * D}}]) -- hc += Wx * r . prev_x
    local hcd = cur_gatesd[{{}, {2 * D + 1, 3 * D}}]:tanh() --hidden candidate : hc = tanh(Wx * r .x + Wh *  prev_h + b)
    next_h:addcmul(cur_x,-1, ud, cur_x)
    next_h:addcmul(ud,hcd)  --next_h = (1-u) . x + u . hc   
    --prev_h = next_h
    
  end

  return self.output
end


function layer:backward(input, gradOutput, scale)
  scale = scale or 1.0
  local h0, x = self:_unpack_input(input)
  
  if not h0 then h0 = self.h0 end

  local grad_h0, grad_x = self.grad_h0, self.grad_x
  local h= self.output
  local ht = self.cell
  local grad_h = gradOutput

  local N, T, D, H = self:_get_sizes(input, gradOutput)
  local Wx = self.weight[{{1, D}}]
  local Wh = self.weight[{{D + 1, D + H}}]
  local grad_Wx = self.gradWeight[{{1, D}}]
  local grad_Wh = self.gradWeight[{{D + 1, D + H}}]
  local grad_b = self.gradBias
  
  local Wxt = Wx[{{},{1, 3 * H}}]
  local Wht = Wh[{{},{1, 3 * H}}]
  local grad_Wxt = grad_Wx[{{},{1, 3 * H}}]
  local grad_Wht = grad_Wh[{{},{1, 3 * H}}]
  local grad_bt = grad_b[{{1, 3 * H}}]
  
  local Wxd = Wx[{{},{3 * H + 1, 3 * H + 3 * D}}]
  local Whd = Wh[{{},{3 * H + 1, 3 * H + 3 * D}}]
  local grad_Wxd = grad_Wx[{{},{3 * H + 1, 3 * H + 3 * D}}]
  local grad_Whd = grad_Wh[{{},{3 * H + 1, 3 * H + 3 * D}}]
  local grad_bd = grad_b[{{3 * H + 1, 3 * H + 3 * D}}]

  grad_h0:resizeAs(h0):zero() 
  grad_x:resizeAs(x):zero()
  local grad_next_h = self.buffer1:resizeAs(h0):zero()
  local temp_buffer = self.buffer2:resizeAs(h0):zero()
  local grad_a_sum = self.buffer3:resize(1,3*H):zero()
  local grad_next_hd = self.buffer1d:resizeAs(x[{{}, 1}]):zero()
  local temp_bufferd = self.buffer2d:resizeAs(x[{{}, 1}]):zero()
  local grad_a_sumd = self.buffer3d:resize(1,3*D):zero()
  for t = T, 1, -1 do
    local next_h= h[{{}, t}]
    local prev_h= nil
    if t == 1 then
      prev_h = h0
    else
      prev_h = ht[{{}, t - 1}]
    end
    local prev_hd = ht[{{}, t}]
    --
    local ud = self.gatesd[{{}, t, {1, D}}]
    local rd = self.gatesd[{{}, t, {D + 1, 2 * D}}]
    local hcd = self.gatesd[{{}, t, {2 * D + 1, 3 * D}}]
    
    
    local grad_ad = self.grad_a_bufferd:resize(N, 3 * D):zero()
    local grad_aud = grad_ad[{{}, {1, D}}]
    local grad_ard = grad_ad[{{}, {D + 1, 2 * D}}]
    local grad_ahcd = grad_ad[{{}, {2 * D + 1, 3 * D}}]
    grad_next_hd:zero():add(grad_h[{{}, t}])
    -- We will use grad_au as temporary buffer
    -- to compute grad_ahc.
    local cur_x = x[{{}, t}]
    local grad_hcd = grad_aud:fill(0):add(grad_next_hd ):cmul(ud)  
    grad_ahcd:fill(1):addcmul(-1, hcd,hcd):cmul(grad_hcd)
    local grad_rd = grad_aud:fill(0):addmm(grad_ahcd, Wxd[{{}, {2 * D + 1, 3 * D}}]:t() ):cmul(cur_x)
    grad_ard:fill(1):add(-1, rd):cmul(rd):cmul(grad_rd)
    temp_bufferd:fill(0):add(hcd):add(-1, cur_x)
    grad_aud:fill(1):add(-1, ud):cmul(ud):cmul(temp_bufferd):cmul(grad_next_hd)   
    grad_h0:mm(grad_ad, Whd:t())
    grad_Whd:addmm(scale, prev_hd:t(), grad_ad)
    grad_Wxd[{{}, {1, 2 * D}}]:addmm(scale, cur_x:t(), grad_ad[{{}, {1, 2 * D}}])
    
    grad_a_sumd:sum(grad_ad, 1)
    grad_bd:add(scale, grad_a_sumd)
    temp_bufferd:fill(0):add(cur_x):cmul(rd)
    grad_Wxd[{{}, {2 * D + 1, 3 * D}}]:addmm(scale, temp_bufferd:t(), grad_ahcd)   
    grad_next_hd:addcmul(-1, ud, grad_next_hd)
    grad_next_hd:addmm(grad_ad[{{}, {1, 2 * D}}], Wxd[{{}, {1, 2 * D}}]:t())
    temp_bufferd:fill(0):addmm(grad_ad[{{}, {2 * D + 1, 3 * D}}], Wxd[{{}, {2 * D + 1, 3 * D}}]:t()):cmul(rd)
    grad_next_hd:add(temp_bufferd)
    --
    
    
    grad_next_h:add(grad_h0)
    --grad_next_h:clamp(-5,5)

    local u = self.gates[{{}, t, {1, H}}]
    local r = self.gates[{{}, t, {H + 1, 2 * H}}]
    local hc = self.gates[{{}, t, {2 * H + 1, 3 * H}}]
    
    
    local grad_a = self.grad_a_buffer:resize(N, 3 * H):zero()
    local grad_au = grad_a[{{}, {1, H}}]
    local grad_ar = grad_a[{{}, {H + 1, 2 * H}}]
    local grad_ahc = grad_a[{{}, {2 * H + 1, 3 * H}}]
    
    -- We will use grad_au as temporary buffer
    -- to compute grad_ahc.
    
    local grad_hc = grad_au:fill(0):add(grad_next_h ):cmul(u)  
    grad_ahc:fill(1):addcmul(-1, hc,hc):cmul(grad_hc)
    local grad_r = grad_au:fill(0):addmm(grad_ahc, Wht[{{}, {2 * H + 1, 3 * H}}]:t() ):cmul(prev_h)
    grad_ar:fill(1):add(-1, r):cmul(r):cmul(grad_r)
    
    temp_buffer:fill(0):add(hc):add(-1, prev_h)
    grad_au:fill(1):add(-1, u):cmul(u):cmul(temp_buffer):cmul(grad_next_h)   
    grad_x[{{}, t}]:mm(grad_a, Wxt:t())
    grad_x[{{}, t}]:add(grad_next_hd)
    grad_Wxt:addmm(scale, x[{{}, t}]:t(), grad_a)
    grad_Wht[{{}, {1, 2 * H}}]:addmm(scale, prev_h:t(), grad_a[{{}, {1, 2 * H}}])
    
    grad_a_sum:sum(grad_a, 1)
    grad_bt:add(scale, grad_a_sum)
    temp_buffer:fill(0):add(prev_h):cmul(r)
    grad_Wht[{{}, {2 * H + 1, 3 * H}}]:addmm(scale, temp_buffer:t(), grad_ahc)   
    grad_next_h:addcmul(-1, u, grad_next_h)
    grad_next_h:addmm(grad_a[{{}, {1, 2 * H}}], Wht[{{}, {1, 2 * H}}]:t())
    temp_buffer:fill(0):addmm(grad_a[{{}, {2 * H + 1, 3 * H}}], Wht[{{}, {2 * H + 1, 3 * H}}]:t()):cmul(r)
    grad_next_h:add(temp_buffer)
  end
  grad_h0:copy(grad_next_h)

  if self._return_grad_h0 then
    self.gradInput = {self.grad_h0, self.grad_x}
  else
    self.gradInput = self.grad_x
  end

  return self.gradInput
end


function layer:updateGradInput(input, gradOutput)
  self:backward(input, gradOutput, 0)
end


function layer:accGradParameters(input, gradOutput, scale)
  self:backward(input, gradOutput, scale)
end

function layer:clearState()
  self.cell:set()
  self.gates:set()
  self.buffer1:set()
  self.buffer2:set()
  self.buffer3:set()
  self.grad_a_buffer:set()
  self.gatesd:set()
  self.buffer1d:set()
  self.buffer2d:set()
  self.buffer3d:set()
  self.grad_a_bufferd:set()

  self.grad_h0:set()
  self.grad_x:set()
  self.output:set()
end
