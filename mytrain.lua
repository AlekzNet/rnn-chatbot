require 'torch'
require 'nn'
require 'optim'

require 'LanguageModel'
require 'util.DataLoader'
-- require 'fifo'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

-- local fifo = Fifo:Create(35)
local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-online',false,'boolean option')
cmd:option('-outfile',"")
-- cmd:option('-input_h5', 'data/alldia.h5')
cmd:option('-input_h5', 'data/alldia-curr.h5')
cmd:option('-input_json', 'data/ascii.json')
cmd:option('-batch_size', 20)
-- cmd:option('-batch_size', 572)
-- cmd:option('-batch_size', 398)
-- cmd:option('-batch_size', 1)
cmd:option('-seq_length', 300)
-- cmd:option('-seq_length', 40)

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 0)
cmd:option('-model_type', 'gru')
-- cmd:option('-model_type', 'lstm')
-- cmd:option('-model_type', 'gridgru')
-- cmd:option('-wordvec_size', 600)
-- cmd:option('-wordvec_size', 64)
cmd:option('-rnn_size', 1000)
-- cmd:option('-rnn_size', 600)
cmd:option('-num_layers', 3)
cmd:option('-dropout', 0.1)
-- cmd:option('-dropout', 0.0)
cmd:option('-batchnorm', 1)

-- Optimization options
cmd:option('-max_epochs', 50)
-- cmd:option('-learning_rate', 1.0e-04)
-- cmd:option('-learning_rate', 1.0e-03)
cmd:option('-learning_rate', 5.0e-06)
cmd:option('-grad_clip', 5)
cmd:option('-lr_decay_every', 5)
cmd:option('-lr_decay_factor', 0.5)

-- Output options
cmd:option('-print_every', 2)
cmd:option('-checkpoint_every', 1500)
cmd:option('-checkpoint_name', '/nas/soft/nn/cp')

-- Benchmark options
cmd:option('-speed_benchmark', 0)
cmd:option('-memory_benchmark', 0)

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')

local opt = cmd:parse(arg)

if opt.model_type == 'gridgru' then
   opt.wordvec_size=opt.rnn_size
else
   opt.wordvec_size=64
end

local online=opt.online
local outfile=opt.outfile

if online then
	opt.batch_size = 1
	local fp = assert(hdf5.open(opt.input_h5), "Can't open file " .. opt.input_h5 )
	opt.seq_length=fp:read('train'):dataspaceSize()[1]-1
	print ("Online mode. Seq_length is set to ", opt.seq_length)
	fp:close()
	opt.lr_decay_factor=0.9
	opt.learning_rate=1.0e-08
	opt.lr_decay_every=20
	
end


-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  -- Memory benchmarking is only supported in CUDA mode
  -- TODO: Time benchmarking is probably wrong in OpenCL mode.
  require 'cltorch'
  require 'clnn'
  cltorch.setDevice(opt.gpu + 1)
  dtype = torch.Tensor():cl():type()
  print(string.format('Running with OpenCL on GPU %d', opt.gpu))
else
  -- Memory benchmarking is only supported in CUDA mode
  opt.memory_benchmark = 0
  print 'Running in CPU mode'
end


-- Initialize the DataLoader and vocabulary
local loader = DataLoader(opt)
local vocab = utils.read_json(opt.input_json)
local idx_to_token = {}
for k, v in pairs(vocab.idx_to_token) do
  idx_to_token[tonumber(k)] = v
end

-- Initialize the model and criterion
local opt_clone = torch.deserialize(torch.serialize(opt))
opt.checkpoint_name = string.format('%s_%dx%d-%s-%d', opt.checkpoint_name, opt.num_layers, opt.rnn_size, opt.dropout, opt.batchnorm)
print (opt.checkpoint_name)

opt_clone.idx_to_token = idx_to_token
local model = nil
local start_i = 0
if opt.init_from ~= '' then
  print('Initializing from ', opt.init_from)
  local checkpoint = torch.load(opt.init_from)
  model = checkpoint.model:type(dtype)
  if opt.reset_iterations == 0 then
    start_i = checkpoint.i
  end
  print('Loaded')

else
  model = nn.LanguageModel(opt_clone):type(dtype)
  print('New model created')
end

print(opt)

if not start_i and not online then
  io.write("Enter the iteration number: ")
  start_i = io.read()
elseif online then
  start_i = 1
end

local params, grad_params = model:getParameters()
print('number of parameters in the model: ' .. params:nElement())
print('grad_params:norm', grad_params:norm())
print('params:norm', params:norm())
print('grad/param norm',  grad_params:norm() / params:norm())

local crit = nn.CrossEntropyCriterion():type(dtype)

-- Set up some variables we will use below
local N, T = opt.batch_size, opt.seq_length
local train_loss_history = {}
local val_loss_history = {}
local val_loss_history_it = {}
local forward_backward_times = {}
local init_memory_usage, memory_usage = nil, {}

if opt.memory_benchmark == 1 then
  -- This should only be enabled in GPU mode
  assert(cutorch)
  cutorch.synchronize()
  local free, total = cutorch.getMemoryUsage(cutorch.getDevice())
  init_memory_usage = total - free
end

-- Loss function that we pass to an optim method
local function f(w)
  assert(w == params)
  grad_params:zero()

  -- Get a minibatch and run the model forward, maybe timing it
  local timer
  local x, y = loader:nextBatch('train')
  x, y = x:type(dtype), y:type(dtype)
  if opt.speed_benchmark == 1 then
    if cutorch then cutorch.synchronize() end
--    timer = torch.Timer()
  end
  local scores = model:forward(x)

  -- Use the Criterion to compute loss; we need to reshape the scores to be
  -- two-dimensional before doing so. Annoying.
  local scores_view = scores:view(N * T, -1)
  local y_view = y:view(N * T)
  local loss = crit:forward(scores_view, y_view)

  -- Run the Criterion and model backward to compute gradients, maybe timing it
  local grad_scores = crit:backward(scores_view, y_view):view(N, T, -1)
  model:backward(x, grad_scores)
  if timer then
    if cutorch then cutorch.synchronize() end
    local time = timer:time().real
    print('Forward / Backward pass took ', time)
    table.insert(forward_backward_times, time)
  end

  -- Maybe record memory usage
  if opt.memory_benchmark == 1 then
    assert(cutorch)
    if cutorch then cutorch.synchronize() end
    local free, total = cutorch.getMemoryUsage(cutorch.getDevice())
    local memory_used = total - free - init_memory_usage
    local memory_used_mb = memory_used / 1024 / 1024
    print(string.format('Using %dMB of memory', memory_used_mb))
    table.insert(memory_usage, memory_used)
  end
  if opt.grad_clip > 0 then
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  return loss, grad_params
end


-- Train the model!
local optim_config = {learningRate = opt.learning_rate}
local num_train = loader.split_sizes['train']
local num_iterations = opt.max_epochs * num_train
model:training()
for i = start_i + 1, num_iterations do
  local epoch = math.floor(i / num_train) + 1

  -- Check if we are at the end of an epoch
  if i % num_train == 0 then
--    model:resetStates() -- Reset hidden states

    -- Maybe decay learning rate
    if epoch % opt.lr_decay_every == 0 then
      model:resetStates()
      local old_lr = optim_config.learningRate
	  opt.learning_rate = opt.learning_rate * opt.lr_decay_factor
      optim_config = {learningRate = old_lr * opt.lr_decay_factor}
      print('learningRate = ', optim_config.learningRate)
    end
  end

  -- Take a gradient step and maybe print
  -- Note that adam returns a singleton array of losses
  local mytimer = torch.Timer()
  local _, loss = optim.adam(f, params, optim_config)
  table.insert(train_loss_history, loss[1])
  local elapsed = mytimer:time().real
  if opt.print_every > 0 and i % opt.print_every == 0 then
    local float_epoch = i / num_train + 1
--    fifo:push(loss[1])
    local msg = 'Epoch %.4f / %d, i = %d / %d, loss = %f, time = %f'
    local args = {msg, float_epoch, opt.max_epochs, i, num_iterations, loss[1], elapsed}
    print(string.format(unpack(args)))
    collectgarbage()
  end

  -- Maybe save a checkpoint
  local check_every = opt.checkpoint_every
  if (check_every > 0 and i % check_every == 0) or i == num_iterations  or (i % num_train == 0 and not online) then
    -- Evaluate loss on the validation set. Note that we reset the state of
    -- the model; this might happen in the middle of an epoch, but that
    -- shouldn't cause too much trouble.
    val_loss = 0.1
    table.insert(val_loss_history, val_loss)
    table.insert(val_loss_history_it, i)
	if online then
--	    model:resetStates()
	end
--    model:training()

    -- First save a JSON checkpoint, excluding the model
    local checkpoint = {
      opt = opt,
      train_loss_history = train_loss_history,
      val_loss_history = val_loss_history,
      val_loss_history_it = val_loss_history_it,
      forward_backward_times = forward_backward_times,
      memory_usage = memory_usage,
      i = i
    }
	local filename = ""
	if online then
		if outfile then
			filename = outfile .. "json"
		else 
			filename = opt.checkpoint_name .. "_online_2.t7.json"
		end
	else
    		filename = string.format('%s_%d.json', opt.checkpoint_name, i)
	end
    -- Make sure the output directory exists before we try to write it
    paths.mkdir(paths.dirname(filename))
    utils.write_json(filename, checkpoint)

    -- Now save a torch checkpoint with the model
    -- Cast the model to float before saving so it can be used on CPU
	--    model:clearState()
	if online then
		if outfile then		
			filename = outfile
		else
			filename = opt.checkpoint_name .. "_online_2.t7"
		end
	else
    	filename = string.format('%s_%s_%d.t7', opt.checkpoint_name, opt.learning_rate, i)
	end
    paths.mkdir(paths.dirname(filename))
    print('Saving a checkpoint')
--	if i == num_iterations then
--		model:resetStates()
--		model:clearState()
--	end
    model:float()
    checkpoint.model = model
    local mytimer = torch.Timer()
    torch.save(filename, checkpoint)
    print ('Checkpoint = ',filename,'Time = ',mytimer:time().real)
    model:type(dtype)
--    params, grad_params = model:getParameters()
    collectgarbage()
  end
end
