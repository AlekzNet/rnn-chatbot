require 'torch'
require 'nn'
require 'optim'

require 'LanguageModel'
--require 'util.DataLoader'

-- local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', 'data/input.txt.h5')
cmd:option('-input_json', 'data/input.txt.json')
cmd:option('-batch_size', 2)
cmd:option('-seq_length', 4)

-- Model options
cmd:option('-init_from', '')
cmd:option('-model_type', 'gridgru')
cmd:option('-wordvec_size', 64)
cmd:option('-rnn_size', 1536)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0.1)
cmd:option('-batchnorm', 1)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 3e-2)
cmd:option('-grad_clip', 5)
cmd:option('-lr_decay_every', 1)
cmd:option('-lr_decay_factor', 0.9)

-- Output options
cmd:option('-print_every', 1)
cmd:option('-checkpoint_every', 10)
cmd:option('-checkpoint_name', '/nas/doc/nn/checkpoint')

-- Benchmark options
cmd:option('-speed_benchmark', 0)
cmd:option('-memory_benchmark', 0)

-- Backend options
cmd:option('-gpu', -1)
cmd:option('-gpu_backend', 'cuda')

local opt = cmd:parse(arg)
local idx_to_token = {}
idx_to_token[1] = 1

-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  -- Memory benchmarking is only supported in CUDA mode
  -- TODO: Time benchmarking is probably wrong in OpenCL mode.
  require 'cltorch'
  require 'clnn'
  cltorch.setDevice(opt.gpu + 1)
  dtype = torch.Tensor():cl():type()
else
  -- Memory benchmarking is only supported in CUDA mode
  opt.memory_benchmark = 0
end
local models={"gru","lstm"}
-- local models={"gru","lstm","gridgru"}
local mod_params={}
local grad_params=nil
local model = nil
for i=4,5 do
for j=2550,4000,50 do
-- for j=50,3000,50 do
opt.num_layers=i
opt.rnn_size=j
for k,m in ipairs(models) do
opt.model_type=m
if opt.model_type == 'gridgru' then
   opt.wordvec_size=j
else
   opt.wordvec_size=64
end
-- Initialize the model and criterion
local opt_clone = torch.deserialize(torch.serialize(opt))
opt_clone.idx_to_token = idx_to_token
  model = nn.LanguageModel(opt_clone):type(dtype)
mod_params[m], grad_params = model:getParameters()
--print (m,mod_params[m]:nElement())
model = nil
collectgarbage()
end -- end models
-- print(opt.num_layers,opt.rnn_size,opt.num_layers*opt.rnn_size,mod_params["gru"]:nElement(),mod_params["lstm"]:nElement(),mod_params["gridgru"]:nElement())
print(opt.num_layers,opt.rnn_size,opt.num_layers*opt.rnn_size,mod_params["gru"]:nElement(),mod_params["lstm"]:nElement())
--collectgarbage()
end -- end model size
end -- end layers
