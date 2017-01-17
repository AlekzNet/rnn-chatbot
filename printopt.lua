require 'torch'
require 'nn'
require 'optim'

require 'LanguageModel'
require 'util.DataLoader'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', 'data/input.txt.h5')
cmd:option('-input_json', 'data/input.txt.json')
cmd:option('-batch_size', 270)
-- cmd:option('-batch_size', 10)
-- cmd:option('-batch_size', 1)
cmd:option('-seq_length', 700)
-- cmd:option('-seq_length', 40)

-- Model options
cmd:option('-init_from', '')
cmd:option('-model_type', 'lstm')
cmd:option('-wordvec_size', 64)
cmd:option('-rnn_size', 1350)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0.1)
cmd:option('-batchnorm', 1)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 1.8e-3)
cmd:option('-grad_clip', 6)
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


local model = nil
if opt.init_from ~= '' then
  print('Initializing from ', opt.init_from)
  model = torch.load(opt.init_from).model:type(dtype)
else
  assert(false,"No file specified")
end
local params, grad_params = model:getParameters()
print('number of parameters in the model: ' .. params:nElement())
print(opt)
