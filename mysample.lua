require 'torch'
require 'nn'
require 'dia'
require 'LanguageModel'


local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'cv/checkpoint_4000.t7')
cmd:option('-length', 1000)
-- cmd:option('-start_text', '\n')
cmd:option('-start_text', '\nHello, how are you doing today?\n')
cmd:option('-sample', 1)
cmd:option('-temperature', 0.7)
cmd:option('-gpu', -1)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-verbose', 0)
local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
print('Loading ', opt.checkpoint)
local model = checkpoint.model
print('Loaded')

local msg
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  model:cuda()
  msg = string.format('Running with CUDA on GPU %d', opt.gpu)
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  require 'cltorch'
  require 'clnn'
  model:cl()
  msg = string.format('Running with OpenCL on GPU %d', opt.gpu)
else
  msg = 'Running in CPU mode'
end

model:evaluate()
local dia = Dia:Create(16)
-- dia:push(opt.start_text)
-- local sample = model:sample(opt)
local sample = nil
-- dia:push(sample)
local reply = nil
local n = 0
while true do
	io.write("Me: ")
	reply = io.read()
	if string.find(reply,"^;") then
		if string.find(reply,";reset") then
			model:resetStates()
			model:clearState()
			dia:clear()
			print("Reset")
		elseif string.find(reply,";clear")then
			dia:clear()
			print("The dialogue array is cleared")
		elseif string.find(reply,";temp=") then
			opt.temperature = string.gsub(reply, ".*temp= *","")
			print("The temp is set to ".. opt.temperature)
		elseif string.find(reply,";curlen") then
			print("The current length is " .. dia:curlen() .. ", the maximum length is " .. dia:maxlen())
		elseif string.find(reply,";print") then
			print(dia:dia())
		elseif string.find(reply,";help") then
			print ("reset, clear, temp=, curlen, setlen=")
		elseif string.find(reply,";setlen=") then
			n = string.gsub(reply, ".*setlen= *","")
                        dia:setlen(tonumber(n))
                        print("The dialogue length is set to " .. n)
		else
			print ("Unknown command")
		end
	else
		
		dia:push(reply)
		opt.start_text = dia:dia()
--		io.write(#opt.start_text, " NN: ")
		io.write("NN: ")
--		print(opt.start_text)
		sample = model:sample(opt)
		dia:push(sample)
		print(sample)
		model:resetStates()
--		model:clearState()
	end
end
