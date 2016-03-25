require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'

local BATCH_SIZE = 1000
local CROP_SIZE = 128
local MAX_STEP = 600

function accuracy (prediction, target)
  local _, yHat = torch.max(prediction, 2)
  return yHat:eq(target):mean()
end

print('# of CUDA devices:', cutorch.getDeviceCount())
print('using device:', cutorch.getDevice())
-- cutorch.setDevice(2)

torch.manualSeed(3)

local model = require './model'
model:cuda()
model:training()
local loss = nn.CrossEntropyCriterion()
loss:cuda()

local train = torch.load('train.t7')
local val = torch.load('val.t7')

local n = train['y']:size(1)
print('# of samples', n)

local learningRate = 0.1
local learningRateDecay = 0.8
local learningRateDecayPeriod = 200

local rgb, d, y
for step = 1, MAX_STEP do
  -- construct mini-batch
  local i = step * BATCH_SIZE % n
  if i < BATCH_SIZE then
    i = 1
  end
  local j = math.min(i + BATCH_SIZE, n)
  rgb = train['x'][1][{{i, j}}]:cuda()
  d = train['x'][2][{{i, j}}]:cuda()
  y = train['y'][{{i, j}}]:cuda()

  -- compute gradient
  model:zeroGradParameters()
  local yHat = model:forward({rgb, d})
  local cost = loss:forward(yHat, y)
  print(step, learningRate, cost)
  local dl = loss:backward(yHat, y)
  model:backward({rgb, d}, dl)
  model:updateParameters(learningRate)

  -- update learning rate
  if step % learningRateDecayPeriod == 0 then
    learningRate = learningRate * learningRateDecay
    model:clearState()
    torch.save('model.'..step..'.t7', model)
  end
end

-- evaluate on val
model:evaluate()

rgb = val['x'][1][{{1,1000}}]:cuda()
d = val['x'][2][{{1,1000}}]:cuda()
y = val['y'][{{1,1000}}]:cuda()
local yHat = model:forward({rgb, d})

local valCost = loss:forward(yHat, y)
print('val entropy:', valCost)
print('val acc:', accuracy(yHat, y))

torch.save('model.'..MAX_STEP..'.t7', model)
