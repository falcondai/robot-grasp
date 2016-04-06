require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'
require 'paths'

local BATCH_SIZE = tonumber(arg[2]) or 200
local CROP_SIZE = 128
local MAX_STEP = tonumber(arg[3]) or 800

function accuracy (prediction, target)
  local _, yHat = torch.max(prediction, 2)
  return yHat:eq(target):mean()
end

print('# of CUDA devices:', cutorch.getDeviceCount())
print('using device:', cutorch.getDevice())
print('saving checkpoint models to:', arg[1])
paths.mkdir(arg[1])

torch.manualSeed(3)

local model = require './modelRGB'
print(model)
model:cuda()
model:training()

local classLoss = nn.CrossEntropyCriterion()
classLoss:cuda()
local depthLoss = nn.MSECriterion()
depthLoss:cuda()
local lambda = 0.1

local train = torch.load('train.t7')
local val = torch.load('val.t7')

local n = train['y']:size(1)
print('# of samples', n)

local mParams, mGrad = model:getParameters()
local classCost, depthCost, cost
function _fgrad (rgb, d, y)
  function fgrad (params)
    mParams:copy(params)
    model:zeroGradParameters()
    local yHat = model:forward(rgb)
    -- classCost = classLoss:forward(yHat[1], y)
    -- depthCost = depthLoss:forward(yHat[2], d)
    depthCost = depthLoss:forward(yHat, d)
    -- cost = classCost + lambda * depthCost
    cost = depthCost
    -- local classDl = classLoss:backward(yHat[1], y)
    -- local depthDl = depthLoss:backward(yHat[2], d)
    local depthDl = depthLoss:backward(yHat, d)
    -- model:backward(rgb, classDl)
    -- model:backward(rgb, lambda * depthDl)
    model:backward(rgb, depthDl)
    return cost, mGrad
  end
  return fgrad
end

local rgb, d, y
local state = {}
for step = 1, MAX_STEP do
  -- construct mini-batch
  local i = step * BATCH_SIZE % n
  if i < BATCH_SIZE then
    i = 1
  end
  local j = math.min(i + BATCH_SIZE - 1, n)
  local size = j - i + 1
  rgb = train['x'][1][{{i, j}}]:cuda()
  d = train['x'][2][{{i, j}}]:reshape(size, CROP_SIZE * CROP_SIZE):cuda()
  y = train['y'][{{i, j}}]:cuda()

  optim.adam(_fgrad(rgb, d, y), mParams, state)
  -- print(step, cost, classCost, depthCost, mGrad:norm())
  print(step, cost, mGrad:norm())

  -- checkpoint the model
  if step % 200 == 0 then
    model:clearState()
    torch.save(arg[1]..'/model.'..step..'.t7', model)
  end
end

-- save the final model
model:clearState()
torch.save(arg[1]..'/model.'..MAX_STEP..'.t7', model)
