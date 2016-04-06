require 'torch'
require 'nn'
require 'cunn'

local CROP_SIZE = 128
local BATCH_SIZE = arg[2] or 800

function accuracy (prediction, target)
  local _, yHat = torch.max(prediction, 2)
  return yHat:eq(target):mean()
end

local model = torch.load(arg[1])
local classLoss = nn.CrossEntropyCriterion():cuda()
local depthLoss = nn.MSECriterion():cuda()

local val = torch.load('val.t7')

-- evaluate on val
model:evaluate()

local nVal = val['y']:size(1)
local totalEnt = 0
local totalCorrect = 0
local totalMSE = 0
print('# of val samples:', nVal)
val['yHat'] = torch.zeros(val['x'][2]:size())
for i = 1, math.ceil(nVal/BATCH_SIZE) do
  local j = (i-1) * BATCH_SIZE + 1
  local k = math.min(j + BATCH_SIZE - 1, nVal)
  local rgb = val['x'][1][{{j, k}}]:cuda()
  local d = val['x'][2][{{j, k}}]:reshape(k-j+1, CROP_SIZE * CROP_SIZE):cuda()
  local y = val['y'][{{j, k}}]:cuda()
  local yHat = model:forward(rgb)

  val['yHat'][{{j, k}}] = yHat[2]:reshape(k-j+1, CROP_SIZE, CROP_SIZE):float()

  local classCost = classLoss:forward(yHat[1], y)
  local depthCost = depthLoss:forward(yHat[2], d)
  local acc = accuracy(yHat[1], y)
  totalEnt = totalEnt + classCost * (k - j + 1)
  totalCorrect = totalCorrect + acc * (k - j + 1)
  totalMSE = totalMSE + depthCost * (k - j + 1)
end
torch.save('val-out.t7', val)
print('val entropy:', totalEnt / nVal)
print('val acc:', totalCorrect / nVal)
print('val MSE:', totalMSE / nVal)
