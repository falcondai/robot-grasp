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
local loss = nn.CrossEntropyCriterion()
loss:cuda()

local val = torch.load('val.t7')

-- evaluate on val
model:evaluate()

local nVal = val['y']:size(1)
local totalEnt = 0
local totalCorrect = 0
print('# of val samples:', nVal)
for i = 1, math.ceil(nVal/BATCH_SIZE) do
  local j = (i-1) * BATCH_SIZE + 1
  local k = math.min(j + BATCH_SIZE - 1, nVal)
  local rgb = val['x'][1][{{j, k}}]:cuda()
  local d = val['x'][2][{{j, k}}]:cuda()
  local y = val['y'][{{j, k}}]:cuda()
  local yHat = model:forward({rgb, d})

  local valCost = loss:forward(yHat, y)
  local acc = accuracy(yHat, y)
  totalEnt = totalEnt + valCost * (k - j + 1)
  totalCorrect = totalCorrect + acc * (k - j + 1)
end
print('val entropy:', totalEnt / nVal)
print('val acc:', totalCorrect / nVal)
