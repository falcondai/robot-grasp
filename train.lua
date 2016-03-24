require 'torch'
require 'nn'
-- require 'cunn'
require 'image'
require 'paths'
npy = require 'npy4th'

local BATCH_SIZE = 128
local CROP_SIZE = 128

torch.manualSeed(3)

local model = require './model'
-- model:cuda()
model:training()
local loss = nn.CrossEntropyCriterion()

local trainY = npy.loadnpy('splits/train_y.npy')
local valY = npy.loadnpy('splits/val_y.npy')

-- lua is 1-index
local trainY = trainY + 1
local valY = valY + 1

-- load the train set
local trainFn = {}
for line in io.open('splits/train_fn.txt'):lines() do
  table.insert(trainFn, line:split(' '))
end
local nTrain = #trainFn
local trainRgb = torch.Tensor(nTrain, 3, CROP_SIZE, CROP_SIZE)
local trainD = torch.Tensor(nTrain, 1, CROP_SIZE, CROP_SIZE)
for i, row in ipairs(trainFn) do
  trainRgb[i] = image.load(row[1])
  trainD[i][1] = npy.loadnpy(row[2])
end
local trainX = {trainRgb, trainD}

-- load the validation set
local valFn = {}
for line in io.open('splits/val_fn.txt'):lines() do
  table.insert(valFn, line:split(' '))
end
local nVal = #valFn
local valRgb = torch.Tensor(nVal, 3, CROP_SIZE, CROP_SIZE)
local valD = torch.Tensor(nVal, 1, CROP_SIZE, CROP_SIZE)
for i, row in ipairs(valFn) do
  valRgb[i] = image.load(row[1])
  valD[i][1] = npy.loadnpy(row[2])
  if i == 10 then
    break
  end
end
local valX = {valRgb, valD}

for i = 1, 100 do
  model:zeroGradParameters()
  local yhat = model:forward(trainX)
  local cost = loss:forward(yhat, trainY)
  print(cost)
  local dl = loss:backward(yhat, trainY)
  model:backward(trainX, dl)
  model:updateParameters(0.1)
end
