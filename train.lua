require 'torch'
require 'nn'
-- require 'cudnn'
require 'image'
require 'paths'
npy = require 'npy4th'

local BATCH_SIZE = 128
local CROP_SIZE = 128

torch.manualSeed(3)

local model = require './model'
local loss = nn.CrossEntropyCriterion()

local trainF = io.open('splits/train_fn.txt')
local valF = io.open('splits/val_fn.txt')
local trainY = npy.loadnpy('splits/train_y.npy')
local valY = npy.loadnpy('splits/val_y.npy')

-- lua is 1-index
local trainY = trainY + 1
local valY = valY + 1

local trainFn = {}
for line in trainF:lines() do
  table.insert(trainFn, line:split(' '))
end
local nTrain = #trainFn
local rgb = torch.Tensor(nTrain, 3, CROP_SIZE, CROP_SIZE)
local d = torch.Tensor(nTrain, 1, CROP_SIZE, CROP_SIZE)
for i, row in ipairs(trainFn) do
  rgb[i] = image.load(row[1])
  d[i][1] = npy.loadnpy(row[2])
end
local trainX = {rgb, d}
print(trainX)

-- local rgb = image.load('processed/pos/0464-000.png')
-- local d = npy.loadnpy('processed/pos/0464-000.npy'):reshape(1, 128, 128)
--
local yhat = model:forward({rgb, d})
local cost = loss:forward(yhat, trainY)
-- local dl = loss:backward(yhat, )
-- model:backward()
--
print(yhat)
-- print(cost)
