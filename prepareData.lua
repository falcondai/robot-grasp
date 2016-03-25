require 'torch'
require 'image'
require 'paths'
npy = require 'npy4th'

local CROP_SIZE = 128
local DEPTH_RESCALE_FACTOR = 1 / 255

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
  trainD[i][1] = npy.loadnpy(row[2]) * DEPTH_RESCALE_FACTOR
end
local trainX = {trainRgb, trainD}
print('train set')
print(trainX)
-- save train set
torch.save('train.t7', { x = trainX, y = trainY })

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
  valD[i][1] = npy.loadnpy(row[2]) * DEPTH_RESCALE_FACTOR
end
local valX = {valRgb, valD}
print('validation set')
print(valX)
-- save validation set
torch.save('val.t7', { x = valX, y = valY })
