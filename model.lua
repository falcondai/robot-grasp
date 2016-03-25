require 'torch'
require 'nn'

local CROP_SIZE = 128

local model = nn.Sequential()

model:add(nn.JoinTable(1, 3))
model:add(nn.SpatialConvolution(4, 32, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(32, 64, 3, 3, 2, 2, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.Reshape(64*64*128))
model:add(nn.Linear(64*64*128, 2))

return model
