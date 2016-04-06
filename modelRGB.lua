require 'torch'
require 'nn'

local CROP_SIZE = 128

local model = nn.Sequential()

model:add(nn.SpatialConvolution(3, 32, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(32, 64, 3, 3, 2, 2, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(64, 128, 3, 3, 2, 2, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(128, 64, 3, 3, 2, 2, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

model:add(nn.SpatialConvolution(64, 32, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))

local concat = nn.ConcatTable()
local depth = nn.Sequential()
local classify = nn.Sequential()

classify:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
classify:add(nn.ReLU(true))

classify:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
classify:add(nn.ReLU(true))

classify:add(nn.SpatialConvolution(64, 32, 3, 3, 1, 1, 1, 1))
classify:add(nn.ReLU(true))

classify:add(nn.Reshape(16*16*32))
classify:add(nn.Linear(16*16*32, 2))

concat:add(classify)

depth:add(nn.SpatialFullConvolution(32, 32, 3, 3, 2, 2, 1, 1, 1, 1))
depth:add(nn.ReLU(true))

depth:add(nn.SpatialFullConvolution(32, 4, 3, 3, 4, 4, 1, 1, 3, 3))
depth:add(nn.ReLU(true))

depth:add(nn.SpatialConvolution(4, 1, 7, 7, 1, 1, 3, 3))
depth:add(nn.ReLU(true))

depth:add(nn.Reshape(CROP_SIZE * CROP_SIZE))

concat:add(depth)

model:add(concat)

return model
