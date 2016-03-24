require 'torch'
require 'nn'
-- require 'cudnn'

local CROP_SIZE = 128

local model = nn.Sequential()

model:add(nn.JoinTable(1, 3))

model:add(nn.SpatialConvolution(4, 32, 3, 3))
model:add(nn.ReLU(true))

-- model:add(nn.SpatialConvolution(32, 64, 3, 3))
-- model:add(nn.ReLU(true))
--
-- model:add(nn.SpatialConvolution(64, 128, 3, 3))
-- model:add(nn.ReLU(true))

-- model:add(nn.Reshape((128-6)*(128-6)*128))
-- model:add(nn.Linear((128-6)*(128-6)*128, 1))

model:add(nn.Reshape((128-2)*(128-2)*32))
model:add(nn.Linear((128-2)*(128-2)*32, 2))
-- model:add(nn.Sigmoid())

-- return model:cuda()
return model
