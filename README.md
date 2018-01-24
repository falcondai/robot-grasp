# Robotic grasp

Use learning to solve the robotic grasp problem.

## approach

We factored the system into 3 components:

- proposer: take in an image (RGB+D) then propose a grasp to be evaluated
- evaluator: score the grasps
- executor: execute the recommended grasps

and we will learn/optimize each component using data collected on and off a robot (Baxter).

## datasets

```
train set
{
  1 : DoubleTensor - size: 6415x3x128x128
  2 : DoubleTensor - size: 6415x1x128x128
}
validation set
{
  1 : DoubleTensor - size: 1604x3x128x128
  2 : DoubleTensor - size: 1604x1x128x128
}

```

### training the grasp evaluator

To train a reasonable grasp evaluator, we treated the evaluation problem as a binary classification task and used the publicly available [Cornell grasp dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php) to train off the robot. Here are some of statistics of this dataset:

- image count: 885
- labeled grasps count: 8019
- positive: 5110 (64%)
- negative: 2909 (36%)
- object count: 244
- object category count: 93

## usage

```
python2 preprocess.py
python2 split.py
th prepareData.lua
python2 train.py logs 64 10
```

### use pre-trained models

### training


## author

Falcon Dai (dai@ttic.edu)

### contributors

[Andrew Hundt](ATHundt@gmail.com)