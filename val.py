import tensorflow as tf
# TODO load from files using queue
import torchfile

import numpy as np
import time, sys

from model import build_model
from util import *

# constants
width = 128
loss_lambda = 0.1
checkpoint_dir = sys.argv[1]

# model
# grasp_class_prediction, depth_prediction, logit, grasp_image_ph, keep_prob_ph = build_model(width)
grasp_class_prediction, logit, grasp_image_ph, keep_prob_ph = build_model(width)
depth_image_ph =  tf.placeholder('float', [None, width, width, 1])
grasp_class_ph =  tf.placeholder('int64', [None])

# loss
grasp_class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit, grasp_class_ph), name='grasp_class_loss')
# depth_loss = tf.reduce_mean(tf.square(depth_image_ph - depth_prediction), name='depth_loss')
# combined_loss = (1. - loss_lambda) * grasp_class_loss + loss_lambda * depth_loss
combined_loss = grasp_class_loss

# evaluation
batch = int(sys.argv[2])
correct_prediction = tf.equal(tf.argmax(grasp_class_prediction, 1), grasp_class_ph)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
with tf.Session() as sess:
    restore_vars(saver, sess, checkpoint_dir)
    val_data = torchfile.load('val.t7')
    n = len(val_data['x'][0])
    print '%d samples' % n
    acc = 0.
    loss = 0.
    for i in xrange(n / batch + 1):
        start = batch * i
        if n == start:
            break
        end = min(start + batch, n)
        rgb_image = val_data['x'][0][start:end].transpose(0, 2, 3, 1)
        grasp_class = val_data['y'][start:end] - 1
        eval_feed = {
            grasp_image_ph: rgb_image,
            grasp_class_ph: grasp_class,
            # depth_image_ph: d_image,
            keep_prob_ph: 1.,
        }

        loss += combined_loss.eval(feed_dict=eval_feed) * (end - start)
        acc += accuracy.eval(feed_dict=eval_feed) * (end - start)
    print acc / n, loss / n
