import tensorflow as tf
# TODO load from files using queue
import torchfile

import numpy as np
import time

from model import build_model
from util import *

# constants
width = 128
loss_lambda = 0.1
checkpoint_dir = 'checkpoints-dev'

# model
grasp_class_prediction, depth_prediction, logit, grasp_image_ph, keep_prob_ph = build_model(width)
depth_image_ph =  tf.placeholder('float', [None, width, width, 1])
grasp_class_ph =  tf.placeholder('int64', [None])

# loss
grasp_class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logit, grasp_class_ph), name='grasp_class_loss')
depth_loss = tf.reduce_mean(tf.square(depth_image_ph - depth_prediction), name='depth_loss')
combined_loss = (1. - loss_lambda) * grasp_class_loss + loss_lambda * depth_loss

# optimization
batch = 32
n_eval_interval = 1
n_train_step = 10**3
global_step = tf.Variable(0, trainable=False, name='global_step')
initial_learning_rate = 0.5
decay_steps = 64
decay_rate = 0.9
momentum = 0.5
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(combined_loss, global_step=global_step)
# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(grasp_class_loss, global_step=global_step)
# train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(grasp_class_loss, global_step=global_step)

# evaluation
correct_prediction = tf.equal(tf.argmax(grasp_class_prediction, 1), grasp_class_ph)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# summary
tf.scalar_summary('learning_rate', learning_rate)
tf.scalar_summary('grasp_loss', grasp_class_loss)
tf.scalar_summary('depth_loss', depth_loss)
tf.scalar_summary('loss', combined_loss)
tf.scalar_summary('accuracy', accuracy)
summary_op = tf.merge_all_summaries()

def main():
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    with tf.Session() as sess:
        tf.set_random_seed(1234)
        np.random.seed(123)

        writer = tf.train.SummaryWriter('tf-log/%d' % time.time(), sess.graph)

        restore_vars(saver, sess, checkpoint_dir)

        # load train data
        train_data = torchfile.load('train.t7')
        n = len(train_data['x'][0])
        for i in xrange(n_train_step):
            # start = i * batch % n
            # end = min(start + batch, n)
            # rgb_image = train_data['x'][0][start:end].transpose(0, 2, 3, 1)
            # grasp_class = train_data['y'][start:end] - 1

            ind = np.random.choice(n, batch, replace=False)
            rgb_image = train_data['x'][0][ind].transpose(0, 2, 3, 1)
            d_image = train_data['x'][1][ind].transpose(0, 2, 3, 1)
            grasp_class = train_data['y'][ind] - 1

            if i % n_eval_interval == 0:
                val_feed = {
                    grasp_image_ph: rgb_image,
                    grasp_class_ph: grasp_class,
                    depth_image_ph: d_image,
                    keep_prob_ph: 1.,
                }
                print 'grasp loss', grasp_class_loss.eval(feed_dict=val_feed)
                print 'depth loss', depth_loss.eval(feed_dict=val_feed)
                print 'accuracy', accuracy.eval(feed_dict=val_feed)
                writer.add_summary(sess.run(summary_op, feed_dict=val_feed), i)

            train_feed = {
                grasp_image_ph: rgb_image,
                grasp_class_ph: grasp_class,
                depth_image_ph: d_image,
                keep_prob_ph: 0.8,
            }
            train_op.run(feed_dict=train_feed)

if __name__ == '__main__':
    main()
