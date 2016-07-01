import tensorflow as tf
import numpy as np

def build_model(width):
    with tf.name_scope('model'):
        grasp_image_ph = tf.placeholder('float', [None, width, width, 3])
        keep_prob_ph = tf.placeholder('float', name='dropout')

        # rgb conv
        a1 = tf.contrib.layers.convolution2d(tf.nn.dropout(grasp_image_ph / 255., keep_prob_ph), 32, (3, 3), tf.nn.relu, weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv1')
        # a2 = tf.contrib.layers.convolution2d(tf.nn.dropout(a1, keep_prob_ph), 64, (3, 3), tf.nn.relu, stride=(2, 2), weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv2')
        # a3 = tf.contrib.layers.convolution2d(tf.nn.dropout(a2, keep_prob_ph), 64, (3, 3), tf.nn.relu, stride=(2, 2), weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv3')
        # a4 = tf.contrib.layers.convolution2d(tf.nn.dropout(a3, keep_prob_ph), 32, (3, 3), tf.nn.relu, stride=(2, 2), weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv4')

        a2 = tf.contrib.layers.convolution2d(tf.nn.dropout(a1, keep_prob_ph), 64, (3, 3), tf.nn.relu, weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv2')
        a2_max = tf.nn.max_pool(a2, (1, 3, 3, 1), (1, 2, 2, 1), 'SAME')
        a3 = tf.contrib.layers.convolution2d(tf.nn.dropout(a2_max, keep_prob_ph), 64, (3, 3), tf.nn.relu, weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv3')
        a3_max = tf.nn.max_pool(a3, (1, 3, 3, 1), (1, 2, 2, 1), 'SAME')
        a4 = tf.contrib.layers.convolution2d(tf.nn.dropout(a3_max, keep_prob_ph), 32, (3, 3), tf.nn.relu, weight_init=tf.contrib.layers.xavier_initializer_conv2d(), name='conv4')
        a4_max = tf.nn.max_pool(a4, (1, 3, 3, 1), (1, 2, 2, 1), 'SAME')

        conv = a4_max

        # flatten
        conv_shape = conv.get_shape().as_list()
        flat_dim = np.product(conv_shape[1:])
        print 'final shape', conv_shape, 'flat_dim', flat_dim
        conv_flat = tf.reshape(conv, [-1, flat_dim])

        # fc
        fc1 = tf.contrib.layers.fully_connected(conv_flat, 2, weight_init=tf.contrib.layers.xavier_initializer(), name='fc1')

        # depth
        with tf.variable_scope('depth'):
            batch_size = tf.shape(grasp_image_ph)[0]
            deconv_shape1 = tf.pack([batch_size, 64, 64, 16])
            f1 = tf.Variable(tf.contrib.layers.xavier_initializer_conv2d()([5, 5, 16, 64]), name='features1')
            d1 = tf.nn.conv2d_transpose(a3, f1, deconv_shape1, (1, 1, 1, 1), name='deconv1')
            deconv_shape2 = tf.pack([batch_size, width, width, 1])
            f2 = tf.Variable(tf.contrib.layers.xavier_initializer_conv2d()([5, 5, 1, 16]), name='features2')
            d2 = tf.nn.conv2d_transpose(tf.nn.relu(d1), f2, deconv_shape2, (1, 2, 2, 1), name='deconv2')

        # prediction
        logit = fc1
        grasp_class_prediction = tf.nn.softmax(fc1)
        depth_prediction = d2

        return grasp_class_prediction, depth_prediction, logit, grasp_image_ph, keep_prob_ph
