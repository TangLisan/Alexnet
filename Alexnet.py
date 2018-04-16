# coding=utf-8

import tensorflow as tf


def Alexnet(input, train, regularizer):
    with tf.variable_scope('layer1_conv1'):
        w_1 = tf.get_variable("weight", [5, 5, 1, 32],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_1 = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input, w_1, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b_1))

        tf.summary.histogram('layer1_weight', w_1)
        tf.summary.histogram('layer1_bias', b_1)
    with tf.name_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3_conv2'):

        w_2 = tf.get_variable("weight", [5, 5, 32, 64],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_2 = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, w_2, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b_2))

        tf.summary.histogram('layer3_weight', w_2)
        tf.summary.histogram('layer3_bias', b_2)

    with tf.name_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 在这里变换成向量的形式看看会不会有所不同。
    pool_shape = pool2.get_shape().as_list()
    num = pool_shape[1] * pool_shape[2] * pool_shape[3]
    repool2 = tf.reshape(pool2, [pool_shape[0], num])

    with tf.variable_scope('layer5_fc1'):
        w_fc1 = tf.get_variable("weight", [num, 512],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(w_fc1))
        b_fc1 = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(repool2, w_fc1) + b_fc1)
        tf.summary.histogram('layer5_fc1_weight', w_fc1)
        tf.summary.histogram('layer5_fc1_bias', b_fc1)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6_fc2'):
        w_fc2 = tf.get_variable("weight", [512, 10],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(w_fc2))
        b_fc2 = tf.get_variable("bias", [10], initializer=tf.constant_initializer(0.1))
        tf.summary.histogram('layer6_fc2_weight', w_fc2)
        tf.summary.histogram('layer3_fc2_bias', b_fc2)
        fc2 = tf.matmul(fc1, w_fc2) + b_fc2

    # 网络的最后的输出不能使经过激活函数的结果，不然就会最后收敛到零
    return fc2
