# coding=utf-8

"""
------------------------------------------------------------------------------------------------------
checkpoint
http://www.360doc.com/content/17/0314/10/10408243_636714291.shtml
from checkpoint restory training
------------------------------------------------------------------------------------------------------
moving average
http://blog.csdn.net/sinat_29957455/article/details/78508793
http://blog.csdn.net/uestc_c2_403/article/details/72235334
moving average 是真实变量值的副本，但是这个副本是和真实的之间有差异的。要想获得变量的滑动变量的值就要使用滑动平均的办法。
要获取滑动平均的保存文件，
使用tensorboard 可以看到整个运行图的结构和每一个变量的变化情况。
监控指标可视化。
关于监控指标的可视化问题 http://blog.csdn.net/gsww404/article/details/78605784
------------------------------------------------------------------------------------------------------
"""

import os

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data 
from Alexnet import Alexnet

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 28, 28, 1])
    # lable
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    tf.summary.image('image', x, 100)
    y = Alexnet.Alexnet(x, True, regularizer)

    tf.summary.image('image',x,100)
    global_step = tf.Variable(0, trainable=False)
    print (tf.trainable_variables())
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)

    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('loss', loss)
    print (tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    tf.summary.scalar('learning_rate', learning_rate)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./model/Mlog", tf.get_default_graph())

        tf.global_variables_initializer().run()
        #        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, 28, 28, 1))


            # print sess.run(y, feed_dict={x: xs})

            summary, _, loss_value, step = sess.run([merged, train_op, loss, global_step],
                                                    feed_dict={x: reshaped_xs, y_: ys})

            writer.add_summary(summary, i)
            if i % 1000 == 0:
                print ("After %d training step(s),loss on training batch is %g." % (step, loss_value))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
    writer.close()


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
