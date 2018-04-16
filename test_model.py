# coding=utf-8
"""
参考资料

http://blog.csdn.net/sinat_29957455/article/details/78508793
moving average
"""
import numpy as np
import tensorflow as tf

import input_data
from MNIST_Alexnet import Alexnet, mnist_train


def test_model(mnist):
    # 在这里从新处理，主要是考虑使用循环的方法对单个数据传入网路

    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [mnist.validation.num_examples, 28, 28, 1])
        y_ = tf.placeholder(tf.float32, [mnist.validation.num_examples, 10])
        # 在这里怎样处理单个数据传入网络

        reshaped_xs = np.reshape(mnist.validation.images, [mnist.validation.num_examples, 28, 28, 1])

        validate_feed = {x: reshaped_xs, y_: mnist.validation.labels}

        y = Alexnet.Alexnet(x, False, None)

        # translate the y into 1 or 0


        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                y = sess.run(y, feed_dict={x: reshaped_xs})
                for i in range(100):
                    print y[i]

                print("After %s training step(s),validation accuracy= %g " % (global_step, accuracy_score))
            else:
                print ("not found the checkpoint")
                return


def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    test_model(mnist)


if __name__ == '__main__':
    tf.app.run()
