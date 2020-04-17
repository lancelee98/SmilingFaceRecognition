#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import time
import utils.tfdata_tools as tt
import matplotlib.pyplot as plt

learning_rate = 0.01


def hidden_layer(input_tensor, resuse=tf.AUTO_REUSE):
    with tf.variable_scope("C1-conv", reuse=resuse):
        filter1 = tf.get_variable("weight", [5, 5, 1, 12],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias1 = tf.get_variable("bias", [12], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, filter1, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))

    with tf.name_scope("S2-max_pool", ):
        maxPool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("C3-conv", reuse=resuse):
        filter2 = tf.get_variable("weight", [5, 5, 12, 24],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias2 = tf.get_variable("bias", [24], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(maxPool1, filter2, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))

    with tf.name_scope("S4-max_pool", ):
        maxPool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 3, 2, 1], padding='VALID')

    with tf.variable_scope("C5-conv", reuse=resuse):
        filter3 = tf.get_variable("weight", [5, 5, 24, 120],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias3 = tf.get_variable("bias", [120], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(maxPool2, filter3, strides=[1, 1, 1, 1], padding='SAME')  # 5*5
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, bias3))

    with tf.variable_scope("layer6-full1", reuse=resuse):
        Full_connection1_weights = tf.get_variable("weight", [5 * 5 * 120, 60],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        Full_connection1_bias = tf.get_variable("bias", [60], initializer=tf.constant_initializer(0.0))
        # 将卷积的输出展开
        flat = tf.reshape(relu3, [-1, 5 * 5 * 120])
        # 神经网络计算，并添加sigmoid激活函数
        Full_1 = tf.nn.relu(tf.matmul(flat, Full_connection1_weights) + Full_connection1_bias)

    with tf.variable_scope("layer7-full2", reuse=resuse):
        Full_connection2_weights = tf.get_variable("weight", [60, 2],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        Full_connection2_bias = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.0))
        y_conv = tf.nn.softmax(tf.matmul(Full_1, Full_connection2_weights) + Full_connection2_bias, name="y_conv")
    return y_conv


x_image = tf.placeholder(tf.float32, [None, 32, 20, 1], name="x-input")  # 32*20
y_ = tf.placeholder(tf.float32, [None, 2], name="y-input")  # 0 1 0号位置为1代表不笑 1号位置为1代表笑

y_pred = hidden_layer(x_image)
# 最后是交叉熵作为损失函数，使用梯度下降来对模型进行训练
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=tf.argmax(y_, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 测试正确率
average_y = hidden_layer(x_image)
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=average_y, labels=tf.argmax(y_, 1)))
correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))  # 求出最大值的坐标并做对比
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# train capacity=10000 smile=5616 not_smile=4384
image_batch, label_batch = tt.get_tfrecord(200, '..//data/firstGround//train//smile_train.tfrecords', capacity=10000)
# train capacity=10000 smile=5616 not_smile=4384
image_total, label_total = tt.get_tfrecord(10000, '..//data/firstGround//train//smile_train.tfrecords', capacity=10000)
# test capacity=2995 smile=1632 not_smile=1363
image_batch_test, label_batch_test = tt.get_tfrecord(2995, '..//data/firstGround//test//smile_test.tfrecords',
                                                     capacity=2995)
save_model = './model/first_ground_model.ckpt'


def draw_leaning_line(iteration_time, training_cost, testing_cost):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(iteration_time, training_cost, 'r', label='training_cost')
    ax.plot(iteration_time, testing_cost, label='testing_cost')
    ax.legend(loc=1)
    ax.set_xlabel('iteration_time')
    ax.set_ylabel('cost')
    ax.set_title('training_cost & testing_cost')
    plt.show()


def draw_leaning_line2(iteration_time, train_accuracy, test_accuracy):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(iteration_time, train_accuracy, 'r', label='train_accuracy')
    ax.plot(iteration_time, test_accuracy, label='test_accuracy')
    ax.legend(loc=1)
    ax.set_xlabel('iteration_time')
    ax.set_ylabel('accuracy')
    ax.set_title('train_accuracy & test_accuracy')
    plt.show()


def trian():
    training_cost, testing_cost, iteration_time, train_accuracy, test_accuracy = [], [], [], [], []
    # tf.add_to_collection('y_pred',average_y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 进行训练
        start_time = time.time()
        b_image_test, b_label_test = sess.run([image_batch_test, label_batch_test])
        b_image_total, b_label_total = sess.run([image_total, label_total])
        for i in range(10):
            # 取训练数据
            b_image, b_label = sess.run([image_batch, label_batch])
            # 训练数据
            train_step.run(feed_dict={x_image: b_image, y_: b_label})
            # 每迭代10个batch,对当前训练数据进行测试，输出当前预测准确率
            if i % 2 == 0:
                trian_c = cost.eval(feed_dict={x_image: b_image_total, y_: b_label_total})
                # print("step %d, training trian_cost %g" % (i, trian_c))
                test_c = cost.eval(feed_dict={x_image: b_image_test, y_: b_label_test})
                # print("total test cost %g" % test_c)
                train_a = accuracy.eval(feed_dict={x_image: b_image_total, y_: b_label_total})
                # print("step %d, training accuracy %g" % (i, train_accuracy))
                test_a = accuracy.eval(feed_dict={x_image: b_image_test, y_: b_label_test})
                print("step %d,total test accuracy %g" % (i, test_a))
                training_cost.append(trian_c)
                testing_cost.append(test_c)
                train_accuracy.append(train_a)
                test_accuracy.append(test_a)
                iteration_time.append(i)
        draw_leaning_line(iteration_time, training_cost, testing_cost)
        draw_leaning_line2(iteration_time, train_accuracy, test_accuracy)
        end_time = time.time()
        print("total test accuracy %g" % test_a)
        print('time:', (end_time - start_time) / 60.0)
        saver = tf.train.Saver()
        saver.save(sess, save_model)
        coord.request_stop()
        coord.join(threads)


# trian()

