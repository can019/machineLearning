import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

# plt.plot(x_data, y_data, 'o')
# plt.xlim(0, 5.3)  # x축 범위
# plt.ylim(0, 5.3)  # y축 범위
# plt.show()

w = tf.Variable(3.0)
b = tf.Variable(1.0)

learning_rate = 0.01

for i in range(20000):
    with tf.GradientTape() as tape:  # library로 gradient 구하기
        hypothesis = w * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis-y_data))

    w_grad, b_grad = tape.gradient(cost, [w,b])

    w.assign_sub(learning_rate * w_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0 :
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, w.numpy(), b.numpy(), cost))
