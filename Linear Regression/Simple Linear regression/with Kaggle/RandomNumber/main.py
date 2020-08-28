import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Random Number.csv')
data.info()

x = data['X']
y = data['Y']

w = tf.Variable(0.6659)  # print를 위해 임의로 조정
b = tf.Variable(0.7)


learning_rate = 0.000001

for i in range(10000):
    with tf.GradientTape() as tape:
        hypothesis = w*x+b
        cost = tf.reduce_mean(tf.square(hypothesis-y))

    w_grad, b_grad = tape.gradient(cost,[w,b])
    w.assign_sub(learning_rate * w_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 1000 == 0:
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, w.numpy(), b.numpy(), cost))

x2 = [-5,150,305]
plt.scatter(x,y)
plt.plot(x2,x2*w+b,c='r')
plt.show()