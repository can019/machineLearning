import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
Height = data['Height']
Weight = data['Weight']
print(Height)
print(Weight)
w = tf.Variable(5.0)
b = tf.Variable(5.0)

learning_rate = 0.01


for i in range (100000):
    with tf.GradientTape() as tape:
        hypothesis = w*Height + b
        cost = tf.reduce_mean(tf.square(hypothesis-Weight))

    w_grad, b_grad = tape.gradient(cost,[w,b])
    w.assign_sub(learning_rate*w_grad)
    b.assign_sub(learning_rate*b_grad)

    if i % 100 == 0:
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, w.numpy(), b.numpy(), cost))
