import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd

# data preprocessing
data = pd.read_csv('kc_house_data.csv')
newdata = data[['bedrooms','bathrooms','sqft_living','sqft_lot', 'floors', 'waterfront','view','price']]
numpyData = np.array(newdata,dtype='float32')

# slice data
x = numpyData[:,:-1]
y = numpyData[:,[-1]]


w = tf.Variable(tf.random.normal([7,1]))  # [입력 feature 수, 출력 feature 수]
b = tf.Variable(tf.random.normal([1]))
print(w.shape)
def predict(x):
    return tf.matmul(x,w) + b  # 반드시 x먼저 와야함. matmul은 행렬곱

learning_rate = 0.000000000001

n_epochs = 200000000
for i in range(n_epochs+1):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean(tf.square(predict(x)-y))

        w_grad, b_grad = tape.gradient(cost, [w,b])

        w.assign_sub(learning_rate * w_grad)
        b.assign_sub(learning_rate * b_grad)

        if i % 1000 ==0:
            print('{:5} | {:10.4f}'.format(i,cost.numpy()))
