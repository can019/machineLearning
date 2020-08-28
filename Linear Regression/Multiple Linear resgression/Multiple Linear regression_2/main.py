import tensorflow as tf
import numpy as np

data = np.array([
    # x1  x2   x3    y
    [73., 80., 75., 152.],
    [93., 88., 93., 185.],
    [89., 91., 90., 180.],
    [96., 98., 100., 196.],
    [73., 66., 70., 142.]
], dtype = np.float32)

# slice data
x = data[:,:-1]
y = data[:,[-1]]

w = tf.Variable(tf.random.normal([3,1]))  # [입력 feature 수, 출력 feature 수]
b = tf.Variable(tf.random.normal([1]))
def predict(x):
    return tf.matmul(x,w) + b  # 반드시 x먼저 와야함. matmul은 행렬곱

learning_rate = 0.000001

n_epochs = 2000

for i in range(n_epochs+1):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean(tf.square(predict(x)-y))

        w_grad, b_grad = tape.gradient(cost, [w,b])

        w.assign_sub(learning_rate * w_grad)
        b.assign_sub(learning_rate * b_grad)

        if i % 100 ==0:
            print('{:5} | {:10.4f}'.format(i,cost.numpy()))