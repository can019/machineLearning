import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Salary_Data.csv')
data.info()
experience = data['YearsExperience']
salary = data['Salary']

w = tf.Variable(2000.0)
b = tf.Variable(500.0)
learning_rate = 0.00001
for i in range(1000000):
    with tf.GradientTape() as tape:
        hypothesis = w*experience + b
        cost = tf.reduce_mean(tf.square(hypothesis-salary))

    w_grad, b_grad = tape.gradient(cost, [w,b])
    w.assign_sub(learning_rate * w_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 10000 == 0:
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, w.numpy(), b.numpy(), cost))

x2 = [1.0, 11.0]
plt.scatter(experience, salary)
plt.plot(x2,x2*w+b, c= 'r')
plt.show()
