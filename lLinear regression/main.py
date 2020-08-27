import tensorflow as tf

W = tf.Variable([5.0])
X = [1.,2.,3.,4.]
Y = [1.,3.,5.,7.]
for step in range(300):
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) -Y ,X))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)
    if step % 10 ==0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))