import tensorflow as tf

W = tf.Variable([5.0])  # w의 초기값 :: w=5부터 경사하강 시작
X = [1.,2.,3.,4.]  # data set
Y = [1.,3.,5.,7.]  # label set

for step in range(300):  # 300번 훈련을 하겠다.
    hypothesis = W * X  # 가설
    cost = tf.reduce_mean(tf.square(hypothesis - Y))  # cost 정의

    alpha = 0.01  # learning rate
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X)-Y, X))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)
    if step % 10 ==0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))
print('\n Model :: H(x) ={:10.6f}x'.format(W.numpy()[0]))
