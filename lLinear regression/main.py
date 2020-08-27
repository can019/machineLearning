import tensorflow as tf

W = tf.Variable([5.0])  # w의 초기값 :: w=5부터 경사하강 시작
X = [1., 2., 3., 4.]  # data set
Y = [1., 3., 5., 7.]  # label set

for step in range(300):  # 300번 훈련을 하겠다.
    hypothesis = W * X  # 가설
    cost = tf.reduce_mean(tf.square(hypothesis - Y))  # cost 정의

    alpha = 0.01  # learning rate
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X)-Y, X))  # 미분
    descent = W - tf.multiply(alpha, gradient)  # 미분값을 토대로 w값 이동. descent에 담아놓음.
    W.assign(descent)  # 변경된 W값 적용
    # 13, 14 line을 합쳐 W.assign(W - tf.multiply(alpha, gradient))로 표현해도 된다.
    # W는 텐서플로우 변수기 때문에 assign을 통하여 값을 바꿔준다.
    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))
print('\n Model :: H(x) ={:10.6f}x'.format(W.numpy()[0]))

# library 함수 설명 ::
# tf.multiply : 곱
# tf.reduce_mean : 평균
