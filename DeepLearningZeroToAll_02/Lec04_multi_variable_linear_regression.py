import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


def test1():
    # Test1: hypothesis not using matrix
    # data and label
    x1 = [73., 93., 89., 96., 73.]
    x2 = [80., 88., 91., 98., 66.]
    x3 = [75., 93., 90., 100., 70.]
    Y = [152., 185., 180., 196., 142.]

    # random weights
    w1 = tf.Variable(tf.random_normal([1]))
    w2 = tf.Variable(tf.random_normal([1]))
    w3 = tf.Variable(tf.random_normal([1]))
    b = tf.Variable(tf.random_normal([1]))

    lr = 0.000001
    epochs = 1000

    for epoch in range(epochs+1):
        # tf.GradientTape() to record the gradient of the cost function
        with tf.GradientTape() as tape:
            h = w1 * x1 + w2 * x2 + w3 * x3 + b
            cost = tf.reduce_mean(tf.square(h - Y))

        # Calculates the gradients of the cost
        w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])

        # Update w1, w2, w3 and b
        w1.assign_sub(lr * w1_grad)
        w2.assign_sub(lr * w2_grad)
        w3.assign_sub(lr * w3_grad)
        b.assign_sub(lr * b_grad)

        if epoch % 50 == 0:
            print("{:5} | {:12.4f}".format(epoch, cost.numpy()))


def test2():
    # Test2: hypothesis using matrix
    data = np.array([
        # x1, x2, x3, y
        [73., 80., 75., 152.],
        [93., 88., 93., 185.],
        [89., 91., 90., 180.],
        [96., 98., 100., 196.],
        [73., 66., 70., 142.]
    ], dtype=np.float32)

    # slice data
    X = data[:, :-1] # shape: (5, 3)
    y = data[:, [-1]] # shape: (5, 1)

    W = tf.Variable(tf.random_normal([3, 1]))
    b = tf.Variable(tf.random_normal([1]))
    lr = 0.000001
    epochs = 2000

    def predict(X):
        return tf.matmul(X, W) + b

    for epoch in range(epochs+1):
        with tf.GradientTape() as tape:
            cost = tf.reduce_mean((tf.square(predict(X) - y)))

        W_grad, b_grad = tape.gradient(cost, [W, b])

        W.assign_sub(lr * W_grad)
        b.assign_sub(lr * b_grad)

        if epoch % 50 == 0:
            print("{:5} | {:12.4f}".format(epoch, cost.numpy()))

if __name__ == '__main__':
    test2()