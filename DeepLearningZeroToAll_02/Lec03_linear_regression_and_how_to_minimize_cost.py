import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


# X = np.array([1, 2, 3])
# Y = np.array([1, 2, 3])

# Cost function in pure Python
def cost_func_python(W, X, Y):
    c = 0
    for i in range(len(X)):
        c += (W * X[i] - Y[i]) ** 2
    return c / len(X)

def use_pure_python():
    for feed_W in np.linspace(-3, 5, 15):
        curr_cost = cost_func_python(feed_W, X, Y)
        print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))


# Cost function using tensorflow
def cost_func_tf(W, X, Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))

def use_tensorflow():
    W_values = np.linspace(-3, 5, 15)
    cost_values = []

    for feed_W in W_values:
        curr_cost = cost_func_tf(feed_W, X, Y)
        cost_values.append(curr_cost)
        print("{:6.3f} | {}".format(feed_W, curr_cost))


# Apply Gradient descent
def apply_gd():
    # tf.set_random_seed(42)
    X = [1., 2., 3., 4.]
    Y = [1., 3., 5., 7.]

    W = tf.Variable([1.0])

    for step in range(3000):
        hypothesis = W * X
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

        lr = 0.01
        gradient = tf.reduce_mean(tf.multiply(tf.multiply(2 * W, X) - Y, X))
        W = W - tf.multiply(lr, gradient)
        # W.assign(descent)

        if step % 10 == 0:
            print('{:5} | {:10.4f} | {:10.6}'.format(step, cost.numpy(), W.numpy()[0]))


if __name__ == '__main__':
    apply_gd()


