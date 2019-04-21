import tensorflow as tf
tf.enable_eager_execution()

x = [1, 2, 3, 4]
y = [1, 2, 3, 4]

W = tf.Variable(2.9) # initial weight
b = tf.Variable(0.5) # initial bias

# learning rate
lr = 0.01

# cost function -> MSE
# minimize cost(loss) function to find the proper weight and bias
# loss = tf.reduce_mean(tf.square(h-y)) # reduce_* : 차원을 줄임. ex) rank(1) -> rank(0)

for epoch in range(100+1): # W, b update
    # Gradient descent
    with tf.GradientTape() as tape: # Record operations for automaatic differentiation
        h = W * x + b # hypothesis
        loss = tf.reduce_mean(tf.square(h - y))

    W_grad, b_grad = tape.gradient(loss, [W, b])
    W.assign_sub(lr * W_grad)
    b.assign_sub(lr * b_grad)
    if epoch % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(epoch, W.numpy(), b.numpy(), loss))

# Test
print("Test our model!")
while True:
    test_val = input()
    if test_val == 'exit':
        break
    pred = W * int(test_val) + b
    print('input: {} \nprediction: {}'.format(test_val, pred))