import numpy as np
import tensorflow as tf


def invoke_simple_function():
    # create static variable placeholders, shape and name are optional
    A = tf.placeholder(tf.float32, shape=(5, 5), name='A')
    v = tf.placeholder(tf.float32)

    # define computation expression, no calculation will happen here
    w = tf.matmul(A, v)

    # create session to reserve memory for variables on graph execution
    with tf.Session() as session:
        # execute graph with defined calculations by passing paras in feed_dict
        # v needs to be of shape=(5, 1) not just shape=(5,)
        output = session.run(w, feed_dict={A: np.random.randn(5, 5), v: np.random.randn(5, 1)})

        # numpy array will be returned
        print(output, type(output))


def find_minimum_of_simple_function():
    shape = (2, 2)

    # create variables that will contain data that will be updated during graph execution
    # a tf variable can be initialized with anything that can be turned into a tf tensor (eg. numpy array or a tf array)
    x = tf.Variable(tf.random_normal(shape))
    # x = tf.Variable(np.random.randn(2, 2))
    t = tf.Variable(0)

    # define variable initialization operation
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        # variables has to be initialized (instantiated)
        # execute variable initialization operation
        out = session.run(init)
        print(out)

        # get data values from tensors
        print(x.eval())
        print(t.eval())


    u = tf.Variable(20.0)
    cost = u*u + u + 1.0

    # define computation graph by setting optimizer with learning rate of 0.3 which minimizes 'cost' function
    train_op = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

    # run a session
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        # Strangely, while the weight update is automated, the loop itself is not.
        # So we'll just call train_op until convergence.
        # This is useful for us anyway since we want to track the cost function.
        for i in range(12):
            # do one optimization step by execution computation graph and update parameters according to chosen optimizer
            session.run(train_op)
            print(f'epoch:{i}, cost = {cost.eval()}, parameter value = {u.eval()}')


if __name__ == '__main__':
    # INVOKE SIMPLE FUNCTION In THEANO
    invoke_simple_function()

    # FIND MINIMUM OF THE SIMPLE FUNCTION In THEANO
    find_minimum_of_simple_function()
