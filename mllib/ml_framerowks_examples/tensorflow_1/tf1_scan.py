import numpy as np
import tensorflow as tf1
import matplotlib.pyplot as plt


def scan1():
    # sequence of elements we want to square
    x = tf1.placeholder(tf1.int32, shape=(None,), name='x')

    # thing to do to every element of the sequence
    # notice how it always ignores the last output
    def square(last, current):
        return current * current

    # this is a "fancy for loop"
    # it says: apply square to every element of x
    square_op = tf1.scan(
        fn=square,
        elems=x,
    )

    # run it!
    with tf1.Session() as session:
        o_val = session.run(square_op, feed_dict={x: [1, 2, 3, 4, 5]})
        print("output:", o_val)


def scan2():
    # N = number of fibonacci numbers we want
    # shape=() means scalar
    N = tf1.placeholder(tf1.int32, shape=(), name='N')

    # recurrence and loop
    # notice how we don't use current_input at all!
    def recurrence(last_output, current_input):
        return (last_output[1], last_output[0] + last_output[1])

    fibonacci = tf1.scan(
        fn=recurrence,
        elems=tf1.range(N),
        initializer=(0, 1),
    )

    # run it!
    with tf1.Session() as session:
        o_val = session.run(fibonacci, feed_dict={N: 8})
        print("output:", o_val)


def scan3():
    # original sequence is a noisy sine wave
    original = np.sin(np.linspace(0, 3 * np.pi, 300))
    X = 2 * np.random.randn(300) + original
    plt.plot(X)
    plt.title("original")
    plt.show()

    # set up placeholders
    decay = tf1.placeholder(tf1.float32, shape=(), name='decay')
    sequence = tf1.placeholder(tf1.float32, shape=(None,), name='sequence')

    # the recurrence function and loop
    def recurrence(last, x):
        return (1.0 - decay) * x + decay * last

    lpf = tf1.scan(
        fn=recurrence,
        elems=sequence,
        initializer=0.0,  # sequence[0] to use the first value of the sequence
    )

    # run it!
    with tf1.Session() as session:
        Y = session.run(lpf, feed_dict={sequence: X, decay: 0.97})

        plt.plot(Y)
        plt.plot(original)
        plt.title("filtered")
        plt.show()


if __name__ == '__main__':
    print('\nRunning scan1...')
    scan1()

    print('\nRunning scan2...')
    scan2()

    print('\nRunning scan3...')
    scan3()
