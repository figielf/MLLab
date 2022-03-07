import numpy as np
import theano
import theano.tensor as T


def invoke_simple_function():
    c = T.scalar('c')
    v = T.vector('v')
    A = T.matrix('A')

    # definition of operation, no calculation will happen now
    w = A.dot(v)

    # creates computation graph creation, still no calculation will be done here
    # the graph to evaluate / function to calculate will take 'inputs' and will produce 'outputs'
    matrix_times_vector = theano.function(inputs=[A, v], outputs=w)

    A_val = np.array([[1, 2], [3, 4]])
    v_val = np.array([5, 6])

    # graph execution, this will do actual calculation
    w_val = matrix_times_vector(A_val, v_val)
    print(w_val)


def find_minimum_of_simple_function():
    # create shared variable which will be updated as a result of theano function execution
    # the first argument is its initial value, the second is its name
    x = theano.shared(20.0, 'x')

    cost = x*x + x + 1

    # T.grad will return a gradient calculated automatically
    x_update = x - 0.3*T.grad(cost, x)

    # create computation graph taking no 'inputs', producing 'outputs' and updating parameters from 'updates'
    # updates takes in a list of tuples of (the shared variable to update, the update expression)
    train = theano.function(inputs=[], outputs=cost, updates=[(x, x_update)])

    for i in range(25):
        # invoke computation graph
        cost_val = train()
        print(cost_val)

    # print the optimal value of x
    print(x.get_value())


if __name__ == '__main__':
    # INVOKE SIMPLE FUNCTION In THEANO
    invoke_simple_function()

    # FIND MINIMUM OF THE SIMPLE FUNCTION In THEANO
    find_minimum_of_simple_function()
