
import numpy as np
import theano
import theano.tensor as T
from matplotlib import pyplot as plt


def scan1(input):
  x = T.vector('x')

  def square(x):
    return x*x

  outputs, updates = theano.scan(
    fn=square,
    sequences=x,
    n_steps=x.shape[0],
  )

  square_op = theano.function(
    inputs=[x],
    outputs=[outputs],
  )

  return square_op(input)


def scan2(input):
  N = T.iscalar('N')

  def recurrence(n, fn_1, fn_2):
    return fn_1 + fn_2, fn_1

  outputs, updates = theano.scan(
    fn=recurrence,
    sequences=T.arange(N),
    n_steps=N,
    outputs_info=[1., 1.]
  )

  fibonacci = theano.function(
    inputs=[N],
    outputs=outputs,
  )

  return fibonacci(input)

def scan3(inputs):
  decay = T.scalar('decay')
  sequence = T.vector('sequence')

  def recurrence(x, last, decay):
    return (1 - decay) * x + decay * last

  outputs, _ = theano.scan(
    fn=recurrence,
    sequences=sequence,
    n_steps=sequence.shape[0],
    outputs_info=[np.float64(0)],
    non_sequences=[decay]
  )

  lpf = theano.function(
    inputs=[sequence, decay],
    outputs=outputs,
  )

  return lpf(inputs[0], inputs[1])


if __name__ == '__main__':
  print('\nRunning scan1...')
  o_val = scan1(np.array([1, 2, 3, 4, 5]))
  print("output:", o_val)

  print('\nRunning scan2...')
  o_val = scan2(8)
  print("output:", o_val)

  print('\nRunning scan3...')
  X = 2 * np.random.randn(300) + np.sin(np.linspace(0, 3 * np.pi, 300))
  plt.plot(X)
  plt.title("original")
  plt.show()
  Y = scan3((X, 0.99))
  print("output:", Y)
  plt.plot(Y)
  plt.title("filtered")
  plt.show()