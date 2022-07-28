import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

from activations import softmax2
from scores import multiclass_cross_entropy, sparse_multiclass_cross_entropy
from tests.utils.nlp_data_utils import tokenize_brown_sentence, get_sequences_with_word2idx_from_brown_corpus, \
    get_sequences_from_sentences_given_word2idx, START_STR, END_STR, split_sentences, generate_random_sequence, \
    get_idx2word_mapping, idx_sqe2sentence


class simple_bigram_model:
    def __init__(self, n_states, start_seq_state, end_seq_state, smoothing=1):
        self.V = n_states
        self.smoothing = smoothing
        self.start_idx = start_seq_state
        self.end_idx = end_seq_state
        self.log_A = None

    def fit(self, x, add_start_end_tokens=True):
        A_count = np.ones((self.V, self.V)) * self.smoothing
        for seq in x:
            if add_start_end_tokens:
                seqence = [self.start_idx] + seq + [self.end_idx]
            for t in range(1, len(seqence)):
                A_count[seqence[t - 1], seqence[t]] += 1
        A = A_count / A_count.sum(axis=1, keepdims=True)
        self.log_A = np.log(A)

        loss = []
        for seq in x:
            seqence = seq
            if add_start_end_tokens:
                seqence = [self.start_idx] + seq + [self.end_idx]

            n_bigrams = len(seqence) - 1
            x_seq = np.zeros((n_bigrams, self.V))
            target = np.zeros((n_bigrams, self.V))
            x_seq[np.arange(n_bigrams), seqence[:-1]] = 1
            target[np.arange(n_bigrams), seqence[1:]] = 1

            y_hat = softmax2(x_seq.dot(self.log_A))
            loss.append(multiclass_cross_entropy(target, y_hat))

        return np.mean(loss)

    def seqence_prob(self, seq, add_start_end_tokens=True):
        seq_prob = 0
        seqence = seq
        if add_start_end_tokens:
            seqence = [self.start_idx] + seq + [self.end_idx]
        counter = 0
        for t in range(1, len(seqence)):
            seq_prob += self.log_A[seqence[t - 1], seqence[t]]
            counter += 1
        return seq_prob / counter


class logistic_bigram_model:
    def __init__(self, V, start_seq_state, end_seq_state):
        self.V = V
        self.start_idx = start_seq_state
        self.end_idx = end_seq_state
        self.W = np.random.randn(self.V, self.V) / np.sqrt(self.V)

    def fit(self, x, learning_rate=1e-1, epochs=1, add_start_end_tokens=True):
        n = len(x)
        history = []
        for epoch in range(epochs):
            x_train = shuffle(x)
            counter = 0
            for i, seq in enumerate(x_train):
                seqence = seq
                if add_start_end_tokens:
                    seqence = [self.start_idx] + seq + [self.end_idx]

                n_bigrams = len(seqence) - 1
                x_batch = np.zeros((n_bigrams, self.V))
                y_batch = np.zeros((n_bigrams, self.V))
                x_batch[np.arange(n_bigrams), seqence[:-1]] = 1
                y_batch[np.arange(n_bigrams), seqence[1:]] = 1

                predictions = softmax2(x_batch.dot(self.W))
                self.W = self.W - learning_rate * x_batch.T.dot(predictions - y_batch)

                y_hat = softmax2(x_batch.dot(self.W))
                loss = multiclass_cross_entropy(y_batch, y_hat)
                history.append(loss)
                if i % (n//100) == 0:
                    counter += 1
                    print(f'{counter}% done - loss: {np.mean(history[-100:])}')
        return history


class logistic_bigram_model_fast:
    def __init__(self, V, start_seq_state, end_seq_state):
        self.V = V
        self.start_idx = start_seq_state
        self.end_idx = end_seq_state
        self.W = np.random.randn(self.V, self.V) / np.sqrt(self.V)

    def fit(self, x, learning_rate=1e-1, epochs=1, add_start_end_tokens=True):
        n = len(x)
        history = []
        for epoch in range(epochs):
            x_train = shuffle(x)
            counter = 0
            for i, seq in enumerate(x_train):
                seqence = seq
                if add_start_end_tokens:
                    seqence = [self.start_idx] + seq + [self.end_idx]

                n_bigrams = len(seqence) - 1
                x_idx = seqence[:-1]
                y_idx = seqence[1:]
                x_batch = np.zeros((n_bigrams, self.V))
                x_batch[np.arange(n_bigrams), x_idx] = 1

                predictions = softmax2(self.W[x_idx])
                loss = sparse_multiclass_cross_entropy(y_idx, predictions)
                history.append(loss)
                if i % (n//100) == 0:
                    counter += 1
                    print(f'{counter}% done - loss: {np.mean(history[-100:])}')

                dOut = predictions
                dOut[np.arange(n_bigrams), y_idx] -= 1
                dW = x_batch.T.dot(dOut)
                self.W = self.W - learning_rate * dW
        return history


class nn_bigram_model:
    def __init__(self, V, hidden_size, start_seq_state, end_seq_state):
        self.V = V
        self.D = hidden_size
        self.start_idx = start_seq_state
        self.end_idx = end_seq_state
        self.W1 = np.random.randn(self.V, self.D) / np.sqrt(self.V)
        self.W2 = np.random.randn(self.D, self.V) / np.sqrt(self.D)

    def fit(self, x, learning_rate=1e-1, epochs=1, add_start_end_tokens=True):
        n = len(x)
        history = []
        for epoch in range(epochs):
            x_train = shuffle(x)
            counter = 0
            for i, seq in enumerate(x_train):
                seqence = seq
                if add_start_end_tokens:
                    seqence = [self.start_idx] + seq + [self.end_idx]

                n_bigrams = len(seqence) - 1
                x_batch = np.zeros((n_bigrams, self.V))
                y_batch = np.zeros((n_bigrams, self.V))
                x_batch[np.arange(n_bigrams), seqence[:-1]] = 1
                y_batch[np.arange(n_bigrams), seqence[1:]] = 1

                z = np.tanh(x_batch.dot(self.W1))
                predictions = softmax2(z.dot(self.W2))
                loss = multiclass_cross_entropy(y_batch, predictions)
                self.W2 = self.W2 - learning_rate * z.T.dot(predictions - y_batch)
                dz = (predictions - y_batch).dot(self.W2.T) * (1 - z * z)
                self.W1 = self.W1 - learning_rate * x_batch.T.dot(dz)

                #z_hat = np.tanh(x_batch.dot(self.W1))
                #y_hat = softmax2(z_hat.dot(self.W2))
                #loss = multiclass_cross_entropy(y_batch, y_hat)
                history.append(loss)
                if i % (n//100) == 0:
                    counter += 1
                    print(f'{counter}% done - loss: {np.mean(history[-100:])}')
        return history


class nn_bigram_model_fast:
    def __init__(self, V, hidden_size, start_seq_state, end_seq_state):
        self.V = V
        self.D = hidden_size
        self.start_idx = start_seq_state
        self.end_idx = end_seq_state
        self.W1 = np.random.randn(self.V, self.D) / np.sqrt(self.V)
        self.W2 = np.random.randn(self.D, self.V) / np.sqrt(self.D)

    def fit(self, x, learning_rate=1e-1, epochs=1, add_start_end_tokens=True):
        n = len(x)
        history = []
        for epoch in range(epochs):
            x_train = shuffle(x)
            counter = 0
            for i, seq in enumerate(x_train):
                seqence = seq
                if add_start_end_tokens:
                    seqence = [self.start_idx] + seq + [self.end_idx]

                n_bigrams = len(seqence) - 1
                x_batch = np.zeros((n_bigrams, self.V))
                x_idx = seqence[:-1]
                y_idx = seqence[1:]
                x_batch[np.arange(n_bigrams), x_idx] = 1

                z = np.tanh(self.W1[x_idx])
                predictions = softmax2(z.dot(self.W2))
                loss = sparse_multiclass_cross_entropy(y_idx, predictions)
                history.append(loss)
                if i % (n//100) == 0:
                    counter += 1
                    print(f'{counter}% done - loss: {np.mean(history[-100:])}')

                dOut = predictions
                dOut[np.arange(n_bigrams), y_idx] -= 1
                dW2 = z.T.dot(dOut)
                self.W2 = self.W2 - learning_rate * dW2
                dz = dOut.dot(self.W2.T) * (1 - z * z)

                # compare the speed of below
                np.subtract.at(self.W1, np.array(x_idx), learning_rate * dz)  # fast way of subtracting few times same rows
                # second option
                #self.W1 = self.W1 - learning_rate * x_batch.T.dot(dz)
                # third option
                #i = 0
                #for idx in x_idx: # don't include end token
                #    selfW1[idx] = selfW1[idx] - learning_rate * dz[i]
                #    i += 1

        return history


def print_results(sentences, sequences, word2idx, probs):
    idx2word = get_idx2word_mapping(word2idx)
    for i in range(len(sequences)):
        if sentences is not None:
            print('original brown sentence:\n', sentences[i])
        print('preprocessed sentence:\n', sequences[i])
        print('recreated sentences:\n', idx_sqe2sentence(sequences[i], idx2word))
        print('sentence log probability by model:\n', probs)


def smooth_loss(x, decay=0.99):
    y = np.zeros(len(x))
    last = 0
    for t in range(len(x)):
      z = decay * last + (1 - decay) * x[t]
      y[t] = z / (1 - decay ** (t + 1))
      last = z
    return y


if __name__ == '__main__':
    sequences, word2idx, sentences = get_sequences_with_word2idx_from_brown_corpus(n_vocab=2000)
    V = len(word2idx)
    print("Vocab size:", V)

    print('\nbigram markov model model:')
    bigram_mm_model = simple_bigram_model(V, start_seq_state=word2idx[START_STR], end_seq_state=word2idx[END_STR], smoothing=0.1)
    bigram_mm_loss = bigram_mm_model.fit(sequences)
    print(f'bigram markov model cross entropy loss: {bigram_mm_loss}')

    #print('\nsentences from brown training corpus:')
    #print_results(sentences[:3], sequences[:3], word2idx, bigram_mm_model)
    print('\nreal hand made sentences:')
    real_sentences = split_sentences(['Man had a dog', 'One child was looking for his mother', 'One child was looking for his mother .'])
    real_sequences = get_sequences_from_sentences_given_word2idx(real_sentences, word2idx, tokenize_method=tokenize_brown_sentence)
    real_sequences_probs = [bigram_mm_model.seqence_prob(s) for s in real_sequences]
    print_results(real_sentences, real_sequences, word2idx, real_sequences_probs)
    print('\nrandom sentences:')
    random_sequences = [generate_random_sequence(len(s), V, start_seq_state=word2idx[START_STR], end_seq_state=word2idx[END_STR]) for s in real_sentences]
    random_sequences_probs = [bigram_mm_model.seqence_prob(s) for s in random_sequences]
    print_results(None, random_sequences, word2idx, random_sequences_probs)

    print('\nbigram logistic regression model:')
    lr_model = logistic_bigram_model_fast(V, start_seq_state=word2idx[START_STR], end_seq_state=word2idx[END_STR])
    lr_history = lr_model.fit(sequences, learning_rate=1e-1, epochs=1)

    print('\nbigram neural network model:')
    nn_model = nn_bigram_model_fast(V, hidden_size=100, start_seq_state=word2idx[START_STR], end_seq_state=word2idx[END_STR])
    nn_history = nn_model.fit(sequences, learning_rate=1e-2, epochs=1)

    plt.figure(figsize=(20, 16))
    plt.plot(smooth_loss(lr_history), label='logistic bigram smoothed cross entropy')
    plt.plot(smooth_loss(nn_history), label='neural network bigram smoothed cross entropy')
    plt.axhline(y=bigram_mm_loss, color='r', linestyle='-', label='markov model bigram cross entropy loss')
    plt.legend()
    plt.show()

    # transition prababilities estimamated by different models
    plt.figure(figsize=(24, 10))
    plt.subplot(1, 3, 1)
    plt.title("bigram markov model model")
    plt.imshow(np.exp(bigram_mm_model.log_A))
    plt.subplot(1, 3, 2)
    plt.title("bigram logistic regression model")
    plt.imshow(softmax2(lr_model.W))
    plt.subplot(1, 3, 3)
    plt.title("bigram neural network model")
    plt.imshow(softmax2(np.tanh(nn_model.W1).dot(nn_model.W2)))
    plt.show()
