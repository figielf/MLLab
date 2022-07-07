import numpy as np
import nltk
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.manifold import TSNE

from clustering.kmeans_soft import Kmeans_soft
from tests.utils.data_utils import get_book_titles_data


def tokenize(txt, lemitizer):
    try:
        result = txt.lower()
        result = nltk.tokenize.word_tokenize(result)
        result = [token for token in result if len(token) > 2]
        result = [lemitizer.lemmatize(token) for token in result]
        result = [token for token in result if token not in stopwords]
        result = [token for token in result if not any(c.isdigit() for c in token)]  # remove any digits, i.e. "3rd edition"
    except Exception as e:
        print(e)
    return result


def build_vocabulary(tokens_vectors):
    word2index = {}
    index2word = []
    current_index = 0
    for txt_tokenized in tokens_vectors:
        for token in txt_tokenized:
            if token not in word2index:
                word2index[token] = current_index
                index2word.append(token)
                current_index += 1
    return word2index, index2word


def perform_count_vectorization(tokens_vectors, word2index):
    token_counts = np.zeros(len(word2index))
    for token in tokens_vectors:
        id = word2index[token]
        token_counts[id] += 1
    return token_counts


def annotate1(X, index_word_map, eps=0.1):
  N, D = X.shape
  placed = np.empty((N, D))
  for i in range(N):
    # if x, y is too close to something already plotted, move it
    close = []

    x, y = X[i]
    for retry in range(3):
      for j in range(i):
        diff = np.array([x, y]) - placed[j]

        # if something is close, append it to the close list
        if diff.dot(diff) < eps:
          close.append(placed[j])

      if close:
        # then the close list is not empty
        x += (np.random.randn() + 0.5) * (1 if np.random.rand() < 0.5 else -1)
        y += (np.random.randn() + 0.5) * (1 if np.random.rand() < 0.5 else -1)
        close = [] # so we can start again with an empty list
      else:
        # nothing close, let's break
        break

    placed[i] = (x, y)

    #plt.annotate(
    #  s=index_word_map[i],
    #  text=index_word_map[i],
    #  xy=(X[i,0], X[i,1]),
    #  xytext=(x, y),
    #  arrowprops={
    #    'arrowstyle': '->',
    #    'color': 'black',
    #  }
    #)


def plot_clusters(X, K, R, index2word):
    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors)
    plt.figure(figsize=(80.0, 80.0))
    plt.scatter(X[:, 0], X[:, 1], s=300, alpha=0.9, c=colors)
    annotate1(X, index2word)
    # plt.show()
    plt.savefig("c:/temp/test.png")


def print_clusters(R, index2word):
    # print out the clusters
    hard_responsibilities = np.argmax(R, axis=1) # is an N-size array of cluster identities
    # let's "reverse" the order so it's cluster identity -> word index
    cluster2word = {}
    for i in range(len(hard_responsibilities)):
      word = index2word[i]
      cluster = hard_responsibilities[i]
      if cluster not in cluster2word:
        cluster2word[cluster] = []
      cluster2word[cluster].append(word)

    # print out the words grouped by cluster
    for cluster, wordlist in cluster2word.items():
      print("cluster", cluster, "->", wordlist)


if __name__ == '__main__':
    # load data
    titles, stopwords = get_book_titles_data()

    stopwords = stopwords.union({
        'introduction', 'edition', 'series', 'application',
        'approach', 'card', 'access', 'package', 'plus', 'etext',
        'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
        'third', 'second', 'fourth', })

    print(f'Num of titles: {len(titles)}. Title example:"{titles[0]}"')
    print(f'Num of stop words: {len(stopwords)}')

    lemitizer = WordNetLemmatizer()
    titles_tokenized = [tokenize(t, lemitizer) for t in titles]

    word2index_map, index2word_map = build_vocabulary(titles_tokenized)

    # vectorize input data
    N = len(titles_tokenized)
    D = len(word2index_map)
    X = np.zeros((D, N))  # terms will go along rows, documents along columns
    i = 0
    for tokens in titles_tokenized:
        X[:, i] = perform_count_vectorization(tokens, word2index_map)
        i += 1

    transformer = TfidfTransformer()
    X_transformed = transformer.fit_transform(X).toarray()
    print('X_transformed.shape:', X_transformed.shape)

    reducer = TSNE()
    X_reduced = reducer.fit_transform(X_transformed)
    print('X_reduced.shape:', X_reduced.shape)

    K = D//10
    shuffle_ids = np.arange(D)
    np.random.shuffle(shuffle_ids)
    centres0 = X_reduced[shuffle_ids[:K]]

    k_means = Kmeans_soft(n_clusters=K)
    clusters_hist, r_hist, centres_hist, cost_hist = k_means.fit(X_reduced, beta=1.0, initial_centres=centres0, max_steps=20, logging_step=5)
    #report_on_kmeans(k_means, X, y, clusters_hist, r_hist, centres_hist, cost_hist)

    plot_clusters(X_reduced, K, r_hist[-1], index2word_map)
    print_clusters(r_hist[-1], index2word_map)
