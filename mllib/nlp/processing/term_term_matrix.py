import numpy as np


def term_term_matrix(sentences, vocab_size, context_size, start_end_tokens_to_add=None):
    # sentences has to be tokenized and indexed

    def get_window(centre_idx, sequence, window_size):
        centre = sequence[centre_idx]
        left = sequence[:centre_idx][-window_size:]
        right = sequence[centre_idx + 1:][:window_size-1]  # TODO replace 'window_size-1' with 'window_size' otherwise left size has context_size tokens but righ_size has context_size-1 tokens
        return centre, left, right

    tt = np.zeros((vocab_size, vocab_size))

    N = len(sentences)
    print(f'number of sentences to process: {N}')
    it = 0
    for sentence in sentences:
        it += 1
        if it % 10000 == 0:
            print(f'processed {it} / {N}')

        if start_end_tokens_to_add:
            # do not make start_token neither end_token a centre
            start_token, end_token = start_end_tokens_to_add
            sentence = [start_token] + sentence + [end_token]
            sent_start_idx, sent_end_idx = 1, len(sentence) - 1
        else:
            sent_start_idx, sent_end_idx = 0, len(sentence)

        for i in range(sent_start_idx, sent_end_idx):
            centre, left, right = get_window(i, sentence, context_size)
            for j, context_token in enumerate(reversed(left)):
                points = 1.0 / (j + 1)
                tt[centre, context_token] += points
                tt[context_token, centre] += points
            for j, context_token in enumerate(right):
                points = 1.0 / (j + 1)
                tt[centre, context_token] += points
                tt[context_token, centre] += points
    return tt





