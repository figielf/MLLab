import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

from tests.transformers.encoder_tests import set_seed, preprocess_hugging_face_dataset, train_model, DEVICE
from transformers_and_attention.decoder import TextGenerationDecoder


def dummy_data_test(model):
    print(f'\n\n----------DUMMY DATA TEST----------\n')
    x = np.random.randint(0, 20_000, size=(8, 512))
    x_t = torch.tensor(x).to(DEVICE)

    mask = np.ones((8, 512))
    mask[:, 256:] = 0
    mask_t = torch.tensor(mask).to(DEVICE)

    y = model(x_t, mask_t)
    print(f'Decoder input shape: {x_t.shape}')
    print(f'Decoder input: {x_t}')
    print(f'\nDecoder output shape: {y.shape}')
    print(f'Decoder output: {y}')
    print(f'\n----------END OF DUMMY DATA TEST----------\n\n')


def get_targets_for_decoder(batch, pad_token):
    # get targets as inputs shifted right by 1
    targets = batch['input_ids'].clone().detach()
    targets = torch.roll(targets, shifts=-1, dims=1)  # first dimension is N
    targets[:, -1] = pad_token
    return targets


def calc_avg_metric(scores, weights):
    score_sum = 0
    weights_sum = 0
    for s, w in zip(scores, weights):
        score_sum += s
        weights_sum += 1
    return score_sum / weights_sum


def evaluate_decoder_model(model, data_loader, pad_token):
    model.eval()

    n_correct = 0
    n_total = 0
    for batch in data_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        y_hat_prob = model(batch['input_ids'], batch['attention_mask'] == 1)
        _, y_hat = torch.max(y_hat_prob, dim=-1)

        targets = get_targets_for_decoder(batch, pad_token)

        comapre_mask = torch.roll(batch['attention_mask'], shifts=-1, dims=1)
        comapre_mask[:, -1] = 0
        n_tokens_to_predict = torch.sum(comapre_mask)

        targets_masked = targets.masked_fill(comapre_mask == 0, -1)
        y_hat_masked = y_hat.masked_fill(comapre_mask == 0, -2)

        n_correct += (y_hat_masked == targets_masked).sum().item()
        n_total += n_tokens_to_predict
    accuracy = n_correct / n_total
    return accuracy


def train_decoder_on_hagging_face_data(model_factory, checkpoint, dataset, batch_size=32, n_epochs=4):
    print(f'\n\n----------REAL DATA TEST----------\n')
    tokenized_datasets, tokenizer, data_collator = preprocess_hugging_face_dataset(checkpoint, dataset, use_labels=False)

    train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    model = model_factory(tokenizer.vocab_size, tokenizer.max_model_input_sizes[checkpoint])
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    shifted_inputs_provider = lambda batch: get_targets_for_decoder(batch, tokenizer.pad_token_id)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    def decoder_criterion(prediction, target):
        # decoder model outputs are N x T x V but PyTorch CrossEntropyLoss expects N x V x T
        return criterion(prediction.transpose(2, 1), target)

    print(f'\nTraining started\n')
    train_losses, test_losses = train_model(model, decoder_criterion, optimizer, n_epochs=n_epochs, targets_provider=shifted_inputs_provider, train_loader=train_loader, test_loader=None, metric_calculators={'loss': calc_avg_metric})
    print(f'\nTraining finished\n')

    plt.plot(train_losses['loss'], label='train set loss')
    plt.legend()
    plt.show()

    train_acc = evaluate_decoder_model(model, train_loader, pad_token=tokenizer.pad_token_id)
    print(f'\nTrain accuracy: {train_acc}')
    print(f'\n----------END OF REAL DATA TEST----------\n\n')
    return model, tokenizer, data_collator


def predict_next_word_of_text_by_decoder(model, tokenizer, data_collator, sentences):
    model.eval()

    predictions = []
    for sent in sentences:
        model_inputs = tokenizer(sent, truncation=True, return_tensors='pt')
        output = model(model_inputs['input_ids'][:, :-1].to(DEVICE), model_inputs['attention_mask'][:, :-1].to(DEVICE))
        last_word_predictions = output[:, -1, :]
        probs = F.softmax(last_word_predictions, dim=-1)
        probability, predicted_word_id = torch.max(probs, dim=-1)  # probs shape=(N, T, n_classes)
        predicted_word = tokenizer.decode(predicted_word_id)
        predictions.append((sent, predicted_word, probability.item()))
    return predictions


def predict_text_following_a_prompt_by_decoder(model, tokenizer, data_collator, sentences, max_words=10):
    model.eval()

    predictions = []
    for sent in sentences:
        model_inputs = tokenizer(sent, truncation=True, return_tensors='pt')
        sentence = model_inputs['input_ids'].to(DEVICE)
        padding_mask = model_inputs['attention_mask'].to(DEVICE)
        predicted_words = []
        for i in range(max_words):
            output = model(sentence, padding_mask)
            last_word_predictions = output[:, -1, :]
            probs = F.softmax(last_word_predictions, dim=-1)
            probability, predicted_word_id = torch.max(probs, dim=-1)  # probs shape=(N, T, n_classes)
            predicted_word = tokenizer.decode(predicted_word_id)
            predicted_words.append(predicted_word)

            print(predicted_word)

            sentence = torch.hstack((sentence, predicted_word_id.view(1, 1)))
            padding_mask = torch.ones_like(sentence)
            if predicted_word_id == tokenizer.sep_token_id:
                break

        predictions.append((sent, ' '.join(predicted_words)))
    return predictions


def get_decoder_model(vs, ml):
    model = TextGenerationDecoder(
        vocab_size=vs,
        max_len=ml,
        d=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout_prob=0.1,
    )
    model.to(DEVICE)
    return model



if __name__ == '__main__':
    set_seed()

    #model = TextGenerationDecoder(20_000, 1024, 16, 64, 4, 2, 20000, 0.1)
    #dummy_data_test(model.to(DEVICE))

    # run model training and prediction on real data
    checkpoint = 'distilbert-base-cased'
    data = load_dataset("glue", "sst2")

    model, tokenizer, data_collator = train_decoder_on_hagging_face_data(get_decoder_model, checkpoint, data, batch_size=128, n_epochs=1)

    print('-----------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------')

    print('\nNext word prediction task:\n')
    test_sentences = ['What a nice day, isn"t it?',
                      'This car is very fast',
                      'Dogs doesn"t like cats',
                      'Mares has a brother and he really likes him']
    predictions = predict_next_word_of_text_by_decoder(model, tokenizer, data_collator, test_sentences)
    print(pd.DataFrame(predictions, columns=['sentence', 'pred', 'probability']))

    prompts = ['What a nice day',
               'This car',
               'Dogs',
               'Mares has a brother and he']
    print('\nPrompt text generation:\n')
    predictions = predict_text_following_a_prompt_by_decoder(model, tokenizer, data_collator, prompts, max_words=10)
    print(pd.DataFrame(predictions, columns=['prompt', 'generated continuation']))
