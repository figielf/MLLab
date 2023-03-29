import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

from tests.transformers.common import set_seed, calc_avg_metric, preprocess_spa_dataset_for_encoder_decoder_translation, train_model, DEVICE
from tests.utils.data_utils import get_data_dir
from transformers_and_attention.encoder_decoder import TextTranslationTransformer


def encoder_decoder_dummy_data_test(model):
    print(f'\n\n----------DUMMY DATA TEST----------\n')
    xe = np.random.randint(0, 20_000, size=(8, 512))
    xe_t = torch.tensor(xe).to(DEVICE)

    xd = np.random.randint(0, 10_000, size=(8, 256))
    xd_t = torch.tensor(xd).to(DEVICE)

    maske = np.ones((8, 512))
    maske[:, 256:] = 0
    maske_t = torch.tensor(maske).to(DEVICE)

    maskd = np.ones((8, 256))
    maskd[:, 128:] = 0
    maskd_t = torch.tensor(maskd).to(DEVICE)

    y = model(xe_t, xd_t, maske_t, maskd_t)

    print(f'Encoder input shape: {xe_t.shape}')
    print(f'Encoder input: {xe_t}')
    print(f'Decoder input shape: {xd_t.shape}')
    print(f'Decoder input: {xd_t}')
    print(f'\nEncoder-Decoder output shape: {y.shape}')
    print(f'Encoder-Decoder output: {y}')
    print(f'\n----------END OF DUMMY DATA TEST----------\n\n')


def get_data_for_encoder_decoder_model(batch, tokenizer):
    # get targets as inputs shifted right by 1
    decoder_input_ids = batch['labels'].clone().detach()
    decoder_input_ids = torch.roll(decoder_input_ids, shifts=1, dims=1)  # first dimension is N
    decoder_input_ids[:, 0] = tokenizer.cls_token_id
    decoder_input_ids = decoder_input_ids.masked_fill(decoder_input_ids == -100, tokenizer.pad_token_id)

    decoder_attention_mask = torch.ones_like(decoder_input_ids)
    decoder_attention_mask = decoder_attention_mask.masked_fill(decoder_input_ids == tokenizer.pad_token_id, 0)
    return [batch['input_ids'], decoder_input_ids, batch['attention_mask'], decoder_attention_mask]


def evaluate_eoncoder_decoder_model(model, data_loader, model_parameters_builder, targets_provider):
    model.eval()

    n_correct = 0
    n_total = 0
    for batch in data_loader:
        if n_total > 500:
            break

        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        input_ids, decoder_input_ids, attention_mask, decoder_attention_mask = model_parameters_builder(batch)
        y_hat_prob = model(input_ids, decoder_input_ids, attention_mask, decoder_attention_mask)
        _, y_hat = torch.max(y_hat_prob, dim=-1)

        targets = targets_provider(batch)

        comapre_mask = decoder_attention_mask
        n_tokens_to_predict = torch.sum(comapre_mask)

        targets_masked = targets.masked_fill(comapre_mask == 0, -1)
        y_hat_masked = y_hat.masked_fill(comapre_mask == 0, -2)

        n_correct += (y_hat_masked == targets_masked).sum().item()
        n_total += n_tokens_to_predict
    accuracy = n_correct / n_total
    return accuracy


def train_encoder_decoder_on_real_data(model_factory, checkpoint, dataset, batch_size=32, n_epochs=4):
    print(f'\n\n----------REAL DATA TEST----------\n')
    tokenized_datasets, tokenizer, vocab_size, data_collator = preprocess_spa_dataset_for_encoder_decoder_translation(checkpoint, dataset)
    print('Vocab size:', vocab_size)

    train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    validation_loader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=data_collator)

    model = model_factory(vocab_size, vocab_size)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    def decoder_criterion(prediction, target):
        # decoder model outputs are N x T x V but PyTorch CrossEntropyLoss expects N x V x T
        return criterion(prediction.transpose(2, 1), target)
    labels_provider = lambda batch: batch['labels']
    model_param_builder = lambda batch: get_data_for_encoder_decoder_model(batch, tokenizer)




    print('\nNext word prediction task:\n')
    text_in_english = ['What a nice day, isn"t it?',
                      'This car is very fast',
                      'Have a nice day',
                      'Mares has a brother and he really likes him']
    predictions = translate_text(model, tokenizer, data_collator, text_in_english, max_words=10)
    print(pd.DataFrame(predictions, columns=['text in english', 'translation in spanish']))




    print(f'\nTraining started\n')
    train_losses, test_losses = train_model(
        model,
        decoder_criterion,
        optimizer,
        n_epochs=n_epochs,
        targets_provider=labels_provider,
        model_parameters_builder=model_param_builder,
        train_loader=train_loader,
        test_loader=validation_loader,
        metric_calculators={'loss': calc_avg_metric})
    print(f'\nTraining finished\n')

    plt.plot(train_losses['loss'], label='train set loss')
    plt.legend()
    plt.show()

    train_acc = evaluate_eoncoder_decoder_model(model, train_loader, model_parameters_builder=model_param_builder, targets_provider=labels_provider)
    test_acc = evaluate_eoncoder_decoder_model(model, validation_loader, model_parameters_builder=model_param_builder, targets_provider=labels_provider)
    print(f'\nTrain accuracy: {train_acc}')
    print(f'\nTest accuracy: {test_acc}')
    print(f'\n----------END OF REAL DATA TEST----------\n\n')
    return model, tokenizer, data_collator


def get_encoder_decoder_model(evs, dvs, eml=512, dml=512):
    transformer = TextTranslationTransformer(
        encoder_vocab_size=evs,
        encoder_max_len=eml,
        decoder_vocab_size=dvs,
        decoder_max_len=dml,
        d=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout_prob=0.1
    )
    return transformer


def translate_text(model, tokenizer, data_collator, sentences, max_words=10):
    model.eval()

    predictions = []
    for sent in sentences:
        model_inputs = tokenizer(sent, truncation=True, return_tensors='pt')
        sentence = model_inputs['input_ids'].to(DEVICE)
        padding_mask = model_inputs['attention_mask'].to(DEVICE)
        encoder_output = model.encoder(sentence, padding_mask)
        decoder_start_sentence = tokenizer('<s>', return_tensors='pt')
        decoder_input = decoder_start_sentence['input_ids'][:, :-1].to(DEVICE)
        decoder_attention_mask = torch.ones_like(decoder_input, device=DEVICE)
        predicted_words = []
        for i in range(max_words):
            output = model.decoder(encoder_output, decoder_input, padding_mask, decoder_attention_mask)
            last_word_predictions = output[:, -1, :]
            probs = F.softmax(last_word_predictions, dim=-1)
            probability, predicted_word_id = torch.max(probs, dim=-1)  # probs shape=(N, T, n_classes)
            predicted_word = tokenizer.decode(predicted_word_id)
            predicted_words.append(predicted_word)

            decoder_input = torch.hstack((decoder_input, predicted_word_id.view(1, 1)))
            decoder_attention_mask = torch.ones_like(decoder_input, device=DEVICE)
            if predicted_word_id == 0:  # == tokenizer('</s>')['input_ids][0][0]
                break

        predictions.append((sent, ' '.join(predicted_words)))
    return predictions


if __name__ == '__main__':
    print('device:', DEVICE)
    set_seed()

    #model = get_encoder_decoder_model(evs=20_000, dvs=10_000)
    #encoder_decoder_dummy_data_test(model.to(DEVICE))

    # run model training and prediction on real data
    checkpoint = 'Helsinki-NLP/opus-mt-en-es'
    data = load_dataset('csv', data_files=get_data_dir('spa_simple_30k_samples.csv'))

    model, tokenizer, data_collator = train_encoder_decoder_on_real_data(get_encoder_decoder_model, checkpoint, data, batch_size=128, n_epochs=1)

    print('-----------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------')

    print('\nEnglish to spanish text translation task:\n')
    text_in_english = ['What a nice day, isn"t it?',
                      'This car is very fast',
                      'Have a nice day',
                      'Mares has a brother and he really likes him']
    predictions = translate_text(model, tokenizer, data_collator, text_in_english, max_words=10)
    print(pd.DataFrame(predictions, columns=['text in english', 'translation in spanish']))
