from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForSeq2Seq, PreTrainedTokenizerFast

from tests.transformers.decoder_tests import evaluate_decoder_model
from tests.transformers.encoder_tests import DEVICE, set_seed
from tests.utils.data_utils import get_data_dir
from transformers_and_attention.encoder import EmbeddingEncoder
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


def preprocess_spa_dataset_for_encoder_decoder_translation(checkpoint, raw_dataset, max_input_length=128, max_target_length=128, split_test_size=0.3):
    print(f'\n-----     "{raw_dataset}" with "{checkpoint}" data preprocessing is about to start     -----\n')

    def tokenize_fn(batch):
        inputs = tokenizer(batch['en'], max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=batch['es'], max_length=max_target_length, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs


    print(f'\nraw spa (english to spanish translation) data before preprocessing:\n{raw_dataset}')
    print(f'example of raw data:\n{raw_dataset["train"][:3]}')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print(f'\ntokenizer vocabulary size: {tokenizer.vocab_size}')
    print(f'\ntokenizer max_model_input_sizes: {tokenizer.max_model_input_sizes}')
    print(f'\ntokenizer spacial token: {tokenizer.all_special_tokens}')
    print(f'tokenizer spacial token ids: {tokenizer.all_special_ids}')

    raw_dataset_splitted = raw_dataset['train'].train_test_split(test_size=split_test_size)
    tokenized_datasets = raw_dataset_splitted.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_dataset_splitted["train"].column_names,
    )
    print(f'\ntokenized data:\n{tokenized_datasets}')
    print(f'example of tokenized data:\n{tokenized_datasets["train"][:3]}')

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    print(f'\ndata collator:\n{data_collator}')
    en_sentences = raw_dataset_splitted["train"][0:3]["en"]
    es_sentences = raw_dataset_splitted["train"][0:3]["es"]
    print(f'\nexample of english sentences:\n{en_sentences}')
    print(f'and its version encoded back to tokens:')
    for input_ids in tokenized_datasets["train"][:3]["input_ids"]:
        print(tokenizer.decode(input_ids))
    print(f'\ntheir translations to spanish:\n{es_sentences}')
    print(f'and its version encoded back to tokens:')
    for input_ids in tokenized_datasets["train"][:3]["labels"]:
        print(tokenizer.decode(input_ids))

    print(f'\n-----     "{raw_dataset}" with "{checkpoint}" data preprocessing finished     -----\n')
    return tokenized_datasets, tokenizer, data_collator


def train_encoder_decoder_on_real_data(model_factory, checkpoint, dataset, batch_size=32, n_epochs=4):
    print(f'\n\n----------REAL DATA TEST----------\n')
    tokenized_datasets, tokenizer, data_collator = preprocess_spa_dataset_for_encoder_decoder_translation(checkpoint, dataset)
    tokenizer.add_special_tokens({"cls_token": "<s>"})  # add START prompt token to represent begining of the translation
    V = tokenizer.vocab_size + 1  # vocab size including one special token that is not counted
    print('Vocab size:', V)

    train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    validation_loader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=data_collator)

    model = model_factory(V, V)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    shifted_inputs_provider = lambda batch: get_targets_for_decoder(batch, tokenizer.pad_token_id)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    def decoder_criterion(prediction, target):
        # decoder model outputs are N x T x V but PyTorch CrossEntropyLoss expects N x V x T
        return criterion(prediction.transpose(2, 1), target)

    print(f'\nTraining started\n')
    train_losses, test_losses = train_model(model, decoder_criterion, optimizer, n_epochs=n_epochs, targets_provider=shifted_inputs_provider, train_loader=train_loader, test_loader=validation_loader, metric_calculators={'loss': calc_avg_metric})
    print(f'\nTraining finished\n')

    plt.plot(train_losses['loss'], label='train set loss')
    plt.legend()
    plt.show()

    train_acc = evaluate_decoder_model(model, train_loader, pad_token=tokenizer.pad_token_id)
    print(f'\nTrain accuracy: {train_acc}')
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
    transformer.encoder.to(DEVICE)
    transformer.decoder.to(DEVICE)
    return transformer


if __name__ == '__main__':
    set_seed()

    model = TextGenerationDecoder(20_000, 1024, 16, 64, 4, 2, 20000, 0.1)
    dummy_data_test(model.to(DEVICE))

    # run model training and prediction on real data
    checkpoint = 'Helsinki-NLP/opus-mt-en-es'
    data = load_dataset('csv', data_files=get_data_dir('spa_simple_30k_samples.csv'))

    model, tokenizer, data_collator = train_encoder_decoder_on_real_data(get_encoder_decoder_model, checkpoint, data, batch_size=128, n_epochs=4)
