from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForSeq2Seq


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed():
    torch.manual_seed(12345)
    np.random.seed(98765)


def calc_batch_avg_metric(scores, weights):
    score_sum = 0
    weights_sum = 0
    for s, w in zip(scores, weights):
        score_sum += s * w
        weights_sum += w
    return score_sum / weights_sum


def calc_avg_metric(scores, weights):
    score_sum = 0
    weights_sum = 0
    for s, w in zip(scores, weights):
        score_sum += s
        weights_sum += 1
    return score_sum / weights_sum


def inspect_data_loader(data_loader):
    for i, batch in enumerate(data_loader):
        if i > 2:
            break
        print(f'\n\ndata batch {i}:\n{batch}')


def preprocess_hugging_face_dataset(checkpoint, raw_datasets, use_labels=True):
    print(f'\n-----     "{raw_datasets}" with "{checkpoint}" data preprocessing is about to start     -----\n')

    def tokenize_fn(batch):
        return tokenizer(batch['sentence'], truncation=True)

    print(f'\nraw glue sst2 data before preprocessing:\n{raw_datasets}')
    print(f'example of raw data:\n{raw_datasets["train"][:3]}')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print(f'\ntokenizer vocabulary size: {tokenizer.vocab_size}')
    print(f'\ntokenizer max_model_input_sizes: {tokenizer.max_model_input_sizes}')

    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
    print(f'\ntokenized data:\n{tokenized_datasets}')
    print(f'example of tokenized data:\n{tokenized_datasets["train"][:3]}')

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print(f'\ndata collator:\n{data_collator}')
    print(f'example of data:\n{tokenized_datasets["train"][:3]}')

    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    if use_labels:
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        print(f'\nall target classes in training data: {set(tokenized_datasets["train"]["labels"])}')
    else:
        tokenized_datasets = tokenized_datasets.remove_columns(["label"])
    print(f'\nfinally preprocessed glue sst2 data:\n{tokenized_datasets}')
    print(f'\nexample of finally preprocessed data:\n{tokenized_datasets["train"][:3]}')
    print(f'\nand its version encoded back to tokens:')
    for input_ids in tokenized_datasets["train"][:3]["input_ids"]:
        print(tokenizer.decode(input_ids))

    print(f'\n-----     "{raw_datasets}" with "{checkpoint}" data preprocessing finished     -----\n')
    return tokenized_datasets, tokenizer, data_collator


def preprocess_spa_dataset_for_encoder_decoder_translation(checkpoint, raw_dataset, max_input_length=128, max_target_length=128, split_test_size=0.3):
    print(f'\n-----     "{raw_dataset}" with "{checkpoint}" data preprocessing is about to start     -----\n')

    def tokenize_fn(batch):
        inputs = tokenizer(batch['en'], max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=batch['es'], max_length=max_target_length, truncation=True)
        inputs['labels'] = labels['input_ids']
        return inputs

    print(f'\nraw spa (english to spanish translation) data before preprocessing:\n{raw_dataset}')
    print(f'example of raw data:\n{raw_dataset["train"][:3]}')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print(f'\ntokenizer basic info:')
    print(f'\tvocabulary size: {tokenizer.vocab_size}')
    print(f'\tmax_model_input_sizes: {tokenizer.max_model_input_sizes}')
    print(f'\tspacial tokens: {tokenizer.all_special_tokens}')
    print(f'\tspacial token ids: {tokenizer.all_special_ids}')
    tokenizer.add_special_tokens({"cls_token": "<s>"})  # add START prompt token to represent begining of the translation
    V = tokenizer.vocab_size + 1  # vocab size including one special token that is not counted
    print(f'\n\tadded new special token "<s>" for use as START token for decoder input:')
    print(f'\tnew spacial tokens: {tokenizer.all_special_tokens}')
    print(f'\tnew spacial token ids: {tokenizer.all_special_ids}')
    print(f'\tnew vocabulary size: {tokenizer.vocab_size}, so we will have to add 1 by hand!')

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
    return tokenized_datasets, tokenizer, V, data_collator


def get_targets_for_decoder(batch, pad_token):
    # get targets as inputs shifted right by 1
    targets = batch['input_ids'].clone().detach()
    targets = torch.roll(targets, shifts=-1, dims=1)  # first dimension is N
    targets[:, -1] = pad_token
    return targets


def train_model(model, criterion, optimizer, n_epochs, targets_provider, model_parameters_builder, train_loader, test_loader=None, metric_calculators=None):
    train_history = {}
    test_history = {}
    for metric in metric_calculators.keys():
        train_history[metric] = []
    if test_loader:
        for metric in metric_calculators.keys():
            test_history[metric] = []

    for i in range(n_epochs):
        model.train()
        t0 = datetime.now()
        epoch_train_scores = []
        epoch_train_scores_weights = []
        for ib, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            targets = targets_provider(batch)
            optimizer.zero_grad()
            y_hat_prob = model(*model_parameters_builder(batch))
            #targets = targets_provider(batch)
            train_batch_loss = criterion(y_hat_prob, targets)

            train_batch_loss.backward()
            optimizer.step()

            batch_size = batch['input_ids'].shape[0]
            epoch_train_scores.append(train_batch_loss.item())
            epoch_train_scores_weights.append(batch_size)

            print(f'\tbatch {ib + 1}, train criterion value: {train_batch_loss.item()}')

        for metric, metric_calculator in metric_calculators.items():
            train_history[metric].append(metric_calculator(epoch_train_scores, epoch_train_scores_weights))

        test_report = ''
        if test_loader:
            model.eval()
            epoch_test_scores = []
            epoch_test_scores_weights = []
            for ib, batch in enumerate(test_loader):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                y_hat_prob = model(*model_parameters_builder(batch))
                targets = targets_provider(batch)
                test_batch_loss = criterion(y_hat_prob, targets)
                batch_size = batch['input_ids'].shape[0]
                epoch_test_scores.append(test_batch_loss.item())
                epoch_test_scores_weights.append(batch_size)
                print(f'\tbatch {ib + 1}, test criterion value: {test_batch_loss.item()}')

            for metric, metric_calculator in metric_calculators.items():
                test_history[metric].append(metric_calculator(epoch_test_scores, epoch_test_scores_weights))

            test_report = f'Test Loss: {test_history["loss"][-1]}, '
        print(f'Epoch {i + 1}/{n_epochs}, Train Loss: {train_history["loss"][-1]}, {test_report}Duration: {datetime.now() - t0}')
    return train_history, test_history
