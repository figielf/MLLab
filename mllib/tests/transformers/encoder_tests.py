from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from transformers_and_attention.encoder import TextClassificationEncoder

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', DEVICE)


def set_seed():
    torch.manual_seed(12345)
    np.random.seed(98765)


def dummy_data_test(model):
    print(f'\n\n----------DUMMY DATA TEST----------\n')
    x = np.random.randint(0, 20_000, size=(8, 512))
    x_t = torch.tensor(x).to(DEVICE)

    mask = np.ones((8, 512))
    mask[:, 256:] = 0
    mask_t = torch.tensor(mask).to(DEVICE)

    y = model(x_t, mask_t)
    print(f'Encoder input shape: {x_t.shape}')
    print(f'Encoder input: {x_t}')
    print(f'\nEncoder output shape: {y.shape}')
    print(f'Encoder output: {y}')
    print(f'\n----------END OF DUMMY DATA TEST----------\n\n')


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
    print(f'example of padded data:\n{tokenized_datasets["train"][:3]}')

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


def evaluate_encoder_model(model, data_loader):
    model.eval()

    n_correct = 0
    n_total = 0
    for batch in data_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        y_hat_prob = model(batch['input_ids'], batch['attention_mask'])

        _, y_hat = torch.max(y_hat_prob, 1)

        targets = batch['labels']
        n_correct += (y_hat == targets).sum().item()
        n_total += targets.shape[0]

    accuracy = n_correct / n_total
    return accuracy


def calc_batch_avg_metric(scores, weights):
    score_sum = 0
    weights_sum = 0
    for s, w in zip(scores, weights):
        score_sum += s * w
        weights_sum += w
    return score_sum / weights_sum


def train_model(model, criterion, optimizer, n_epochs, targets_provider, train_loader, test_loader=None, metric_calculators=None):
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

            optimizer.zero_grad()
            y_hat_prob = model(batch['input_ids'], batch['attention_mask'])
            targets = targets_provider(batch)
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
                y_hat_prob = model(batch['input_ids'], batch['attention_mask'])
                targets = targets_provider(batch)
                test_batch_loss = criterion(y_hat_prob, targets)
                batch_size = batch['input_ids'].shape[0]
                epoch_test_scores.append(test_batch_loss.item())
                epoch_test_scores_weights.append(batch_size)
                #print(f'\tbatch {ib + 1}, test criterion value: {test_batch_loss.item()}')

            for metric, metric_calculator in metric_calculators.items():
                test_history[metric].append(metric_calculator(epoch_test_scores, epoch_test_scores_weights))

            test_report = f'Test Loss: {test_history["loss"][-1]}, '
        print(f'Epoch {i + 1}/{n_epochs}, Train Loss: {train_history["loss"][-1]}, {test_report}Duration: {datetime.now() - t0}')
    return train_history, test_history


def train_encoder_on_hagging_face_data(model_factory, checkpoint, dataset, n_epochs=4, batch_size=32):
    print(f'\n\n----------REAL DATA TEST----------\n')
    tokenized_datasets, tokenizer, data_collator = preprocess_hugging_face_dataset(checkpoint, dataset)

    train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    validation_loader = DataLoader(tokenized_datasets['validation'], batch_size=batch_size, collate_fn=data_collator)

    model = model_factory(tokenizer.vocab_size, tokenizer.max_model_input_sizes[checkpoint])
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    labels_provider = lambda batch: batch['labels']

    print(f'\nTraining started\n')
    train_losses, test_losses = train_model(model, criterion, optimizer, n_epochs=n_epochs, targets_provider=labels_provider, train_loader=train_loader, test_loader=validation_loader, metric_calculators={'loss': calc_batch_avg_metric})
    print(f'\nTraining finished\n')

    plt.plot(train_losses['loss'], label='train set loss')
    plt.plot(test_losses['loss'], label='validation set loss')
    plt.legend()
    plt.show()

    train_acc = evaluate_encoder_model(model, train_loader)
    validation_acc = evaluate_encoder_model(model, validation_loader)
    print(f'\nTrain acc: {train_acc}, Validation acc: {validation_acc}')
    print(f'\n----------END OF REAL DATA TEST----------\n\n')
    return model, tokenizer, data_collator


def predict_text_sentiment_by_encoder(model, tokenizer, sentences):
    model.eval()

    predictions = []
    for sent in sentences:
        model_inputs = tokenizer(sent, truncation=True, return_tensors='pt')
        output = model(model_inputs['input_ids'].to(DEVICE), model_inputs['attention_mask'].to(DEVICE))
        probs = F.softmax(output, dim=-1)
        probability, predicted_class = torch.max(probs, dim=-1)
        predictions.append((sent, predicted_class.item(), probability.item()))
    return predictions


def inspect_data_loader(data_loader):
    for i, batch in enumerate(data_loader):
        if i > 2:
            break
        print(f'\n\ndata batch {i}:\n{batch}')


def get_encoder_model(vs, ml):
    model = TextClassificationEncoder(
        vocab_size=vs,
        max_len=ml,
        d_k=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_classes=2,
        dropout_prob=0.1,
    )
    #model.to(DEVICE)
    return model


if __name__ == '__main__':
    set_seed()

    #model = TextClassificationEncoder(20_000, 1024, 16, 64, 4, 2, 5, 0.1)
    #dummy_data_test(model.to(DEVICE))

    # run model training and prediction on real data
    checkpoint = 'distilbert-base-cased'
    data = load_dataset("glue", "sst2")

    model, tokenizer, data_collator = train_encoder_on_hagging_face_data(get_encoder_model, checkpoint, data, n_epochs=4, batch_size=128)

    print('-----------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------')
    test_sentences = ['What a nice day, isn"t it?',
                      'This is really great',
                      'I hate this problematic thing, someone should fix that',
                      'I don"t have opinion about it, I dont know it well']
    predictions = predict_text_sentiment_by_encoder(model, tokenizer, test_sentences)
    print(pd.DataFrame(predictions, columns=['sentence', 'pred', 'probability']))
