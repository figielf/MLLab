import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader

from tests.transformers.common import set_seed, calc_batch_avg_metric, preprocess_hugging_face_dataset, train_model, DEVICE
from transformers_and_attention.encoder import TextClassificationEncoder


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


def evaluate_encoder_model(model, data_loader, model_parameters_builder):
    model.eval()

    n_correct = 0
    n_total = 0
    for batch in data_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        y_hat_prob = model(*model_parameters_builder(batch))

        _, y_hat = torch.max(y_hat_prob, 1)

        targets = batch['labels']
        n_correct += (y_hat == targets).sum().item()
        n_total += targets.shape[0]

    accuracy = n_correct / n_total
    return accuracy


def train_encoder_on_hagging_face_data(model_factory, checkpoint, dataset, n_epochs=4, batch_size=32):
    print(f'\n\n----------REAL DATA TEST----------\n')
    tokenized_datasets, tokenizer, data_collator = preprocess_hugging_face_dataset(checkpoint, dataset)

    train_loader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    validation_loader = DataLoader(tokenized_datasets['validation'], batch_size=batch_size, collate_fn=data_collator)

    model = model_factory(tokenizer.vocab_size, tokenizer.max_model_input_sizes[checkpoint])
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    labels_provider = lambda batch: batch['labels']
    model_param_builder = lambda batch: [batch['input_ids'], batch['attention_mask']]

    print(f'\nTraining started\n')
    train_losses, test_losses = train_model(
        model,
        criterion,
        optimizer,
        n_epochs=n_epochs,
        targets_provider=labels_provider,
        model_parameters_builder=model_param_builder,
        train_loader=train_loader,
        test_loader=validation_loader,
        metric_calculators={'loss': calc_batch_avg_metric})
    print(f'\nTraining finished\n')

    plt.plot(train_losses['loss'], label='train set loss')
    plt.plot(test_losses['loss'], label='validation set loss')
    plt.legend()
    plt.show()

    train_acc = evaluate_encoder_model(model, train_loader, model_parameters_builder=model_param_builder)
    validation_acc = evaluate_encoder_model(model, validation_loader, model_parameters_builder=model_param_builder)
    print(f'\nTrain acc: {train_acc}, Validation acc: {validation_acc}')
    print(f'\n----------END OF REAL DATA TEST----------\n\n')
    return model, tokenizer, data_collator


def predict_text_sentiment_by_encoder(model, tokenizer, sentences):
    model.eval()

    predictions = []
    for sent in sentences:
        model_inputs = tokenizer(sent, truncation=True, return_tensors='pt')
        logits = model(model_inputs['input_ids'].to(DEVICE), model_inputs['attention_mask'].to(DEVICE), return_logits=True)
        probability, predicted_class = torch.max(logits, dim=-1)
        predictions.append((sent, predicted_class.item(), probability.item()))
    return predictions


def get_encoder_model(vs, ml):
    model = TextClassificationEncoder(
        vocab_size=vs,
        max_len=ml,
        d=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_classes=2,
        dropout_prob=0.1,
    )
    #model.to(DEVICE)
    return model


if __name__ == '__main__':
    print('device:', DEVICE)
    set_seed()

    model = TextClassificationEncoder(20_000, 1024, 16, 64, 4, 2, 5, 0.1)
    dummy_data_test(model.to(DEVICE))

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
