import random
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import sys
import argparse
import itertools
from collections import defaultdict

import utils
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint, get_langs_and_paths, set_lr
from torch.utils.tensorboard.writer import SummaryWriter  # to print to tensorboard
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from Network import Encoder, Decoder, Seq2Seq
from utils import srcField, trgField, device, reinflection2TSV, plt, showAttention, REINFLECTION_STR, INFLECTION_STR

def concat_to_file(fn, s):
    with open(fn, "a+", encoding='utf8') as f: f.write(s)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('lang', type=str, help='Language')
    parser.add_argument('--hall', action='store_true', help='Use hallucinated data')
    parser.add_argument('--nn', action='store_true', help='Use hallucinated NN data')
    parser.add_argument('--agg', action='store_true', help='Precision@k aggregation at test')
    parser.add_argument('--embed-size', type=int, default=300, help='Embedding size')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')   
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')   
    return parser.parse_args()

total_timer = datetime.now()
# Definition of tokenizers, Fields and device were moved to utils

datafields = [("src", srcField), ("trg", trgField)]

params = parse_args()

# Generate new datasets for Inflection:
training_mode = 'LEMMA' # choose either 'FORM' or 'LEMMA'.
# data_dir = os.path.join('data',f'{training_mode}-SPLIT')
data_dir = 'data'
tsv_dir = f'data_{training_mode}_TSV'
if params.nn:
    tsv_dir = f'data_NN_TSV'

langs, files_paths, lang2family = get_langs_and_paths(data_dir=data_dir, use_hall=params.hall)
os.makedirs(f"log.SIG20/{training_mode}", exist_ok=True)
if not os.path.exists(tsv_dir): os.mkdir(tsv_dir)
results_df = pd.DataFrame(columns=["Family", "Language", "Accuracy", "ED"])

# log_file = f"log.SIG20/{training_mode}/log_file0.txt"

lang = params.lang

outputs_dir = f"outputs.SIG20/{training_mode}/{lang}"
os.makedirs(outputs_dir, exist_ok=True)
log_file = f"{outputs_dir}/log_file.txt"

lang_t0 = datetime.now()
starter_s = f"Starting to train a new model on Language={lang}, from Family={lang2family[lang]}, at {str(datetime.now())}\n"
concat_to_file(log_file, starter_s)
print(starter_s)

# Add here the datasets creation, using TabularIterator (add custom functions for that)
if not params.nn:
    train_file, dev_file, test_file = reinflection2TSV([f"{data_dir}/{fn}" for fn in files_paths[lang]], dir_name=tsv_dir, mode=INFLECTION_STR)
else:
    train_file = f"{tsv_dir}/{lang}.trn.tsv"
    dev_file = f"{tsv_dir}/{lang}.dev.tsv"
    test_file = f"{tsv_dir}/{lang}.tst.tsv"

# train_data, dev_data, test_data = TabularDataset.splits(path="", train=train_file, validation=dev_file, test=test_file, fields=datafields, format='tsv')
train_data, dev_data, test_data = TabularDataset.splits(path="", train=train_file, validation=dev_file, test=test_file, fields=datafields, format='tsv')
print(f"Found training examples: {len(train_data)}")
concat_to_file(log_file, f"Found training examples: {len(train_data)}")


print("- Building vocabularies")
srcField.build_vocab(train_data) # no limitation of max_size or min_freq is needed.
trgField.build_vocab(train_data) # no limitation of max_size or min_freq is needed.

print("- Starting to train the model:")
print("- Defining hyper-params")

### We're ready to define everything we need for training our Seq2Seq model ###
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 50 # if choice in {1,2} else 10
learning_rate = params.lr
batch_size = params.batch_size

# Model hyperparameters
input_size_encoder = len(srcField.vocab)
input_size_decoder = len(trgField.vocab)
output_size = len(trgField.vocab)
encoder_embedding_size = params.embed_size
decoder_embedding_size = params.embed_size
hidden_size = params.hidden_size
num_layers = 1
enc_dropout = params.dropout
dec_dropout = params.dropout
measure_str = 'Edit Distance'
comment = f"epochs={num_epochs} lr={learning_rate} batch={batch_size} embed={encoder_embedding_size} hidden_size={hidden_size}"

print(f"Hyper-Params: {comment}")
concat_to_file(log_file, f"Hyper-Params: {comment}\n")
print("- Defining a SummaryWriter object")
# Tensorboard to get nice loss plot
writer = SummaryWriter(os.path.join(outputs_dir,"runs"), comment=comment)
step = 0

print("- Generating BucketIterator objects")
train_iterator, dev_iterator, test_iterator = BucketIterator.splits(
    (train_data, dev_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

print("- Constructing networks")
encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout,).to(device)
model = Seq2Seq(encoder_net, decoder_net).to(device)
print("- Defining some more stuff...")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = srcField.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load(os.path.join(outputs_dir,"my_checkpoint.pth.tar")), model, optimizer)

# for Oracle 1.5: aggregating test instances with the same lemma and tags
agg_dict = {}
if params.agg:
    agg_dict = defaultdict(list)
    for i in range(len(test_data)):
        ex = test_data.examples[i]
        lemma = "".join([list(group) for k, group in itertools.groupby(ex.src, lambda x: x == "&") if not k][0])
        tags = ";".join([list(group) for k, group in itertools.groupby(ex.src, lambda x: x == "$") if not k][1])
        agg_dict[lemma + "$" + tags].append(i)

random.seed(42)
indices = random.sample(range(len(test_data)), k=10)
accs, eds = [], []
# examples_for_printing = random.sample(test_data.examples,k=10)
# validation_sentences = test_data.examples[indices]

print("Let's begin training!\n")
concat_to_file(log_file,"Training...\n")

# prev_best_dev_ed = None
# prev_best_epoch = 0
# patience = 10
# cur_attempt = 0

# epoch = 0
# while epoch < num_epochs:
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]  (lang={lang})")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    model.eval()
    examples_for_printing = [test_data.examples[i] for i in indices] # For a more sufficient evaluation, we apply translate_sentence on 10 samples.
    translated_sentences = [translate_sentence(model, ex.src, srcField, trgField, device, max_length=50) for ex in examples_for_printing]
    # print(f"Translated example sentence: \n {translated_sentences}")
    for i,translated_sent in enumerate(translated_sentences):
        ex = examples_for_printing[i]
        if translated_sent[-1]=='<eos>': translated_sent = translated_sent[:-1]
        src_print = ''.join(ex.src)
        trg_print = ''.join(ex.trg)
        pred_print = ''.join(translated_sent)
        ed_print = utils.editDistance(trg_print, pred_print)
        print(f"{i+1}. input: {src_print} ; gold: {trg_print} ; pred: {pred_print} ; ED = {ed_print}")

    # train_result, train_accuracy = bleu(train_data, model, srcField, trgField, device, measure_str=measure_str, agg=agg_dict)
    # writer.add_scalar("Train Accuracy", train_accuracy, global_step=epoch)
    # print(f"train avgED = {train_result}; train avgAcc = {train_accuracy}")
    dev_result, dev_accuracy = bleu(dev_data, model, srcField, trgField, device, measure_str=measure_str, agg=agg_dict)
    writer.add_scalar("Dev Accuracy", dev_accuracy, global_step=epoch)
    print(f"dev avgED = {dev_result}; dev avgAcc = {dev_accuracy}")
    test_result, test_accuracy = bleu(test_data, model, srcField, trgField, device, measure_str=measure_str, agg=agg_dict)
    writer.add_scalar("Test Accuracy", test_accuracy, global_step=epoch)
    print(f"test avgED = {test_result}; test avgAcc = {test_accuracy}")

    # if prev_best_dev_ed and result >= prev_best_dev_ed:
    #     cur_attempt += 1
    #     if cur_attempt > patience:
    #         print("Early stopping: exhausted patience")
    #         load_checkpoint(torch.load(os.path.join(outputs_dir,"my_checkpoint.pth.tar")), model, optimizer)
    #         break
    #     else:
    #         print(f"No improvement in dev accuracy, attempt: {cur_attempt}")
    #         load_checkpoint(torch.load(os.path.join(outputs_dir,"my_checkpoint.pth.tar")), model, optimizer)
    #         learning_rate *= 0.75
    #         set_lr(optimizer, learning_rate)
    #         print(optimizer)
    # else:
    #     prev_best_dev_ed = result
    #     prev_best_epoch = epoch
    #     epoch += 1
    #     cur_attempt = 0
    #     # load_checkpoint(torch.load(os.path.join(outputs_dir,"my_checkpoint.pth.tar")), model, optimizer)

    if save_model:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),}
        save_checkpoint(checkpoint, os.path.join(outputs_dir, "my_checkpoint.pth.tar"))


    print()
    accs.append(dev_accuracy)
    eds.append(dev_result)

# running on entire test data takes a while
# score = bleu(test_data[1:100], model, srcField, trgField, device, measure_str='ed')
result, accuracy = bleu(test_data, model, srcField, trgField, device, measure_str=measure_str, agg=agg_dict)
lang_runtime = datetime.now() - lang_t0
output_s = f"Results for Language={lang} from Family={lang2family[lang]}:\n {lang} {measure_str} score on test set" \
            f" is {result:.2f}.\n {lang} Average Accuracy is {accuracy*100:.2f}.\n Elapsed time is {lang_runtime}.\n\n"
concat_to_file(log_file, output_s) # write to log file
print(output_s) # write to screen

tot_runtime_s = f'\nTotal runtime: {str(datetime.now() - total_timer)}\n'

concat_to_file(log_file, tot_runtime_s)
print(tot_runtime_s)
