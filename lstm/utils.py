import os.path
# import io

import numpy as np
import torch
# import spacy
from torchtext.data.metrics import bleu_score
# from torchtext.legacy.data.example import Example
# import sys
import random
from torchtext.legacy.data import Field
from copy import deepcopy

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

REINFLECTION_STR = 'reinflection'
INFLECTION_STR = 'inflection'

# print("- Loading tokenizers")
# spacy_eng = spacy.load('en_core_web_sm')
# spacy_ger = spacy.load('de_core_news_sm')

# def tokenize_ger(text):
#     return [tok.text for tok in spacy_ger.tokenizer(text)]
# def tokenize_eng(text):
#     return [tok.text for tok in spacy_eng.tokenizer(text)]
src_tokenizer = lambda x: x.split(',')
trg_tokenizer = lambda x: x.split(',')

# print("- Defining Field objects")
# german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
# english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
srcField = Field(tokenize=src_tokenizer,init_token="<sos>",eos_token="<eos>")
trgField = Field(tokenize=trg_tokenizer,init_token="<sos>",eos_token="<eos>")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def translate_sentence(model, sentence, german, english, device, max_length=50, return_attn=False):
    # Load german tokenizer
    # spacy_ger = spacy.load("de_core_news_sm")
    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    # if type(sentence) == str:
    #     tokens = [token.text.lower() for token in spacy_ger(sentence)]
    # else:
    #     tokens = [token.lower() for token in sentence]
    assert type(sentence) == list
    tokens = deepcopy(sentence)

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        outputs_encoder, hiddens, cells = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]
    attention_matrix = torch.zeros(max_length, max_length)
    for i in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        with torch.no_grad():
            output, hiddens, cells, attn = model.decoder(previous_word, outputs_encoder, hiddens, cells, return_attn=return_attn)
            best_guess = output.argmax(1).item()
            if return_attn: attention_matrix[i] = torch.cat((attn.squeeze(), torch.zeros(max_length-attn.shape[0]).to(device)),dim=0)

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    if return_attn:
        return translated_sentence[1:], attention_matrix[:len(translated_sentence)+2, :len(sentence)+2]
    else:
        return translated_sentence[1:]

def bleu(data, model, german, english, device, measure_str='bleu', agg={}): # measure_str can be either 'bleu' or 'ed'
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)
    if measure_str=='bleu':
        acc = "Undefined"
        res = bleu_score(outputs, targets)
    else:
        # Count also Accuracy. Ignore <eos>, obviously.
        targets = [t[0] for t in targets]
        # Aggregation based on edit distance
        if agg:
            eval_indices = []
            for indices in agg.values():
                ed_per_lemma = [editDistance(targets[i], outputs[i]) for i in indices]
                best_idx = indices[np.argmin(ed_per_lemma)]
                eval_indices.append(best_idx)
                print("Gold:", set(["".join(targets[i]) for i in indices]))
                print("Predictions:", set(["".join(outputs[i]) for i in indices]))
                print("Best:", "".join(outputs[best_idx]))
                print("----")
        else:
            eval_indices = range(len(targets))
        acc = np.array([targets[i]==outputs[i] for i in eval_indices]).mean()
        # acc = (np.array(targets, dtype=object)==np.array(outputs, dtype=object)).sum()/len(data)
        res = np.mean([editDistance(targets[i], outputs[i]) for i in eval_indices])
    return res, acc

def editDistDP(str1, str2, m, n):
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] # Create a table to store results of subproblems
    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
            # If first string is empty, only option is to insert all characters of second string
            if i == 0:
                dp[i][j] = j  # Min. operations = j
            # If second string is empty, only option is to remove all characters of second string
            elif j == 0:
                dp[i][j] = i  # Min. operations = i

            # If last characters are same, ignore last char and recur for remaining string
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # If last character are different, consider all possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                   dp[i - 1][j],  # Remove
                                   dp[i - 1][j - 1])  # Replace
    return dp[m][n]

def editDistance(s1,s2):
    return editDistDP(s1,s2,len(s1),len(s2))

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer, verbose=True):
    if verbose: print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def reinflection2sample(line, mode=REINFLECTION_STR):
    assert mode in {REINFLECTION_STR, INFLECTION_STR}
    if mode==REINFLECTION_STR:
        # The line format is [src_feat, src_form, trg_feat, trg_form]
        src_feat, src_form, trg_feat, trg_form = line
        src_feat, trg_feat = src_feat.split(";"), trg_feat.split(";")
        src_form, trg_form = list(src_form), list(trg_form)
        src = ','.join(src_feat + ['+'] + src_form + ['+'] + trg_feat)
        trg = ','.join(trg_form)
    else: # inflection mode
        lemma, form, feat = line
        feat = feat.split(";")
        lemma, form = list(lemma), list(form)
        src = ','.join(lemma + ['$'] + feat) # Don't use '+' as it is part of some tags (see the file tags.yaml).
        trg = ','.join(form)
    return src, trg

def reinflection2TSV(fn, dir_name="data", mode=REINFLECTION_STR):
    """
    Convert a file in the Reinflection format (src_feat\tsrc_form\ttrg_feat\ttrg_form) to a TSV file of the format
    src\ttrg, each one consists of CSV strings. If mode=inflection, then count the data as SIGMORPHON format, and fn must be a tuple of 3 paths.
    :param mode: can be either 'inflection' or 'reinflection'.
    :param dir_name:
    :param fn: if mode=reinflection, then fn: str. else, fn:Tuple(str)
    :return: The two paths of the TSV files.
    """
    assert mode in {REINFLECTION_STR, INFLECTION_STR}
    if mode==REINFLECTION_STR:
        fn = os.path.join(dir_name,fn)
        fn_wo_ext = os.path.splitext(fn)[0]
        new_fn = fn_wo_ext+".tsv"

        data = open(fn, encoding='utf8').read().split('\n')
        data = [line.split('\t') for line in data]

        examples = []
        for e in data:
            if e[0] == '': continue
            src, trg = reinflection2sample(e, mode=mode) # the only modification for supporting Inflection as well.
            examples.append(f"{src}\t{trg}")

        open(new_fn, mode='w', encoding='utf8').write('\n'.join(examples))
    else:
        train_fn, test_fn = fn[0], fn[2] # file paths without parent-directories prefix
        new_train_fn = os.path.join(dir_name, os.path.basename(train_fn)+".tsv") # use the paths without parent-directories prefixes
        new_test_fn = os.path.join(dir_name, os.path.basename(test_fn)+".tsv")

        # Removed so that the files are overwritten each time
        # if os.path.isfile(new_train_fn) and os.path.isfile(new_test_fn): return [new_train_fn, new_test_fn]

        for fn,new_fn in zip([train_fn, test_fn], [new_train_fn, new_test_fn]):
            data = open(fn, encoding='utf8').read().split('\n')
            data = [line.split('\t') for line in data]

            examples = []
            for e in data:
                if e[0] == '': continue
                src, trg = reinflection2sample(e, mode=mode) # the only modification for supporting Inflection as well.
                examples.append(f"{src}\t{trg}")

            open(new_fn, mode='w', encoding='utf8').write('\n'.join(examples))
        new_fn = [new_train_fn, new_test_fn]
        # The result is a directory "data\\LEMMA_TSV_FORMAT" with 180 files of the format 'lang.{trn|tst}.tsv'
    return new_fn

def get_langs_and_paths(data_dir='', use_hall=False):
    """
    Return a list of the languages, and a dictionary of tuples: {lang: (train_path,dev_path,test_paht)}.
    :param data_dir:
    :return:
    """
    develop_paths = {}
    surprise_paths = {}
    test_paths = {}
    lang2family = {} # a dictionary that indicates the family of every language

    for family in os.listdir(data_dir):
        for filename in os.listdir(os.path.join(data_dir, family)):
            lang, ext = os.path.splitext(filename)
            filename = os.path.join(family,filename)
            hall_file = False
            if lang.endswith('.hall'):
                lang = lang[:-5]
                hall_file = True
            if ext=='.trn':
                if hall_file == use_hall:
                    develop_paths[lang] = filename
            elif ext=='.dev':
                surprise_paths[lang] = filename
            elif ext=='.tst':
                test_paths[lang] = filename
            family_name = os.path.split(family)[1]
            if lang not in lang2family: lang2family[lang] = family_name

    langs = lang2family.keys()
    if use_hall: langs = develop_paths.keys() # not all languages have hall data
    files_paths = {k:(develop_paths[k],surprise_paths[k],test_paths[k]) for k in langs}
    return langs, files_paths, lang2family

def showAttention(input_sentence, output_words, attentions, fig_name="Attention Weights.png"):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence + ['<eos>'])#, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # plt.show()
    plt.savefig(fig_name)
