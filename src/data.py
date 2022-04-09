from collections import namedtuple
import os
import nltk
from nltk.stem.snowball import SnowballStemmer
import torch
import treetaggerwrapper

BOUNDARY = '#'
CONTEXT_SIZE = 3

features = [
    'close_', 'close_mid', 'open_mid', 'open_', 'high', 'mid', 'low', 'front', 'back', 'wide', 'round_',
    'tenuis', 'media', 'sibilant', 'spirant', 'nasal', 'tremulant', 'lateral', 'semivowel',
    'bilabial', 'labiodental', 'pro', 'medio', 'post', 'palatal', 'velar', 'laryng', 'voiced', 'consonant',
    'non_palatalization', 'palatalization']

Phone = namedtuple('Phone', features, defaults=(0,) * len(features))

finnish_letters = 'abcdefghijklmnopqrstuvwxyzåäö'

finnish_phones = {
    'a': Phone(open_=1, low=1, back=1, wide=1),
    'b': Phone(media=1, bilabial=1, voiced=1, consonant=1),
    'c': Phone(tenuis=1, velar=1),
    'd': Phone(media=1, medio=1, voiced=1),
    'e': Phone(close_mid=1, open_mid=1, mid=1, front=1, wide=1),
    'é': Phone(close_mid=1, open_mid=1, mid=1, front=1, wide=1),
    'f': Phone(spirant=1, labiodental=1, consonant=1),
    'g': Phone(media=1, velar=1, voiced=1, consonant=1),
    'h': Phone(spirant=1, laryng=1, consonant=1),
    'i': Phone(close_=1, high=1, front=1, wide=1),
    'j': Phone(semivowel=1, palatal=1, consonant=1),
    'k': Phone(tenuis=1, velar=1, consonant=1),
    'l': Phone(lateral=1, pro=1, medio=1, post=1, consonant=1),
    'm': Phone(nasal=1, bilabial=1, voiced=1, consonant=1),
    'n': Phone(nasal=1, pro=1, medio=1, post=1, voiced=1, consonant=1),
    'o': Phone(close_mid=1, open_mid=1, mid=1, back=1, round_=1),
    'p': Phone(tenuis=1, bilabial=1, consonant=1),
    'q': Phone(tenuis=1, velar=1, consonant=1),
    'r': Phone(tremulant=1, pro=1, medio=1, post=1, voiced=1, consonant=1),
    's': Phone(sibilant=1, pro=1, medio=1, consonant=1),
    't': Phone(tenuis=1, pro=1, consonant=1),
    'u': Phone(close_=1, high=1, back=1, round_=1),
    'v': Phone(semivowel=1, labiodental=1, voiced=1, consonant=1),
    'w': Phone(semivowel=1, labiodental=1, consonant=1),
    'x': Phone(tenuis=1, velar=1, consonant=1),
    'y': Phone(close_=1, high=1, front=1, round_=1),
    'z': Phone(sibilant=1, pro=1, medio=1, voiced=1, consonant=1),
    'å': Phone(close_mid=1, open_mid=1, mid=1, back=1, round_=1),
    'ä': Phone(open_=1, low=1, front=1, wide=1),
    'ö': Phone(close_mid=1, open_mid=1, mid=1, front=1, round_=1),
    'ij': Phone(semivowel=1, palatal=1, consonant=1),
    'ng': Phone(nasal=1, velar=1, consonant=1),
    'ts': (Phone(tenuis=1, pro=1, consonant=1), Phone(sibilant=1, pro=1, medio=1, consonant=1)),
    'sh': Phone(sibilant=1, medio=1, wide=1, consonant=1),
    BOUNDARY: Phone()
}

def all_finnish_letters(word):
    return all([finnish_letters.find(letter) != -1 for letter in word])

def flattened_phone_vector(word, phones):
    pv = []
    index = 0
    while index < len(word):
        phone = None
        if index < len(word) - 1:
            if word[index + 1] in phones:
                next_phone = phones[word[index + 1]]
                current_phone = phones[word[index]]
                if isinstance(next_phone, Phone):
                    if getattr(next_phone, 'non_palatalization') == 1:
                        if isinstance(current_phone, Phone):
                            phone = current_phone._replace(palatal=0)
                        else:
                            phone = list(current_phone)
                            phone[-1] = phone[-1]._replace(palatal=0)
                            phone = tuple(phone)
                        index += 1
                    elif getattr(next_phone, 'palatalization') == 1:
                        if isinstance(current_phone, Phone):
                            phone = current_phone._replace(palatal=1)
                        else:
                            phone = list(current_phone)
                            phone[-1] = phone[-1]._replace(palatal=1)
                            phone = tuple(phone)
                        index += 1
            if not phone:
                letters = word[index:index + 2]
                if letters in phones:
                    phone = phones[letters]
                    index += 1
        if not phone:
            phone = phones[word[index]]
        if isinstance(phone, Phone):
            for value in phone:
                pv.append(value)
        else:
            for a_phone in phone:
                for value in a_phone:
                    pv.append(value)
        index += 1
    return pv

def read_gutenberg_tokens(fn, data, vocab, stemmer, tagger, line_number, verify_tokens):
    text_started = False
    last_line = None
    first_line_skipped = False
    tokens = []
    with open(fn, encoding='utf-8') as file:
        for line in file:
            line_number += 1
            if line_number % 100 == 0:
                print(line_number, end='\r')
            line = line.strip('\n')
            if not line:
                continue
            if not text_started:
                if line.startswith('*** START OF THIS PROJECT GUTENBERG EBOOK '):
                    text_started = True
                continue
            if not first_line_skipped:
                first_line_skipped = True
                continue
            if line.startswith('*** END OF THIS PROJECT GUTENBERG EBOOK'):
                break
            if last_line:
                word_tokens = nltk.tokenize.word_tokenize(last_line)
                line_started = False
                for word_token in word_tokens:
                    if not line_started and all(char.isupper() or not char.isalpha() for char in word_token):
                        continue
                    line_started = True
                    if word_token.isalnum():
                        token = word_token.lower()
                        tokens.append(token)
                    else:
                        tokens.append(BOUNDARY)
            last_line = line
    print(line_number, end='\r')
    context = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == BOUNDARY:
            if len(context) == CONTEXT_SIZE:
                data.append(context)
                if verify_tokens:
                    if context[int(CONTEXT_SIZE / 2)] in verify_tokens:
                        print(context)
            context = []
        else:
            tags = tagger.tag_text(text=token, tagonly=True)
            tag = tags[0].split('\t')[1]
            if not tag.startswith('N_') and not tag.startswith('V_'):
                index += 1
                continue
            token = stemmer.stem(token)
            context.append(token)
            if len(context) == CONTEXT_SIZE:
                data.append(context)
                if verify_tokens:
                    if context[int(CONTEXT_SIZE / 2)] in verify_tokens:
                        print(context)
                index -= CONTEXT_SIZE - 1
                for token in context:
                    vocab.update(get_subtokens(token))
                context = []
        index += 1
    return line_number

def get_subtokens(token):
    subtokens = set()
    for ngram_length in range(3, 7):
        for ngram_index in range(0, len(token) - ngram_length + 1, 1):
            ngram = token[ngram_index:ngram_index + ngram_length]
            if ngram_index == 0:
                ngram = BOUNDARY + ngram
            if ngram_index + ngram_length == len(token):
                ngram = ngram + BOUNDARY
            subtokens.add(ngram)
    subtokens.add(BOUNDARY + token + BOUNDARY)
    return subtokens

def read_datasets(prefix, data_dir, embeds_only=False):
    stemmer = SnowballStemmer('finnish')
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='fi')
    fns = os.listdir(data_dir)
    data = []
    vocab = set()
    line_number = 0
    for fn in fns:
        if fn.startswith(prefix) and fn.endswith('.txt'):
            print(fn)
            line_number = read_gutenberg_tokens(os.path.join(data_dir, fn), data, vocab, stemmer, tagger, line_number, ['haastelev', 'peräk'])
    vocab = list(vocab)
    charmap = {char: index for index, char in enumerate(
        {letter for token in vocab for letter in token})}
    tokenmap = {token: index for index, token in enumerate(vocab)}
    for index in range(0, len(data)):
        token_context = data[index]
        context = token_context + ([BOUNDARY] * (CONTEXT_SIZE - len(token_context)))
        data[index] = (torch.tensor([[tokenmap[BOUNDARY + context[token_index] + BOUNDARY] for token_index in list(range(0, int(len(token_context) / 2))) + list(range(int(len(token_context) / 2) + 1, len(context)))]]), torch.tensor(tokenmap[BOUNDARY + context[int(len(token_context) / 2)] + BOUNDARY]))
    tokens = [None] * len(vocab)
    max_len = len(max(vocab, key=len))
    for index in range(0, len(vocab)):
        word = vocab[index]
        token = word + (BOUNDARY * (max_len - len(word)))
        tokens[index] = [charmap[char] for char in token]
    tokens = torch.tensor(tokens)
    if embeds_only:
        dist_data_index1 = 0
        dist_data_index2 = 0
    else:
        dist_data_index1 = int(len(data) * 0.15)
        dist_data_index2 = int(len(data) * 0.3)
    datasets = {'training': data[dist_data_index2:],
                'dev': data[:dist_data_index1],
                'test': data[dist_data_index1:dist_data_index2]}
    print('Finished reading datasets')
    return datasets, tokens, vocab, tokenmap

if __name__=='__main__':
    from paths import data_dir
    d, t, v, m = read_datasets('pg', data_dir)


