import numpy as np
import tensorflow as tf
from tensorflow.keras import layers , activations , models
from tensorflow.keras import preprocessing , utils
import os
import io
import json
import yaml
import re

regex = '(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )'

######## Reading in the data
data_path = "./chatbot_nlp/data"
files = os.listdir(data_path+"/")

questions = list()
answers = list()

for file_name in files:
    stream = open(data_path+"/"+file_name)
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for conv in conversations:
        replies = conv[1:]
        ans = ''
        for rep in replies:
            if type(rep) is str:
                ans += ' ' + rep
        if ans is not '':
            answers.append(re.sub(regex, r' ', ans))
            questions.append(re.sub(regex, r' ', conv[0]))

for i in range(len(answers)):
    answers[i] = "<START> " + answers[i] + " <END>"

# Tokenizing
t = preprocessing.text.Tokenizer(filters='')
t.fit_on_texts(questions+answers)
vocab_size = len(t.word_index)+1
print("Vocab size: ", vocab_size)

tokenizer_json = t.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

print("Saved tokenizer")

######## Creating embedding matrix
# creating a dictionary of all words
embedding_index = dict()
file = open('./data/glove.6B.200d.txt', encoding="utf8")
for line in file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs

print('Loaded %s word vectors.' % len(embedding_index))

# creating the matrix itself
embedding_matrix = np.zeros((vocab_size, 200))
for word, i in t.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector

np.save('./data/embedding_matrix.npy', embedding_matrix)
print("Saved matrix")
