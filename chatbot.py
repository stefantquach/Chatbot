import numpy as np
import tensorflow as tf
from tensorflow.keras import layers , activations , models
from tensorflow.keras import preprocessing , utils
import os
import json
import io
import yaml
import re
import h5py

regex = '(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )'
# vocab_size=1912
#
# # import tokenizer
# with open('tokenizer.json') as f:
#     data = json.load(f)
#     tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

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
question_maxlen = max([len(x) for x in questions])
answer_maxlen = max([len(x) for x in answers])

def str_to_tokens( sentence : str ):
    sentence_ = re.sub(regex, r' ', sentence)
    words = sentence_.lower().split()
    tokens_list = list()
    for word in words:
        # try:
        a = t.word_index[ word ]
        # except KeyError:
        #     a = 2
        tokens_list.append( a )
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=question_maxlen , padding='post')


def create_inference_model(path_enc, path_dec):
    encoder_model = tf.keras.models.load_model(path_enc)
    decoder_model = tf.keras.models.load_model(path_dec)
    return encoder_model , decoder_model


enc_model , dec_model = create_inference_model('./encoder_model.h5', './decoder_model.h5')

for _ in range(10):
    states_values = enc_model.predict( str_to_tokens( input( 'Enter question : ' ) ) )
    # print(states_values)
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = t.word_index['<start>']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in t.word_index.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word )
                sampled_word = word

        if sampled_word == '<end>' or len(decoded_translation.split()) > answer_maxlen:
            stop_condition = True

        empty_target_seq = np.zeros( ( 1 , 1 ) )
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ]

    print( decoded_translation )
