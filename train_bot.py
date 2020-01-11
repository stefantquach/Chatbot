import numpy as np
import tensorflow as tf
from tensorflow.keras import layers , activations , models
from tensorflow.keras import preprocessing , utils
import os
import yaml
import re

regex = '(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )'

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

# encoder input
tokenized_q = t.texts_to_sequences(questions)
question_maxlen = max([len(x) for x in questions])
padded_q = preprocessing.sequence.pad_sequences(tokenized_q, maxlen=question_maxlen, padding='post')
encoder_input_data = np.array(padded_q)
print(encoder_input_data.shape, question_maxlen)

# decoder input
tokenized_ans = t.texts_to_sequences(answers)
answer_maxlen = max([len(x) for x in answers])
padded_ans = preprocessing.sequence.pad_sequences(tokenized_ans, maxlen=answer_maxlen, padding='post')
decoder_input_data = np.array(padded_ans)
print(decoder_input_data.shape, answer_maxlen)

# decoder output
tokenized_ans = t.texts_to_sequences(answers)
for i in range(len(tokenized_ans)):
    tokenized_ans[i]=tokenized_ans[i][1:]
padded_ans = preprocessing.sequence.pad_sequences(tokenized_ans, maxlen=answer_maxlen, padding='post')
onehot = utils.to_categorical(padded_ans, vocab_size)
decoder_output_data = np.array(onehot)
print(decoder_output_data.shape)

# Saving
np.save( './data/enc_in_data.npy' , encoder_input_data )
np.save( './data/dec_in_data.npy' , decoder_input_data )
np.save( './data/dec_tar_data.npy' , decoder_output_data )

##### Building training model
embedding_matrix = np.load('./data/embedding_matrix.npy')

embedding = layers.Embedding(vocab_size, 200, mask_zero=True, weights=[embedding_matrix], trainable=False, name='shared_embedding')

encoder_input = layers.Input(shape=(None, ), name='encoder_input')
encoder_lstm = layers.LSTM(200, return_state=True, name='encoder_lstm')
encoder_output , state_h1, state_c1 = encoder_lstm(embedding(encoder_input))
encoder_states = [state_h1, state_c1]

decoder_input = layers.Input(shape=(None, ), name='decoder_input')
decoder_lstm = layers.LSTM(200, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_output, _, _ = decoder_lstm(embedding(decoder_input), initial_state=encoder_states)
decoder_hidden = layers.Dense(vocab_size/2, activation='relu', name='decoder_hidden')
decoder_dense = layers.Dense(vocab_size, activation='softmax', name='decoder_dense')
output = decoder_dense(decoder_hidden(decoder_output))

train_model = models.Model([encoder_input, decoder_input], output)
train_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

train_model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=50, epochs=75 )
train_model.save( './models/training_model.h5' )
# train_model.load_weights('training_model.h5')

### Building inference model
encoder_model = models.Model(encoder_input, encoder_states)
encoder_model.save('./models/encoder_model.h5')

decoder_state_input_h = layers.Input(shape=( 200 ,))
decoder_state_input_c = layers.Input(shape=( 200 ,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# decoder_embedding = layers.Embedding(vocab_size, 200, weights=[embedding_matrix], trainable=False, name='decoder_embedding')
decoder_outputs, state_h, state_c = decoder_lstm(
    embedding(decoder_input) , initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_hidden(decoder_outputs))
decoder_model = tf.keras.models.Model(
    [decoder_input] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

decoder_model.save('./models/decoder_model.h5')


# def str_to_tokens( sentence : str ):
#     sentence_ = re.sub(regex, r' ', sentence)
#     words = sentence_.lower().split()
#
#     tokens_list = list()
#     for word in words:
#         try:
#             a = t.word_index[ word ]
#         except KeyError:
#             a = 2
#         tokens_list.append( a )
#     return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=question_maxlen , padding='post')
#
# for _ in range(10):
#     states_values = encoder_model.predict( str_to_tokens( input( 'Enter question : ' ) ) )
#     empty_target_seq = np.zeros( ( 1 , 1 ) )
#     empty_target_seq[0, 0] = t.word_index['<start>']
#     stop_condition = False
#     decoded_translation = ''
#     while not stop_condition :
#         dec_outputs , h , c = decoder_model.predict([ empty_target_seq ] + states_values )
#         sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
#         sampled_word = None
#         for word , index in t.word_index.items() :
#             if sampled_word_index == index :
#                 decoded_translation += ' {}'.format( word )
#                 sampled_word = word
#
#         if sampled_word == '<end>' or len(decoded_translation.split()) > answer_maxlen:
#             stop_condition = True
#
#         empty_target_seq = np.zeros( ( 1 , 1 ) )
#         empty_target_seq[ 0 , 0 ] = sampled_word_index
#         states_values = [ h , c ]
#
#     print( decoded_translation )
