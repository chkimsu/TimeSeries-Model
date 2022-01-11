#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking
from tensorflow.keras.models import Model
from keras_self_attention import SeqSelfAttention
from keras_self_attention import SeqWeightedAttention
from tensorflow.keras.layers import concatenate

from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import pandas as pd
import ast

def get_word2vec_info(word2vec_embed_file):

    embed_data =  json.loads(open(word2vec_embed_file,"r").read())
    max_index = max(list(map(int, embed_data['reverse_dictionary'].keys())))  ## 198   -- 문자열을 int형으로 변환

    # data preprocess
    dictionary = embed_data['dictionary']
    reverse_dictionary = embed_data['reverse_dictionary']
    embedded_list = embed_data['embeddings']
    embedded_size = len(embedded_list[0])
    embedded_list = embedded_list[:-1]  ## 마지막 nan 하나 빼줘야한다. 
    embedded_list = np.append(embedded_list, [np.random.normal(size=[embedded_size])], axis=0)
    
    
    # add start, end, pad symbol
    reverse_dictionary[max_index+1] = "SOS"
    dictionary["SOS"] = max_index+1

    max_index = max_index+1
    
    embedded_list = np.append(embedded_list,[np.random.normal(size=[embedded_size])], axis=0)
    reverse_dictionary[max_index+1] = "EOS"
    dictionary["EOS"] = max_index+1
    vocabulary_size = len(embed_data['dictionary'])  # 201

    max_index = max_index+1

    embedded_list = np.append(embedded_list,[np.random.normal(size=[embedded_size])], axis=0)
    reverse_dictionary[max_index+1] = "PAD"
    dictionary["PAD"] = max_index+1
    vocabulary_size = len(embed_data['dictionary'])  # 202

    index_embed = dict(zip(list(map(int, reverse_dictionary.keys())), embedded_list))

    return dictionary, index_embed



dictionary, index_embed = get_word2vec_info("./legacy/20201229.json")


def load_dataset(path, num_examples):
  import ast
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  sequence = [ast.literal_eval(l.split('\t')[2])  for l in lines[:num_examples]]
  encoder_input = sequence
  decoder_input = list(map(lambda x : [dictionary['SOS']] + x, sequence))
  decoder_output = list(map(lambda x : [dictionary['EOS']] + x, sequence))

  return encoder_input, decoder_input, decoder_output 



def preprocess_padding(tensor):
  from tensorflow.keras.preprocessing.sequence import pad_sequences
  
  max_len = np.max(list(map(lambda x: len(x),tensor)))
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=max_len, value = dictionary['PAD'])
  return tensor




def create_tensor(path, num_examples=None):
  # creating cleaned input, output pairs
  encoder_input, decoder_input, decoder_output = load_dataset(path, num_examples)
  encoder_input = preprocess_padding(encoder_input)
  decoder_input = preprocess_padding(decoder_input)
  decoder_output = preprocess_padding(decoder_output)

  return encoder_input, decoder_input, decoder_output



encoder_input, decoder_input, decoder_output = create_tensor("./legacy/train.tsv",10000)
max_len = np.max(list(map(lambda x: len(x),encoder_input)))



class CustomEmbedding(tf.keras.layers.Layer):
  
  def __init__(self, input_dim, output_dim, mask_zero=False, **kwargs):
    super(CustomEmbedding, self).__init__(**kwargs)
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.mask_zero = mask_zero
    
  def build(self, input_shape):
    self.embeddings = self.add_weight(
      shape=(self.input_dim, self.output_dim),
      initializer='random_normal',
      dtype='float32')
    
  def call(self, inputs):
    return tf.nn.embedding_lookup(self.embeddings, inputs)
  
  def compute_mask(self, inputs, mask=None):
    if not self.mask_zero:
      return None
    return tf.not_equal(inputs, dictionary['PAD'])

def define_model():
    encoder_inputs = Input(shape=(len(encoder_input[0])), dtype = 'int32')
    customlayer = CustomEmbedding(input_dim = len(index_embed.keys()),output_dim =8,
                                  mask_zero = True, 
                                  weights = [pd.DataFrame(index_embed).transpose().values],
                                  trainable = False)
    x = customlayer(encoder_inputs)

    encoder_lstm = LSTM(units=32, return_state=True, name = 'embed')
    encoder_outputs, state_h, state_c = encoder_lstm(x)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(len(decoder_input[0])), dtype = 'int32')
    customlayer2 = CustomEmbedding(input_dim = len(index_embed.keys()),output_dim =8, mask_zero = True, weights = [pd.DataFrame(index_embed).transpose().values],
                                  trainable = False)

    y = customlayer2(decoder_inputs)

    decoder_lstm = LSTM(units=32, return_sequences=True)
    decoder_outputs= decoder_lstm(y, initial_state=encoder_states)
    decoder_outputs = SeqWeightedAttention(return_attention=True, name = 'seq')(decoder_outputs)
    decoder_outputs = concatenate(decoder_outputs)

    # 디코더의 첫 상태를 인코더의 은닉 상태, 셀 상태로 합니다.
    decoder_softmax_layer = Dense(586, activation='softmax')
    decoder_outputs = decoder_softmax_layer(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model



model = define_model()
data_tuple = tuple(zip(encoder_input, decoder_input, decoder_output))
encoder_input.shape


def generator_inputs():
    for e_i, d_i, d_o in data_tuple: 
        e_i = np.reshape(e_i, [1,max_len])
        d_i = np.reshape(d_i, [1,max_len+1])
        d_o = np.reshape(d_o, [1,max_len+1])
        yield (e_i, d_i), d_o


tfdata_gen = tf.data.Dataset.from_generator(generator_inputs, output_shapes = (([None,max_len],[None,max_len+1]),[None,max_len+1]), output_types = ((tf.int32,tf.int32),tf.int32 ) )

model.compile(optimizer="rmsprop", loss="mse")
model.fit(tfdata_gen, steps_per_epoch = 10, epochs = 10)


## 압축된 임베딩 결과 확인하는것. 

from tensorflow.keras.models import Model

layer_name = 'embed'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(encoder_input[0])

intermediate_output[0]
