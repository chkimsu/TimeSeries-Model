import math
import os
from os import environ
import csv
import numpy as np
import pandas as pd
import sys
import datetime
import time
import configparser
import json
import collections
from six.moves import xrange
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)

class Log2Vec:
    def __init__(self, dictionary, embedding_dim=32,bag_window=2,batch_size=200, optimizer='sgd',epochs=10000):
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.bag_window = bag_window
        self.batch_size = batch_size
        self.alpha = 0.8
        self.dictionary_data = pd.read_csv(dictionary, header=None, delimiter='\t')
        self.reverse_dictionary = {}
        self.dictionary = {}
        for i in self.dictionary_data.itertuples():
            self.reverse_dictionary[i[1]] = i[2]
            self.dictionary[i[2]] = i[1]

        self.vocab_size = len(self.dictionary)

        if optimizer=='adam':
            self.optimizer = tf.optimizers.Adam()
        else:
            self.optimizer = tf.optimizers.SGD(learning_rate=0.1)
    
    @tf.function
    def train(self, x_train=None, y_train=None):
        # Look up embeddings for inputs.
        print("train start")
        self.W1 = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_dim]))
        self.b1 = tf.Variable(tf.random.normal([self.embedding_dim]))

        self.W2 = tf.Variable(tf.random.normal([self.embedding_dim,self.vocab_size]))
        self.b2 = tf.Variable(tf.random.normal([self.vocab_size]))
        
        dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(100)

        for _ in range(self.epochs):
            for xTrain,yTrain in dataset:

                with tf.GradientTape() as t:
                    hidden_layer = tf.add(tf.matmul(xTrain,self.W1),self.b1)
                    output_layer = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, self.W2), self.b2))
                    cross_entropy_loss = tf.reduce_mean(-tf.math.reduce_sum(yTrain * tf.math.log(output_layer), axis=[1]))
                grads = t.gradient(cross_entropy_loss, [self.W1,self.b1, self.W2, self.b2])
                self.optimizer.apply_gradients(zip(grads, [self.W1,self.b1, self.W2, self.b2]))
                if(_ % 10 == 0):
                    print(cross_entropy_loss)

    def vectorized(self, idx):
        return (self.W1+self.b1)[idx]
    def all_vector(self):
        return (self.W1+self.b1).numpy()
    
    def to_one_hot(self,index,vocab_size):
        temp = np.zeros(vocab_size)
        if index != vocab_size:
            temp[index]=1
        else:
            temp[index]=0
        return temp

    def to_one_hot_weight(self,index,vocab_size,weight):
        temp = np.zeros(vocab_size)
        if index != vocab_size:
            temp[index] = 1*weight
        else:
            temp[index] = 0
        return temp

    def time2io(self,seq_data,time_data, V, window_size):
        alpha=self.alpha
        t_idx = 0
        res_batch = []
        res_label = []
        res_len = []
        for words in seq_data:
          time_list = time_data[t_idx]
          L = len(words)
          for index, word in enumerate(words):
            contexts = []
            labels = []
            s = index - window_size
            e = index + window_size + 1
            contexts.append([words[i] for i in range(s, e) if 0 <= i < L and i != index])
            res_len.append([len(contexts[0])])
            times = []
            for i in range(s, e):
              if 0 <= i < L and i != index:
                if i > index:
                    weight = np.sum(time_list[index+1:i+1])
                if i < index:
                    weight = np.sum(time_list[i:index+1])
                times.append(pow(alpha,weight))
            times_sum = np.sum(times,dtype=np.float32)
            while (len(contexts[0])!=window_size*2) :
              contexts[0].append(self.vocab_size)
              times.append(0)
            labels.append(word)
            with np.errstate(divide='ignore') :
              time_weight = np.true_divide(times,times_sum)
            one_hot_v = []
            v_idx = 0
            for i in contexts[0]:
                if i == self.vocab_size:
                    one_hot_v.append(np.zeros(self.vocab_size))
                else:
                    one_hot_v.append(self.to_one_hot_weight(index=int(i), vocab_size=self.vocab_size, weight=time_weight[v_idx]))
                v_idx = v_idx+1
            sum_v = [sum(i) for i in zip(*one_hot_v)]
            res_batch.append(sum_v)
            label_v = self.to_one_hot(index=labels, vocab_size = self.vocab_size)
            res_label.append(label_v)
          t_idx=t_idx+1

        return res_batch,res_len,res_label


    def generate_batch(self,data_dict):
        try:
          data = list()
          # print("bstart",time.time())
          span = 2 * self.bag_window + 1
          batch = np.ndarray(shape=(self.batch_size, span - 1), dtype=np.int32)
          for i in data_dict["log_seq"]:
            data.append(list(map(int,i.replace("[","").replace("]","").replace(" ","").split(","))))
          data_time = list()
          for i in data_dict["log_gap_time"]:
            tmp_time = list(map(float,i.replace("[","").replace("]","").replace(" ","").split(",")))
            tmp_time = np.log10(tmp_time)
            tmp_time[np.isneginf(tmp_time)] = 0
            data_time.append(tmp_time)
          res_batch,res_len,labels= self.time2io(data,data_time, self.vocab_size,self.bag_window)
            
          return res_batch,res_len,labels
        except Exception as e :
          print("failed generating batch")
          print(e)



train_data_path = sys.argv[1]
dictionary_data_path = sys.argv[2]

data = pd.read_csv(train_data_path,delimiter="\t")
data.columns = ["log_seq","log_gap_time"]


L2V = Log2Vec(dictionary_data_path, embedding_dim=32,bag_window=2, optimizer='sgd',epochs=1000)
seq,seq_len,seq_label = L2V.generate_batch(data)


seq = np.asarray(seq, dtype='float32')
seq_label = np.asarray(seq_label, dtype='float32')

L2V.train(x_train=seq,y_train=seq_label)

pickle_data = {
                'embeddings': L2V.all_vector().tolist(),
                'dictionary': L2V.dictionary,
                'reverse_dictionary': L2V.reverse_dictionary
        }
with open('l2v_results.json' , "w" , encoding="utf-8") as f :
    json.dump(pickle_data,f,ensure_ascii=False)
