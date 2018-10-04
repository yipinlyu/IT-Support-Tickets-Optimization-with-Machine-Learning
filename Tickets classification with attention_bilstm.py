# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:19:09 2018

@author: yipin
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import time
import os
os.chdir("D:/Intern/Project/")
os.getcwd()


MAX_DOCUMENT_LENGTH = 25
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 64
ATTENTION_SIZE = 64
lr = 1e-3
BATCH_SIZE = 256
KEEP_PROB = 0.5
LAMBDA = 0.0001

MAX_LABEL = 14
epochs = 10




def to_one_hot(y):
    n_class = len(pd.unique(y))
    return np.eye(n_class)[y.astype(int)]


def data_preprocessing(train, test, max_len):
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
    x_transform_train = vocab_processor.fit_transform(train)
    x_transform_test = vocab_processor.transform(test)
    vocab = vocab_processor.vocabulary_
    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)
    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)
    
    return x_train, x_test, vocab, vocab_size


def split_dataset(x_test, y_test, dev_ratio):
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    print(dev_size)
    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]
    return x_test, x_dev, y_test, y_dev, dev_size, test_size - dev_size


def fill_feed_dict(data_X, data_Y, batch_size):
    # Shuffle data first.
    perm = np.random.permutation(data_X.shape[0])
    data_X = data_X[perm]
    data_Y = data_Y[perm]
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = data_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = data_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch
        
def main():
    df = pd.read_csv('./Data/correct_cleane_all.csv',encoding = 'cp1252')
    X = df['short_desc_cleaned']
    y = df['category_id']
    
    y = to_one_hot(y)
    ##--initial data processing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    x_train, x_test, vocab, vocab_size = data_preprocessing(X_train.astype(str), X_test.astype(str), MAX_DOCUMENT_LENGTH)
    
    print(vocab_size)
    print(y.shape)
    
    x_test, x_dev, y_test, y_dev, dev_size, test_size = \
    split_dataset(x_test, y_test, 0.1)
    print("Validation size: ", dev_size)
    
    graph = tf.Graph()
    
    with graph.as_default():
        
        batch_x = tf.placeholder(tf.int32, [None, MAX_DOCUMENT_LENGTH])
        batch_y = tf.placeholder(tf.float32, [None, MAX_LABEL])
        keep_prob = tf.placeholder(tf.float32)
        
        embeddings_var = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0), trainable=True)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_x)
        W = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=0.1))
        # print(batch_embedded.shape)  # (?, 256, 100)
        
        rnn_outputs, _ = bi_rnn(BasicLSTMCell(HIDDEN_SIZE), BasicLSTMCell(HIDDEN_SIZE),
                                inputs=batch_embedded, dtype=tf.float32)
        # Attention
        fw_outputs = rnn_outputs[0]
        print(fw_outputs.shape)
        bw_outputs = rnn_outputs[1]
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H) # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
        print(M.shape)
        #alpha (bs * sl, 1)
        alpha = tf.nn.softmax(tf.matmul(tf.reshape(M, [-1, HIDDEN_SIZE]), tf.reshape(W, [-1, 1])))
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, MAX_DOCUMENT_LENGTH, 1])) # supposed to be (batch_size * HIDDEN_SIZE, 1)
        print(r.shape)
        r = tf.squeeze(r)
        h_star = tf.tanh(r) # (batch , HIDDEN_SIZE
        #attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
        
        
        drop = tf.nn.dropout(h_star, keep_prob)
        # shape = drop.get_shape()
        # print(shape)
        
        # Fully connected layer（dense layer)
        W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE, MAX_LABEL], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[MAX_LABEL]))
        y_hat = tf.nn.xw_plus_b(drop, W, b)
        # print(y_hat.shape)
        
        # y_hat = tf.squeeze(y_hat)
        
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=batch_y))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        
        # Accuracy metric
        prediction = tf.argmax(tf.nn.softmax(y_hat), 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(batch_y, 1)), tf.float32))
        
        steps = 10001
        
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        print("Initialized! ")
    
        print("Start trainning")
        start = time.time()
        for e in range(epochs):
    
            epoch_start = time.time()
            print("Epoch %d start !" % (e + 1))
            for x_batch, y_batch in fill_feed_dict(x_train, y_train, BATCH_SIZE):
                fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: KEEP_PROB}
                l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)
    
            epoch_finish = time.time()
            print("Validation accuracy and loss: ", sess.run([accuracy, loss], feed_dict={
                batch_x: x_dev,
                batch_y: y_dev,
                keep_prob: 1.0
            }))
    
        print("Training finished, time consumed : ", time.time() - start, " s")
        print("Start evaluating:  \n")
        cnt = 0
        test_acc = 0
        for x_batch, y_batch in fill_feed_dict(x_test, y_test, BATCH_SIZE):
                fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: 1.0}
                acc = sess.run(accuracy, feed_dict=fd)
                test_acc += acc
                cnt += 1        
        
        print("Test accuracy : %f %%" % ( test_acc / cnt * 100))



    
    
