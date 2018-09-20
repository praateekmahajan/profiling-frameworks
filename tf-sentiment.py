import numpy as np
import os
import sys
import tensorflow as tf
from utils import *

EPOCHS=3
BATCHSIZE=64
EMBEDSIZE=125
NUMHIDDEN=100
DROPOUT=0.2
LR=0.001
BETA_1=0.9
BETA_2=0.999
EPS=1e-08
MAXLEN=150 #maximum size of the word sequence
MAXFEATURES=30000 #vocabulary size
GPU=True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_symbol(CUDNN=True, 
                  maxf=MAXFEATURES, edim=EMBEDSIZE, nhid=NUMHIDDEN, batchs=BATCHSIZE):
    word_vectors = tf.contrib.layers.embed_sequence(X, vocab_size=maxf, embed_dim=edim)
    word_list = tf.unstack(word_vectors, axis=1)
    
    if not CUDNN:
        cell = tf.contrib.rnn.GRUCell(nhid)
        outputs, states = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)
    else:
        # Using cuDNN since vanilla RNN
        from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
        cudnn_cell = cudnn_rnn_ops.CudnnGRU(num_layers=1, 
                                            num_units=nhid, 
                                            input_size=edim, 
                                            input_mode='linear_input')
        params_size_t = cudnn_cell.params_size()
        params = tf.Variable(tf.random_uniform([params_size_t], -0.1, 0.1), validate_shape=False)   
        input_h = tf.Variable(tf.zeros([1, batchs, nhid]))
        outputs, states = cudnn_cell(input_data=word_list,
                                     input_h=input_h,
                                     params=params)
        logits = tf.layers.dense(outputs[-1], 2, activation=None, name='output')
    return logits


def init_model(m, y, lr=LR, b1=BETA_1, b2=BETA_2, eps=EPS):
    # Single-class labels, don't need dense one-hot
    # Expects unscaled logits, not output of tf.nn.softmax
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=m, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(lr, b1, b2, eps)
    training_op = optimizer.minimize(loss)
    return training_op

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    print(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)

    # Place-holders
    X = tf.placeholder(tf.int32, shape=[None, MAXLEN])
    y = tf.placeholder(tf.int32, shape=[None])
    sym = create_symbol()

    model = init_model(sym, y)
    init = tf.global_variables_initializer()
    sess =  tf.Session()  
    sess.run(init)

    correct = tf.nn.in_top_k(sym, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    for j in range(EPOCHS):
        for data, label in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):
            sess.run(model, feed_dict={X: data, y: label})
        # Log
        acc_train = sess.run(accuracy, feed_dict={X: data, y: label})
        print(j, "Train accuracy:", acc_train)


    n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE
    y_guess = np.zeros(n_samples, dtype=np.int)
    y_truth = y_test[:n_samples]
    c = 0
    for data, label in yield_mb(x_test, y_test, BATCHSIZE):
        pred = tf.argmax(sym, 1)
        output = sess.run(pred, feed_dict={X: data})
        y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = output
        c += 1 

    print("Accuracy: ", 1.*sum(y_guess == y_truth)/len(y_guess))

