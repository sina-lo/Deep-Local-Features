from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from random import shuffle
import argparse
import sys
import scipy
import tempfile
from scipy import io
import h5py
from math import ceil
import math
import numpy as np
import tensorflow as tf
import os
from layers import batch_norm, conv_2d, norm_spatial_subtractive, pool_l2
from loss import loss_matching, loss_non_matching
from eval_PR import ErrorRateAt95Recall


FLAGS = None
#batch_size = 200
#margin_nonmatch = 0.3
#margin_match = 0.3
margin_match_list = [0.1]*100
margin_nonmatch_list = [0.4]*100

match_loss_threshold = 0.01
nonmatch_loss_threshold = 0.01
#load_model = 0
#save_model = 1
lr = [1e-2]*10 + [1e-3]*10 + [1e-4]*20 + [1e-5]*30 + [1e-6]*40

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_noPad(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_3x3(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(.01, shape=shape)
  return tf.Variable(initial)

def CNN(inputs):

  with tf.name_scope('whitening'):
    inputs_w = inputs /255

  with tf.name_scope('conv0'):
    W_conv0 = weight_variable([7, 7, 1, 24])
    b_conv0 = bias_variable([24])
    h_conv0 = tf.nn.relu((conv2d(inputs_w, W_conv0) + b_conv0))

  with tf.name_scope('pool0'):
    h_pool0 = max_pool_3x3(h_conv0)

  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 24, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu((conv2d(h_pool0, W_conv1) + b_conv1))
    #h_conv1 = tf.nn.relu(tf.layers.batch_normalization((conv2d(h_pool0, W_conv1) + b_conv1),training=True))

  with tf.name_scope('pool1'):
    h_pool1 = max_pool_3x3(h_conv1)

  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, 64, 96])
    b_conv2 = bias_variable([96])
    h_conv2 = tf.nn.relu((conv2d(h_pool1, W_conv2) + b_conv2))

  with tf.name_scope('pool2'):
    h_pool2 = max_pool_3x3(h_conv2)

  with tf.name_scope('conv3'):
    W_conv3 = weight_variable([3, 3, 96, 96])
    b_conv3 = bias_variable([96])
    h_conv3 = tf.nn.relu((conv2d(h_pool2, W_conv3) + b_conv3))

  with tf.name_scope('conv4'):
    W_conv4 = weight_variable([3, 3, 96, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.relu((conv2d(h_conv3, W_conv4) + b_conv4))

  with tf.name_scope('pool4'):
    h_pool4 = max_pool_3x3(h_conv4)

  with tf.name_scope('FC1'):
    W_FC1 = weight_variable([4, 4, 64, 128])
    b_FC1 = bias_variable([128])
    #h_FC1 = tf.nn.relu(tf.layers.batch_normalization((conv2d_noPad(h_pool4, W_FC1) + b_FC1),training=True))
    h_FC1 =tf.nn.relu(conv2d_noPad(h_pool4, W_FC1) + b_FC1)

  res = tf.nn.l2_normalize(tf.reshape(h_FC1, (-1, 128)),axis=1,epsilon=1e-7)
  #res = tf.reshape(h_FC1, (-1, 64))
  return res

def label_cluster(label):
  current_index = -1
  start_index = -1
  count = 0
  list_out = []
  y=label.shape
  for i in range(y[1]):
    if (label[0,i] != current_index):
      if (start_index!=-1):
        list_out.append((start_index,count))
      current_index = label[0,i]
      start_index = i
      count =  1;
    else:
      count=count +1

  return list_out


def main(args):

  # parsing the argument

  batch_size = args.batch_size
  load_model = args.load_model
  save_model = args.save_model

  # Import data
  f_train_loss = open("summary/train_loss.txt", "w+")
  f_train_error = open("summary/train_error.txt", "w+")
  f_test_loss = open("summary/test_loss.txt", "w+")
  f_test_error = open("summary/test_error.txt", "w+")


  patch = tf.placeholder(tf.float32, [None, 64,64,1])


  match_index_1= tf.placeholder (tf.int32, [None])
  match_index_2= tf.placeholder (tf.int32, [None])
  NonMatch_index_1= tf.placeholder (tf.int32, [None])
  NonMatch_index_2= tf.placeholder (tf.int32, [None])


  descs = CNN(patch)

  #read dataset
  f_train1 =  h5py.File('liberty.mat', 'r')
  patches_train1 = f_train1['data'][()]
  info_train1 = f_train1['info'][()].astype(np.int)
  f_train1.close()

  f_train2 =  h5py.File('notredame.mat', 'r')
  patches_train2 = f_train2['data'][()]
  info_train2 = f_train2['info'][()].astype(np.int)
  f_train2.close()

  patches_train = np.concatenate((patches_train1, patches_train2), axis=0)
  info_train = np.concatenate((info_train1, info_train2 + info_train1[0,-1] +1 ), axis=1)
  lList_train = label_cluster(info_train)


  f_test =  h5py.File('yosemite.mat', 'r')
  patches_test = f_test['data'][()]
  info_test = f_test['info'][()].astype(np.int)
  f_test.close()
  lList_test = label_cluster(info_test)

  with tf.name_scope('loss'):
    d1 = tf.gather(descs,match_index_1)
    d2 = tf.gather(descs,match_index_2)
    d3 = tf.gather(descs,NonMatch_index_1)
    d4 = tf.gather(descs,NonMatch_index_2)

    margin_match = tf.placeholder(tf.float32)
    margin_nonmatch = tf.placeholder(tf.float32)
    loss_desc_match = loss_matching(d1,d2,margin=margin_match)
    loss_desc_nonmatch = loss_non_matching(d3,d4,margin=margin_nonmatch)
    loss_hinge = loss_desc_match + loss_desc_nonmatch
    loss_hinge = tf.reduce_mean(loss_hinge)
    loss_desc_match_mean = tf.reduce_mean(loss_desc_match)
    loss_desc_nonmatch_mean = tf.reduce_mean(loss_desc_nonmatch)
    distance_match = 1-tf.norm(d1-d2,axis=1)/2
    distance_nonmatch = 1-tf.norm(d3-d4,axis=1)/2

  with tf.name_scope('adam_optimizer'):
    learning_rate = tf.placeholder(tf.float32, shape=[])
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_hinge)

  with tf.name_scope('accuracy'):
    correct_prediction_match = tf.less(loss_desc_match, match_loss_threshold)
    correct_prediction_match = tf.cast(correct_prediction_match, tf.float32)
    correct_prediction_nonmatch = tf.less(loss_desc_nonmatch, nonmatch_loss_threshold)
    correct_prediction_nonmatch = tf.cast(correct_prediction_nonmatch, tf.float32)

  accuracy = tf.reduce_mean(tf.concat([correct_prediction_match,correct_prediction_nonmatch],axis = 0))

  with tf.Session() as sess:
    saver = tf.train.Saver()
    if(load_model):
      saver.restore(sess, tf.train.latest_checkpoint('./log'))
    else:
      sess.run(tf.global_variables_initializer())
    epoch = 100
    point_size_train = int(info_train[0,-1])
    point_size_test = int(info_test[0,-1])


    #calculating one hot representation of the labels

    #calculating random performance

    print('Random Test Mode : ')
    labels_test = []
    scores_test = []
    epoch_loss = 0
    i = 0
    for j in range(0, point_size_test - 2 * batch_size, 2 * batch_size):
      index1 = [int(k[0]) for k in lList_test[j:j + batch_size]]
      index2 = [int(k[0] + 1) for k in lList_test[j:j + batch_size]]
      index3 = [int(k[0]) for k in lList_test[j + batch_size:j + 2 * batch_size]]
      index4 = [int(k[0] + 1) for k in lList_test[j + batch_size:j + 2 * batch_size]]

      A = np.asarray(list(range(0, batch_size)) + list(range(2 * batch_size, 3 * batch_size)))
      B = np.asarray(list(range(batch_size, 2 * batch_size)) + list(range(3 * batch_size, 4 * batch_size)))
      C = np.asarray(list(range(0, batch_size)) + list(range(2 * batch_size, 3 * batch_size)))
      D = np.asarray(list(range(3 * batch_size, 4 * batch_size)) + list(range(batch_size, 2 * batch_size)))

      p = patches_test[index1 + index2 + index3 + index4, :, :].astype(np.float32)
      p = np.expand_dims(p, 3)
      fd = {match_index_1: A, match_index_2: B, NonMatch_index_1: C, NonMatch_index_2: D, patch: p,
            margin_match: margin_match_list[i], margin_nonmatch: margin_nonmatch_list[i]}
      fet = (accuracy, loss_hinge, distance_match, distance_nonmatch, loss_desc_match_mean,
             loss_desc_nonmatch_mean, loss_desc_match, loss_desc_nonmatch, correct_prediction_match,
             correct_prediction_nonmatch)
      temp = sess.run(fet, fd)
      labels_test = labels_test + [1] * 2 * batch_size + [0] * 2 * batch_size
      scores_test = scores_test + temp[2].tolist() + temp[3].tolist()
      epoch_loss += temp[1]

    print('Random Test Loss = %g ' % epoch_loss)
    print('Random Test Error at 95%% recall is:  %g ' % (ErrorRateAt95Recall(labels=labels_test,
                                                                             scores=scores_test)))



    for i in range(epoch):
      labels_train =[]
      scores_train =[]
      epoch_loss=0
      print ('Training Mode : ')
      shuffle(lList_train)
      print(i)


      for j in range(0,point_size_train-2*batch_size,2*batch_size):
        index1 = [int(k[0]) for k in lList_train[j:j+batch_size]]
        index2 = [int(k[0]+1) for k in lList_train[j:j+batch_size]]
        index3 = [int(k[0]) for k in lList_train[j+batch_size:j+2*batch_size]]
        index4 = [int(k[0]+1) for k in lList_train[j+batch_size:j+2*batch_size]]

        A = np.asarray(list(range(0,batch_size)) + list(range(2*batch_size,3*batch_size)))
        B = np.asarray(list(range(batch_size,2*batch_size)) + list(range(3*batch_size,4*batch_size)))
        C = np.asarray(list(range(0,batch_size)) + list(range(2*batch_size,3*batch_size)))
        D = np.asarray(list(range(3*batch_size,4*batch_size))+list(range(batch_size,2*batch_size)))

        p = patches_train[index1+index2+index3+index4,:,:].astype(np.float32)
        p = np.expand_dims(p,3)
        fd = {match_index_1:A,match_index_2:B,NonMatch_index_1:C,NonMatch_index_2:D,patch:p,margin_match:margin_match_list[i],margin_nonmatch:margin_nonmatch_list[i],learning_rate:lr[i]}
        '''if (j%50000 == 0):
          #train_accuracy = accuracy.eval(feed_dict=fd)
          fet = (accuracy, loss_hinge, distance_match , distance_nonmatch, loss_desc_match_mean,
                 loss_desc_nonmatch_mean, loss_desc_match, loss_desc_nonmatch, correct_prediction_match, correct_prediction_nonmatch)
          temp = sess.run(fet,fd)
          #print('step %d, training accuracy %g' % (j, temp[0]))
          print('step %d, loss %g' % (j, temp[1]))
          print('step %d, match loss %g' % (j, temp[4]))
          print('step %d, non-match loss %g' % (j, temp[5]))
          print ('Error at 95%% recall is: %g ' %(ErrorRateAt95Recall(labels = [1]*2*batch_size + [0]*2*batch_size,scores = temp[2].tolist()+temp[3].tolist() )))

        train_step.run(feed_dict=fd)'''

        '''fet = (accuracy, loss_hinge, distance_match, distance_nonmatch, loss_desc_match_mean,
               loss_desc_nonmatch_mean, loss_desc_match, loss_desc_nonmatch, correct_prediction_match,
               correct_prediction_nonmatch)
        temp = sess.run(fet, fd)'''

        _, c, d_m, d_nm = sess.run([train_step, loss_hinge, distance_match, distance_nonmatch ], feed_dict=fd)
        labels_train = labels_train + [1] * 2 * batch_size + [0] * 2 * batch_size
        scores_train = scores_train + d_m.tolist() + d_nm.tolist()
        epoch_loss += c

      if save_model == 1:
        saver.save(sess, './log/', global_step=i, write_meta_graph=False)

      train_error = (ErrorRateAt95Recall(labels=labels_train,
                                                                        scores=scores_train))
      print('Train Loss = %g ' % epoch_loss)
      print('Train Error at 95%% recall is:  %g ' %train_error)
      f_train_error.write('%g\n!' %train_error)
      f_train_loss.write('%g\n!' %epoch_loss)


      print('Test Mode : ')
      labels_test=[]
      scores_test =[]
      epoch_loss=0
      for j in range(0,point_size_test-2*batch_size,2*batch_size):
        index1 = [int(k[0]) for k in lList_test[j:j + batch_size]]
        index2 = [int(k[0] + 1) for k in lList_test[j:j + batch_size]]
        index3 = [int(k[0]) for k in lList_test[j + batch_size:j + 2 * batch_size]]
        index4 = [int(k[0] + 1) for k in lList_test[j + batch_size:j + 2 * batch_size]]

        A = np.asarray(list(range(0, batch_size)) + list(range(2 * batch_size, 3 * batch_size)))
        B = np.asarray(list(range(batch_size, 2 * batch_size)) + list(range(3 * batch_size, 4 * batch_size)))
        C = np.asarray(list(range(0, batch_size)) + list(range(2 * batch_size, 3 * batch_size)))
        D = np.asarray(list(range(3 * batch_size, 4 * batch_size)) + list(range(batch_size, 2 * batch_size)))

        p = patches_test[index1 + index2 + index3 + index4, :, :].astype(np.float32)
        p = np.expand_dims(p, 3)
        fd = {match_index_1: A, match_index_2: B, NonMatch_index_1: C, NonMatch_index_2: D, patch: p,margin_match:margin_match_list[i],margin_nonmatch:margin_nonmatch_list[i]}
        fet = (accuracy, loss_hinge, distance_match, distance_nonmatch, loss_desc_match_mean,
               loss_desc_nonmatch_mean, loss_desc_match, loss_desc_nonmatch, correct_prediction_match,
               correct_prediction_nonmatch)
        temp = sess.run(fet, fd)
        labels_test = labels_test + [1] * 2 * batch_size + [0] * 2 * batch_size
        scores_test = scores_test + temp[2].tolist() + temp[3].tolist()
        epoch_loss += temp[1]

      test_error = ErrorRateAt95Recall(labels=labels_test, scores=scores_test)
      print('Test Loss = %g ' % epoch_loss)
      print('Test Error at 95%% recall is:  %g ' % test_error)
      f_test_error.write('%g\n!' %test_error)
      f_test_loss.write('%g\n!' %epoch_loss)

    f_train_loss.close()
    f_train_error.close()
    f_test_loss.close()
    f_test_error.close()


if __name__ == '__main__':

  parser =  argparse.ArgumentParser(description='Network and Display options')

  parser.add_argument('--batch_size', type=int, default=200)
  parser.add_argument('-load_model', '--load_model', action='store_true')
  parser.add_argument('-save_model', '--save_model', action='store_true')

  args = parser.parse_args()
  main(args)



 


