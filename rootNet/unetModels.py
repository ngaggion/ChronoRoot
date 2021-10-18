#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:55:32 2019

@author: ngaggion
"""
import tensorflow as tf

from .modelUtils import conv2d, upsample2d, dropout, resUnit
from .modelUtils import pixel_wise_softmax


class ResUNetDS(object):
  """
    Input has to be multiple of 4 (CHECK THIS).
  """
  def __init__(self, name, finetuneLayers = None, dropout = 0.5):
    self.name = name
    self.reuse = None
    self.finetuneLayers = finetuneLayers
    self.dropout = dropout

  def __call__(self, x, isTrain):
    "If finetuneLayers is None, all the layers will be finetuned. Otherwise, only those which match the names in finetuneLayers list will be trained."

    with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
      nonlinearity = tf.nn.elu

      add1 = resUnit(x, 1, 16, 3, 1, "SAME", nonlinearity, isTrain)
      pool1 = tf.nn.max_pool2d(add1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
      #drop1 = dropout(add1, self.dropout, isTrain, "drop1")
           
      add2 = resUnit(pool1, 2, 32, 3, 1, "SAME", nonlinearity, isTrain)
      pool2 = tf.nn.max_pool2d(add2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
      #drop2 = dropout(pool2, self.dropout, isTrain, "drop2")
            
      add3 = resUnit(pool2, 3, 64, 3, 1, "SAME", nonlinearity, isTrain)
      pool3 = tf.nn.max_pool2d(add3, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
      #drop3 = dropout(pool3, self.dropout, isTrain, "drop3")
      
      add4 = resUnit(pool3, 4, 128, 3, 1, "SAME", nonlinearity, isTrain)
      drop4 = dropout(add4, self.dropout, isTrain, "drop4")
      
      up4 = conv2d(upsample2d(drop4, 2), "deconv4", 64, 3, 1, "SAME", bn=False, af=None, is_train=isTrain)
      concat5 = tf.add(up4, add3)
    
      add5 = resUnit(concat5, 5, 64, 3, 1, "SAME", nonlinearity, isTrain)
      drop5 = dropout(add5, self.dropout, isTrain, "drop5")
           
      up5 = conv2d(upsample2d(drop5, 2), "deconv5", 32, 3, 1, "SAME", bn=False, af=None, is_train=isTrain)
      concat6 = tf.add(up5, add2)

      add6 = resUnit(concat6, 6, 32, 3, 1, "SAME", nonlinearity, isTrain)
      drop6 = dropout(add6, self.dropout, isTrain, "drop6")
 
      up6 = conv2d(upsample2d(drop6, 2), "deconv6", 16, 3, 1, "SAME", bn=False, af=None, is_train=isTrain)
      concat7 = tf.add(up6, add1)

      add7 = resUnit(concat7, 7, 16, 3, 1, "SAME", nonlinearity, isTrain)
      
      unet_out1 = conv2d(add7, "unet_out", 2, 3, 1, "SAME", bn=True, af=None, is_train=isTrain)
      unet_out = pixel_wise_softmax(unet_out1)
      
      stack = tf.concat([x, unet_out], axis = 3)
       
      conv8 = resUnit(stack, 8, 16, 3, 1, "SAME", nonlinearity, isTrain)
      pool8 = tf.nn.max_pool2d(conv8, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
      drop8 = dropout(pool8, self.dropout, isTrain, "drop8")
       
      conv9 = resUnit(drop8, 9, 32, 3, 1, "SAME", nonlinearity, isTrain)
      drop9 = dropout(conv9, self.dropout, isTrain, "drop9")
       
      up9 = conv2d(upsample2d(drop9, 2), "deconv9", 16, 3, 1, "SAME", bn=False, af=None, is_train=isTrain)
      
      add9 = tf.add(up9, conv8)
      conv10 = resUnit(add9, 10, 16, 3, 1, "SAME", nonlinearity, isTrain)
     
      conv_out = conv2d(conv10, "conv_out", 2, 3, 1, "SAME", bn=True, af=None, is_train=isTrain)
      
    if self.reuse is None:

      if self.finetuneLayers is None:
          self.varList = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      else:
          self.varList = []

          for scope in self.finetuneLayers:
            self.varList.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope))

      self.saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name))

      self.reuse = True

    return conv_out, unet_out

  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)


class ResUNet(object):
  """
    Input has to be multiple of 4 (CHECK THIS).
  """
  def __init__(self, name, finetuneLayers = None, dropout = 0.5):
    self.name = name
    self.reuse = None
    self.finetuneLayers = finetuneLayers
    self.dropout = dropout

  def __call__(self, x, isTrain):
    "If finetuneLayers is None, all the layers will be finetuned. Otherwise, only those which match the names in finetuneLayers list will be trained."

    with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
      nonlinearity = tf.nn.elu

      add1 = resUnit(x, 1, 16, 3, 1, "SAME", nonlinearity, isTrain)
      pool1 = tf.nn.max_pool2d(add1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
      #drop1 = dropout(add1, self.dropout, isTrain, "drop1")
      
      # print('Add 1', add1.shape)
      
      add2 = resUnit(pool1, 2, 32, 3, 1, "SAME", nonlinearity, isTrain)
      pool2 = tf.nn.max_pool2d(add2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
      #drop2 = dropout(pool2, self.dropout, isTrain, "drop2")
      
      # print('Add 2', add2.shape)
      # print('Pool 2', pool2.shape)
      
      add3 = resUnit(pool2, 3, 64, 3, 1, "SAME", nonlinearity, isTrain)
      pool3 = tf.nn.max_pool2d(add3, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
      #drop3 = dropout(pool3, self.dropout, isTrain, "drop3")
      
      # print('Add 3', add3.shape)
      # print('Pool 3', pool3.shape)
      
      add4 = resUnit(pool3, 4, 128, 3, 1, "SAME", nonlinearity, isTrain)
      drop4 = dropout(add4, self.dropout, isTrain, "drop4")
      
      # print('Add 4', add4.shape)
      # print('Pool 4', pool4.shape)
      
      up4 = conv2d(upsample2d(drop4, 2), "deconv4", 64, 3, 1, "SAME", bn=False, af=None, is_train=isTrain)
      # print('Up 4', up4.shape)
      concat5 = tf.add(up4, add3)
      add5 = resUnit(concat5, 5, 64, 3, 1, "SAME", nonlinearity, isTrain)
      drop5 = dropout(add5, self.dropout, isTrain, "drop5")
      # print('Add 5', add5.shape)
      
      up5 = conv2d(upsample2d(drop5, 2), "deconv5", 32, 3, 1, "SAME", bn=False, af=None, is_train=isTrain)
      # print('Up 5', up5.shape)
      
      concat6 = tf.add(up5, add2)
      add6 = resUnit(concat6, 6, 32, 3, 1, "SAME", nonlinearity, isTrain)
      drop6 = dropout(add6, self.dropout, isTrain, "drop6")
      # print('Add 6', add6.shape)
      
      up6 = conv2d(upsample2d(drop6, 2), "deconv6", 16, 3, 1, "SAME", bn=False, af=None, is_train=isTrain)
      # print('Up 6', up6.shape)
      concat7 = tf.add(up6, add1)
      add7 = resUnit(concat7, 7, 16, 3, 1, "SAME", nonlinearity, isTrain)
      # print('Add 7', add7.shape)
      
      conv_out = conv2d(add7, "unet_out", 2, 3, 1, "SAME", bn=True, af=None, is_train=isTrain)

    if self.reuse is None:

      if self.finetuneLayers is None:
          self.varList = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      else:
          self.varList = []

          for scope in self.finetuneLayers:
            self.varList.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope))

      self.saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name))

      self.reuse = True

    return conv_out

  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)
    
class UNet(object):
  """
    Input has to be multiple of 4 (CHECK THIS).
  """

  def __init__(self, name, finetuneLayers = None, dropout = 0.5):
    self.name = name
    self.reuse = None
    self.finetuneLayers = finetuneLayers
    self.dropout = dropout

  def __call__(self, x, isTrain):
    "If finetuneLayers is None, all the layers will be finetuned. Otherwise, only those which match the names in finetuneLayers list will be trained."

    with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
      nonlinearity = tf.nn.elu

      conv1_a = conv2d(x, "conv1_a", 16, 3, 1, "SAME", bn=True, af=nonlinearity, is_train=isTrain)
      conv1_b= conv2d(conv1_a, "conv1_b", 16, 3, 1, "SAME", bn=True, af=None, is_train=isTrain)
      act1 = nonlinearity(conv1_b)
      pool1 = tf.nn.avg_pool2d(act1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

      conv2_a = conv2d(pool1, "conv2_a", 32, 3, 1, "SAME", bn=True, af=nonlinearity, is_train=isTrain)
      conv2_b = conv2d(conv2_a, "conv2_b", 32, 3, 1, "SAME", bn=True, af=None, is_train=isTrain)
      act2 = nonlinearity(conv2_b)
      pool2 = tf.nn.avg_pool2d(act2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

      conv3_a = conv2d(pool2, "conv3_a", 64, 3, 1, "SAME", bn=True, af=nonlinearity, is_train=isTrain)
      conv3_b = conv2d(conv3_a, "conv3_b", 64, 3, 1, "SAME", bn=True, af=None, is_train=isTrain)
      act3 = nonlinearity(conv3_b)
      pool3 = tf.nn.avg_pool2d(act3, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

      conv4_a = conv2d(pool3, "conv4_a", 128, 3, 1, "SAME", bn=True, af=nonlinearity, is_train=isTrain)
      conv4_b = conv2d(conv4_a, "conv4_b", 128, 3, 1, "SAME", bn=True, af=None, is_train=isTrain)
      act4 = nonlinearity(conv4_b)

      # Dropout
      drop4 = tf.layers.dropout(act4, rate=self.dropout, training=isTrain, name="drop4")

      # Up-scaling and convolving without non-linearity
      deconv4= conv2d(upsample2d(drop4, 2), "deconv4", 64, 3, 1, "SAME", bn=True, af=None, is_train=isTrain)

      concat5 = nonlinearity(tf.add(deconv4, conv3_b))
      conv5_a = conv2d(concat5, "conv5_a", 64, 3, 1, "SAME", bn=True, af=nonlinearity, is_train=isTrain)
      conv5_b = conv2d(conv5_a, "conv5_b", 64, 3, 1, "SAME", bn=True, af=nonlinearity, is_train=isTrain)
      drop5 = tf.layers.dropout(conv5_b, rate=self.dropout, training=isTrain, name="drop5")
      deconv5= conv2d(upsample2d(drop5, 2), "deconv5", 32, 3, 1, "SAME", bn=True, af=None, is_train=isTrain)

      concat6 = nonlinearity(tf.add(deconv5, conv2_b))
      conv6_a = conv2d(concat6, "conv6_a", 32, 3, 1, "SAME", bn=True, af=nonlinearity, is_train=isTrain)
      conv6_b = conv2d(conv6_a, "conv6_b", 32, 3, 1, "SAME", bn=True, af=nonlinearity, is_train=isTrain)
      drop6 = tf.layers.dropout(conv6_b, rate=self.dropout, training=isTrain, name="drop6")
      deconv6= conv2d(upsample2d(drop6, 2), "deconv6", 16, 3, 1, "SAME", bn=True, af=None, is_train=isTrain)

      concat7 = nonlinearity(tf.add(deconv6, conv1_b))
      conv7_a = conv2d(concat7, "conv7_a", 16, 3, 1, "SAME", bn=True, af=nonlinearity, is_train=isTrain)
      conv7_b = conv2d(conv7_a, "conv7_b", 16, 3, 1, "SAME", bn=True, af=nonlinearity, is_train=isTrain)
      conv_out = conv2d(conv7_b, "conv_out", 2, 3, 1, "SAME", bn=True, af=None, is_train=isTrain)

    if self.reuse is None:

      if self.finetuneLayers is None:
          self.varList = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      else:
          self.varList = []

          for scope in self.finetuneLayers:
            self.varList.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope))

      self.saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name))

      self.reuse = True

    return conv_out

  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)

