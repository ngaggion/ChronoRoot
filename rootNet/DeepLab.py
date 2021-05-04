""" DeepLab.py
    Implementation of DeepLab v3
    Architecture taken from: https://github.com/rishizek/tensorflow-deeplab-v3
"""

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.slim.python.slim.nets import resnet_utils


_BATCH_NORM_DECAY = 0.9
_OUTPUT_STRIDE = 16


def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
  """Atrous Spatial Pyramid Pooling.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    is_training: A boolean denoting whether the input is for training.
    depth: The depth of the ResNet unit output.
  Returns:
    The atrous spatial pyramid pooling output.
  """
  with tf.compat.v1.variable_scope("aspp"):
    if output_stride not in [8, 16]:
      raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [6, 12, 18]
    if output_stride == 8:
      atrous_rates = [2*rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
        conv_3x3_1 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
        conv_3x3_2 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
        conv_3x3_3 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

        # (b) the image-level features
        with tf.compat.v1.variable_scope("image_level_features"):
          # global average pooling
          image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
          # 1x1 convolution with 256 filters( and batch normalization)
          image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
          # bilinearly upsample features
          image_level_features = tf.compat.v1.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

        return net


class DeepLab(object):
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
    base_model = resnet_v2.resnet_v2_101
    
    with tf.compat.v1.variable_scope(self.name, reuse=self.reuse):
        logits, end_points = base_model(x,
                                        num_classes=None,
                                        is_training=isTrain,
                                        global_pool=False,
                                        output_stride=8)
    
        inputs_size = tf.shape(x)[1:3]
        net = end_points['DeepLab/resnet_v2_101/block4']
        net = atrous_spatial_pyramid_pooling(net, _OUTPUT_STRIDE, _BATCH_NORM_DECAY, isTrain)

        net = layers_lib.conv2d(net, 2, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
        logits = tf.compat.v1.image.resize_bilinear(net, inputs_size, name='upsample')
              
    if self.reuse is None:

      if self.finetuneLayers is None:
          self.varList = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      else:
          self.varList = []

          for scope in self.finetuneLayers:
            self.varList.extend(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope))

      self.saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name))

      self.reuse = True

    return logits

  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)
