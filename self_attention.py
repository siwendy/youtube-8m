# Copyright 2018 Juhan Bae All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences."""
# noinspection PyUnresolvedReferences
#import pathmagic
from frame_level_models import *
import numpy as np
from modeling import transformer_model,get_activation,create_initializer
#import modeling
#from tensorflow import flags
#import tensorflow as tf
#import video_level_models
#import models
#import modules
import common
FLAGS = common.FLAGS


#flags.DEFINE_integer("self_attention_filter_size", 2,
#                     "The filter multiplier size for deep context gate.")
#flags.DEFINE_integer("self_attention_n_head", 8,
#                     "The multi head num.")
#flags.DEFINE_integer("self_attention_n_layer", 1,
#                     "layer num.")
#flags.DEFINE_integer("self_attention_hidden_size", 1024,
#                     "The number of units after attention cluster layer.")
#flags.DEFINE_float("self_attention_hidden_dropout", 0.5,
#                   "Dropout rate for clustering operation")
#flags.DEFINE_float("self_attention_ff_dropout", 0.5,
#                   "Dropout rate for Feed Forward operation")
#flags.DEFINE_float("self_attention_attention_dropout", 0.5,
#                   "Dropout rate for Feed Forward operation")
#flags.DEFINE_bool("self_avg_embed", True,
#                     "layer num.")
#
#FLAGS = flags.FLAGS


def shape_list(x):
  """
  deal with dynamic shape in tensorflow cleanly
  """
  ps = x.get_shape().as_list()
  ts = tf.shape(x)
  return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

class SelfAttentionModule():
    def __init__(self, feature_size, max_frames, n_head, n_layer, hidden_dropout, attention_dropout, is_training, avg_emb):
        """ Initialize SelfAttentionModule.
        :param hidden_dropout: float
        :param cluster_size: int
        :param is_training: bool
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.hidden_dropout = hidden_dropout 
        self.attention_dropout = attention_dropout 
        self.n_head = n_head;
        self.n_layer = n_layer;
        self.avg_emb = avg_emb;

    def forward(self, inputs, **unused_params):
        """ Forward method for SelfAttentionModule.
        :param inputs: 3D Tensor of size 'batch_size x max_frames x feature_size'
        :return: 2D Tensor of size 'batch_size x (max_frames * feature_size)
        """
        inputs = tf.reshape(inputs, [-1, self.feature_size])
        h = tf.reshape(inputs, [-1, self.max_frames, self.feature_size])
        if self.avg_emb:
          print("avg_emb")
          sequence_output = transformer_model(
              input_tensor=h,
              attention_mask=None,
              hidden_size=self.feature_size,
              num_hidden_layers=self.n_layer,
              num_attention_heads=self.n_head,
              intermediate_size=3*self.feature_size,
                      intermediate_act_fn=get_activation('relu'),
                      hidden_dropout_prob=self.hidden_dropout if self.is_training else 0,
                      attention_probs_dropout_prob=self.attention_dropout if self.is_training else 0,
                      initializer_range=0.02,
                      do_return_all_layers=False)
          return tf.reduce_mean(sequence_output, 1)
        else:
          cls = tf.get_variable("cls", [self.feature_size], initializer=tf.random_normal_initializer(stddev=0.02))
          h = tf.map_fn(lambda x: tf.concat(([cls], x), axis=0), h)
          sequence_output = transformer_model(
              input_tensor=h,
              attention_mask=None,
              hidden_size=self.feature_size,
              num_hidden_layers=self.n_layer,
              num_attention_heads=self.n_head,
              intermediate_size=3*self.feature_size,
                      intermediate_act_fn=get_activation('relu'),
                      hidden_dropout_prob=self.hidden_dropout if self.is_training else 0,
                      attention_probs_dropout_prob=self.attention_dropout if self.is_training else 0,
                      initializer_range=0.02,
                      do_return_all_layers=False)
          with tf.variable_scope("pooler"):
              # We "pool" the model by simply taking the hidden state corresponding
              # to the first token. We assume that this has been pre-trained
              first_token_tensor = tf.squeeze(
                  sequence_output[:, 0:1, :], axis=1)
              return tf.layers.dense(
                  first_token_tensor,
                  self.feature_size,
                  activation=tf.tanh,
                  kernel_initializer=create_initializer(0.02))


class SelfAttentionModel(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     hidden_size=None,
                     is_training=True,
                     gating_reduction=None,
                     **unused_params):
        hidden_dropout = FLAGS.self_attention_hidden_dropout
        hidden_size = FLAGS.self_attention_hidden_size
        hidden1_size = hidden_size
        gating_reduction = gating_reduction or FLAGS.gating_reduction
        n_head = FLAGS.self_attention_n_head
        attention_dropout = FLAGS.self_attention_attention_dropout
        n_layer = FLAGS.self_attention_n_layer
        avg_embeeding  = FLAGS.self_avg_embed

        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        # Differentiate video & audio features.
        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]
        video_features = tf.nn.l2_normalize(video_features, 1)
        audio_features = tf.nn.l2_normalize(audio_features, 1)
        video_features = tf.reshape(video_features, [-1, max_frames, 1024])
        audio_features = tf.reshape(audio_features, [-1, max_frames, 128])

        video_cluster = SelfAttentionModule( feature_size = 1024,
                                              max_frames = max_frames,
                                              n_head = n_head,
                                              n_layer = n_layer,
                                              hidden_dropout=hidden_dropout,
                                              attention_dropout=attention_dropout,
                                              avg_emb=avg_embeeding,
                                               is_training=is_training)
        audio_cluster = SelfAttentionModule(feature_size = 128,
                                              max_frames = max_frames,
                                              n_head = n_head,
                                              n_layer = n_layer,
                                              hidden_dropout=hidden_dropout,
                                              attention_dropout=attention_dropout,
                                              avg_emb=avg_embeeding,
                                               is_training=is_training)
        with tf.variable_scope("video"):
            video_cluster_activation = video_cluster.forward(video_features)

        with tf.variable_scope("audio"):
            audio_cluster_activation = audio_cluster.forward(audio_features)

        activation = tf.concat([video_cluster_activation, audio_cluster_activation], 1)
        act_dim = activation.get_shape().as_list()[1]
        print("act dimension", act_dim)
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [act_dim, hidden1_size],
                                          initializer=slim.variance_scaling_initializer())

        activation = tf.matmul(activation, hidden1_weights)
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn",
            fused=False)

        if FLAGS.enable_gate:
          gating_weights_1 = tf.get_variable("gating_weights_1",
                                             [hidden1_size, hidden1_size // gating_reduction],
                                             initializer=slim.variance_scaling_initializer())

          gates = tf.matmul(activation, gating_weights_1)

          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              activation_fn=slim.nn.relu,
              scope="gating_bn")

          gating_weights_2 = tf.get_variable("gating_weights_2",
                                             [hidden1_size // gating_reduction, hidden1_size],
                                             initializer=slim.variance_scaling_initializer()
                                             )
          gates = tf.matmul(gates, gating_weights_2)

          gates = tf.sigmoid(gates)
          tf.summary.histogram("final_gates", gates)

          activation = tf.multiply(activation, gates)
        else:
          print("no final gate")

        aggregated_model = getattr(video_level_models,
                                   "MoeModel")

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


class MixSelfAttentionModel(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     hidden_size=None,
                     gating_reduction=None,
                     is_training=True,
                     **unused_params):
        hidden_dropout = FLAGS.self_attention_hidden_dropout
        hidden_size = FLAGS.self_attention_hidden_size
        hidden1_size = hidden_size
        n_head = FLAGS.self_attention_n_head
        attention_dropout = FLAGS.self_attention_attention_dropout
        n_layer = FLAGS.self_attention_n_layer
        avg_embeeding  = FLAGS.self_avg_embed
        gating_reduction = gating_reduction or FLAGS.gating_reduction

        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])

        # Differentiate video & audio features.
        video_features = reshaped_input[:, 0:1024]
        audio_features = reshaped_input[:, 1024:]
        video_features = tf.nn.l2_normalize(video_features, 1)
        audio_features = tf.nn.l2_normalize(audio_features, 1)
        mix_features = tf.concat([video_features, audio_features], 1)

        video_cluster = SelfAttentionModule( feature_size = 1152,
                                              max_frames = max_frames,
                                              n_head = n_head,
                                              n_layer = n_layer,
                                              hidden_dropout=hidden_dropout,
                                              attention_dropout=attention_dropout,
                                              avg_emb=avg_embeeding,
                                               is_training=is_training)
        with tf.variable_scope("mix_video_audio"):
            activation = video_cluster.forward(mix_features)

        act_dim = activation.get_shape().as_list()[1]
        print("act dimension", act_dim)
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [act_dim, hidden1_size],
                                          initializer=slim.variance_scaling_initializer())

        activation = tf.matmul(activation, hidden1_weights)
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn",
            fused=False)

        if FLAGS.enable_gate:
          gating_weights_1 = tf.get_variable("gating_weights_1",
                                             [hidden1_size, hidden1_size // gating_reduction],
                                             initializer=slim.variance_scaling_initializer())

          gates = tf.matmul(activation, gating_weights_1)

          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=is_training,
              activation_fn=slim.nn.relu,
              scope="gating_bn")

          gating_weights_2 = tf.get_variable("gating_weights_2",
                                             [hidden1_size // gating_reduction, hidden1_size],
                                             initializer=slim.variance_scaling_initializer()
                                             )
          gates = tf.matmul(gates, gating_weights_2)

          gates = tf.sigmoid(gates)
          tf.summary.histogram("final_gates", gates)

          activation = tf.multiply(activation, gates)
        else:
          print("no final gate")

        aggregated_model = getattr(video_level_models,
                                   "MoeModel")

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)
