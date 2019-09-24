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


flags.DEFINE_integer("self_attention_filter_size", 2,
                     "The filter multiplier size for deep context gate.")
flags.DEFINE_integer("self_attention_n_head", 8,
                     "The multi head num.")
flags.DEFINE_integer("self_attention_n_layer", 1,
                     "layer num.")
flags.DEFINE_integer("self_attention_hidden_size", 1024,
                     "The number of units after attention cluster layer.")
flags.DEFINE_float("self_attention_cluster_dropout", 0.5,
                   "Dropout rate for clustering operation")
flags.DEFINE_float("self_attention_ff_dropout", 0.5,
                   "Dropout rate for Feed Forward operation")
flags.DEFINE_float("self_attention_resid_dropout", 0.5,
                   "Dropout rate for Feed Forward operation")
flags.DEFINE_bool("self_avg_embed", True,
                     "layer num.")

FLAGS = flags.FLAGS


def shape_list(x):
  """
  deal with dynamic shape in tensorflow cleanly
  """
  ps = x.get_shape().as_list()
  ts = tf.shape(x)
  return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

class SelfAttentionModule():
    def __init__(self, feature_size, max_frames, n_head, n_layer, dropout_rate, resid_dropout_rate, is_training, avg_emb):
        """ Initialize SelfAttentionModule.
        :param dropout_rate: float
        :param cluster_size: int
        :param is_training: bool
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.dropout_rate = dropout_rate
        self.resid_dropout_rate = resid_dropout_rate 
        self.n_head = n_head;
        self.n_layer = n_layer;
        self.avg_emb = avg_emb;

    def _norm(self, x, g=None, b=None, e=1e-5, axis=[1]):
        u = tf.reduce_mean(x, axis=axis, keep_dims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
        x = (x - u) * tf.rsqrt(s + e)
        if g is not None and b is not None:
            x = x*g + b
        return x
    
    def norm(self, x, scope, axis=[-1]):
        with tf.variable_scope(scope):
            n_state = shape_list(x)[-1]
            g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
            b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
            return self._norm(x, g, b, axis=axis)
    
    def dropout(self, x, pdrop, train):
        if train and pdrop > 0:
            x = tf.nn.dropout(x, 1-pdrop)
        return x
    
    def mask_attn_weights(w):
        n = shape_list(w)[-1]
        b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)
        b = tf.reshape(b, [1, 1, n, n])
        w = w*b + -1e9*(1-b)
        return w
    
    def _attn(self, q, k, v, train=False, scale=False):
        #w=[-1,head,n_ctx,n_ctx]
        w = tf.matmul(q, k)
    
        if scale:
            n_state = shape_list(v)[-1]
            w = w*tf.rsqrt(tf.cast(n_state, tf.float32))
    
        #w = mask_attn_weights(w)
        w = tf.nn.softmax(w)
    
        w = self.dropout(w, self.dropout_rate, train)
    
        #w=[-1,head,n_ctx,n_ctx],v=[-1,head,n_ctx,emb]
        a = tf.matmul(w, v)
        return a
    
    def split_states(self, x, n):
        x_shape = shape_list(x)
        m = x_shape[-1]
        new_x_shape = x_shape[:-1]+[n, m//n]
        return tf.reshape(x, new_x_shape)
    
    def merge_states(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
        return tf.reshape(x, new_x_shape)
    
    def split_heads(self, x, n, k=False):
        #[-1,n_ctx,head,head_emb]
        if k:
            return tf.transpose(self.split_states(x, n), [0, 2, 3, 1])
        else:
            return tf.transpose(self.split_states(x, n), [0, 2, 1, 3])
    
    def merge_heads(self, x):
        #x:[-1,head,n_ctx,emb]
        #return:[-1,n_ctx,head*emb]
        return self.merge_states(tf.transpose(x, [0, 2, 1, 3]))
    
    def conv1d(self, x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), pad='VALID', train=False):
        with tf.variable_scope(scope):
            #x = [-1,n_ctx,512]
            nx = shape_list(x)[-1]
            #rf = 1,nx=emb,nf=3*emb
            w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
            b = tf.get_variable("b", [nf], initializer=b_init)
            if rf == 1: #faster 1x1 conv
                c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
            else: #was used to train LM
                c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
            return c
    
    def attn(self, x, scope, n_state, n_head, train=False, scale=False):
        assert n_state%n_head==0
        with tf.variable_scope(scope):
            #c [-1,n_ctx,3*emb]
            c = self.conv1d(x, 'c_attn', n_state*3, 1, train=train)
            #q,k,v [-1,n_ctx,emb]
            q, k, v = tf.split(c, 3, 2)
            #q [-1,head,n_ctx,emb] v [-1,head,emb,n_ctx] v [-1,head,n_ctx,emb]
            q = self.split_heads(q, n_head)
            k = self.split_heads(k, n_head, k=True)
            v = self.split_heads(v, n_head)
            #a [-1,head,n_ctx,emb]
            a = self._attn(q, k, v, train=train, scale=scale)
            #a [-1,n_ctx,head,emb]
            a = self.merge_heads(a)
            #a [-1,n_ctx,head*emb]
            a = self.conv1d(a, 'c_proj', n_state, 1, train=train)
            #a [-1,n_ctx,emb]
            a = self.dropout(a, self.resid_dropout_rate, train)
            return a
    
    def mlp(self, x, scope, n_state, train=False):
        with tf.variable_scope(scope):
            nx = shape_list(x)[-1]
            #act = act_fns[afn]
            #act = tf.nn.relu
            h = tf.nn.relu(self.conv1d(x, 'c_fc', n_state, 1, train=train))
            h2 = self.conv1d(h, 'c_proj', nx, 1, train=train)
            h2 = self.dropout(h2, self.resid_dropout_rate, train)
            return h2

    def block(self, x, scope, train=False, scale=False):
      with tf.variable_scope(scope):
        #nx = emb_size
        nx = shape_list(x)[-1]
        #a [-1,n_ctx,emb]
        a = self.attn(x, 'attn', nx, self.n_head, train=train, scale=scale)
        n = self.norm(x+a, 'ln_1')
        m = self.mlp(n, 'mlp', nx*4, train=train)
        h = self.norm(n+m, 'ln_2')
        return h

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
                      hidden_dropout_prob=0.1 if self.is_training else 0,
                      attention_probs_dropout_prob=0.1 if self.is_training else 0,
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
                      hidden_dropout_prob=0.1 if self.is_training else 0,
                      attention_probs_dropout_prob=0.1 if self.is_training else 0,
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
                     **unused_params):
        cluster_dropout = FLAGS.self_attention_cluster_dropout
        ff_dropout = FLAGS.self_attention_ff_dropout
        filter_size = FLAGS.self_attention_filter_size
        hidden_size = FLAGS.self_attention_hidden_size
        n_head = FLAGS.self_attention_n_head
        resid_dropout = FLAGS.self_attention_resid_dropout
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
                                              dropout_rate=cluster_dropout,
                                              resid_dropout_rate=resid_dropout,
                                              avg_emb=avg_embeeding,
                                               is_training=is_training)
        audio_cluster = SelfAttentionModule(feature_size = 128,
                                              max_frames = max_frames,
                                              n_head = n_head,
                                              n_layer = n_layer,
                                              dropout_rate=cluster_dropout,
                                              resid_dropout_rate=resid_dropout,
                                              avg_emb=avg_embeeding,
                                               is_training=is_training)
        with tf.variable_scope("video"):
            video_cluster_activation = video_cluster.forward(video_features)

        with tf.variable_scope("audio"):
            audio_cluster_activation = audio_cluster.forward(audio_features)

        concat_activation = tf.concat([video_cluster_activation, audio_cluster_activation], 1)
        activation = tf.layers.dense(concat_activation, hidden_size, use_bias=False, activation=None)
        activation = tf.layers.batch_normalization(activation, training=is_training)

        # Deep context gating.
        gating_weights1 = tf.layers.dense(activation, hidden_size * filter_size,
                                          use_bias=False, activation=tf.nn.relu)
        gating_weights1 = tf.layers.batch_normalization(gating_weights1, training=is_training)
        if is_training:
            gating_weights1 = tf.nn.dropout(gating_weights1, ff_dropout)

        gating_weights2 = tf.layers.dense(gating_weights1, hidden_size, use_bias=False, activation=None)
        gating_weights2 = tf.layers.batch_normalization(gating_weights2, training=is_training)
        gating_weights2 = tf.sigmoid(gating_weights2)
        activation = tf.multiply(activation, gating_weights2)

        aggregated_model = getattr(video_level_models,
                                   "MoeModel")

        return aggregated_model().create_model(
            model_input=activation,
            filter_size=filter_size,
            vocab_size=vocab_size,
            is_training=is_training,
            ff_dropout=ff_dropout,
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
                     is_training=True,
                     **unused_params):
        cluster_dropout = FLAGS.self_attention_cluster_dropout
        ff_dropout = FLAGS.self_attention_ff_dropout
        filter_size = FLAGS.self_attention_filter_size
        hidden_size = FLAGS.self_attention_hidden_size
        n_head = FLAGS.self_attention_n_head
        resid_dropout = FLAGS.self_attention_resid_dropout
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
        mix_features = tf.concat([video_features, audio_features], 1)

        video_cluster = SelfAttentionModule( feature_size = 1152,
                                              max_frames = max_frames,
                                              n_head = n_head,
                                              n_layer = n_layer,
                                              dropout_rate=cluster_dropout,
                                              resid_dropout_rate=resid_dropout,
                                              avg_emb=avg_embeeding,
                                               is_training=is_training)
        with tf.variable_scope("mix_video_audio"):
            activation = video_cluster.forward(mix_features)
        activation = tf.layers.dense(activation, hidden_size, use_bias=False, activation=None)
        activation = tf.layers.batch_normalization(activation, training=is_training)

        # Deep context gating.
        gating_weights1 = tf.layers.dense(activation, hidden_size * filter_size,
                                          use_bias=False, activation=tf.nn.relu)
        gating_weights1 = tf.layers.batch_normalization(gating_weights1, training=is_training)
        if is_training:
            gating_weights1 = tf.nn.dropout(gating_weights1, ff_dropout)

        gating_weights2 = tf.layers.dense(gating_weights1, hidden_size, use_bias=False, activation=None)
        gating_weights2 = tf.layers.batch_normalization(gating_weights2, training=is_training)
        gating_weights2 = tf.sigmoid(gating_weights2)
        activation = tf.multiply(activation, gating_weights2)

        aggregated_model = getattr(video_level_models,
                                   "MoeModel")

        return aggregated_model().create_model(
            model_input=activation,
            filter_size=filter_size,
            vocab_size=vocab_size,
            is_training=is_training,
            ff_dropout=ff_dropout,
            **unused_params)
