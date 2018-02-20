# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:26:51 2018

@author: dandrews

Actor Critic model pair
"""
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, Multiply
from keras import optimizers
from keras import backend as K
import numpy as np


class DeepQ():
    TAU = 0.1
    _learning_rate = 1e-4 #use change_learning_rate(new_rate)

    def __init__(self, input_shape, action_shape, num_aux_rewards):

        #create models
        shared_base = self._create_shared_based(input_shape)
        q_model = self._create_q_model(shared_base, action_shape)
        auxillary = self._create_auxillary_rewards_model(shared_base, action_shape, num_aux_rewards)
        self.auxillary = auxillary
        self.base = shared_base
        self.q_model = q_model
        K.set_value(self.q_model.optimizer.lr, self._learning_rate)

    def change_learing_rate(self, learning_rate):
        self._learning_rate = learning_rate
        K.set_value(self.q_model.optimizer.lr, self._learning_rate)

    def get_learning_rate(self):
        return self._learning_rate

    def _create_shared_based(self, input_shape):
                state = Sequential([
                    ################################
                    # Atari Solving network layout #
                    ################################
                   Conv2D(32, kernel_size=8,
                          strides=4,
                          input_shape=((input_shape)),
                          activation='relu',
                          padding='same',
                          name='Conv2d_1'
                          ),
                   Conv2D(64, kernel_size=4, strides=2, activation='relu', name='Conv2d_2', padding='same'),
                   Conv2D(64, kernel_size=3, strides=1, activation='relu', name='Conv2d_3', padding='same'),
                   Dense(128, activation='relu', name='shared_dense_1'),
                   Flatten(),
                   Dense(64, activation='linear', name='shared_output_1' )
                   ])

                return state

    def _create_q_model(self, shared_state, action_input_shape):
        action =  Sequential([
                Dense(shared_state.layers[-1].output_shape[1], activation='relu',input_shape=action_input_shape, name='action_dense_1'),
                ])

        mult = Multiply()([action.output, shared_state.output])
        merged = Dense(64, activation='relu', name='merged_dense')(mult)
        merged = Dense(32, activation='relu', name='critic_dense')(merged)
        merged = Dense(1, activation='tanh', name='critic_out')(merged)
        model = Model(inputs=[shared_state.input, action.input], outputs=merged)
        opt = optimizers.Adam(clipnorm=1.)
        model.compile(optimizer=opt, loss='mse')
        return model

    def _create_auxillary_rewards_model(self, shared_state, action_input_shape, num_aux_rewards):
        """
        Create a seperate 'head' for predicting domain knowledge
        AKA Auxillary features

        "auxillary Reward Architecture for Reinforcement Learning",
        https://arxiv.org/pdf/1706.04208.pdf, Seijen et al.

        "REINFORCEMENT LEARNING WITH UNSUPERVISED AUXILIARY TASKS"
        https://arxiv.org/pdf/1611.05397.pdf, Jaderberg et al.

        Create a second head that will be trained on domain knowledge of some
        kind, like pixel control or other domain knowlege.
        """
        action =  Sequential([
                Dense(shared_state.layers[-1].output_shape[1], activation='relu',input_shape=action_input_shape, name='auxillary_dense_1'),
                ])

        mult = Multiply()([action.output, shared_state.output])

        merged = Dense(64, activation='relu', name='auxillary_merged_dense')(mult)
        merged = Dense(32, activation='relu', name='auxillary_dense')(merged)
        merged = Dense(num_aux_rewards, activation='tanh', name='auxillary_out')(merged)
        model = Model(inputs=[shared_state.input, action.input], outputs=merged)
        opt = optimizers.Adam(clipnorm=1.)
        model.compile(optimizer=opt, loss='logcosh') # Similar to huber loss.
        return model


    def train_q_model(self, s_batch, a_batch, r_batch):
        loss = self.q_model.train_on_batch([s_batch, a_batch], r_batch)
        return loss

    def train_auxillary(self, s_batch, a_batch, aux_batch):
        new_shape = (aux_batch.shape[0], aux_batch.shape[1] *aux_batch.shape[2])
        loss = self.auxillary.train_on_batch([s_batch, a_batch], aux_batch.reshape(new_shape))
        return loss

    def target_train(self, target):
        """
        Nudges target model weights towards this model
        """
        self._target_train(self.q_model, target.q_model)


    def _target_train(self, source, target):
        tau = self.TAU
        source_weights = np.array(source.get_weights())
        target_weights = np.array(target.get_weights())
        new_weights = tau * source_weights + (1 - tau) * target_weights
        target.set_weights(new_weights)

    def scale_learning_rate(self, scale=0.1):
        lr = K.get_value(self.q_model.optimizer.lr)
        K.set_value(self.q_model.optimizer.lr, lr*scale)
        lr = K.get_value(self.actor.optimizer.lr)
        K.set_value(self.actor.optimizer.lr, lr*scale)
        return("New learning rate: {} ".format(
                K.get_value(self.q_model.optimizer.lr),
                ))


if __name__ == '__main__':
    input_shape = (84,84,3)
    action_shape = (9,)
    aux_num = 5
    batch_size = 10

    q_model = DeepQ(input_shape, action_shape, aux_num)
    s_batch = np.random.random_sample((batch_size,) + input_shape)
    a_batch = np.random.random_sample((batch_size,) + action_shape)
    r_batch = np.random.random_sample((batch_size,1))
    q_model.train_q_model(s_batch, a_batch, r_batch)

    aux_batch = np.random.random_sample((batch_size, aux_num))
    q_model.train_auxillary(s_batch, a_batch, aux_batch)
    print(q_model.q_model.predict([s_batch, a_batch]))
