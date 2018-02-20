# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:31:13 2018

@author: dandrews
"""

from keras_deep_q import DeepQ
import numpy as np
from matplotlib import pyplot as plt


class Grid_World():
    """
    Sample grid that more or less mimics AI Gym environments
    """

    def __init__(self, x_dim, y_dim):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self._create_coord_grid()
        self.reset()
        self.diagonal = np.sqrt(x_dim * y_dim)

    def reset(self):
        # pick 3 differnt points
        p, g, f = np.random.choice(len(self._point_list), 3)

        self.failure = self._point_list[f]
        self.goal = self._point_list[g]
        self.player = self._point_list[p]
        return self.data()

    def step(self, action):
        """
        A more complete implementation would move the player and
        calculate a reward.
        For this example we just want to generate data so will just call reset
        """
        s_ = self.reset()
        r = np.linalg.norm(self.player-self.goal)/ self.diagonal # q value is % of distance to goal
        t = False
        info = self.auxillary()
        return s_, r, t, info




    def data(self):
        output = np.zeros((self.x_dim, self.y_dim, 3))
        output[tuple(self.failure) + (0,)] = 1 # r
        output[tuple(self.goal)    + (1,)] = 1 # g
        output[tuple(self.player)  + (2,)] = 1 # b
        return output


    def auxillary(self):
        player_dist = np.linalg.norm(self.player-self._coord_grid, axis=2)
        prob_location = self.softmax(-player_dist)
        return prob_location


    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()



    def _create_coord_grid(self):
        """
        creates an (x,y,2) grid where g[x,y] = [x,y]
        for linalg operations on the entire grid
        """
        x = np.arange(self.x_dim, dtype=np.int32)
        y = np.arange(self.y_dim, dtype=np.int32)
        g = np.array(np.meshgrid(x,y))
        g = np.moveaxis(g, 0, -1)
        g = np.moveaxis(g, 0, 1)

        self._point_list = g[g[:,:,0]> -1]
        self._coord_grid = g


class Predict():
    action_shape = (1,)
    epochs = 1000
    batch_size = 1000
    a_loss = []
    q_loss = []
    use_aux = False


    def __init__(self, gw: Grid_World):
        self.gw = gw
        self.deep_q = DeepQ((gw.x_dim, gw.y_dim, 3), self.action_shape, gw.x_dim * gw.y_dim)

    def get_samples(self, n):
        """
        A more complete implementation would, for example, use reinforcement
        learning to train the Q values as well as the aux_values.
        """
        s_batch = []
        a_batch = []
        r_batch = []
        aux_batch = []
        for i in range(n):
            action = 0
            s_, r, t, info = gw.step(action)
            s_batch.append(s_)
            a_batch.append(action)
            aux_batch.append(info)
            r_batch.append(r)

        return (np.array(s_batch, dtype=np.float32),
                np.array(a_batch, dtype=np.float32),
                np.array(r_batch, dtype=np.float32),
                np.array(aux_batch, dtype=np.float32)
                )


    def train(self):
        for epoch in range(self.epochs):
            s_batch, a_batch, r_batch, aux_batch = self.get_samples(self.batch_size)
            if self.use_aux:
                al = self.deep_q.train_auxillary(s_batch, a_batch, aux_batch)
                self.a_loss.append(al)
            ql = self.deep_q.train_q_model(s_batch, a_batch, r_batch)

            self.q_loss.append(ql)
            print(ql, end=", ")

        plt.title("q loss")
        plt.plot(self.q_loss)
        plt.show()
        if self.use_aux:
            plt.title("aux loss")
            plt.plot(self.a_loss)
            plt.show()



if __name__ == '__main__':
    gw = Grid_World(3,4)
    plt.imshow(gw.data())
    plt.show()
    plt.imshow(gw.auxillary())
    plt.show()

    predict = Predict(gw)
    predict.train()

