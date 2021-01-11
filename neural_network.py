# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:15:37 2020

@author: barreau
"""

import logging
import os

# Delete some warning messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf

import numpy as np

class NeuralNetwork():

    def __init__(self, t, x, u, v, X_f, t_g, u_v,
                 layers_density, layers_trajectories, layers_speed, layers_acceleration,
                 init_density=[[], []], init_trajectories=[[], []], init_speed=[[], []]):
     
        '''
        Initialize a neural network for regression purposes.

        Parameters
        ----------
        t : list of N numpy array of shape (?,)
            standardized time coordinate of training points.
        x : list of N numpy array of shape (?,)
            standardized space coordinate of training points.
        u : list of N numpy array of shape (?,)
            standardized density values at training points.
        v : list of N numpy array of shape (?,)
            standardized velocity values at training points.
        X_f : 2D numpy array of shape (N_F, 2)
            standardized (space, time) coordinate of F physics training points.
        t_g : 1D numpy array of shape (N_G, 1)
            standardized time coordinate of G physics training points.
        u_v : 1D numpy array of shape (N_v, 1)
            standardized u coordinate of dV physics training points.
        layers_density : list of int (size N_L)
            List of integers corresponding to the number of neurons in each
            for the neural network Theta.
        layers_trajectories : list of int
            List of integers corresponding to the number of neurons in each 
            layer for the neural network Phi.
        layers_speed : list of int
            List of integers corresponding to the number of neurons in each 
            layer for the neural network V.
        init_density : list of two lists, optional
            Initial values for the weight and biases of Theta. 
            The default is [[], []].
        init_trajectories : list of two lists, optional
            Initial values for the weight and biases of Phi. 
            The default is [[], []].
        init_speed : list of two lists, optional
            Initial values for the weight and biases of V. 
            The default is [[], []].

        Returns
        -------
        None.

        '''

        self.t = t
        self.x = x 
        self.u = u 
        self.v = v

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        self.t_g = t_g
        self.u_v = u_v
        
        self.N = len(self.x) # Number of agents

        self.gamma_var = tf.Variable(tf.random.truncated_normal([1,1], mean=0, 
                                                         stddev=0.01, dtype=tf.float32), 
                                     dtype=tf.float32, trainable=True)
        self.noise_rho_bar = [tf.Variable(tf.random.truncated_normal([1,1], mean=0, 
                                                         stddev=0.01, dtype=tf.float32), 
                                     dtype=tf.float32, trainable=True)  for _ in range(self.N)]

        # Initilization of the neural networks
        
        # Theta neural network
        self.weights_density, self.biases_density = self.initialize_neural_network(layers_density, init_density[0], init_density[1], act="tanh")
        list_var_density = self.weights_density + self.biases_density
        list_var_density.append(self.gamma_var)
        list_var_density = list_var_density + self.noise_rho_bar
        
        # Phi neural network
        self.weights_trajectories = []
        self.biases_trajectories = []
        for i in range(self.N):
            weights_trajectories, biases_trajectories = self.initialize_neural_network(layers_trajectories, 
                                                                                                 initWeights=init_trajectories[0], 
                                                                                                 initBias=init_trajectories[1], 
                                                                                                 act="tanh")
            self.weights_trajectories.append(weights_trajectories)
            self.biases_trajectories.append(biases_trajectories)
                        
        # V neural network
        self.weights_speed, self.biases_speed = self.initialize_neural_network(layers_speed, init_speed[0], init_speed[1], act="tanh")
        self.weights_dspeed, self.biases_dspeed = self.initialize_neural_network(layers_acceleration, act="tanh")
        list_var_speed = self.weights_speed + self.biases_speed + self.weights_dspeed + self.biases_dspeed
        
        # Start a TF session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # PDE part     
        self.t_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(self.N)]
        self.x_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(self.N)]
        self.u_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(self.N)]
        self.v_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(self.N)]
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])
        self.u_v_tf = tf.placeholder(tf.float32, shape=[None, self.u_v.shape[1]])
        
        self.u_pred = [self.net_u(self.t_tf[i], self.net_x_pv(self.t_tf[i], i)) - self.noise_rho_bar[i]
                       for i in range(self.N)] 
        self.f_pred = self.net_f(self.t_f_tf, self.x_f_tf)        
        
        # Agents part
        self.t_g_tf = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(self.N)]
        
        self.x_pred = self.net_x(self.t_tf)
        self.g_pred = self.net_g(self.t_g_tf)

        # MSE part
        self.MSEu1 = tf.reduce_mean(tf.square(tf.concat(self.u_tf, 0) 
                                              - self.net_u(tf.concat(self.t_tf, 0),
                                                          tf.concat(self.x_tf, 0)))*tf.exp(tf.concat(self.u_tf, 0)))
        self.MSEu2 = tf.reduce_mean(tf.square(tf.concat(self.u_tf, 0)
                                              - tf.concat(self.u_pred, 0)))
        self.MSEf = tf.reduce_mean(tf.square(self.f_pred))
        
        self.MSEtrajectories = tf.reduce_mean(tf.square(tf.concat(self.x_tf, 0)
                                                        - tf.concat(self.x_pred, 0))*tf.exp(tf.concat(self.u_tf, 0)))
        self.MSEg = tf.reduce_mean(tf.square(tf.concat(self.g_pred, 0)))
            
        self.MSEv1 = tf.reduce_mean(tf.square(tf.concat(self.v_tf, 0) - self.net_v(tf.concat(self.u_tf, 0))))
        self.MSEv2 = tf.reduce_mean(tf.square(tf.concat(self.v_tf, 0) - self.net_v(tf.concat(self.u_pred, 0))))
        self.MSEv = tf.reduce_mean(tf.square(self.net_ddf(self.u_v_tf) - self.net_ddfNeg(self.u_v_tf)))
        
        # # Old training procedure
        # 
        # self.loss_trajectories = self.MSEtrajectories + 0.1*self.MSEg \
        #     + 0.1*self.MSEv1 + self.MSEv
        # self.loss = self.MSEu1 + 0.1*self.MSEf + 0.25*self.MSEg \
        #     + 0.1*self.MSEv1 + self.MSEv
        # self.loss_precise = 0.5*self.MSEu1 + 0.5*self.MSEu2 + self.MSEf \
        #     + self.MSEtrajectories + 0.5*self.MSEg + \
        #         0.1*self.MSEv1 + 0*self.MSEv2 + self.MSEv \
        #             + 0.1*tf.square(self.gamma_var)
        # self.optimizer = []
        # self.optimizer.append(OptimizationProcedure(self, self.loss_trajectories, 100, {'maxiter': 1000,
        #                                                                   'maxfun': 1000,
        #                                                                   'maxcor': 50,
        #                                                                   'maxls': 50,
        #                                                                   'ftol': 5.0 * np.finfo(float).eps}))
        # self.optimizer.append(OptimizationProcedure(self, self.loss, 1000, {'maxiter': 4000,
        #                                                                   'maxfun': 4000,
        #                                                                   'maxcor': 50,
        #                                                                   'maxls': 20,
        #                                                                   'ftol': 5.0 * np.finfo(float).eps}, 
        #                                             var_list=list_var_density))
        # self.optimizer.append(OptimizationProcedure(self, self.loss_precise, 0, {'maxiter': 10000,
        #                                                                   'maxfun': 50000,
        #                                                                   'maxcor': 150,
        #                                                                   'maxls': 75,
        #                                                                   'ftol': 1.0 * np.finfo(float).eps}))
        
        # New training procedure
        
        self.loss_normal = self.MSEtrajectories + self.MSEu1 + self.MSEv1 + self.MSEv
        self.loss_rough = 0.9*self.loss_normal + 0.1*(self.MSEu2 + self.MSEv2) + 0.5*(self.MSEg + self.MSEf)
        self.loss_precise = 0.5*self.loss_normal + 0.5*(self.MSEu2 + self.MSEv2) + 0.7*(self.MSEg + self.MSEf) + 0.1*tf.square(self.gamma_var)
        
        self.optimizer = []
        self.optimizer.append(OptimizationProcedure(self, self.MSEtrajectories + self.MSEv1, 5000, {'maxiter': 1000,
                                                                          'maxfun': 2000,
                                                                          'maxcor': 100,
                                                                          'maxls': 40,
                                                                          'ftol': 1.0 * np.finfo(float).eps}))
        self.optimizer.append(OptimizationProcedure(self, self.loss_normal, 5000, {'maxiter': 2000,
                                                                          'maxfun': 2000,
                                                                          'maxcor': 40,
                                                                          'maxls': 20,
                                                                          'ftol': 5.0 * np.finfo(float).eps}))
        self.optimizer.append(OptimizationProcedure(self, self.loss_rough, 5000, {'maxiter': 1000,
                                                                          'maxfun': 2000,
                                                                          'maxcor': 50,
                                                                          'maxls': 20,
                                                                          'ftol': 5.0 * np.finfo(float).eps}, 
                                                    var_list=list_var_density + list_var_speed))
        self.optimizer.append(OptimizationProcedure(self, self.loss_precise, 5000, {'maxiter': 2000,
                                                                          'maxfun': 4000,
                                                                          'maxcor': 50,
                                                                          'maxls': 40,
                                                                          'ftol': 1.0 * np.finfo(float).eps}))

        # Specific training procedure
        # self.optimizer = []
        # self.optimizer.append(OptimizationProcedure(self, self.MSEtrajectories, 10000, {'maxiter': 2000,
        #                                                                   'maxfun': 20000,
        #                                                                   'maxcor': 100,
        #                                                                   'maxls': 50,
        #                                                                   'ftol': 1.0 * np.finfo(float).eps}))


        # Initialize the TF session
        init = tf.global_variables_initializer() 
        self.sess.run(init)
        
    def initialize_neural_network(self, layers, initWeights=[], initBias=[], act="tanh"):
        '''
        Initialize a neural network

        Parameters
        ----------
        layers : list of integers of length NL
            List of number of nodes per layer.
        initWeights : list, optional
            List of matrices corresponding to the initial weights in each layer. 
            The default is [].
        initBias : list, optional
            List of matrices corresponding to the initial biases in each layer. 
            The default is [].
        act : string, optional
            Activation function. Can be tanh or relu. The default is "tanh".

        Returns
        -------
        weights : list of tensors
            List of weights as tensors with initial value.
        biases : list of tensors
            List of weights as tensors with initial value.

        '''
        
        weights, biases = [], []
        
        num_layers = len(layers)
        if len(initWeights) == 0:
            initWeights = [np.nan]*num_layers
            initBias = [np.nan]*num_layers
            
        for l in range(num_layers-1):
            
            if np.isnan(initWeights[l]).any():
                initWeights[l] = np.zeros((layers[l], layers[l+1]), dtype=np.float32)
                initBias[l] = np.zeros((1, layers[l+1]), dtype=np.float32)
                
            W = self.xavier_initializer(size=[layers[l], layers[l+1]], init=initWeights[l], act=act)
            b = tf.Variable(initBias[l], dtype=tf.float32) 

            weights.append(W)
            biases.append(b)
            
        return weights, biases
    
    def xavier_initializer(self, size, init, act="tanh"):
        '''
        Return random values in accordance with xavier initialization if tanh
        or he initialization if relu

        Parameters
        ----------
        size : list of integers
            size of the variable.
        init : numpy array
            initial value.
        act : string, optional
            Activation function, can be tanh or relu. The default is "tanh".

        Returns
        -------
        Tensor
            Initialized tensor.

        '''
        
        in_dim = size[0]
        out_dim = size[1]

        xavier_stddev = np.sqrt(4/(in_dim + out_dim))
        xavier_bound = np.sqrt(6/(in_dim + out_dim))
        
        if act == "relu":
            return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], mean=init, stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)
        else:
            return tf.Variable(init + tf.random.uniform([in_dim, out_dim], minval=init-xavier_bound, maxval=init+xavier_bound, dtype=tf.float32), dtype=tf.float32)
            #return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], mean=init, stddev=xavier_bound*np.sqrt(2/6), dtype=tf.float32), dtype=tf.float32)
           
    
    def neural_network(self, X, weights, biases, act=tf.nn.tanh):
        '''
        Compute the output of a given neural network in terms of tensor.

        Parameters
        ----------
        X : tensor
            Input.
        weights : list of tensors
            list of weights.
        biases : list of tensors
            list of biases.
        act : TF activation function, optional
            tf.nn.relu or tf.nn.tanh. The default is tf.nn.tanh.

        Returns
        -------
        tensor
            output of the neural network.

        '''
        
        num_layers = len(weights) + 1

        H = X
        for l in range(num_layers - 2):
            W, b = weights[l], biases[l]
            H = act(tf.add(tf.matmul(H, W), b))
            
        W, b = weights[-1], biases[-1]
        return tf.add(tf.matmul(H, W), b)
    
    def net_v(self, u):
        '''
        Standardized velocity

        Parameters
        ----------
        u : float32
            Standardized density.

        Returns
        -------
        TYPE
            Standardized velocity.

        '''
        return tf.square(self.neural_network(u, self.weights_speed, 
                                self.biases_speed, act=tf.nn.tanh))*(1-u)
    
    def net_ddf(self, u):
        '''
        Standardized second derivative of the flux

        Parameters
        ----------
        u : float32
            Standardized density.

        Returns
        -------
        TYPE
            Standardized second derivative of the flux.

        '''
        f = u*self.net_v(u)
        df = tf.gradients(f, u)[0]
        ddf = tf.gradients(df, u)[0]
        return ddf
    
    def net_ddfNeg(self, u):
        '''
        Positive estimation of the standardized acceleration

        Parameters
        ----------
        u : float32
            Standardized density.

        Returns
        -------
        TYPE
            Positive estimation of the standardized acceleration.

        '''
        return -tf.square(self.neural_network(u, self.weights_dspeed, 
                                self.biases_dspeed, act=tf.nn.tanh))
    
    def net_F(self, u):
        '''
        Characteristic speed

        Parameters
        ----------
        u : float32
            standardized density.

        Returns
        -------
        TYPE
            standardized characteristic speed.

        '''
        v = self.net_v(u)
        v_u = tf.gradients(v, u)[0]
        return v + (u+1)*v_u 

    def net_u(self, t, x):
        '''
        return the standardized value of rho hat at position (t, x)

        Parameters
        ----------
        t : tensor
            standardized time location.
        x : tensor
            standardized space location.

        Returns
        -------
        u_tanh : tensor
            standardized estimated density tensor.

        '''
        
        u_tanh = self.neural_network(tf.concat([t,x],1), self.weights_density, 
                                self.biases_density, act=tf.nn.tanh)
        return u_tanh

    def net_f(self, t, x):
        '''
        return the physics function f at position (t,x)

        Parameters
        ----------
        t : tensor
            standardized time location.
        x : tensor
            standardized space location.

        Returns
        -------
        tensor
            normalized estimated physics f tensor.

        '''
        
        u = self.net_u(t, x)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + self.net_F(u) * u_x - self.gamma_var**2 * u_xx
        return f
    
    def net_x_pv(self, t, i=0):
        '''
        return the standardized position of the agent i
        Parameters
        ----------
        t : tensor (NOT A LIST)
            standardized time.
        i : int, optional
            Number of the agent. The default is 0.
        Returns
        -------
        tensor
            standardized position of the agent i.
        '''
        x_tanh = self.neural_network(t, self.weights_trajectories[i], 
                                    self.biases_trajectories[i], act=tf.nn.tanh)
        return x_tanh
    
    def net_x(self, t):
        '''
        return the standardized position of each agent
        Parameters
        ----------
        t : list of tensors
            standardized time.
        Returns
        -------
        output : list of tensors
            list of standardized positions of all agents at given time.
        '''
        output = [self.net_x_pv(t[i], i) for i in range(self.N)]
        return output
    
    def net_g(self, t):
        '''
        return the physics function g for all agents at time t

        Parameters
        ----------
        t : tensor
            standardized time.

        Returns
        -------
        list of tensor
            list of standardized estimated physics g tensor.

        '''
        
        x_trajectories = self.net_x(t) 
        g = []
        for i in range(len(x_trajectories)):
            x_t = tf.gradients(x_trajectories[i], t[i])[0]
            u = self.net_u(t[i], x_trajectories[i])
            g.append(x_t - self.net_v(u))
        return g

    def loss_callback(self, MSEu1, MSEu2, MSEf, MSEtrajectories, MSEg, MSEv1, MSEv2, MSEv, total_loss, gamma):
        
        if self.epoch%10 == 1:
            print('Epoch: %.0f | MSEu1: %.5e | MSEu2: %.5e | MSEf: %.5e | MSEtrajectories: %.5e | MSEg: %.5e | MSEv1: %.5e | MSEv2: %.5e | MSEv: %.5e | Gamma: %.5e | Total: %.5e' %
                  (self.epoch, MSEu1, MSEu2, MSEf, MSEtrajectories, MSEg, MSEv1, MSEv2, MSEv, gamma**2, total_loss))
            
        self.epoch += 1

    def train(self):
        '''
        Train the neural networks

        Returns
        -------
        None.

        '''
        
        tf_dict = { }
        
        for k, v in zip(self.x_tf, self.x):
            tf_dict[k] = v
            
        for k, v in zip(self.t_tf, self.t):
            tf_dict[k] = v
            
        for k, v in zip(self.u_tf, self.u):
            tf_dict[k] = v
            
        for k, v in zip(self.v_tf, self.v):
            tf_dict[k] = v
            
        for k, v in zip(self.t_g_tf, self.t_g):
            tf_dict[k] = v
            
        tf_dict[self.t_f_tf] = self.t_f
        tf_dict[self.x_f_tf] = self.x_f
        tf_dict[self.u_v_tf] = self.u_v
        
        for i in range(len(self.optimizer)):
            print('---> STEP %.0f' % (i+1))
            self.epoch = 1
            self.optimizer[i].train(tf_dict)    
    
    def predict(self, t, x):
        '''
        Return the standardized estimated density at (t, x)

        Parameters
        ----------
        t : numpy array (?, )
            standardized time coordinate.
        x : numpy array (?, )
            standardized space coordinate.

        Returns
        -------
        numpy array
            standardized estimated density.

        '''
        t = np.float32(t)
        x = np.float32(x)

        return np.minimum(np.maximum(self.sess.run(self.net_u(t, x)), -1), 1)
    
    def predict_speed(self, u):
        '''
        Return the standardized estimated speed at u

        Parameters
        ----------
        u : numpy array (?, )
            standardized density.

        Returns
        -------
        numpy array
            standardized estimated speed.

        '''
        u = np.float32(u)
        return self.sess.run(self.net_v(u))
    
    def predict_dspeed(self, u):

        u = np.float32(u)
        u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        return (self.sess.run(self.net_ddf(u_tf), feed_dict={u_tf: u}), self.sess.run(self.net_ddfNeg(u)))
    
    def predict_F(self, u):
        '''
        Return the standardized estimated characteristic speed at u

        Parameters
        ----------
        u : numpy array (?, )
            standardized density.

        Returns
        -------
        numpy array
            standardized estimated characteristic speed.

        '''
        u = np.float32(u)
        u_tf = tf.placeholder(tf.float32, shape=[None, 1])
        return self.sess.run(self.net_F(u_tf), feed_dict={u_tf: u})
    
    def predict_trajectories(self, t):
        '''
        Return the standardized estimated agents' locations at t
        Parameters
        ----------
        t : N numpy arrays of size (?,)
            standardized time coordinate.
        Returns
        -------
        lif of N numpy arrays
            standardized estimated agents location.
        '''
        tf_dict = {}
        i = 0
        for k, v in zip(self.t_tf, t):
            tf_dict[k] = v
            i = i+1
        return self.sess.run(self.x_pred, tf_dict)
    
class OptimizationProcedure():
    
    def __init__(self, mother, loss, epochs, options, var_list=None):
        self.loss = loss
        self.optimizer_adam = tf.train.AdamOptimizer().minimize(loss, var_list=var_list)
        self.optimizer_BFGS = tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=var_list,
                                                                         method='L-BFGS-B', 
                                                                         options=options)

        self.mother = mother
        self.epochs = epochs
        
        
    def train(self, tf_dict):
        mother = self.mother
        print('------> ADAM')
        nbEpoch = 10
        last_epochs = [10]*nbEpoch
        nb_supp = 0
        for epoch in range(self.epochs):
            mother.epoch = epoch + 1
            if epoch%10 == 0:
                mother.loss_callback(mother.sess.run(mother.MSEu1, tf_dict), 
                                mother.sess.run(mother.MSEu2, tf_dict), 
                                mother.sess.run(mother.MSEf, tf_dict), 
                                mother.sess.run(mother.MSEtrajectories, tf_dict), 
                                mother.sess.run(mother.MSEg, tf_dict), 
                                mother.sess.run(mother.MSEv1, tf_dict), 
                                mother.sess.run(mother.MSEv2, tf_dict), 
                                mother.sess.run(mother.MSEv, tf_dict), 
                                mother.sess.run(self.loss, tf_dict), 
                                mother.sess.run(mother.gamma_var))
            mother.sess.run(self.optimizer_adam, tf_dict)
            loss = mother.sess.run(self.loss, tf_dict)
            
            if loss > np.amax(last_epochs):
                nb_supp = nb_supp + 1
            elif loss < np.amin(last_epochs):
                nb_supp = max(nb_supp - 1, 0)
                
            if nb_supp > np.ceil(nbEpoch/2):
                print('Early stop.')
                mother.sess.run(self.optimizer_adam, tf_dict)
                break
            last_epochs[epoch%nbEpoch] = loss
            
        mother.loss_callback(mother.sess.run(mother.MSEu1, tf_dict), 
                             mother.sess.run(mother.MSEu2, tf_dict), 
                             mother.sess.run(mother.MSEf, tf_dict), 
                             mother.sess.run(mother.MSEtrajectories, tf_dict), 
                             mother.sess.run(mother.MSEg, tf_dict), 
                             mother.sess.run(mother.MSEv1, tf_dict), 
                             mother.sess.run(mother.MSEv2, tf_dict), 
                             mother.sess.run(mother.MSEv, tf_dict), 
                             mother.sess.run(self.loss, tf_dict), 
                             mother.sess.run(mother.gamma_var))
            
        print('------> BFGS')
        self.optimizer_BFGS.minimize(mother.sess,
                                feed_dict=tf_dict,
                                fetches=[mother.MSEu1, mother.MSEu2, mother.MSEf, mother.MSEtrajectories, 
                                         mother.MSEg, mother.MSEv1, mother.MSEv2, mother.MSEv,
                                         self.loss, mother.gamma_var],
                                loss_callback=mother.loss_callback)