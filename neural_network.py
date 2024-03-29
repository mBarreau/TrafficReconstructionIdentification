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
                 layers_density, layers_trajectories, layers_speed, max_speed=None,
                 init_density=[[], []], init_trajectories=[[], []], init_speed=[[], []],
                 beta=0.05, N_epochs=1000, N_lambda=10, sigmas=[], opt=0):
     
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
        
        tf.reset_default_graph()
        
        self.beta = beta
        self.N_epochs = N_epochs
        self.N_lambda = N_lambda
        self.opt = opt
        self.sigmas = sigmas

        self.t = t
        self.x = x 
        self.u = u 
        self.v = v

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        self.t_g = t_g
        self.u_v = u_v
        
        self.N = len(self.x)  # Number of agents
        self.max_speed = max_speed

        self.gamma_var = tf.Variable(1e-2, dtype=tf.float32, trainable=True)
        self.noise_rho_bar = [tf.Variable(tf.random.truncated_normal([1,1], mean=0, 
                                                          stddev=0.001, dtype=tf.float32), 
                                      dtype=tf.float32, trainable=True) for _ in range(self.N)]
        # self.noise_rho_bar = [tf.Variable(0., dtype=tf.float32, trainable=False) for _ in range(self.N)]

        # Initilization of the neural networks
        self.weights = {}
        self.biases = {}
        layers = {'speed':layers_speed, 'trajectories':layers_trajectories, 
                  'density':layers_density}
        init = {'speed':init_speed, 'trajectories':init_trajectories, 
                'density':init_density}
        
        # Theta neural network
        self.weights['density'], self.biases['density'] = self.initialize_neural_network(layers['density'],
                                                                                         init['density'][0],
                                                                                         init['density'][1],
                                                                                         act="tanh")
        list_var_density = self.weights['density'] + self.biases['density']
        # list_var_density = list_var_density + self.noise_rho_bar
        
        # Phi neural network
        self.weights['trajectories'] = []
        self.biases['trajectories'] = []
        list_var_trajectories = []
        for i in range(self.N):
            weights_trajectories, biases_trajectories = self.initialize_neural_network(layers['trajectories'],
                                                                                       initWeights=init['trajectories'][0],
                                                                                       initBias=init['trajectories'][1],
                                                                                       act="tanh")
            self.weights['trajectories'].append(weights_trajectories)
            self.biases['trajectories'].append(biases_trajectories)
            list_var_trajectories += weights_trajectories + biases_trajectories
                        
        # V neural network
        self.weights['speed'], self.biases['speed'] = self.initialize_neural_network(layers['speed'],
                                                                                     initWeights=init['speed'][0],
                                                                                     initBias=init['speed'][1],
                                                                                     act="tanh")
        list_var_speed = self.weights['speed'] + self.biases['speed']
        
        self.encoder1_weights = self.xavier_initializer([2, layers_density[1]], init=np.zeros((2, layers_density[1])))
        self.encoder1_biases = self.xavier_initializer([1, layers_density[1]], init=np.zeros((1, layers_density[1])))
        self.encoder2_weights = self.xavier_initializer([2, layers_density[1]], init=np.zeros((2, layers_density[1])))
        self.encoder2_biases = self.xavier_initializer([1, layers_density[1]], init=np.zeros((1, layers_density[1])))
        
        # list_var_density = list_var_density + [self.encoder1_weights, self.encoder1_biases, self.encoder2_weights, self.encoder2_biases]

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
        self.MSEu1 = tf.reduce_mean(tf.square(tf.concat(self.u_tf, 0) - self.net_u(tf.concat(self.t_tf, 0),tf.concat(self.x_tf, 0)))*tf.exp(0*tf.concat(self.u_tf, 0)))
        self.MSEu2 = tf.reduce_mean(tf.square(tf.concat(self.u_tf, 0) - tf.concat(self.u_pred, 0)))
        self.MSEf = tf.reduce_mean(tf.square(self.f_pred))
        
        self.MSEtrajectories = tf.reduce_mean(tf.square(tf.concat(self.x_tf, 0) - tf.concat(self.x_pred, 0))*tf.exp(0*tf.concat(self.u_tf, 0)))
        self.MSEg = tf.reduce_mean(tf.square(tf.concat(self.g_pred, 0)))
            
        self.MSEv1 = tf.reduce_mean(tf.square(tf.concat(self.v_tf, 0) - self.net_v(tf.concat(self.u_tf, 0))))
        self.MSEv2 = tf.reduce_mean(tf.square(tf.concat(self.v_tf, 0) - self.net_v(tf.concat(self.u_pred, 0))))
        self.MSEv = tf.reduce_mean(tf.square(tf.nn.relu(self.net_ddf(self.u_v_tf)))) #+ tf.reduce_mean(tf.square(tf.nn.relu(self.net_v(self.u_v_tf) - max_speed)))

        # Lambda update procedure
        self.losses = [self.MSEu1, self.MSEu2, self.MSEf,       # density loss functions
                       self.MSEv1, self.MSEv2, self.MSEv,       # speed loss functions
                       self.MSEtrajectories, self.MSEg,         # trajectory loss functions
                       tf.square(self.gamma_var)]               # viscosity parameter
        self.constraints = [0, 1, 1, 0, 1, 1, 0, 1, 1]
 
        self.lambdas_tf = [tf.placeholder(tf.float32, shape=()) for _ in self.losses]
        self.lambdas_init = [0] * len(self.losses)
        for i in range(len(self.losses)):
            if self.constraints[i] == 0:
                if len(sigmas) != len(self.losses):
                    # By default, the weighting terms are set to 1
                    self.lambdas_init[i] = 1
                else:
                    self.lambdas_init[i] = sigmas[i]
            else:
                self.lambdas_init[i] = 0
                    
        self.saved_lambdas = [[lambda_init] for lambda_init in self.lambdas_init]

        self.loss = np.dot(np.array(self.lambdas_tf), np.array(self.losses))
        
        if opt == 1 or opt == 2 or opt == 9: # We use the penalty method
            self.c_tf = tf.placeholder(tf.float32, shape=()) 
            for i in range(len(self.losses)):
                self.loss = self.loss + self.c_tf * self.constraints[i] * tf.square(self.losses[i]) / 2
        
        variables = list_var_density + list_var_speed + list_var_trajectories
        self.dL = []
        for g in tf.gradients(self.loss, variables):
            if g is not None:
                self.dL.append(g)
        
        self.losses_values = []
        self.loss_values = []
        
        self.optimizer = []   
        self.optimizer.append(OptimizationProcedure(self, self.MSEv1 + self.MSEtrajectories, 
                                                        0, 
                                                        {'maxiter': 40000, 
                                                         'maxfun': 40000,
                                                         'maxcor': 25,
                                                         'maxls': 20,
                                                         'ftol': 5.0 * np.finfo(float).eps},
                                                        opt=0))         
        self.optimizer.append(OptimizationProcedure(self, self.loss, 
                                                        self.N_epochs, 
                                                        {'maxiter': 40000, 
                                                         'maxfun': 40000,
                                                         'maxcor': 75,
                                                         'maxls': 50,
                                                         'ftol': 1.0 * np.finfo(float).eps},
                                                        opt=opt))
        
    def start(self):
         # Start a TF session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # Initialize the TF session
        init = tf.global_variables_initializer() 
        self.sess.run(init)      
        
    def close(self):
        self.sess.close()
        tf.reset_default_graph()
        
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

    def neural_network(self, X, weights, biases, act=tf.nn.tanh, encoders=False):
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
        
        if encoders:
            encoder1 = tf.nn.tanh(tf.add(tf.matmul(X, self.encoder1_weights), self.encoder1_biases))
            encoder2 = tf.nn.tanh(tf.add(tf.matmul(X, self.encoder2_weights), self.encoder2_biases))
        else:
            encoder1 = 1
            encoder2 = 0

        H = X
        for l in range(num_layers - 2):
            W, b = weights[l], biases[l]
            Z = act(tf.add(tf.matmul(H, W), b))
            H = tf.math.multiply(Z, encoder1) + \
                    tf.math.multiply(1 - Z, encoder2)
            
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
        
        v_tanh = tf.square(self.neural_network(u, self.weights['speed'],
                    self.biases['speed'], act=tf.nn.tanh))
        if self.max_speed is None:
            return v_tanh*(1-u)
        else:
            return (v_tanh*(1+u)/2 + self.max_speed)*(1-u)/2
        # return self.max_speed*(1-u)/2
    
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
        # return -self.max_speed * u

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
        
        u_tanh = self.neural_network(tf.concat([t,x],1), self.weights['density'], 
                                self.biases['density'], act=tf.nn.tanh, encoders=True)
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
        x_tanh = self.neural_network(t, self.weights['trajectories'][i], 
                                    self.biases['trajectories'][i], act=tf.nn.tanh)
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
        
        if self.opt == 1 or self.opt == 2 or self.opt == 9:
            tf_dict[self.c_tf] = 1e-2
        
        for i in range(len(self.optimizer)):
            for j in range(len(self.lambdas_tf)):
                tf_dict[self.lambdas_tf[j]] = self.lambdas_init[j]
            print('---> STEP %.0f' % (i+1))
            self.epoch = 1
            self.saved_lambdas = self.optimizer[i].train(tf_dict, i+1)    
    
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
        # return self.net_v(u)
    
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
    
    def __init__(self, mother, loss, N_epochs, options, var_list=None, opt=0):
        self.loss = loss
        self.opt = opt
        
        if self.opt >= 2: # We use ADAM
                self.optimizer_gd = tf.train.AdamOptimizer().minimize(loss, var_list=var_list)
        
        if self.opt == 0 or self.opt == 1 or self.opt >= 6:
            # Define BFGS
            self.optimizer_BFGS = tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=var_list,
                                                                             method='L-BFGS-B', 
                                                                             options=options)
        self.mother = mother
        self.N_epochs = N_epochs
        
    def train(self, tf_dict, step=1):
        
        mother = self.mother
        # N_f = len(tf_dict[mother.t_f_tf])
        saved_lambdas = [[tf_dict[mother.lambdas_tf[i]]] for i in range(len(mother.lambdas_tf))]
        
        if self.opt >= 2: # Use of first order scheme
                
            L = mother.sess.run(mother.losses, tf_dict)

            for epoch in range(self.N_epochs):
                
                mother.epoch = epoch + 1
                #mother.loss_callback(L, mother.sess.run(self.loss, tf_dict))
                
                mother.sess.run(self.optimizer_gd, tf_dict)
                
                if self.opt == 4 or self.opt == 6: # ADAMBU or pretraining BFGS: Update of lambda
                    if epoch % mother.N_lambda == 0:
                        lambdas = mother.sess.run(mother.lambdas, tf_dict)
                        for i in range(len(saved_lambdas)):
                            if len(mother.sigmas) > 0:
                                tf_dict[mother.lambdas_tf[i]] = min(lambdas[i], mother.sigmas[i])
                            else:
                                tf_dict[mother.lambdas_tf[i]] = lambdas[i]
                        # t_f, x_f = latin_hypercube_sampling(N_f)
                        # tf_dict[mother.t_f_tf] = t_f
                        # tf_dict[mother.x_f_tf] = x_f
                elif self.opt == 3 or self.opt == 7: # Primal-dual
                    if epoch % mother.N_lambda == 0:
                        for i in range(len(saved_lambdas)):
                            if mother.constraints[i] == 0:
                                continue
                            if mother.sigmas[i] <= 0: # This is a hard constraint
                                lr = 1e-3 * 1 / (1 + epoch / self.N_epochs)
                                Li = mother.sess.run(mother.losses[i], tf_dict)
                                new_lambda = tf_dict[mother.lambdas_tf[i]] + lr * Li
                                tf_dict[mother.lambdas_tf[i]] = new_lambda
                            else: # This is a soft constraint
                                lr = 1e-2 * (2 * mother.sigmas[i] - tf_dict[mother.lambdas_tf[i]]) / mother.sigmas[i]
                                Li = mother.sess.run(mother.losses[i], tf_dict)
                                new_lambda = tf_dict[mother.lambdas_tf[i]] + lr * Li
                                tf_dict[mother.lambdas_tf[i]] = min(new_lambda, mother.sigmas[i])
                                
                elif self.opt == 2 or self.opt == 9: # ADMM
                    if epoch % mother.N_lambda == 0:
                        for i in range(len(saved_lambdas)):
                            if mother.constraints[i] == 0:
                                continue
                            if mother.sigmas[i] <= 0: # This is a hard constraint
                                Li = mother.sess.run(mother.losses[i], tf_dict)
                                new_lambda = tf_dict[mother.lambdas_tf[i]] + tf_dict[mother.c_tf] * Li
                                tf_dict[mother.lambdas_tf[i]] = new_lambda
                            else: # this is a soft constraint
                                tf_dict[mother.c_tf] = 1e-2 * (2 * mother.sigmas[i] - tf_dict[mother.lambdas_tf[i]]) / mother.sigmas[i]
                                Li = mother.sess.run(mother.losses[i], tf_dict)
                                new_lambda = tf_dict[mother.lambdas_tf[i]] + tf_dict[mother.c_tf] * Li
                                tf_dict[mother.lambdas_tf[i]] = min(new_lambda, mother.sigmas[i])                                
                                
                for i in range(len(saved_lambdas)):
                    saved_lambdas[i].append(tf_dict[mother.lambdas_tf[i]])
                        
                Lnew = mother.sess.run(mother.losses, tf_dict)
                coef_L = 0.
                for i in range(len(Lnew)):
                    coef_L = max(coef_L, abs(L[i] - Lnew[i]) / max(L[i], Lnew[i], 1))
                    
                dL = mother.sess.run(mother.dL, tf_dict)
                max_dL = 0.
                for i in range(len(dL)):
                    max_dL = max(max_dL, np.amax(abs(dL[i])))
                    
                if np.max(Lnew) <= 1e-6:
                    print('Low value of the cost.')
                    break
                
                if coef_L <= 1e-8:
                    print('No evolution of the cost.')
                    break
                
                if max_dL <= 1e-8:
                    print('Gradient is almost zero.')
                    break
                
                L = Lnew
                
        if self.opt == 0 or self.opt >= 6:
            self.optimizer_BFGS.minimize(mother.sess,
                                    feed_dict=tf_dict,
                                    fetches=[mother.losses, self.loss])
            
        if self.opt == 1: # Real penaly method
            for i in range(self.N_epochs):
                self.optimizer_BFGS.minimize(mother.sess,
                                        feed_dict=tf_dict,
                                        fetches=[mother.losses, self.loss])
                
                max_difference_lambdas = 0
                for i in range(len(saved_lambdas)):
                    if mother.constraints[i] == 0:
                        saved_lambdas[i].append(1.)
                        continue
                    old_lambdai = tf_dict[mother.lambdas_tf[i]]
                    Li = mother.sess.run(mother.losses[i], tf_dict)
                    tf_dict[mother.lambdas_tf[i]] = old_lambdai + tf_dict[mother.c_tf] * Li
                    saved_lambdas[i].append(tf_dict[mother.lambdas_tf[i]])
                    relative_diff = abs(old_lambdai - tf_dict[mother.lambdas_tf[i]]) / max(old_lambdai, tf_dict[mother.lambdas_tf[i]], 1)
                    max_difference_lambdas = max(max_difference_lambdas, relative_diff)
                    
                tf_dict[mother.c_tf] = 1.1 * tf_dict[mother.c_tf]
                    
                if max_difference_lambdas <= 1e-3:
                    break
            
        return saved_lambdas