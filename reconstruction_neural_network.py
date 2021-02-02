# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:15:37 2020

@author: barreau
"""

import numpy as np
#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from time import time
from pyDOE import lhs
from neural_network import NeuralNetwork
        
def hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print('{:.0f}h {:.0f}m {:.0f}s'.format(h, m, s))
    
def amin(l):
    min_list = [np.amin(l[i]) for i in range(len(l))]
    return np.amin(min_list)

def amax(l):
    min_list = [np.amax(l[i]) for i in range(len(l))]
    return np.amax(min_list)

class ReconstructionNeuralNetwork():
    
    def __init__(self, t, x, rho, v, L, Tmax, v_max=None, N_f=1000, N_g=100, N_v=50):
        '''
        Initialize a neural network for density reconstruction

        Parameters
        ----------
        t : List of N numpy array of shape (?,)
            time coordinate of training points.
        x : list of N numpy array of shape (?,)
            space coordinate of training points.
        rho : list of N numpy array of shape (?,)
            density values at training points.
        v : list of N numpy array of shape (?,)
            velocity values at training points.
        L : float64
            Length of the spacial domain.
        Tmax : float64
            Length of the temporal domain.
        N_f : integer, optional
            Number of physical points for F. The default is 1000.
        N_g : integer, optional
            Number of physical points for G. The default is 100.

        Returns
        -------
        None.

        '''
        
        self.Nxi = len(x) # Number of agents
        
        self.rho = rho
        self.v = v
        self.t = t
        
        num_hidden_layers = min(max(int(4*Tmax), 5), 15)
        num_nodes_per_layer = int(5*L) 
        layers_density = [2] # There are two inputs: space and time
        for _ in range(num_hidden_layers):
            layers_density.append(num_nodes_per_layer)
        layers_density.append(1)  
        
        num_hidden_layers = min(max(int(3*L), 4), 15)
        num_nodes_per_layer = 10
        layers_trajectories = [1] # There are two inputs: space and time
        # for _ in range(num_hidden_layers):
        #     layers_trajectories.append(num_nodes_per_layer)
        layers_trajectories.append(1)
        
        t_train, x_train, u_train, v_train, X_f_train, t_g_train, u_v_train, v_max = self.createTrainingDataset(t, x, rho, v, v_max, L, Tmax, N_f, N_g, N_v) # Creation of standardized training dataset
        
        self.neural_network = NeuralNetwork(t_train, x_train, u_train, v_train, 
                                            X_f_train, t_g_train, u_v_train,
                                            layers_density=layers_density, 
                                            layers_trajectories=layers_trajectories, 
                                            layers_speed=(1, 5, 5, 5, 5, 1),
                                            max_speed=v_max) # Creation of the neural network
            
    def createTrainingDataset(self, t, x, rho, v, v_max, L, Tmax, N_f, N_g, N_v):       
        '''
        Standardize the dataset

        Parameters
        ----------
        t : list of N arrays of float64 (?,)
            Time coordinate of agents.
        x : list of N arrays of float64 (?,)
            Position of agents along time.
        rho : list of N arrays of float64 (?,)
            Density measurement from each agent.
        v : list of N arrays of float64 (?,)
            Velocity measurement from each agent.
        L : float
            Length of the road.
        Tmax : float
            Time-window.
        N_f : int
            Number of physical points for f.
        N_g : int
            Number of physical points for g.

        Returns
        -------
        t : list of N arrays of float64 (?,)
            Standardized time coordinate of agents.
        x : list of N arrays of float64 (?,)
            Standardized position of agents along time.
        u : list of N arrays of float64 (?,)
            Standardized density measurement from each agent.
        v : list of N arrays of float64 (?,)
            Standardized velocity measurement from each agent.
        X_f : 2D array of shape (N_f, 2)
            Standardized location of physical points for f.
        t_g : list of float64
            List of standardized physical points for g.

        '''
        
        self.lb = np.array([amin(x), amin(t)])
        self.ub = np.array([amax(x), amax(t)])
        self.lb[0], self.lb[1] = 0, 0
        
        x = [2*(x_temp - self.lb[0])/(self.ub[0] - self.lb[0]) - 1 for x_temp in x]
        t = [2*(t_temp - self.lb[1])/(self.ub[1] - self.lb[1]) - 1 for t_temp in t]
        rho = [2*rho_temp-1 for rho_temp in rho]
        v = [v_temp*(self.ub[1] - self.lb[1]) / (self.ub[0] - self.lb[0]) for v_temp in v]
        if v_max is not None:
            v_max = v_max*(self.ub[1] - self.lb[1]) / (self.ub[0] - self.lb[0])
        
        X_f = np.array([2, 2])*lhs(2, samples=N_f)
        X_f = X_f - np.ones(X_f.shape)
        np.random.shuffle(X_f)
        
        t_g = []
        for i in range(self.Nxi):
            tgi = np.amin(t[i]) + lhs(1, samples=N_g)*(np.amax(t[i]) - np.amin(t[i]))
            np.random.shuffle(tgi)
            t_g.append(tgi)
            
        u_v = lhs(1, samples=N_v)*2-1
        np.random.shuffle(u_v)
        
        return (t, x, rho, v, X_f, t_g, u_v, v_max)

    def train(self):
        '''
        Train the neural network

        Returns
        -------
        None.

        '''
        start = time()
        self.neural_network.train()
        hms(time() - start)
        
    def predict(self, t, x):
        '''
        Return the estimated density at (t, x)

        Parameters
        ----------
        t : numpy array (?, )
            time coordinate.
        x : numpy array (?, )
            space coordinate.

        Returns
        -------
        numpy array
            estimated density.

        '''
        
        x = 2*(x - self.lb[0])/(self.ub[0] - self.lb[0])-1
        t = 2*(t - self.lb[1])/(self.ub[1] - self.lb[1])-1
        
        return self.neural_network.predict(t, x)/2+0.5
    
    def predict_speed(self, rho):
        '''
        Return the estimated speed at rho

        Parameters
        ----------
        rho : numpy array (?, )
            density.

        Returns
        -------
        numpy array
            estimated speed.

        '''
        
        u = 2*rho-1
        
        return self.neural_network.predict_speed(u)*(self.ub[0] - self.lb[0]) / (self.ub[1] - self.lb[1])

    
    def predict_F(self, rho):
        '''
        Return the estimated characteristic speed at rho

        Parameters
        ----------
        rho : numpy array (?, )
            density.

        Returns
        -------
        numpy array
            estimated characteristic speed.

        '''
        
        u = 2*rho-1
        
        return self.neural_network.predict_F(u)*(self.ub[0] - self.lb[0]) / (self.ub[1] - self.lb[1])
    
    def predict_trajectories(self, t):
        '''
        Return the estimated agents' locations at t

        Parameters
        ----------
        t : list of N numpy arrays of size (?, 1)
            time coordinate.

        Returns
        -------
        list of N numpy arrays
            estimated agents location.

        '''
        
        t = [2*(t[i] - self.lb[1])/(self.ub[1] - self.lb[1])-1 for i in range(self.Nxi)]
        
        output = self.neural_network.predict_trajectories(t)
        output = [(output[i]+1)*(self.ub[0] - self.lb[0])/2 + self.lb[0] for i in range(self.Nxi)]
        return output
    
    
    def plot(self, axisPlot, rho):
        '''
        

        Parameters
        ----------
        axisPlot : tuple of two 1D-numpy arrays of shape (?,)
            Plot mesh.
        rho : 2D numpy array
            Values of the real density at axisPlot.

        Returns
        -------
        list of three Figures
            return the speed, reconstruction and error figures.

        '''
        
        x = axisPlot[0]
        t = axisPlot[1]

        Nx = len(x)
        Nt = len(t)
            
        XY_prediction = np.zeros((Nx * Nt, 2))
        k = 0
        for i in range(0, Nx):
            for j in range(0, Nt):
                XY_prediction[k] = np.array([t[j], x[i]])
                k = k + 1
        tstar = XY_prediction[:, 0:1]
        xstar = XY_prediction[:, 1:2]
        
        rho_prediction = self.predict(tstar, xstar).reshape(Nx, Nt)
        t_pred = [(np.linspace(np.amin(self.t[i]), np.amax(self.t[i]), t.shape[0])).reshape(-1, 1) for i in range(self.Nxi)]
        X_prediction = self.predict_trajectories(t_pred)
        rho_speed = np.linspace(0, 1).reshape(-1,1)
        v_prediction = self.predict_speed(rho_speed).reshape(-1,1)
        F_prediction = self.predict_F(rho_speed).reshape(-1,1)
        
        figSpeed = plt.figure(figsize=(7.5, 5))
        plt.plot(rho_speed, v_prediction, rasterized=True, label=r'NN approximation of $V$')
        plt.plot(rho_speed, F_prediction, rasterized=True, label=r'NN approximation of $F$')
        densityMeasurements = np.empty((0,1))
        speedMeasurements = np.empty((0,1))
        for i in range(self.Nxi):
            densityMeasurements = np.vstack((densityMeasurements, self.rho[i]))
            speedMeasurements = np.vstack((speedMeasurements, self.v[i]))
        plt.scatter(densityMeasurements, speedMeasurements, rasterized=True, 
                    c='k', s=1, label=r'Data')
        plt.xlabel(r'Normalized Density')
        plt.ylabel(r'Speed [km/min]')
        # plt.ylim(-v_prediction[0], v_prediction[0])
        plt.xlim(0, 1)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        # plt.title('Reconstruction')
        figSpeed.savefig('speed.eps', bbox_inches='tight')

        figReconstruction = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(t, x)
        plt.pcolor(X, Y, rho_prediction, vmin=0.0, vmax=1.0, shading='auto', 
                   cmap='rainbow', rasterized=True)
        for i in range(self.Nxi):
            plt.plot(t_pred[i], X_prediction[i], color="saddlebrown")
        plt.xlabel(r'Time [min]')
        plt.ylabel(r'Position [km]')
        plt.xlim(min(t), max(t))
        plt.ylim(min(x), max(x))
        plt.colorbar()
        plt.tight_layout()
        # plt.title('Reconstruction')
        figReconstruction.savefig('reconstruction.eps', bbox_inches='tight')
        
        figLambda = plt.figure(figsize=(7.5, 5))
        color_plot = plt.rcParams['axes.prop_cycle'].by_key()['color']
        style_plot = ["-", "--"]
        epochs = np.arange(len(self.neural_network.saved_lambdas[0])) * self.neural_network.nbEpoch
        for i in range(len(self.neural_network.saved_lambdas)):
            plt.plot(epochs, self.neural_network.saved_lambdas[i], label='$\lambda_{i}$'.format(i=i+1), 
                     linestyle=style_plot[i%2],
                     color=color_plot[int(i/2)])
        plt.xlabel(r'Epoch')
        plt.ylabel(r'Lambda values')
        plt.grid()
        plt.xlim(0, max(epochs))
        plt.ylim(0, 1)
        plt.legend(loc='best')
        plt.tight_layout()
        # plt.title('Absolute error')
        figLambda.savefig('lambda.eps', bbox_inches='tight') 
        
        figError = plt.figure(figsize=(7.5, 5))
        X, Y = np.meshgrid(t, x)
        plt.pcolor(X, Y, np.abs(rho_prediction-rho), vmin=0.0, vmax=1.0, 
                   shading='auto', cmap='rainbow', rasterized=True)
        for i in range(self.Nxi):
            plt.plot(t_pred[i], X_prediction[i], color="saddlebrown")
        plt.xlabel(r'Time [min]')
        plt.ylabel(r'Position [km]')
        plt.xlim(min(t), max(t))
        plt.ylim(min(x), max(x))
        plt.colorbar()
        plt.tight_layout()
        print("Normalized L^2 error: ", np.mean(np.square(rho_prediction-rho)))
        # plt.title('Absolute error')
        # figError.savefig('error.eps', bbox_inches='tight') 
        
        return [figSpeed, figReconstruction, figError]