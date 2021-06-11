# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:17:25 2020

@author: barreau
"""

import numpy as np
np.random.seed(12345)
import godunov as g
import reconstruction_neural_network as rn
from pyDOE import lhs
import matplotlib.pyplot as plt

#####################################
####     General parameters     #####
#####################################

Vf = 1.5 # Maximum car speed in km.min^-1
gamma = 0 # dissipativity coefficient (0 by default, discrepencies may occur if very small but not zero)
Tmax = 2 # simulation time in min
p = 1/20 # Probability that a car is a PV
L = 5 # Length of the road in km
rhoBar = 0.5 # Average density of cars on the road
rhoSigma = 0.45 # initial condition standard deviation
rhoMax = 120 # Number of vehicles per kilometer
noise = True # noise on the measurements and on the trajectories
greenshield = True # Type of flux function used for the numerical simulation
Ncar = rhoBar*rhoMax*L # Number of cars
Npv = int(Ncar*p) # Number of PV

# Initial position and time of probes vehicles
xiPos = L*lhs(1, samples=Npv).reshape((Npv,))
xiPos = np.flip(np.sort(xiPos))
xiT = np.array([0]*Npv)

# Godunov simulation of the PDE
simu_godunov = g.SimuGodunov(Vf, gamma, xiPos, xiT, L=L, Tmax=Tmax,
                             zMin=0, zMax=1, Nx=500, greenshield=greenshield,
                             rhoBar=rhoBar, rhoSigma=rhoSigma)
rho = simu_godunov.simulation()
simu_godunov.plot()
axisPlot = simu_godunov.getAxisPlot()

# collect data from PV
t_train, x_train, rho_train, v_train = simu_godunov.getMeasurements(selectedPacket=-1, totalPacket=-1, noise=noise)

trained_neural_network = rn.ReconstructionNeuralNetwork(t_train, x_train, rho_train, v_train,
                                                    L, Tmax, v_max=Vf, N_f=1000, N_g=50, N_v=30, opt=7)
trained_neural_network.start() 
trained_neural_network.train()

[_, _, figError] = trained_neural_network.plot(axisPlot, rho)
simu_godunov.pv.plot()
# figError.savefig('error.eps', bbox_inches='tight')

plt.show()
trained_neural_network.close()