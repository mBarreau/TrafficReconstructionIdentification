# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:17:25 2020

@author: barreau
"""

import numpy as np
np.random.seed(12345)
import godunov as g
import reconstruction_neural_network as rn
import matplotlib.pyplot as plt
from pyDOE import lhs

#####################################
####     General parameters     #####
#####################################

Vf = 1.5 # Maximum car speed in km.min^-1
gamma = 0 # dissipativity coefficient (0 by default, discrepencies may occur if very small and not zero)
Tmax = 1.4 # simulation time in min
p = 1/15 # Probability that a car is a PV
L = 5 # Length of the road in km
rhoBar = 0.4 # Average density of cars on the road
rhoMax = 120 # Number of vehicles per kilometer
rhoSigma = 0.25 # initial condition standard deviation
noise = False # noise on the measurements and on the trajectories
greenshield = False # Type of flux function used for the numerical simulation
Ncar = rhoBar*rhoMax*L # Number of cars
Npv = int(Ncar*p) # Number of PV

# Initial position and time of probes vehicles
xiPos = L*lhs(1, samples=Npv).reshape((Npv,))
xiPos = np.flip(np.sort(xiPos))
xiT = np.array([0]*Npv)

# Godunov simulation of the PDE
simu_godunov = g.SimuGodunov(Vf, gamma, xiPos, xiT, L=L, Tmax=Tmax,
                             zMin=0, zMax=1, Nx=1000, greenshield=greenshield,
                             rhoBar=rhoBar, rhoSigma=rhoSigma)
rho = simu_godunov.simulation()
simu_godunov.plot()
axisPlot = simu_godunov.getAxisPlot()

# collect data from PV
t_train, x_train, rho_train, v_train = simu_godunov.getMeasurements(selectedPacket=-1, totalPacket=-1, noise=noise)

trained_neural_network = rn.ReconstructionNeuralNetwork(t_train, x_train, rho_train, v_train,
                                                    L, Tmax, N_f=7500, N_g=150)

[_, _, figError] = trained_neural_network.plot(axisPlot, rho)
simu_godunov.pv.plot()
plt.show()
figError.savefig('error.eps', bbox_inches='tight')


