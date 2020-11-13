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

Vf = 25 # Maximum car speed in m.s^-1
gamma = 0 # dissipativity coefficient (0 by default, discrepencies may occur if very small and not zero)
Tmax = 100 # simulation time
p = 1/15 # Probability that a car is a PV
L = 5000 # Length of the road
rhoBar = 0.2 # Average density of cars on the road
rhoMax = 120 # Number of vehicles per kilometer
rhoSigma = 0.6 # initial condition standard deviation
noise = True # noise on the measurements and on the trajectories
greenshield = True # Type of flux function used for the numerical simulation

Vbar = Vf*(1-rhoBar) # Average speed
Lplus = Tmax*(Vbar+0.1*Vf)/1.1 # Additionnal length
Ltotal = L + Lplus

Ncar = rhoBar*rhoMax*Ltotal/1000 # Number of cars
Npv = int(Ncar*p) # Number of PV

# Initial position and time of probes vehicles
xiPos = L*lhs(1, samples=Npv).reshape((Npv,))
xiPos = np.flip(np.sort(xiPos))
xiT = np.array([0]*Npv)

# Godunov simulation of the PDE
simu_godunov = g.SimuGodunov(Vf, gamma, xiPos, xiT, L=Ltotal, Tmax=Tmax,
                             zMin=0, zMax=1, Nx=1000, greenshield=greenshield,
                             rhoBar=rhoBar, rhoSigma=rhoSigma)
rho = simu_godunov.simulation()
simu_godunov.plot()
axisPlot = simu_godunov.getAxisPlot()

# collect data from PV
t_train, x_train, rho_train, v_train = simu_godunov.getMeasurements(selectedPacket=-1, totalPacket=-1, noise=noise)

trained_neural_network = rn.ReconstructionNeuralNetwork(t_train, x_train, rho_train, v_train,
                                                    Ltotal, Tmax, N_f=7500, N_g=150)

[_, _, figError] = trained_neural_network.plot(axisPlot, rho)
simu_godunov.pv.plot(axisPlot[1])
plt.show()
figError.savefig('error.eps', bbox_inches='tight')


