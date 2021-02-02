# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:17:25 2020

@author: barreau
"""

import numpy as np
np.random.seed(12345)
import reconstruction_neural_network as rn
import sumo as s

#####################################
####     General parameters     #####
#####################################

scenario = 'highway'

sumo = s.Sumo(scenario)
Nx, Nt = sumo.Nx, sumo.Nt
L, Tmax = sumo.L, sumo.Tmax

rho = sumo.getDensity()  # density(time, position) (300, 1000)
sumo.plotDensity()
t_train, x_train, rho_train, v_train = sumo.getMeasurements()
axisPlot = sumo.getAxisPlot()
Vf = np.amax(v_train[14])

trained_neural_network = rn.ReconstructionNeuralNetwork(t_train, x_train, rho_train, v_train,
                                                    L, Tmax, N_f=1000, N_g=100)
trained_neural_network.train()

[_, _, figError] = trained_neural_network.plot(axisPlot, rho)
sumo.plotProbeVehicles()
figError.savefig('error.eps', bbox_inches='tight')