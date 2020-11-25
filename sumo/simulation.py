# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:44:36 2020

@author: barreau
"""

import os, sys, csv

if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:
     sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt

from scipy import signal
from probe_vehicle import ProbeVehicleArray

scenario = "highway"

traci.start(["sumo", "-c", scenario+"/"+scenario+".sumocfg"])

deltaX = 0.010 # in km, more than a vehicle
L = 3 # in km
deltaT = traci.simulation.getDeltaT()/60 # in min
Tmax = 11 # in min
Tstart = 8 # in min
sigma = 0.01 # in km
tau = 0.06 # in min

Nt = int(np.ceil(Tmax/deltaT))
NtStart = int(np.floor(Tstart/deltaT))
Nx = int(np.ceil(L/deltaX))

numberOfVehicles = np.zeros((Nx, Nt-NtStart))
trafficLightPhase = 0
PVList = ProbeVehicleArray()

for n in range(Nt):
   if n%100 == 0:
      print("step", n)
   if n >= NtStart:
       for vehID in traci.vehicle.getIDList():
           if traci.vehicle.getRouteID(vehID) == 'route_0' or traci.vehicle.getRouteID(vehID) == 'route_1':
               vehPos = traci.vehicle.getPosition(vehID)[0]
               vehSpeed = traci.vehicle.getSpeed(vehID)
               i = int(np.floor(vehPos/(1000*deltaX)))
               if 0 <= i < Nx:
                   numberOfVehicles[i,n-NtStart] += 1
                   if traci.vehicle.getTypeID(vehID) == 'PV':
                       PVList.update(vehID, n*deltaT, vehPos/1000, vehSpeed/3.6)        
   traci.simulationStep()
print("step", n)
traci.close()

t = np.linspace(Tstart, Tmax, Nt-NtStart)
x = np.linspace(0, L, Nx)
X, Y = np.meshgrid(t, x)

fig = plt.figure(figsize=(7.5, 5))
plt.pcolor(X, Y, numberOfVehicles, shading='auto', cmap='rainbow')
plt.xlabel('Time [min]')
plt.ylabel('Position [km]')
plt.xlim(min(t), max(t))
plt.ylim(min(x), max(x))
plt.colorbar()
plt.tight_layout()

maxI = int(np.ceil(5*sigma/deltaX))
maxJ = int(np.ceil(5*tau/deltaT))
kernel = np.zeros((2*maxI+1, 2*maxJ+1))
for i in range(2*maxI+1):
    for j in range(2*maxJ+1):
        newI = i-maxI-1
        newJ = j-maxJ-1
        kernel[i,j] = np.exp(-abs(newI)*deltaX/sigma - abs(newJ)*deltaT/tau)
N = kernel.sum()
density = signal.convolve2d(numberOfVehicles, kernel, boundary='symm', mode='same')/N
densityMax = np.amax(density)
density = density/densityMax

fig = plt.figure(figsize=(7.5, 5))
plt.pcolor(X, Y, density, shading='auto', vmin=0, vmax=1, cmap='rainbow')
plt.xlabel('Time [min]')
plt.ylabel('Position [km]')
plt.xlim(min(t), max(t))
plt.ylim(min(x), max(x))
plt.colorbar()
plt.tight_layout()

def load_csv(file):
    data = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data = np.array(data).astype(np.float)
    return data

tVar = []
tVarPlot = []
xVar = []
rhoPV = []
vPV = []
for pv in PVList.pvs:
    tVar = tVar + pv.getT((NtStart-1)/60)
    tVarPlot = tVarPlot + pv.t
    xVar = xVar + pv.x
    vPV = vPV + pv.v
    for k in range(len(pv.t)):
        j = int(pv.t[k]/deltaT)
        i = int(pv.x[k]/deltaX)
        rhoPV.append(density[i,j-NtStart])
plt.scatter(tVarPlot, xVar, color='red', s=0.4)

with open(scenario+'/spaciotemporal.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows([[L, (Tmax-Tstart)]])
    writer.writerows(density)
    
with open(scenario+'/pv.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(np.array([xVar, tVar, rhoPV, vPV]).T)
    