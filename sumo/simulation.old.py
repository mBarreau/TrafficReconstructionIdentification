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
from matplotlib.animation import FuncAnimation, PillowWriter  

from scipy import signal
from probe_vehicle import ProbeVehicleArray

scenario = "highway"

traci.start(["sumo", "-c", scenario+"/"+scenario+".sumocfg"])

deltaX = 10 # in meters, more than a vehicle
L = 3000
deltaT = traci.simulation.getDeltaT()
Tmax = 1500 # in seconds
Tstart = 500 # in seconds

Nt = int(np.ceil(Tmax/deltaT))
NtStart = int(Tstart*Nt/Tmax)
Nx = int(np.ceil(L/deltaX))

numberOfVehicles = np.zeros((Nx, Nt-NtStart))
trafficLightPhase = 0
PVList = ProbeVehicleArray()
time_position_vehicles = []
time_position_pv = []

for n in range(Nt):
   if n%100 == 0:
      print("step", n)
    
   if traci.trafficlight.getPhase("gneJ4") != trafficLightPhase:
       trafficLightPhase = traci.trafficlight.getPhase("gneJ4")
       if traci.trafficlight.getPhase("gneJ4") == 0:
           traci.trafficlight.setPhaseDuration("gneJ4", np.random.randint(30, 40))
       if traci.trafficlight.getPhase("gneJ4") == 2:
           if n *deltaT > 2000:
               traci.trafficlight.setPhaseDuration("gneJ4", np.random.randint(70, 150))
           else:
               traci.trafficlight.setPhaseDuration("gneJ4", np.random.randint(10, 150))
   if n >= NtStart:
       position_vehicles = []
       position_pv = []
       for vehID in traci.vehicle.getIDList():
           if traci.vehicle.getRouteID(vehID) == 'route_0' or traci.vehicle.getRouteID(vehID) == 'route_1':
               vehPos = traci.vehicle.getPosition(vehID)[0]
               i = int(np.floor(vehPos/deltaX))
               if 0 <= i < Nx:
                   numberOfVehicles[i,n-NtStart] += 1
                   if traci.vehicle.getTypeID(vehID) == 'PV':
                       PVList.update(vehID, n*deltaT, vehPos)
                       position_pv.append(vehPos)
                   else:
                       position_vehicles.append(vehPos)
                       
       time_position_vehicles.append(position_vehicles)
       time_position_pv.append(position_pv)          
               
   traci.simulationStep()
   
traci.close()

t = np.linspace(Tstart, Tmax, Nt-NtStart)
x = np.linspace(0, L, Nx)
fig = plt.figure(figsize=(7.5, 5))
X, Y = np.meshgrid(x, t)
plt.pcolor(X, Y, numberOfVehicles.T, shading='auto', cmap='rainbow')
plt.ylabel('Time [s]')
plt.xlabel('Position [m]')
plt.ylim(min(t), max(t))
plt.xlim(min(x), max(x))
plt.colorbar()
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(7.5, 5))
X, Y = np.meshgrid(t, x)
plt.pcolor(X, Y, numberOfVehicles, shading='auto', cmap='rainbow')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.xlim(min(t), max(t))
plt.ylim(min(x), max(x))
plt.colorbar()
plt.tight_layout()
plt.show()

sigma = 20
tau = 4
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
densityMax = 3
density = density/densityMax

def load_csv(file):
    data = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data = np.array(data).astype(np.float)
    return data

# u_pred = load_csv(scenario+'/spaciotemporalReconstruction.csv') 
# tPlot = 0
# fig, axs = plt.subplots(2, figsize=(7.5, 5))
# ln1 = axs[0].plot([], [], color='b', linestyle='None', marker='o', 
#                   markersize=2, label='Vehicle')[0]
# ln2 = axs[0].plot([], [], color='r', linestyle='None', marker='o', 
#                   markersize=3, label='Probe Vehicle')[0]
# ln3 = axs[1].plot([], [], label='Real density')[0]
# ln4 = axs[1].plot([], [], label='Reconstructed density')[0]
# axs[0].set_xlim([min(x), max(x)])
# axs[0].legend()
# axs[0].grid(True)
# axs[1].set_xlim([min(x), max(x)])
# axs[1].set_ylim([0, 1])
# axs[1].legend(loc='upper left')
# axs[1].grid(True)
# axs[1].set_xlabel('Position [m]')
# axs[1].set_ylabel('Normalized Density')
# plt.tight_layout()
# 
# def update(i):  
#     i = int(6*i)
#     ln1.set_data(time_position_vehicles[i], [0]*len(time_position_vehicles[i]))  
#     ln2.set_data(time_position_pv[i], [0]*len(time_position_pv[i]))  
#     ln3.set_data(x[::2], density[::2,i])  
#     ln4.set_data(x[::2], u_pred[::2,i])  
# 
#     return ln1, ln2, ln3, ln4
# 
# ani = FuncAnimation(fig, update, interval=1, frames=int((Nt-NtStart)/6))  
# plt.show()
# writer = PillowWriter(fps=10)  
# ani.save("road.gif", writer=writer)  

fig = plt.figure(figsize=(7.5, 5))
plt.pcolor(X, Y, density, shading='auto', cmap='rainbow')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.xlim(min(t), max(t))
plt.ylim(min(x), max(x))
plt.colorbar()
plt.tight_layout()
plt.show()

tVar = []
tVarPlot = []
xVar = []
rhoPV = []
for pv in PVList.pvs:
    tVar = tVar + pv.getT(NtStart-1)
    tVarPlot = tVarPlot + pv.t
    xVar = xVar + pv.x
    for k in range(len(pv.t)):
        j = int(pv.t[k]/deltaT)
        i = int(pv.x[k]/deltaX)
        rhoPV.append(density[i,j-NtStart])
plt.scatter(tVarPlot, xVar, color='red', s=0.6)

with open(scenario+'/spaciotemporal.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(density)
    
with open(scenario+'/pv.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(np.array([xVar, tVar, rhoPV]).T)
    