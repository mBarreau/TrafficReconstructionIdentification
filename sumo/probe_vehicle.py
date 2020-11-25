# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:52:32 2020

@author: barreau
"""

class ProbeVehicle():
    
    def __init__(self, vehID, t=0, x=0, v=-1):
        self.vehID = vehID
        self.x = [x]
        self.t = [t]
        self.v = [v]
        
    def addPos(self, t, x, v):
        self.x.append(x)
        self.t.append(t)
        self.v.append(v)
        
    def getT(self, tStart=0):
        newT = []
        for t in self.t:
            newT.append(t-tStart)
        return newT
        
class ProbeVehicleArray():
    
    def __init__(self):
        self.pvs = []
        
    def update(self, vehID, t, x, v):
        for pv in self.pvs:
            if pv.vehID == vehID:
                pv.addPos(t, x, v)
                return
        self.pvs.append(ProbeVehicle(vehID, t, x, v))