import csv
import numpy as np
import matplotlib.pyplot as plt

class Sumo():

    def __init__(self, scenario):
        u = self.load_csv('sumo/'+scenario+'/spaciotemporal.csv')  # density(time, position) (1000, 300)
        self.L, self.Tmax = float(u[0][0]), float(u[0][1])
        self.u = np.array(u[1:]).astype(np.float)
        
        data_train = self.load_csv('sumo/'+scenario+'/pv.csv')  # (measurements, features) features=(position, time, density, speed)
        data_train = np.array(data_train).astype(np.float)
        self.probe_t, self.probe_x, self.probe_u, self.probe_v = self.process_probe_data(data_train)  # probe_density(vehicle, position, time, density)

        self.Nx, self.Nt = self.u.shape

    def load_csv(self, file):
        data = []
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        # data = np.array(data).astype(np.float)
        return data

    def process_probe_data(self, data):
        probe_x = []
        probe_t = []
        probe_u = []
        probe_v = []
        
        pv_x = []
        pv_t = []
        pv_u = []
        pv_v = []
        
        t_prev = 0
        for meas in data:
            t = meas[1]
            if t - t_prev < 0:
                probe_x.append(np.array(pv_x).reshape((-1,1)))
                pv_x = []
                probe_t.append(np.array(pv_t).reshape((-1,1)))
                pv_t = []
                probe_u.append(np.array(pv_u).reshape((-1,1)))
                pv_u = []
                probe_v.append(np.array(pv_v).reshape((-1,1)))
                pv_v = []
            pv_x.append(meas[0])
            pv_t.append(meas[1])
            pv_u.append(meas[2])
            pv_v.append(meas[3])
            t_prev = t
        probe_x.append(np.array(pv_x).reshape((-1,1)))
        probe_t.append(np.array(pv_t).reshape((-1,1)))
        probe_u.append(np.array(pv_u).reshape((-1,1)))
        probe_v.append(np.array(pv_v).reshape((-1,1)))
        
        return probe_t, probe_x, probe_u, probe_v

    def getMeasurements(self):
        return self.probe_t, self.probe_t, self.probe_u, self.probe_v

    def getDensity(self):
        return self.u
    
    def getAxisPlot(self):
        t = np.linspace(0, self.Tmax, self.Nt)
        x = np.linspace(0, self.L, self.Nx)
        return (x, t)

    def plotDensity(self):
        plt.figure('density_true', figsize=(7.5, 5))
        plt.imshow(np.flipud(self.u), extent=[0, self.Tmax, 0, self.L], cmap='rainbow', vmin=0.0, vmax=1, aspect='auto')
        plt.colorbar()
        for (t,x) in zip(self.probe_t, self.probe_x):
            plt.scatter(t, x, s=1, c='k')
        # plt.title('Density')
        plt.xlabel('Time [min]')
        plt.ylabel('Position [km]')
        plt.ylim(0, self.L)
        plt.xlim(0, self.Tmax)
        plt.tight_layout()

    def plotProbeVehicles(self):
        for (t,x) in zip(self.probe_t, self.probe_x):
            plt.scatter(t, x, s=1, c='k')
