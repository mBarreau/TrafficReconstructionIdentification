#!/usr/bin/env python3

from argparse import ArgumentParser
import csv
import numpy as np
from matplotlib import pyplot as plt
import seaborn
import pandas


def parse_args():
    ap = ArgumentParser()

    ap.add_argument("godunov_data", type=str)
    ap.add_argument("godunov_np_data", type=str)
    ap.add_argument("sumo_data", type=str)
    ap.add_argument("sumo_np_data", type=str)

    return ap.parse_args()


def get_data(path):
    data = {'time': [], 'rmse': []}

    with open(path, 'r') as datafile:
        reader = csv.reader(datafile)
        next(reader, None)  # skip header

        for row in reader:
            data['time'].append(float(row[0]))
            data['rmse'].append(float(row[1]))

    return data


def main():
    args = parse_args()

    godunov = get_data(args.godunov_data)
    godunov_np = get_data(args.godunov_np_data)
    sumo = get_data(args.sumo_data)
    sumo_np = get_data(args.sumo_np_data)

    colnames = ['Godunov', 'Godunov (NP)', 'SUMO', 'SUMO (NP)']

    error = pandas.DataFrame(
            np.array([
                godunov['rmse'], godunov_np['rmse'], sumo['rmse'],
                sumo_np['rmse']
                ]).T,
            columns=colnames)

    time = pandas.DataFrame(
            np.array([
                godunov['time'], godunov_np['time'], sumo['time'],
                sumo_np['time']
                ]).T,
            columns=colnames)

    for var, label, axlabel, title in zip(
            (time, error), ('time', 'rmse'), ('Time [sec]', 'RMSE'),
            ('Computation time', 'Density error')):
        fig, ax = plt.subplots()
        seaborn.boxplot(data=var, ax=ax)
        ax.grid()
        ax.set_ylabel(axlabel)
        ax.set_title(title)
        fig.savefig(label + '.eps', dpi=200)


if __name__ == "__main__":
    main()
