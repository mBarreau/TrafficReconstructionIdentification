#!/usr/bin/env python3

from argparse import ArgumentParser
import csv
import numpy as np
from matplotlib import pyplot as plt
import seaborn
import pandas
import scipy.stats as stats


def parse_args():
    ap = ArgumentParser()

    ap.add_argument("greenshields_data", type=str)
    ap.add_argument("greenshields_np_data", type=str)
    ap.add_argument("newelldaganzo_data", type=str)
    ap.add_argument("newelldaganzo_np_data", type=str)
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

    greenshields = get_data(args.greenshields_data)
    greenshields_np = get_data(args.greenshields_np_data)
    newelldaganzo = get_data(args.newelldaganzo_data)
    newelldaganzo_np = get_data(args.newelldaganzo_np_data)
    sumo = get_data(args.sumo_data)
    sumo_np = get_data(args.sumo_np_data)

#     colnames = [
#         'Greenshields', 'Greenshields (NP)', 'ND', 'ND (NP)', 'SUMO',
#         'SUMO (NP)']

    colnames = ['Error', 'Time', 'Case', 'Pretraining']

    data = [ (error, time, 'Greenshields', 'Yes') for error, time in
                zip(greenshields['rmse'], greenshields['time'])
            ] + [ (error, time, 'Greenshields', 'No') for error, time in
                zip(greenshields_np['rmse'], greenshields_np['time'])
            ] + [ (error, time, 'Newell-Daganzo', 'Yes') for error, time in
                zip(newelldaganzo['rmse'], newelldaganzo['time'])
            ] + [ (error, time, 'Newell-Daganzo', 'No') for error, time in
                zip(newelldaganzo_np['rmse'], newelldaganzo_np['time'])
            ] + [ (error, time, 'SUMO', 'Yes') for error, time in
                zip(sumo['rmse'], sumo['time'])
            ] + [ (error, time, 'SUMO', 'No') for error, time in
                zip(sumo_np['rmse'], sumo_np['time']) ]

    alldata = pandas.DataFrame(data, columns=colnames)

    # Remove outliers in simulation time
    score = np.abs(stats.zscore(alldata['Time']))
    alldata = alldata[(score < 4)]

    fig, ax = plt.subplots(2, 1, sharex=True)
    seaborn.boxplot(data=alldata, x='Case', y='Time', hue='Pretraining',
            ax=ax[0])
    seaborn.boxplot(data=alldata, x='Case', y='Error', hue='Pretraining',
            ax=ax[1])

    ax[0].set_ylabel('Time [sec]')
    ax[1].set_ylabel('Generalization Error')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[1].get_legend().remove()
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    plt.show()


if __name__ == "__main__":
    main()
