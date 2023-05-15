#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:51:47 2018

@author: Greg Fisher

This is Greg's library
"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import smtplib
import os
import math
import copy
import random
import plotly.graph_objs as go
import plotly
from plotly.subplots import make_subplots
import scipy
import gc
import time

import plotly.express as px

from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate     # COMMASPACE  
from email import encoders

from plotly.subplots import make_subplots


def create_supply_demand_charts(params, num_res_founts, supply_demand_array, fountain_population, agent_population, print_dets, print_fine_dets, run_folder, day, dbs, rounds, trade_moves, SD_charts_freq, daily_succ_trans, pop):
    """This function organises the printing of the supply and demand charts for all of the resource combinations."""

    # print('\n starting create_supply_demand_charts')

    if print_fine_dets == 1:
        print('\n\n supply_demand_array[0]:\n\n', supply_demand_array[0])
        print('\n\n supply_demand_array[1]:\n\n', supply_demand_array[1])

    # Now for Supply & Demand charts
    for res_1 in np.arange(2):

        if res_1 == 0:
            res_2 = 1

        else:
            res_2 = 0

        # Now we're ready to send the data to print_chart
        labels_array = ['Supply', 'Demand']
        y_axis_label = 'Price'
        line_width = 2
        colors = ['blue', 'red', 'black', 'green', 'aqua', 'teal', 'navy', 'fuchsia', 'purple']

        weighted_mean_traded_price = dbs.mean_price_history[res_1][res_2][day]
        net_supply = dbs.net_net_transs_db[day][res_1]

        title = 'Day %s - Res %d: Supply and Demand' % (day, res_1)
        filename = 'Day %s - Supply of & Demand for Res %d' % (day, res_1)
        labels_array = ['Supply', 'Demand']

        if print_fine_dets:

            if res_1 == 0:
                print('\n ---> printing chart: Day %d - Supply & Demand curves for Res A' % day)
            else:
                print('\n ---> printing chart: Day %d - Supply & Demand curves for Res B' % day)

        print_SD_chart(supply_demand_array[res_1], title, run_folder, filename, labels=labels_array, traded_data=[weighted_mean_traded_price, net_supply], show_old_curves=0, print_fine_dets=0)

        if print_fine_dets == 1:
            print('\n dbs.optimal_bskt_turnover[day] =', dbs.optimal_bskt_turnover[day])
            print(' dbs.net_net_transs_db[day] =', dbs.net_net_transs_db[day])

            print(' dbs.net_turnover_prop[round] =', dbs.net_turnover_prop[day])

    # if we are fixing prices, we want to generate charts of S & D curves (res A only) to show how they change.  Note we only do this when len(dbs.saved_SD_curves) == 2:
    if len(dbs.saved_SD_curves) == 2:

        prev_round = day - 10
        current_round = day

        title = 'Day %d - Res 0: Supply and Demand changes' % day
        filename = 'Day %d - Supply of & Demand changes for Res 0' % day
        labels_array = ['Supply Day %d' % prev_round, 'Demand Day %d' % prev_round, 'Supply Day %d' % current_round, 'Demand Day %d' % current_round]

        # print('\n dbs.saved_SD_curves =\n', dbs.saved_SD_curves)

        SD_history_present_data = [dbs.saved_SD_curves[0][1]]
        SD_history_present_data.append(dbs.saved_SD_curves[1][1])

        print_SD_chart(SD_history_present_data, title, run_folder, filename, labels=labels_array, traded_data=[], show_old_curves=1, print_fine_dets=0)

    # Here we print the daily scatter diagrams of agents' MRSs - with and without a red 'journey' agent
    # insert the y-axis data
    dbs.MRS_moves_array_2 = np.insert(dbs.MRS_moves_array_2, 0, np.arange(params.trade_moves + 1, dtype=int), 0)

    title = 'Day %s - MRSs During Trading Moves: Res 0 vs Res 1' % (day)
    filename = 'Day %d - moves_MRS_scatter_%d_vs_%d' % (day, res_1, res_2)

    print_chart_MRSs(dbs.MRS_moves_array_2, title, run_folder, filename, data_type='x', show_journey=0)

    # now with the 'journey agent'
    title = 'MRS Journey'
    filename = 'Day %d - moves_MRS_journey' % day

    print_chart_MRSs(dbs.MRS_moves_array_2, title, run_folder, filename, data_type='x', show_journey=1)

    if print_fine_dets:

       pause()


def plot_policy_lines_and_errors(dpi='high'):
    x_axis = np.arange(1, 41) * 1000

    # 0 is default parameters; 1 is res = 200; 2 is mem = 20; 3 is more commes; and 4 is all

    lines_array = np.array([[14.58, 14.95, 15.28, 14.36, 18.40, 13.60, 14.56, 14.35, 14.81, 17.16, 16.41, 14.37, 14.14, 14.42, 14.61, 16.56, 16.80, 17.10, 16.96, 13.52],
                            [51.69, 65.06, 77.29, 89.53, 90.67, 95.20, 97.29, 97.08, 97.44, 97.59, 97.07, 97.31, 96.85, 97.50, 97.99, 97.87, 98.07, 98.09, 97.71, 98.07]])
    errors_array = np.array([[7.62, 6.39, 6.37, 6.30, 7.58, 6.99, 6.73, 6.15, 6.45, 6.76, 6.34, 6.33, 6.19, 7.20, 6.36, 6.28, 6.71, 6.45, 6.78, 6.46],
                             [9.14, 7.41, 6.99, 2.74, 2.28, 1.21, 0.35, 0.64, 0.19, 0.24, 0.97, 0.71, 0.51, 0.53, 0.39, 0.13, 0.10, 0.12, 0.35, 0.10]])

    colours_array = ['black', 'red', 'blue', 'green', 'orange']

    labels = ['(1) Default', '(2) With Policy', '(3) Longer Memory', '(4) More Communications', '(5) All Changes']

    num_lines = len(lines_array)

    print('\n x_axis =\n', x_axis)
    print('\n lines_array =\n', lines_array)
    print('\n errors_array =\n', errors_array)

    y_min = 0.0
    y_max = 100.0

    x_min = 0.0
    x_max = 21000.0

    data_folder = '%s/single_runs/' % (directory)

    filename = 'ad_hoc_chart_with_policy'

    plt.figure()
    ax = plt.subplot(111)
    plt.axis("auto")
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.xlabel("Round")
    plt.ylabel("Market Coverage Ratio")

    for line_num in range(num_lines):
        y_values = lines_array[line_num]
        y_errors = errors_array[line_num]

        ax.errorbar(x_axis, y_values, y_errors, marker='^', color=colours_array[line_num], label=labels[line_num])  # linestyle='None',

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11),
              ncol=2, fontsize='small')

    #    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #          ncol=3, fancybox=True, shadow=True, fontsize='small')

    # Put a legend to the right of the current axis
    #    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')

    # if we want to show the chart:
    # plt.show()

    # if we want to save the chart:
    if dpi == 'high':
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (data_folder, filename, dt.datetime.now().year,
                       dt.datetime.now().month, dt.datetime.now().day,
                       dt.datetime.now().hour, dt.datetime.now().minute,
                       dt.datetime.now().second,
                       dt.datetime.now().microsecond / 100000), dpi=500)

    else:
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (data_folder, filename, dt.datetime.now().year,
                       dt.datetime.now().month, dt.datetime.now().day,
                       dt.datetime.now().hour, dt.datetime.now().minute,
                       dt.datetime.now().second,
                       dt.datetime.now().microsecond / 100000))

    # if we want to close the chart:
    plt.close()


def plot_non_policy_lines_and_errors(dpi='high'):
    num_dat_points = 40

    x_axis = np.arange(1, num_dat_points + 1) * 1000

    # 0 is default parameters; 1 is res = 200; 2 is mem = 20; 3 is more comms; and 4 is all
    def_mean = np.mean([14.58, 14.95, 15.28, 14.36, 18.40, 13.60, 14.56, 14.35, 14.81, 17.16, 16.41, 14.37, 14.14, 14.42, 14.61, 16.56, 16.80, 17.10, 16.96, 13.52])
    def_std = np.mean([7.62, 6.39, 6.37, 6.30, 7.58, 6.99, 6.73, 6.15, 6.45, 6.76, 6.34, 6.33, 6.19, 7.20, 6.36, 6.28, 6.71, 6.45, 6.78, 6.46])

    lines_array = np.array([[14.58, 14.95, 15.28, 14.36, 18.40, 13.60, 14.56, 14.35, 14.81, 17.16, 16.41, 14.37, 14.14, 14.42, 14.61, 16.56, 16.80, 17.10, 16.96, 13.52, \
                             def_mean, def_mean, def_mean, def_mean, def_mean, def_mean, def_mean, def_mean, def_mean, def_mean, \
                             def_mean, def_mean, def_mean, def_mean, def_mean, def_mean, def_mean, def_mean, def_mean, def_mean],

                            [19.82, 22.49, 30.17, 35.16, 42.48, 46.99, 51.00, 58.16, 62.94, 68.56, 73.29, 76.31, 79.21, 81.94, 84.66, 86.22, 86.60, 88.63, 88.72, 90.72, \
                             91.32, 93.02, 92.79, 93.19, 94.62, 94.58, 94.72, 94.58, 94.58, 94.62, 94.86, 94.86, 94.86, 94.86, 94.86, 94.86, 94.86, 94.86, 94.86, 94.86],

                            [49.28, 43.57, 46.29, 49.29, 48.29, 50.83, 51.70, 55.42, 56.31, 59.00, 61.45, 63.27, 65.10, 70.14, 71.42, 73.44, 75.80, 75.70, 78.26, 79.27, \
                             79.71, 78.72, 81.85, 82.76, 82.96, 83.70, 84.33, 85.22, 84.68, 84.18, 83.39, 84.53, 84.55, 84.89, 86.08, 85.18, 85.42, 87.78, 88.76, 87.79],

                            [51.36, 63.06, 68.41, 73.42, 75.05, 76.91, 77.59, 79.38, 80.83, 82.34, 83.19, 85.35, 85.97, 85.91, 86.03, 86.17, 86.03, 86.56, 86.52, 86.03, \
                             87.10, 87.55, 87.10, 87.31, 87.52, 88.38, 88.89, 88.91, 89.01, 88.93, 88.99, 89.23, 89.37, 89.93, 89.78, 89.71, 90.51, 90.23, 90.49, 90.75],

                            [81.35, 87.66, 89.10, 90.00, 90.56, 90.83, 90.83, 90.83, 91.11, 91.11, 91.11, 91.11, 91.11, 91.11, 91.11, 91.11, 91.11, 91.11, 91.11, 91.11, \
                             91.11, 91.11, 91.11, 91.11, 91.11, 91.11, 91.11, 91.25, 91.25, 91.25, 91.25, 91.25, 91.25, 91.25, 91.25, 91.25, 91.25, 91.25, 91.25, 91.25]])

    errors_array = np.array([[7.62, 6.39, 6.37, 6.30, 7.58, 6.99, 6.73, 6.15, 6.45, 6.76, 6.34, 6.33, 6.19, 7.20, 6.36, 6.28, 6.71, 6.45, 6.78, 6.46, \
                              def_std, def_std, def_std, def_std, def_std, def_std, def_std, def_std, def_std, def_std, \
                              def_std, def_std, def_std, def_std, def_std, def_std, def_std, def_std, def_std, def_std],

                             [6.26, 6.55, 5.35, 4.13, 3.71, 3.95, 3.07, 3.13, 2.95, 2.03, 1.36, 2.00, 1.47, 1.08, 1.00, 0.75, 0.44, 0.40, 0.47, 0.42, \
                              0.12, 0.29, 0.06, 0.00, 0.09, 0.00, 0.00, 0.00, 0.00, 0.09, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],

                             [6.00, 6.08, 5.58, 5.04, 4.63, 4.09, 5.27, 4.29, 4.54, 3.58, 3.71, 3.66, 2.91, 2.96, 2.96, 2.22, 2.34, 2.72, 1.92, 1.61, \
                              1.55, 1.42, 1.05, 0.73, 0.94, 0.96, 0.78, 1.09, 1.06, 1.02, 1.19, 1.08, 0.77, 0.77, 1.04, 0.99, 0.58, 0.47, 0.65, 0.68],

                             [3.44, 2.61, 1.81, 1.40, 1.55, 0.88, 1.30, 0.72, 0.73, 0.76, 0.39, 0.83, 0.53, 0.27, 0.40, 0.55, 0.50, 0.64, 0.65, 0.46, \
                              0.30, 0.46, 0.30, 0.51, 0.44, 0.39, 0.42, 0.49, 0.34, 0.14, 0.31, 0.21, 0.38, 0.57, 0.42, 0.23, 0.28, 0.52, 0.32, 0.11],

                             [1.04, 0.44, 0.18, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, \
                              0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])

    colours_array = ['black', 'red', 'blue', 'green', 'orange']

    labels = ['(1) Default', '(2) More Resources', '(3) Longer Memory', '(4) More Communications', '(5) All Changes']

    num_lines = len(lines_array)

    print('\n x_axis =\n', x_axis)
    print('\n lines_array =\n', lines_array)
    print('\n errors_array =\n', errors_array)

    y_min = 0.0
    y_max = 100.0

    x_min = 0.0
    x_max = float((num_dat_points + 1) * 1000)

    data_folder = '%s/single_runs/' % (directory)

    filename = 'ad_hoc_chart'

    plt.figure()
    ax = plt.subplot(111)
    plt.axis("auto")
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.xlabel("Round")
    plt.ylabel("Market Coverage Ratio")

    for line_num in range(num_lines):
        y_values = lines_array[line_num]
        y_errors = errors_array[line_num]

        ax.errorbar(x_axis, y_values, y_errors, marker='^', color=colours_array[line_num], label='%s' % (labels[line_num]))  # linestyle='None',

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.11),
              ncol=3, fontsize='small')

    #    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #          ncol=3, fancybox=True, shadow=True, fontsize='small')

    # Put a legend to the right of the current axis
    #    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')

    # if we want to show the chart:
    # plt.show()

    # if we want to save the chart:
    if dpi == 'high':
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (data_folder, filename, dt.datetime.now().year,
                       dt.datetime.now().month, dt.datetime.now().day,
                       dt.datetime.now().hour, dt.datetime.now().minute,
                       dt.datetime.now().second,
                       dt.datetime.now().microsecond / 100000), dpi=500)

    else:
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (data_folder, filename, dt.datetime.now().year,
                       dt.datetime.now().month, dt.datetime.now().day,
                       dt.datetime.now().hour, dt.datetime.now().minute,
                       dt.datetime.now().second,
                       dt.datetime.now().microsecond / 100000))

    # if we want to close the chart:
    plt.close()


def log(t):

    """This is for checking the logistic equation: input a value of t and it returns the number of rounds it would take for p to reach 1 to 2dp."""

    p_ceil = 1.0
    p_floor = 0.2
    p_ch = 0
    p = 0.5

    i = 0

    while i < 400 and '%1.2f' % (p) != '0.75':

        p += p_ch
#        print('round', i, ' p = %1.2f' % (p))
        p_ch = (t * 2 * (p - p_floor) * (p_ceil - p)) / float(p_ceil - p_floor)

        i += 1

#    print(' round reached 1 to 2dp =', i - 1)

    return i - 1


def print_scatter_1d_MRS_moves(database, rounds, agent_population, title, y_axis_label, line_width, colors, data_folder, filename, labels, dbs, trade_moves, dpi):

    print('---> printing chart: %s ' % title)

#    print('\n database\n\n', database)

    x_axis = np.zeros(shape=(len(dbs.agent_list)))

    y_range = np.max(database) - np.min(database)

    max_y = np.max(database) + (y_range * 0.05)
    min_y = np.min(database) - (y_range * 0.05)

    min_x = -0.5
    max_x = trade_moves + 1 - 0.5

    # now build the chart:
    plt.figure()
    plt.title("%s" % title)
    plt.axis("auto")
    plt.xlim(xmin=min_x, xmax=max_x)
    plt.ylim(ymin=min_y, ymax=max_y)
    plt.ylabel("Marginal Rates of Substitution")
    plt.xlabel("Moves")

    x_axis_array = list(range(trade_moves + 1))
    x_axis_array[0] = 'Start'

    for move in np.arange(trade_moves + 1):

        plt.plot(x_axis + move, database[move], 'x', color='black', label='%s' % (x_axis_array[move]))

#    plt.legend(loc='upper center', fontsize='small')

    # if we want to show the chart:
    # plt.show()

    # if we want to save the chart:
    if dpi == 'high':
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (data_folder, filename, dt.datetime.now().year,
                    dt.datetime.now().month, dt.datetime.now().day,
                    dt.datetime.now().hour, dt.datetime.now().minute,
                    dt.datetime.now().second,
                    dt.datetime.now().microsecond / 100000), dpi=500)

    else:
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (data_folder, filename, dt.datetime.now().year,
                    dt.datetime.now().month, dt.datetime.now().day,
                    dt.datetime.now().hour, dt.datetime.now().minute,
                    dt.datetime.now().second,
                    dt.datetime.now().microsecond / 100000))

    # if we want to close the chart:
    plt.clf()
    plt.close()
    gc.collect()


def print_scatter_1d(database, database_2, rounds, agent_population, title, y_axis_label, line_width, colors, data_folder, filename, labels, dbs, dpi):

    print('---> printing chart: %s ' % (title))

    val_1 = -1.0
    val_2 = 1.0

    x_axis = np.zeros(shape=(len(agent_population.pop)))

    max_y = np.max(database) * 1.05
    min_y = np.min(database) * 0.95

    min_x = -2.0
    max_x = 2.0

    # now build the chart:
    plt.figure()
    plt.title("%s" % (title))
    plt.axis("auto")
    plt.xlim(xmin=min_x, xmax=max_x)
    plt.ylim(ymin=min_y, ymax=max_y)
#    plt.xlabel("Quantity Supplied or Demanded")
#    plt.ylabel("%s" % (y_axis_label))

    plt.plot(x_axis + val_1, database, 'x', color='red', label=labels[0])

    plt.plot(x_axis + val_2, database_2, 'x', color='blue', label=labels[1])

    plt.legend(loc='upper center', fontsize='small')

    # if we want to show the chart:
    # plt.show()

    # if we want to save the chart:
    if dpi == 'high':
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (data_folder, filename, dt.datetime.now().year,
                    dt.datetime.now().month, dt.datetime.now().day,
                    dt.datetime.now().hour, dt.datetime.now().minute,
                    dt.datetime.now().second,
                    dt.datetime.now().microsecond / 100000), dpi=500)

    else:
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (data_folder, filename, dt.datetime.now().year,
                    dt.datetime.now().month, dt.datetime.now().day,
                    dt.datetime.now().hour, dt.datetime.now().minute,
                    dt.datetime.now().second,
                    dt.datetime.now().microsecond / 100000))

    # if we want to close the chart:
    plt.clf()
    plt.close()
    gc.collect()


def print_histogram(reserves_array, title, data_folder, filename, color):

    print('---> printing chart: %s ' % (title))

    max_num = int(math.ceil(np.max(reserves_array)))
    min_num = int(np.min(reserves_array))

    bins = list(range(min_num - 1, max_num + 2))

    plt.figure()
    plt.title("%s" % (title))
    plt.hist(reserves_array, bins, facecolor=color)

    plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                % (data_folder, filename, dt.datetime.now().year,
                dt.datetime.now().month, dt.datetime.now().day,
                dt.datetime.now().hour, dt.datetime.now().minute,
                dt.datetime.now().second,
                dt.datetime.now().microsecond / 100000))

    # if we want to close the chart:
    plt.clf()
    plt.close()
    gc.collect()


def find_best_box(surf_1, surf_2, prices_array_0_2, prices_array_1_2, print_fine_dets):
    """This function takes two surfaces and finds the grid squares in which the solution might be present (where both surfaces = 0).  It then creates a 'box' of all potential solution squares."""

    num_price_points = len(surf_1)

    price_squares_record = []
    price_grid_sqs = []

    for i in np.arange(num_price_points - 1):

        for j in np.arange(num_price_points - 1):

            # the first condition means the first surface (0_2) goes through the 0 contour in either of the price dimensions and in either direction
            if ((surf_1[i][j] < 0 and surf_1[i + 1][j] > 0) or (surf_1[i][j] > 0 and surf_1[i + 1][j] < 0)) or ((surf_1[i][j] < 0 and surf_1[i][j + 1] > 0) or (surf_1[i][j] > 0 and surf_1[i][j + 1] < 0)) or (
                    (surf_1[i][j] < 0 and surf_1[i + 1][j + 1] > 0) or (surf_1[i][j] > 0 and surf_1[i + 1][j + 1] < 0)):

                # the second condition means the first surface (0_2) goes through the 0 contour in either of the price dimensions and in either direction
                if ((surf_2[i][j] < 0 and surf_2[i + 1][j] > 0) or (surf_2[i][j] > 0 and surf_2[i + 1][j] < 0)) or ((surf_2[i][j] < 0 and surf_2[i][j + 1] > 0) or (surf_2[i][j] > 0 and surf_2[i][j + 1] < 0)) or (
                        (surf_2[i][j] < 0 and surf_2[i + 1][j + 1] > 0) or (surf_2[i][j] > 0 and surf_2[i + 1][j + 1] < 0)):

                    # We want to reject all instances where the surfaces meet the criteria above but where one set of price points is all above or all below the other i.e. where the surfaces are parallel
                    if ((surf_1[i][j] > surf_2[i][j]) and (surf_1[i + 1][j] > surf_2[i + 1][j]) and (surf_1[i][j + 1] > surf_2[i][j + 1]) and (surf_1[i + 1][j + 1] > surf_2[i + 1][j + 1])) == False and (
                            (surf_1[i][j] < surf_2[i][j]) and (surf_1[i + 1][j] < surf_2[i + 1][j]) and (surf_1[i][j + 1] < surf_2[i][j + 1]) and (surf_1[i + 1][j + 1] < surf_2[i + 1][j + 1])) == False:
                        p_02_low = prices_array_0_2[i]
                        p_02_high = prices_array_0_2[i + 1]

                        p_12_low = prices_array_1_2[j]
                        p_12_high = prices_array_1_2[j + 1]

                        price_squares_record.append([[p_02_low, p_02_high], [p_12_low, p_12_high]])
                        price_grid_sqs.append([[i, i + 1], [j, j + 1]])

    if print_fine_dets == 1:
        print('\n from find_best_box -> price_squares_record =', price_squares_record)

    return price_squares_record, price_grid_sqs


def plot_3d_SD_planes_plotly(town_grid, search_points_0_2, search_points_1_2, surf_1, surf_2, prices_array_0_2, prices_array_1_2, data_folder, res, day, title):

    trace_1 = go.Surface(x=prices_array_0_2, y=prices_array_1_2, z=np.transpose(surf_1), contours=dict(z=dict(show=True, color='black', project=dict(y=True), highlightwidth=2, width=1)))
    trace_2 = go.Surface(x=prices_array_0_2, y=prices_array_1_2, z=np.transpose(surf_2), contours=dict(z=dict(show=True, color='black', project=dict(y=True), highlightwidth=2, width=1)))

    data = [trace_1, trace_2]

    if len(search_points_0_2) > 0:
        trace_3 = go.Scatter3d(x=search_points_0_2[0], y=search_points_0_2[1], z=search_points_0_2[2], marker=dict(symbol="diamond", color="black", size=4))

        data.append(trace_3)

    if len(search_points_1_2) > 0:
        trace_4 = go.Scatter3d(x=search_points_1_2[0], y=search_points_1_2[1], z=search_points_1_2[2], marker=dict(symbol="diamond", color="red", size=4))

        data.append(trace_4)

    layout = go.Layout(title=title, scene=dict(xaxis=dict(title="Price 0 2", gridcolor='black'), yaxis=dict(title="Price 1 2", gridcolor='black'), zaxis=dict(title="Net Oversupply")))

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig,
             filename='python_docs/SD_day_%d_res_%d_day_%d - %d - %d - %d - %d - %d - %d'
                      % (day, res, dt.datetime.now().year,
                         dt.datetime.now().month, dt.datetime.now().day,
                         dt.datetime.now().hour, dt.datetime.now().minute,
                         dt.datetime.now().second,
                         dt.datetime.now().microsecond / 100000))


#    py.iplot([
#        dict(z=supply_2d, showscale=False, type='surface'),
#        dict(z=demand_2d, showscale=False, opacity=0.9, type='surface')],
#        filename='python_docs/SD_day_%d_res_%d_day_%d - %d - %d - %d - %d - %d - %d'
#                        % (day, res, dt.datetime.now().year,
#                        dt.datetime.now().month, dt.datetime.now().day,
#                        dt.datetime.now().hour, dt.datetime.now().minute,
#                        dt.datetime.now().second,
#                        dt.datetime.now().microsecond / 100000))

#    raw_input("Press Enter to continue...")


def plot_single_plane_plotly(town_grid, search_points, three_d_data, prices_array_0_2, prices_array_1_2, data_folder, day, res_1, res_2, title):

    xGrid, yGrid = np.meshgrid(prices_array_0_2, prices_array_1_2)

    #    print '\n xGrid =\n', xGrid
    #    print '\n yGrid =\n', yGrid

    trace_1 = go.Surface(x=xGrid, y=yGrid, z=np.transpose(three_d_data), contours=dict(z=dict(show=True, color='black', project=dict(y=True), highlightwidth=2, width=1)))  # y=dict(show=True), x=dict(show=True)

    data = [trace_1]

    if len(search_points) > 0:
        trace_2 = go.Scatter3d(x=search_points[0], y=search_points[1], z=search_points[2], marker=dict(symbol="diamond", color="black", size=4))

        data.append(trace_2)

    layout = go.Layout(title=title, scene=dict(xaxis=dict(title="Price 0 2", gridcolor='black'), yaxis=dict(title="Price 1 2", gridcolor='black'), zaxis=dict(title="Net Oversupply")))

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig,
             filename='python_docs/SD_day_%d_res_%d_res_%d_day_%d - %d - %d - %d - %d - %d - %d'
                      % (day, res_1, res_2, dt.datetime.now().year,
                         dt.datetime.now().month, dt.datetime.now().day,
                         dt.datetime.now().hour, dt.datetime.now().minute,
                         dt.datetime.now().second,
                         dt.datetime.now().microsecond / 100000))


def print_3d_SD_planes(num_res_founts, town_grid, three_d_SD_data, aggr_three_d_data, prices_array_0_2, prices_array_1_2, data_folder, day, res, fountain_population):
    print('---> printing surface charts')

    # we have to decide where we want to save the chart and its default name:
    filepath = data_folder

    x_data, y_data = np.meshgrid(prices_array_0_2, prices_array_1_2)

    net_supply_array = []

    for res in np.arange(num_res_founts):

        title = 'Day %s: S & D planes for Resource %s' % (day, res)
        filename = '3d_planes_day_%s_res_%s' % (day, res)

        supply_2d = three_d_SD_data[res][0]
        demand_2d = three_d_SD_data[res][1]

        #        print '\nsupply_2d =\n', supply_2d
        #        print '\ndemand_2d =\n', demand_2d

        net_supply_array.append(supply_2d - demand_2d)

        max_supply = np.max([np.max(supply_2d), np.max(demand_2d)])
        min_supply = np.min([np.min(supply_2d), np.min(demand_2d)])

        # Create a figure for plotting the data as a 3D histogram.

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf_S = ax.plot_surface(x_data, y_data, supply_2d, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(min_supply, max_supply)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        #    ax_D = fig.gca(projection='3d')
        surf_D = ax.plot_surface(x_data, y_data, demand_2d, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        fig.colorbar(surf_S, shrink=0.5, aspect=5)
        fig.colorbar(surf_D, shrink=0.5, aspect=5)

        plt.title(title)

        plt.xlabel("Price of Res 0 v Res 2")
        plt.ylabel("Price of Res 1 v Res 2")

        # plt.show()

        # if we want to save the chart:
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (filepath, filename, dt.datetime.now().year,
                       dt.datetime.now().month, dt.datetime.now().day,
                       dt.datetime.now().hour, dt.datetime.now().minute,
                       dt.datetime.now().second,
                       dt.datetime.now().microsecond / 100000))

        plt.close()

        #        plot_3d_SD_planes_plotly(town_grid, supply_2d, demand_2d, prices_array_0_2, prices_array_1_2, data_folder, res, day)

        # Now plot a chart for differences
        diff_2d = supply_2d - demand_2d

        filename = '3d_planes_day_%s_res_%s_diff' % (day, res)

        max_diff = np.max(diff_2d)
        min_diff = np.min(diff_2d)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf_diff = ax.plot_surface(x_data, y_data, diff_2d, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(min_diff, max_diff)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf_diff, shrink=0.5, aspect=5)

        plt.title('Supply minus Demand - Day %s Res %s' % (day, res))

        plt.xlabel("Price of Res 0 v Res 2")
        plt.ylabel("Price of Res 1 v Res 2")

        # plt.show()

        # if we want to save the chart:
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (filepath, filename, dt.datetime.now().year,
                       dt.datetime.now().month, dt.datetime.now().day,
                       dt.datetime.now().hour, dt.datetime.now().minute,
                       dt.datetime.now().second,
                       dt.datetime.now().microsecond / 100000))

        plt.close()

        # Create and save aggregate net supply chart
        if res == num_res_founts - 1:
            filename = 'Day %s - fitness_landscape' % (day)

            max_aggr = np.max(aggr_three_d_data)
            min_aggr = np.min(aggr_three_d_data)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            surf_aggr = ax.plot_surface(x_data, y_data, aggr_three_d_data, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_zlim(min_aggr, max_aggr)

            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            fig.colorbar(surf_aggr, shrink=0.5, aspect=5)

            plt.title('Total Net Oversupply - Day %s' % (day))

            plt.xlabel("Price of Res 0 v Res 2")
            plt.ylabel("Price of Res 1 v Res 2")

            # plt.show()

            # if we want to save the chart:
            plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                        % (filepath, filename, dt.datetime.now().year,
                           dt.datetime.now().month, dt.datetime.now().day,
                           dt.datetime.now().hour, dt.datetime.now().minute,
                           dt.datetime.now().second,
                           dt.datetime.now().microsecond / 100000))

            plt.close()

    # Create and save aggr resource pari net supply
    for res_1 in np.arange(num_res_founts):

        for res_2 in np.arange(num_res_founts):

            if res_1 != res_2:
                data_2d = net_supply_array[res_1] - net_supply_array[res_2]

                #            print '\nnet_supply_array =\n', net_supply_array
                #            print '\nnet_supply_array[res_1] =\n', net_supply_array [res_1]
                #            print '\nnet_supply_array[res_2] =\n', net_supply_array[res_2]

                filename = 'Day %s - Net Supply res %s v res %s' % (day, res_1, res_2)

                max_aggr = np.max(data_2d)
                min_aggr = np.min(data_2d)

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                surf_NS = ax.plot_surface(x_data, y_data, data_2d, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
                ax.set_zlim(min_aggr, max_aggr)

                ax.zaxis.set_major_locator(LinearLocator(10))
                ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

                fig.colorbar(surf_NS, shrink=0.5, aspect=5)

                plt.title('Day %s - Net Supply res %s v res %s' % (day, res_1, res_2))

                plt.xlabel("Price of Res 0 v Res 2")
                plt.ylabel("Price of Res 1 v Res 2")

                # plt.show()

                # if we want to save the chart:
                plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                            % (filepath, filename, dt.datetime.now().year,
                               dt.datetime.now().month, dt.datetime.now().day,
                               dt.datetime.now().hour, dt.datetime.now().minute,
                               dt.datetime.now().second,
                               dt.datetime.now().microsecond / 100000))

                plt.close()


#                if (res_1 == 0 and res_2 == 2) or (res_1 == 1 and res_2 == 2):

#                    plot_single_plane_plotly(town_grid, search_points, aggr_three_d_data, prices_array_0_2, prices_array_1_2, data_folder, day, res_1, res_2)

#                    print '\nprinting fancy shmancey chart for resources %s and %s' % (res_1, res_2)

#                raw_input("Press Enter to continue...")


def create_heat_map(dimen, database, data_folder, color, title, filename, dpi):

    print('---> printing chart: %s ' % (title))

    # print('\n database :\n', database)

    plotly_data = go.Heatmap(z=database, colorscale=color)     # layout={'xaxis' : {'visible' : True, 'color' : 'black', 'linewidth' : 3}}

    layout = go.Layout(autosize=True, yaxis={'scaleanchor' : "x", 'scaleratio' : 1, 'range' : [0, 50]}, xaxis={'range' : [0, 50]})       # margin=dict(l=50, r=50, b=100, t=100, pad=4)

    fig = go.Figure(data=plotly_data, layout=layout)

    margin_dict = {'l': 50, 'r': 50, 'b': 100, 't': 100, 'pad': 4}
    # margin_dict = {'l': 0, 'r': 0, 'b': 0, 't': 0, 'pad': 0}

    fig.update_layout(margin=margin_dict, width=1200, height=1200, autosize=False)
    fig.update_xaxes(range=[0, 50], zeroline=False, tickson='boundaries')
    fig.update_yaxes(range=[0, 50], zeroline=False, tickson='boundaries')

    file_name_fig = "%s/%s %d - %d - %d - %d - %d - %d - %d.html" % (data_folder, filename, dt.datetime.now().year,
                                                   dt.datetime.now().month, dt.datetime.now().day,
                                                   dt.datetime.now().hour, dt.datetime.now().minute,
                                                   dt.datetime.now().second,
                                                   dt.datetime.now().microsecond / 100000)

    plotly.offline.plot(fig, filename=file_name_fig, auto_open=False)

    # pause()

    # below is the old approach using matplotlib which used up huge data in ram!

#     # we have to decide where we want to save the chart and its default name:
#     filepath = data_folder
#
#     plt.pcolor(database, cmap=color)        # , cmap=plt.cm.Blues         ,edgecolors='k'
#     plt.colorbar()
#     plt.title(title)
# #    plt.invert_xaxis()
# #    plt.invert_yaxis()
#
#     # plt.show()
#     plt.axis([0, dimen, 0, dimen])
#     plt.gca().invert_yaxis()
#
# #    plt.gca().xaxis.tick_top()
#
#     # if we want to save the chart:
#     if dpi == 'low':
#
#         plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
#                     % (filepath, filename, dt.datetime.now().year,
#                     dt.datetime.now().month, dt.datetime.now().day,
#                     dt.datetime.now().hour, dt.datetime.now().minute,
#                     dt.datetime.now().second,
#                     dt.datetime.now().microsecond / 100000))
#
#     elif dpi == 'high':
#
#         plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
#                     % (filepath, filename, dt.datetime.now().year,
#                     dt.datetime.now().month, dt.datetime.now().day,
#                     dt.datetime.now().hour, dt.datetime.now().minute,
#                     dt.datetime.now().second,
#                     dt.datetime.now().microsecond / 100000), dpi=500)
#
#     plt.clf()
#     plt.close()
#     gc.collect()


def create_heat_map_double(dimen, database, database_2, data_folder, color, color_2, title, filename, dpi):

    print('---> printing chart: %s ' % (title))

    # we have to decide where we want to save the chart and its default name:
    filepath = data_folder

    plt.pcolor(database, cmap=color)        # , cmap=plt.cm.Blues         ,edgecolors='k'
    plt.colorbar()
    plt.contour(database_2, cmap=color_2)
    plt.title(title)
#    plt.invert_xaxis()
#    plt.invert_yaxis()

    # plt.show()
    plt.axis([0, dimen, 0, dimen])
    plt.gca().invert_yaxis()

#    plt.gca().xaxis.tick_top()

    # if we want to save the chart:
    if dpi == 'high':
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (filepath, filename, dt.datetime.now().year,
                    dt.datetime.now().month, dt.datetime.now().day,
                    dt.datetime.now().hour, dt.datetime.now().minute,
                    dt.datetime.now().second,
                    dt.datetime.now().microsecond / 100000), dpi=500)


    else:

        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (filepath, filename, dt.datetime.now().year,
                    dt.datetime.now().month, dt.datetime.now().day,
                    dt.datetime.now().hour, dt.datetime.now().minute,
                    dt.datetime.now().second,
                    dt.datetime.now().microsecond / 100000))

    plt.clf()
    plt.close()
    gc.collect()


def create_heat_map_double_plotly(dimen, database, database_2, data_folder, color, color_2, title, filename, day, agent_population):

    print('---> printing plotly chart: %s ' % (title))

    trace_1 = go.Heatmap(x=np.arange(dimen), y=np.arange(dimen), z=database, colorscale=[[0, 'white'], [1, 'green']])

    x_home_coords = []
    y_home_coords = []

    for x_coord in np.arange(dimen):
        for y_coord in np.arange(dimen):

            if database_2[x_coord][y_coord] > 0:
                x_home_coords.append(x_coord)
                y_home_coords.append(y_coord)

    trace_2 = go.Scatter(x=x_home_coords, y=y_home_coords, line=dict(width=0), marker=dict(color='black', symbol='square'), showlegend=False)  # x=np.arange(dimen), y=np.arange(dimen),

    data = [trace_1, trace_2]

    for agent in agent_population.pop:

        x_home_coords = [agent.home[0]]
        y_home_coords = [agent.home[1]]

        for loc in agent.trade_loc_rec:

            x_home_coords.append(loc[0])
            y_home_coords.append(loc[1])

        new_trace = go.Scatter(x=x_home_coords, y=y_home_coords, line=dict(width=0.5), marker=dict(size=2, color='black', symbol='square'), showlegend=False)

        data.append(new_trace)

#    layout = go.Layout(title=title, scene=dict(xaxis=dict(title="x coord"), yaxis=dict(title="y coord")), annotations=dict(arrowwidth=0.5))
#
#    fig = go.Figure(data=data, layout=layout)

#    layout = go.Layout(title=title, scene=dict(xaxis=dict(title="x coord"), yaxis=dict(title="y coord")))
#
#    fig = go.Heatmap(data=data, layout=layout)

    py.plot(data,
            filename='python_docs/mkt_catchments_day_%d_day_%d - %d - %d - %d - %d - %d - %d'
            % (day, dt.datetime.now().year,
            dt.datetime.now().month, dt.datetime.now().day,
            dt.datetime.now().hour, dt.datetime.now().minute,
            dt.datetime.now().second,
            dt.datetime.now().microsecond / 100000))


def print_3d_histogram(dimen, data_2d, data_folder, title, filename):

    print('---> printing chart: %s' % (title))

    # we have to decide where we want to save the chart and its default name:
    filepath = data_folder

    data_array = np.array(data_2d)

    # Create a figure for plotting the data as a 3D histogram.

    fig = plt.figure()
    plt.title(title)

    ax = fig.gca(projection='3d')

    # Create an X-Y mesh of the same dimension as the 2D data. You can
    # think of this as the floor of the plot.

    x_data, y_data = np.meshgrid(np.arange(dimen), np.arange(dimen))

    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays:
    # x_data, y_data, z_data. The following call boils down to picking
    # one entry from each array and plotting a bar to from
    # (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).

    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()

    ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data, color='white')

    plt.show

    # if we want to save the chart:
    plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                % (filepath, filename, dt.datetime.now().year,
                dt.datetime.now().month, dt.datetime.now().day,
                dt.datetime.now().hour, dt.datetime.now().minute,
                dt.datetime.now().second,
                dt.datetime.now().microsecond / 100000))

    plt.clf()
    plt.close()
    gc.collect()


def within_striking_dist(wait_at_target_til_end, town_grid, location, wait_at_tgt_moves, agent_vision, poss_tgt_location, move, has_acted, print_dets):

    # print_dets = 1

    x_loc, y_loc = poss_tgt_location

    # We set wait_at_tgt_moves = 0 when the agent waits at its target until the end, otherwise we use whatever is given as a value [changed this - not sure why here, seems wrong]
    # if wait_at_target_til_end:
    #     wait_at_tgt_moves = 0

    if print_dets == 1:
        print('\n --> within_striking_dist function starts <--')
        print('\n town_grid.trade_moves =', town_grid.trade_moves)
        print(' agent.location =', location)
        print(' poss_tgt_location =', poss_tgt_location)
        print(' town_grid.dimen =', town_grid.dimen)
        print(' x_loc =', x_loc)
        print(' y_loc =', y_loc)
        print(' move =', move)
        print(' has_acted =', has_acted)
        print(' wait_at_tgt_moves =', wait_at_tgt_moves)
        print(' agent_vision =', agent_vision)

    # We work out if the location is within striking distance if x_dist and y_dist are both <= max_movement

    # First find max_movement, which is the maximum distance on the grid the agent can move in either the x or y direction
    # There are two scenarios: the agent has already acted during the move e.g. traded, which means they will not move this
    # round; and the agent has yet to act so it can move this round.
    if has_acted == 1:

        max_movement = (town_grid.trade_moves - 1 - move - wait_at_tgt_moves) * agent_vision

    else:

        try:

            max_movement = (town_grid.trade_moves - move - wait_at_tgt_moves) * agent_vision

        except TypeError:

            print('\n function is within_striking_dist. town_grid.trade_moves ', town_grid.trade_moves, 'move ', move, 'wait_at_tgt_moves ', wait_at_tgt_moves, 'agent_vision', agent_vision)

            pause()

    if print_dets == 1:
        print('\n max_movement =', max_movement)

    #    print('x_loc', x_loc, 'location', location)

    x_dist = math.fabs(x_loc - location[0])

    if x_dist > town_grid.dimen / 2.0:
        x_dist = town_grid.dimen - x_dist

    y_dist = math.fabs(y_loc - location[1])

    if y_dist > town_grid.dimen / 2.0:
        y_dist = town_grid.dimen - y_dist

    if print_dets == 1:
        print('\n x_dist =', x_dist)
        print(' y_dist =', y_dist)

    if x_dist <= max_movement and y_dist <= max_movement:

        if print_dets == 1:
            print('\n => location is within striking distance')
            print('\n --> within_striking_dist function ends <--\n')

            # pause()

        return 1

    else:

        if print_dets == 1:
            print('\n=> location is NOT within striking distance')
            print('\n--> within_striking_dist function ends <--\n')

            # pause()

        return 0


def clustering_coefficient(two_d_array, dimen, print_dets):
    """This function measures the degree of clustering on a 2d grid: perfect clustering = 0 and perfect dispersion is
    dimen / 2.  The main input is a 2d matrix with numbers in the cells.  The function returns a coefficient of
    clustering."""

    # the coefficient works like this: (i) we find a weighted average of x and y coordinates to form a (weighted) mean
    # location (the weight is given by the numbers in the 2d array); (ii) we determine the distance between each populated
    # grid location (x weight) and add these up.  The total gives us a weighted average distance between the locations.
    # Note, however, that there is a challenge with doing this with a torus: we need to look at two different 2d arrays (the
    # original array and then one corresponding to its mirror image).  We find the coefficient for each of
    # these and use the smallest (note that it might appear there are 4 mirror arrays in a torus but all of these would be
    # identical! so we only need to fine one mirror).

    # we create an array to record the two cluster centres
    clust_centre_array = []

    #    if print_dets == 1:
    #        print '\ntwo_d_array =\n'
    #        for i in np.arange(dimen):
    #            print two_d_array[i]

    # original 2d_array: find total sales, total x dimen and total y dimen
    tot_sales = 0
    av_x = 0
    av_y = 0

    for i in np.arange(dimen):
        for j in np.arange(dimen):
            if two_d_array[i][j] > 0:
                tot_sales += two_d_array[i][j]

    for k in np.arange(dimen):
        for l in np.arange(dimen):
            if two_d_array[k][l] > 0:
                av_x += (two_d_array[k][l] / float(tot_sales)) * k
                av_y += (two_d_array[k][l] / float(tot_sales)) * l

    if print_dets == 1:
        print('\nav_x =', av_x)
        print('av_y =', av_y)

    clust_centre_array.append([av_x, av_y])

    # now we need to find the (direct) distance between this average and each point, weighted
    tot_w_dist = 0
    for i in np.arange(dimen):
        for j in np.arange(dimen):
            if two_d_array[i][j] > 0:
                #                if print_dets == 1:
                #                    print '\ni =', i
                #                    print '\nj =', j
                # distance between average and [i][j]:
                dist = math.sqrt((i - av_x) ** 2 + (j - av_y) ** 2)
                #                if print_dets == 1:
                #                    print 'dist =', dist
                # weighted distance:
                #                if print_dets == 1:
                #                    print 'weight =', two_d_array[i][j] / float(tot_sales)
                w_dist = (two_d_array[i][j] / float(tot_sales)) * dist
                #                if print_dets == 1:
                #                    print 'w_dist =', w_dist
                # tot_weighted distance:
                tot_w_dist += w_dist

    if print_dets == 1:
        print('\ntot_w_dist =', tot_w_dist)

    # now we scale this by the maximum value this could take
    max_coeff = (dimen - 1) / math.sqrt(2)

    orig_coeff = tot_w_dist / max_coeff

    if print_dets == 1:
        print('\nmax_coeff =', max_coeff)
        print('\norig_coeff =', orig_coeff)

    # do the same with the mirror array:
    two_d_array_mir = np.zeros(shape=(dimen, dimen))

    # to create the mirror array we have to cut the original array in to 4 segments and move these around
    for m in np.arange(int(dimen / 2)):
        for n in np.arange(int(dimen / 2)):
            two_d_array_mir[m][n] = two_d_array[int((dimen / 2) + m)][int((dimen / 2) + n)]
            two_d_array_mir[int((dimen / 2) + m)][int((dimen / 2) + n)] = two_d_array[m][n]
            two_d_array_mir[m][int((dimen / 2) + n)] = two_d_array[int((dimen / 2) + m)][n]
            two_d_array_mir[int((dimen / 2) + m)][n] = two_d_array[m][int((dimen / 2) + n)]

    #    if print_dets == 1:
    #        print '\n\ntwo_d_array_mir =\n'
    #        for i in np.arange(dimen):
    #            print two_d_array_mir[i]

    av_x_mir = 0
    av_y_mir = 0

    for k in np.arange(dimen):
        for l in np.arange(dimen):
            if two_d_array_mir[k][l] > 0:
                av_x_mir += (two_d_array_mir[k][l] / float(tot_sales)) * k
                av_y_mir += (two_d_array_mir[k][l] / float(tot_sales)) * l

    if print_dets == 1:
        print('\nav_x_mir =', av_x_mir)
        print('av_y_mir =', av_y_mir)

    # note that the mirror array will give a cluster centre which needs to be translated in to the original array's
    # coordinates by subtracing dimen / 2 from av_x_mir and av_y_mir and then % dimen in case these are negative

    av_x_mir_conv = (av_x_mir - dimen / 2) % dimen
    av_y_mir_conv = (av_y_mir - dimen / 2) % dimen

    clust_centre_array.append([av_x_mir_conv, av_y_mir_conv])

    # now we need to find the (direct) distance between this average and each point, weighted
    tot_w_dist_mir = 0
    for i in np.arange(dimen):
        for j in np.arange(dimen):
            if two_d_array_mir[i][j] > 0:
                #                if print_dets == 1:
                #                    print '\ni =', i
                #                    print '\nj =', j
                # distance between average and [i][j]:
                dist = math.sqrt((i - av_x_mir) ** 2 + (j - av_y_mir) ** 2)
                #                if print_dets == 1:
                #                    print 'dist =', dist
                # weighted distance:
                #                if print_dets == 1:
                #                    print 'weight =', two_d_array_mir[i][j] / float(tot_sales)
                w_dist_mir = (two_d_array_mir[i][j] / float(tot_sales)) * dist
                #                if print_dets == 1:
                #                    print 'w_dist_mir =', w_dist_mir
                # tot_weighted distance:
                tot_w_dist_mir += w_dist_mir

    if print_dets == 1:
        print('\ntot_w_dist_mir =', tot_w_dist_mir)

    # now we scale this by the maximum value this could take

    mirr_coeff = tot_w_dist_mir / max_coeff

    if print_dets == 1:
        print('\nmirr_coeff =', mirr_coeff)

    # we find the smallest of the two clustering coefficients and return this
    min_cluster_coeff = np.min([orig_coeff, mirr_coeff])

    if min_cluster_coeff == orig_coeff:
        clust_centre = clust_centre_array[0]

    elif min_cluster_coeff == mirr_coeff:
        clust_centre = clust_centre_array[1]

    if print_dets == 1:
        print('returned: =', [min_cluster_coeff, clust_centre])

    return [min_cluster_coeff, clust_centre]


def abs_dist_on_torus(loc_1, loc_2, dimen):

    """This function finds the absolute distance between two locations on a torus (in 2 dimensions)."""

#    print(' loc_1', loc_1)
#    print(' loc_2', loc_2)
#
#    print('\n np.abs(loc_1[0] - loc_2[0]) =', np.abs(loc_1[0] - loc_2[0]))
#    print('\n np.abs(loc_1[1] - loc_2[1]) =', np.abs(loc_1[1] - loc_2[1]))

    dist_x = np.abs(loc_1[0] - loc_2[0])

    if dist_x > dimen / 2.0:

        dist_x = int(dimen - dist_x)

    dist_y = np.abs(loc_1[1] - loc_2[1])

    if dist_y > dimen / 2.0:

        dist_y = int(dimen - dist_y)

    return [dist_x, dist_y]


def find_location_furthest_away(town_grid, combined_dict, print_fine_dets, complete=0):

    """This function finds a location on a torus furthest from any negative location provided in a dictionary.  It returns the location.  It is in draft form."""

    # print_fine_dets = 1

    max_dist_from_fight = -1
    best_location = [None, None]

    if complete:

        for x_coord in range(town_grid.dimen):
            for y_coord in range(town_grid.dimen):

                for entry in combined_dict:

                    min_dist = copy.copy(town_grid.dimen)

                    if combined_dict[entry] < 0.0:

                        exec('town_grid.entry_location = %s' % entry)

                        x_dist = np.abs(x_coord - town_grid.entry_location[0])
                        y_dist = np.abs(y_coord - town_grid.entry_location[1])

                        if x_dist > town_grid.dimen / 2.0:
                            x_dist = town_grid.dimen - x_dist

                        if y_dist > town_grid.dimen / 2.0:
                            y_dist = town_grid.dimen - y_dist

                        travel_dist = np.max([x_dist, y_dist])

                        if travel_dist < min_dist:

                            if print_fine_dets:
                                print('\n x', x_coord, 'y', y_coord, 'entry', entry, 'travel_dist', travel_dist, 'replacing old max_dist_from_fight of', max_dist_from_fight)

                            min_dist = travel_dist

                    if min_dist > max_dist_from_fight:
                        max_dist_from_fight = min_dist
                        best_location = [x_coord, y_coord]

    # if we are not taking the complete approach, which takes a long time, we will use a partial approach: we create 10 random locations and choose the location which is furthest from any -ve loc
    else:

        if print_fine_dets:
            print('len(combined_dict)', len(combined_dict))

        # we don't want to be stuck in a while loop so we give it 10 opportunities to find a solution not dimen ** 2
        counter = 0

        while counter < 10:

            random_loc = [random.randint(0, town_grid.dimen - 1), random.randint(0, town_grid.dimen - 1)]

            if print_fine_dets:
                print('\n counter =', counter)
                print('\n random_loc =', random_loc)

            min_dist = copy.copy(town_grid.dimen)

            for entry in combined_dict:

                if combined_dict[entry] < 0.0:

                    exec('town_grid.entry_location = %s' % entry)

                    x_dist = np.abs(random_loc[0] - town_grid.entry_location[0])
                    y_dist = np.abs(random_loc[1] - town_grid.entry_location[1])

                    if x_dist > town_grid.dimen / 2.0:
                        x_dist = town_grid.dimen - x_dist

                    if y_dist > town_grid.dimen / 2.0:
                        y_dist = town_grid.dimen - y_dist

                    travel_dist = np.max([x_dist, y_dist])

                    if travel_dist < min_dist:

                        if print_fine_dets:
                            print('\n x', random_loc[0], 'y', random_loc[1], 'entry', entry, 'travel_dist', travel_dist, 'replacing old min_dist of', min_dist)

                        min_dist = travel_dist

            if min_dist > max_dist_from_fight:

                max_dist_from_fight = min_dist
                best_location = random_loc

            if print_fine_dets:
                print('\n End of while loop - max_dist_from_fight =', max_dist_from_fight, 'best_location so far =', best_location)

            counter += 1

    if print_fine_dets:
        pause()

    return best_location


def print_prices_charts(num_res_founts, print_dets, print_fine_dets, dbs, rounds, fountain_population, data_folder, Walrasian_Trading, two_tribes):
    """This function organises the data for printing charts which show a time series of various price data, including actual
    mean prices, partial equilibrium prices, and general equilibrium prices."""

    #    prices_db = np.array(shape=())

    #    print_fine_dets = 1

    if print_fine_dets == 1:
        print('\n dbs.optimal_price_array =\n', dbs.optimal_price_array)
        print('\n dbs.mean_price_history =\n', dbs.mean_price_history)

    for res_1 in np.arange(num_res_founts):

        for res_2 in np.arange(num_res_founts):

            if res_1 != res_2:

                trans_optimal_price_array = []

                for i in np.arange(rounds):
                    trans_optimal_price_array.append(1 / dbs.optimal_price_array[i][res_1][res_2])

                if two_tribes:

                    trans_optimal_price_array_sharks = []
                    trans_optimal_price_array_jets = []

                    for i in np.arange(rounds):
                        trans_optimal_price_array_sharks.append(1 / dbs.optimal_price_array_sharks[i][res_1][res_2])
                        trans_optimal_price_array_jets.append(1 / dbs.optimal_price_array_jets[i][res_1][res_2])

                if print_fine_dets == 1:
                    print('\n trans_optimal_price_array =\n', trans_optimal_price_array)

                prices_db = [[], [], []]
                chart_for_paper_db = [[], []]

                if two_tribes:
                    prices_db_sharks = [[], [], []]
                    chart_for_paper_db_sharks = [[], []]

                    prices_db_jets = [[], [], []]
                    chart_for_paper_db_jets = [[], []]

                price_day = 0

                # If there were no transactions in a particular day, we must ignore price data - we search through
                # dbs.mean_price_history[res_1][res_2] and ignore days where == None
                for day_mean_chart_price in dbs.mean_price_history[res_1][res_2]:

                    if day_mean_chart_price != 1000 and day_mean_chart_price != 0:
                        prices_db[0].append(price_day)
                        prices_db[1].append(day_mean_chart_price)
                        prices_db[2].append(trans_optimal_price_array[price_day])

                        chart_for_paper_db[0].append(price_day)
                        chart_for_paper_db[1].append(day_mean_chart_price)

                    price_day += 1

                if print_fine_dets == 1:
                    print('\nprices_db\n', prices_db)

                if two_tribes:

                    # sharks
                    price_day = 0

                    for day_mean_chart_price in dbs.mean_price_history_sharks[res_1][res_2]:

                        if day_mean_chart_price != 1000 and day_mean_chart_price != 0:
                            prices_db_sharks[0].append(price_day)
                            prices_db_sharks[1].append(day_mean_chart_price)
                            prices_db_sharks[2].append(trans_optimal_price_array_sharks[price_day])

                            chart_for_paper_db_sharks[0].append(price_day)
                            chart_for_paper_db_sharks[1].append(day_mean_chart_price)

                        price_day += 1

                    # jets
                    price_day = 0

                    # If there were no transactions in a particular day, we must ignore price data - we search through
                    # dbs.mean_price_history[res_1][res_2] and ignore days where == None
                    for day_mean_chart_price in dbs.mean_price_history_jets[res_1][res_2]:

                        if day_mean_chart_price != 1000 and day_mean_chart_price != 0:
                            prices_db_jets[0].append(price_day)
                            prices_db_jets[1].append(day_mean_chart_price)
                            prices_db_jets[2].append(trans_optimal_price_array_jets[price_day])

                            chart_for_paper_db_jets[0].append(price_day)
                            chart_for_paper_db_jets[1].append(day_mean_chart_price)

                        price_day += 1

                labels_array = ['mean actual', 'general equ', 'partial equ']

                title = 'Prices over time: Res %s vs Res %s' % (res_1, res_2)
                y_axis_label = 'Price'
                line_width = 2
                colors = ['blue', 'green', 'aqua', 'black', 'teal', 'navy', 'fuchsia', 'purple']
                filename = 'prices_%s_vs_%s' % (res_1, res_2)

                if len(prices_db[0]) > 1:  # only print this chart if there is data

                    print_chart_prices(prices_db, labels_array, title, y_axis_label, line_width, colors, data_folder, filename, data_type='-', dpi='low')

                    line_width = 3
                    title = ''
                    filename = 'price_paper_%s_vs_%s' % (res_1, res_2)

                    print_chart_prices(chart_for_paper_db, labels_array, title, y_axis_label, line_width, ['black', 'red'], data_folder, filename, data_type='-', dpi='high')

                if two_tribes:

                    labels_array = ['mean actual', 'general equ', 'partial equ']

                    title = 'Prices over time - Sharks: Res %s vs Res %s' % (res_1, res_2)
                    y_axis_label = 'Price'
                    line_width = 2
                    colors = ['blue', 'green', 'aqua', 'black', 'teal', 'navy', 'fuchsia', 'purple']
                    filename = 'prices_sharks_%s_vs_%s' % (res_1, res_2)

                    if len(prices_db_sharks[0]) > 1:  # only print this chart if there is data

                        print_chart_prices(prices_db_sharks, labels_array, title, y_axis_label, line_width, colors, data_folder, filename, data_type='-', dpi='low')

                        line_width = 3
                        title = ''
                        filename = 'price_paper_sharks_%s_vs_%s' % (res_1, res_2)

                        print_chart_prices(chart_for_paper_db_sharks, labels_array, title, y_axis_label, line_width, ['black', 'red'], data_folder, filename, data_type='-', dpi='high')

                    labels_array = ['mean actual', 'general equ', 'partial equ']

                    title = 'Prices over time - Jets: Res %s vs Res %s' % (res_1, res_2)
                    y_axis_label = 'Price'
                    line_width = 2
                    colors = ['blue', 'green', 'aqua', 'black', 'teal', 'navy', 'fuchsia', 'purple']
                    filename = 'prices_jets_%s_vs_%s' % (res_1, res_2)

                    if len(prices_db_jets[0]) > 1:  # only print this chart if there is data

                        print_chart_prices(prices_db_jets, labels_array, title, y_axis_label, line_width, colors, data_folder, filename, data_type='-', dpi='low')

                        line_width = 3
                        title = ''
                        filename = 'price_paper_jets_%s_vs_%s' % (res_1, res_2)

                        print_chart_prices(chart_for_paper_db_jets, labels_array, title, y_axis_label, line_width, ['black', 'red'], data_folder, filename, data_type='-', dpi='high')

                if Walrasian_Trading == 1:  # only print this chart if there is data

                    walrasian_prices_paper = [[], []]

                    walrasian_prices_paper[0] = range(rounds)
                    walrasian_prices_paper[1] = trans_optimal_price_array

                    line_width = 3
                    title = ''
                    filename = 'price_paper_%s_vs_%s' % (res_1, res_2)

                    print_chart_prices(walrasian_prices_paper, labels_array, title, y_axis_label, line_width, ['black', 'red'], data_folder, filename, data_type='-', dpi='high')


def print_turnover_charts(num_res_founts, print_dets, print_fine_dets, dbs, rounds, fountain_population, data_folder, gen_equ_thresh, constitutional_voting, start_const_proces, const_proc_test_period):

    net_turnover_db = []

    for res in range(num_res_founts + 1):

        net_turnover_db.append([])

    net_turnover_trans = np.transpose(dbs.net_turnover_prop[:rounds])

    if print_fine_dets == 1:

        print('\n net_turnover_trans =\n', net_turnover_trans)
        print('\n dbs.optimal_bskt_errors=\n', dbs.optimal_bskt_errors)

    means_array = [[], []]

    for i in np.arange(rounds):

        if np.all(np.abs(dbs.optimal_bskt_errors[i]) < gen_equ_thresh):

            # Put the round in the [0] arrays
            net_turnover_db[0].append(i)
            means_array[0].append(i)

            # Create array to record resource numbers in each round (we need to take a mean)
            all_res_array = np.zeros(shape=(num_res_founts))

            # Iterate through resources
            for res in np.arange(num_res_founts):

                net_turnover_db[res + 1].append(net_turnover_trans[res][i])

                all_res_array[res] = net_turnover_trans[res][i]

                if res == num_res_founts - 1:       # then we can take an av of all_res_array

                    mean_to = np.mean(all_res_array)

                    means_array[1].append(mean_to)

    moving_averages_array = np.zeros(shape=(2, len(means_array[0])), dtype=float)
    moving_averages_array[0] = np.array(means_array[0])

#    print('\n means_array =', means_array)
#    print('\n means_array[1][0] =', means_array[1][0])
#    print('\n type(means_array[1][0]):', type(means_array[1][0]))

    moving_averages_array[1][0] = means_array[1][0]

    moving_av_length = 10

#    print('\n len(means_array[0]) =', len(means_array[0]))

    for day in range(1, len(means_array[0])):

        start_day = np.max([0, day - moving_av_length])

        moving_averages_array[1][day] = np.mean(means_array[1][start_day:day + 1])

        if constitutional_voting == 1 and day >= start_const_proces + (const_proc_test_period * 4):

            moving_averages_array[1][day] = 1.0

    if print_fine_dets == 1:
        print('\nnet_turnover_db =', net_turnover_db)

    labels_array = []

    for res in np.arange(num_res_founts):

        labels_array.append('Res %s' % (res))

    labels_array.append('mean')

    title = 'Mean Turnover of Resources as a proportion of Market Clearing Turnover'
    axis_labels = ['Rounds', 'Ratio']
    line_width = 2
    colors = ['blue', 'green', 'aqua', 'black', 'teal', 'navy', 'fuchsia', 'purple']
    filename = 'turnover_time_series_mean'

    print_chart(net_turnover_db, labels_array, title, axis_labels, line_width, colors, data_folder, filename, data_type='-', dpi='low')

    print_chart(means_array, ['Mean'], '', axis_labels, 3, ['black'], data_folder, filename, data_type='-', dpi='low')

    print_chart(moving_averages_array, [''], '', axis_labels, 2, ['black'], data_folder, filename, data_type='-', dpi='high')



def print_turnover_breakdown_charts(num_res_founts, print_dets, print_fine_dets, dbs, rounds, fountain_population, data_folder, gen_equ_thresh):
    """This function prints a chart for each resource, showing the optimal and actual turnover."""

    for res in np.arange(num_res_founts):

        turnover_db = [[], [], []]

        for i in np.arange(rounds):

            if np.sum(np.abs(dbs.optimal_bskt_errors[i])) < gen_equ_thresh:
                turnover_db[0].append(i)

                turnover_db[1].append(dbs.net_net_transs_db[i][res])

                turnover_db[2].append(dbs.optimal_bskt_turnover[i][res])

        if print_fine_dets == 1:
            print('\nturnover_db =\n', turnover_db)

        labels_array = ['Actual', 'Optimal']

        title = 'Turnover of Resources: Actual and Optimal'
        axis_labels = ['Rounds', 'Turnover']
        line_width = 2
        colors = ['blue', 'green', 'aqua', 'black', 'teal', 'navy', 'fuchsia', 'purple']
        filename = 'turnover_res_time_series_res_%d' % res

        print_chart(turnover_db, labels_array, title, axis_labels, line_width, colors, data_folder, filename, data_type='-', dpi='low', show_legend=True)


def gini(array):

    """Calculate the Gini coefficient of a numpy array.  This was taken from https://github.com/oliviaguest/gini."""

    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    array = array.flatten() #all values are treated equally, arrays must be 1d

    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative

    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element

    n = array.shape[0]#number of array elements

    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


def plot_scatter_2d(data_x, data_y, title, data_folder, filename, dpi):

    min_x = 0   # np.min(data_x) * 0.95
    max_x = 20  # np.max(data_x) * 1.05

    min_y = np.min(data_y) * 0.95
    max_y = np.max(data_y) * 1.05

    plt.figure()
    plt.title("%s" % (title))
    plt.axis("auto")
    plt.xlim(xmin=min_x, xmax=max_x)
    plt.ylim(ymin=min_y, ymax=max_y)
    plt.xlabel("Distance From Target")
    plt.ylabel("Resource Level")

    plt.scatter(data_x, data_y)

    plt.legend(loc='upper center', fontsize='small')

    # if we want to show the chart:
    # plt.show()

    # if we want to save the chart:
    if dpi == 'high':
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (data_folder, filename, dt.datetime.now().year,
                    dt.datetime.now().month, dt.datetime.now().day,
                    dt.datetime.now().hour, dt.datetime.now().minute,
                    dt.datetime.now().second,
                    dt.datetime.now().microsecond / 100000), dpi=500)

    else:
        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (data_folder, filename, dt.datetime.now().year,
                    dt.datetime.now().month, dt.datetime.now().day,
                    dt.datetime.now().hour, dt.datetime.now().minute,
                    dt.datetime.now().second,
                    dt.datetime.now().microsecond / 100000))

    # if we want to close the chart:
    plt.clf()
    plt.close()
    gc.collect()


def generate_MA_array(array, N=5, might_contain_None=False, last_datum=False):

    """This function takes a 1d array and returns a version of the array with moving averages (window of N).  It ensures the first data are the closest to the MA we want by calculating the MA for up to N."""

    if might_contain_None == False:

        # if we import a list, convert it to numpy array
        if type(array) == list:

            array = np.array(array, dtype=float)

        MA_array = np.zeros(shape=len(array), dtype=float)

        for day in range(0, len(array)):

            start_point = np.max([0, day - N + 1])

            MA_array[day] = np.mean(array[start_point:day + 1])

            # print(' day ', day, 'start_point', start_point, 'np.mean(array[start_point:day + 1]', np.mean(array[start_point:day + 1]))

    if might_contain_None:

        if type(array) == list:

            MA_array = np.zeros(shape=len(array), dtype=float)

            for day in range(len(array)):

                start_point = np.max([0, day - N + 1])

                if last_datum:
                    end_point = np.min([last_datum + 1, day + 1])
                    # print('\n start_point =', start_point)
                    # print(' end_point =', end_point)
                else:
                    end_point = day + 1

                # if last_datum:
                #     print('\n array[start_point:end_point] :', array[start_point:end_point])

                if last_datum is False or (last_datum and day <= last_datum):

                    if None not in array[start_point:end_point]:
                        MA_array[day] = np.mean(array[start_point:end_point])

                    else:       # then we have to deal with Nones
                        part_array = []
                        for entry in array[start_point:end_point]:
                            if entry is not None:
                                part_array.append(entry)
                        if len(part_array) > 0:
                            MA_array[day] = np.mean(part_array)
                        else:
                            MA_array[day] = None

                else:
                    MA_array[day] = None

        elif type(array) == np.ndarray:

            MA_array = np.zeros(shape=len(array), dtype=float)

            for day in range(len(array)):

                start_point = np.max([0, day - N + 1])

                if last_datum:
                    end_point = np.min([last_datum + 1, day + 1])
                    # print('\n day =', day, ' start_point =', start_point)
                    # print(' end_point =', end_point)
                else:
                    end_point = day + 1

                # if last_datum:
                #     print('\n array[start_point:end_point] :', array[start_point:end_point])

                part_array = []

                if last_datum is False or (last_datum and day <= last_datum):
                    for item in array[start_point:end_point]:

                        if math.isnan(item) == False:
                            part_array.append(item)

                if len(part_array) > 0:
                    MA_array[day] = np.mean(part_array)
                else:
                    MA_array[day] = None

    return MA_array


def length_of_time(delta):

    """"This function takes a time delta (from datetime) and returns values in years, days, hours, minutes, and seconds.  Note years are assumed to be 365 days and microseconds are added to seconds."""

    years = delta.days // 365
    days = delta.days % 365
    hours = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60
    seconds = (delta.seconds % 3600 % 60) + (delta.microseconds / 1000000.0)

    return years, days, hours, minutes, seconds


def print_chart(database, labels_array=[], title='', axis_labels=[], line_width=3, colors=[], data_folder='folder_name', filename='name', data_type=None, dpi=None, show_legend=False, keep_name=0, special_data=None, font_size=10):

    """A function for printing a chart with multiple lines."""

    print('---> printing chart: %s ' % title)

    # print('\n len(labels_array) =', len(labels_array), 'len(colors) =', len(colors), 'len(database) - 1', len(database) - 1)

    x = database[0]

    ar = []

    col_counter = 0
    for line_dat in database[1:]:

        if len(database) == 2:

            if len(labels_array) > 0:
                line_name = labels_array[0]
            else:
                line_name = ''

            gs1 = go.Scatter(
                    x=x,
                    y=line_dat,
                    name=line_name,
                    line=dict(color='black', width=3),
                    mode='lines'
                )

        elif len(labels_array) == len(colors) == len(database) - 1:      # we have the same number of data points as colours and labels

            gs1 = go.Scatter(
                    x=x,
                    y=line_dat,
                    name=labels_array[col_counter],
                    line=dict(color=colors[col_counter], width=3),
                    mode='lines'
                )

        else:
            if len(labels_array) >= len(database[1:]):

                gs1 = go.Scatter(
                        x=x,
                        y=line_dat,
                        name=labels_array[col_counter],
                        line=dict(width=3),
                        mode='lines'
                    )

            else:

                gs1 = go.Scatter(
                        x=x,
                        y=line_dat,
                        line=dict(width=3),
                        mode='lines'
                    )

        ar.append(gs1)

        col_counter += 1

    if special_data is not None:

        if 'dash' not in special_data:
            dash = 'solid'
        else:
            dash = special_data['dash']

        gs_spec = go.Scatter(
                x=x,
                y=special_data['data'],  # 'line_col'
                name=special_data['name'],
                line=dict(color=special_data['line_col'], width=special_data['line_width'], dash=dash),
                mode='lines'
            )

        ar.append(gs_spec)

    fig = go.Figure(ar)

    if len(axis_labels) > 0:

        fig.update_layout(
                          xaxis_title=axis_labels[0],
                          yaxis_title=axis_labels[1],
                          )

    # print('\n filename =', filename)
    # print(' show_legend =', show_legend)

    # if show_legend:

    fig.update_layout(showlegend=show_legend, font_size=font_size)

    if keep_name:
        file_name_fig = "%s/%s.html" % (data_folder, filename)

    else:
        file_name_fig = "%s/%s %d - %d - %d - %d - %d - %d - %d.html" % (data_folder, filename, dt.datetime.now().year,
                                                       dt.datetime.now().month, dt.datetime.now().day,
                                                       dt.datetime.now().hour, dt.datetime.now().minute,
                                                       dt.datetime.now().second,
                                                       dt.datetime.now().microsecond / 100000)

    plotly.offline.plot(fig, filename=file_name_fig, auto_open=False)





    # Old approach:

#     # we have to decide where we want to save the chart and its default name:
#     filepath = data_folder
#
#     # number of lines in the chart will probably be this (i.e. excluding the
#     # round data)
#     num_lines = len(database) - 1
#
#     # set up x_min and x_max (likely to be just start & end of rounds):
#     x_min = database[0][0]
#     x_max = database[0][-1]
#
# #    print('\n database[1:] =\n', database[1:])
#
#     # note 'y' corresponds to the line data:
#     y_min = np.min(database[1:])
#     y_max = np.max(database[1:])
#
#     if y_min == y_max:
#
#         y_min -= 1.0
#         y_max += 1.0
#
#     # create 5% buffers above and below lines:
#     # create 5% buffers above and below lines:
#     if y_min > 0:
#         if data_type == 'x':
#             y_min = y_min * 0.99
#         else:
#             y_min = y_min * 0.95
#     elif y_min <= 0:   # i.e. y_min is negative
#         y_min = y_min * 1.05
#
#     if math.isnan(y_min) or y_min == np.inf:
#         y_min = 0.0
#
#     if y_max > 0:
#         if data_type == 'x':
#             y_max = y_max * 1.01
#         else:
#             y_max = y_max * 1.05
#     elif y_max <= 0:   # i.e. y_max is negative
#         y_max = y_max * 0.95
#
#     if math.isnan(y_max) or y_max == np.inf:
#         y_max = 5.0
#
#     if len(labels_array) > 0 and labels_array[0] == 'Probability Threshold':
#
#         y_min = 0
#         y_max = 1.4
#
#     # Override y_min and y_max is it's this chart (sometimes we get a v high number, which is wrong)
#     if title == 'Turnover of Resources as a proportion of Gen Equ turnover':
#
#         y_min = 0.0
#         y_max = 1.5
#
#     if math.isnan(y_min) or y_min == np.inf or math.isnan(y_max) or y_max == np.inf:
#
#         print('\n PROBLEM')
#         print('y_min =', y_min)
#         print(' y_min == np.nan', y_min == np.nan, ' y_min == np.inf', y_min == np.inf)
#         print('y_max =', y_max)
#         print(' y_max == np.nan', y_max == np.nan, ' y_max == np.inf', y_max == np.inf)
#         pause()
#
#     # now build the chart:
#     plt.figure()
#     plt.title("%s" % title)
#     plt.axis("auto")
#     plt.xlim(xmin=x_min, xmax=x_max)
#     plt.ylim(ymin=y_min, ymax=y_max)
#
#     if data_type == 'x':
#         plt.xlabel("Time Period")
#     else:
#         plt.xlabel("Round")
#     plt.ylabel("%s" % (y_axis_label))
#
#     # print('\n num_lines =', num_lines)
#
#     if len(labels_array) > 0 or data_type == 'x':
#
#         for i in range(num_lines):
#
#             if i < len(labels_array):
#
#                 label = "%s" % (labels_array[i])
#
#             else:
#
#                 label = ''
#
#             col_num = i % len(colors)
#             if len(colors) == 1:
#                 col = colors[0]
#             else:
#                 col = colors[col_num]
#
#             if data_type != 'x' or (data_type == 'x' and i != 0):
#
#                 plt.plot(database[0],
#                          database[i + 1], data_type,
#                          color=col,
#                          label=label,
#                          linewidth=line_width
#                          )
#
#         # if we're printing MRSs then we show the first agent as a red X, also larger
#         if data_type == 'x':
#
#             plt.plot(database[0],
#                      database[1], 'o',
#                      color='red',
#                      label='',
#                      linewidth=3
#                      )
#
#     else:
#
#         for i in range(num_lines):
#             plt.plot(database[0],
#                      database[i + 1], data_type,
#                      color=colors[i],
#                      linewidth=line_width)
#
#     if len(labels_array) > 0 and labels_array[0] == 'Probability Threshold':
#
# #        print('\n y_min =', y_min)
# #        print('\n y_max =', y_max)
#         plt.legend(loc=0, fontsize='medium')
#
#     else:
#
#         plt.legend(loc=0, fontsize='small')
#
#     # if we want to show the chart:
#     # plt.show()
#
#     # if we want to save the chart:
#     if dpi == 'high':
#
#         plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
#                     % (filepath, filename, dt.datetime.now().year,
#                     dt.datetime.now().month, dt.datetime.now().day,
#                     dt.datetime.now().hour, dt.datetime.now().minute,
#                     dt.datetime.now().second,
#                     dt.datetime.now().microsecond / 100000), dpi=500)
#
#     else:
#
#         plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
#                     % (filepath, filename, dt.datetime.now().year,
#                     dt.datetime.now().month, dt.datetime.now().day,
#                     dt.datetime.now().hour, dt.datetime.now().minute,
#                     dt.datetime.now().second,
#                     dt.datetime.now().microsecond / 100000))
#
#     # if we want to close the chart:
#     plt.clf()
#     plt.close()
#     gc.collect()


def print_contributions_to_total_chart(database, labels_array=[], title='', axis_labels=[], line_width=3, colors=[], data_folder='folder_name', filename='name', data_type=None, dpi=None, show_legend=False, keep_name=0, special_data=None):

    """A function for printing a chart with multiple lines."""

    colors = ['blue', 'red', 'green', 'purple']

    print('---> printing chart: %s ' % title)

    # print('\n len(labels_array) =', len(labels_array), 'len(colors) =', len(colors), 'len(database) - 1', len(database) - 1)

    # plot = px.Figure()

    plot = go.Figure()

    num_lines = len(database) - 1

    for line_num in range(num_lines):

        plot.add_trace(go.Scatter(
            name=labels_array[line_num],
            x=database[0],
            y=database[line_num + 1],
            stackgroup='one',
            # color=colors[line_num]
        ))

        # plot.add_trace(go.Scatter(
        #     name='Data 2',
        #     x=x,
        #     y=[56, 123, 982, 213],
        #     stackgroup='one'
        # )
        # )

    if keep_name:
        file_name_plot = "%s/%s.html" % (data_folder, filename)

    else:
        file_name_plot = "%s/%s %d - %d - %d - %d - %d - %d - %d.html" % (data_folder, filename, dt.datetime.now().year,
                                                       dt.datetime.now().month, dt.datetime.now().day,
                                                       dt.datetime.now().hour, dt.datetime.now().minute,
                                                       dt.datetime.now().second,
                                                       dt.datetime.now().microsecond / 100000)

    plotly.offline.plot(plot, filename=file_name_plot, auto_open=False)


def print_contributions_to_tot_2_axes(database, labels_array=[], title='', axis_labels=[], line_width=3, colors=[], data_folder='folder_name', filename='name', data_type=None, dpi=None, show_legend=False, keep_name=0, font_size=10, special_data=None):

    """A function for printing a chart with multiple filled lines and a second axis for net total."""

    # database enters as [0] for x-axis data and then a number of `stacks'

    # Create figure with secondary y-axis
    # fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig = go.Figure()

    # stack_nums = ['one', 'two', 'three']

    x_axis_data = database[0]

    line_num_index = 0
    stack_num = 0

    for stack in database[1]:

        num_lines = len(stack)

        # Add traces
        for line_num in range(num_lines):

            fig.add_trace(go.Scatter(
                name=labels_array[line_num_index],
                x=x_axis_data,
                y=stack[line_num],
                line=dict(width=0.1, color=colors[line_num_index]),
                # pattern_shape='nation',
                # pattern_shape_sequence=[".", "x", "+"],
                stackgroup=stack_num
                )#, secondary_y=True
            )

            line_num_index += 1

        stack_num += 1


    # fig.add_trace(
    #     go.Scatter(x=[1, 2, 3], y=[40, 50, 60], name="yaxis data"),
    #     secondary_y=False,
    # )

    if special_data is not None:

        fig.add_trace(
            go.Scatter(x=database[0],
                       y=special_data,
                       name="Propensity to Steal",
                       line=dict(color='black', width=4)
            )
                       #, secondary_y=False
        )

    # Add figure title
    fig.update_layout(
        title_text=title,
        font_size=font_size,
        showlegend=False
    )

    # Set x-axis title
    fig.update_xaxes(title_text=axis_labels[0])

    # Set y-axes titles
    fig.update_yaxes(title_text=axis_labels[1])
    # fig.update_yaxes(title_text="Stock", secondary_y=False)
    # fig.update_yaxes(title_text="Change", secondary_y=True)

    # fig.show()


    #
    #
    # colors = ['blue', 'red', 'green', 'purple']
    #
    # print('---> printing chart: %s ' % title)
    #
    # # print('\n len(labels_array) =', len(labels_array), 'len(colors) =', len(colors), 'len(database) - 1', len(database) - 1)
    #
    # # plot = px.Figure()
    #
    # plot = go.Figure()
    #
    # num_lines = len(database) - 1
    #
    # for line_num in range(num_lines):
    #
    #     plot.add_trace(go.Scatter(
    #         name=labels_array[line_num],
    #         x=database[0],
    #         y=database[line_num + 1],
    #         stackgroup='one',
    #         # color=colors[line_num]
    #     ))
    #
    #     # plot.add_trace(go.Scatter(
    #     #     name='Data 2',
    #     #     x=x,
    #     #     y=[56, 123, 982, 213],
    #     #     stackgroup='one'
    #     # )
    #     # )

    if keep_name:
        file_name_plot = "%s/%s.html" % (data_folder, filename)

    else:
        file_name_plot = "%s/%s %d - %d - %d - %d - %d - %d - %d.html" % (data_folder, filename, dt.datetime.now().year,
                                                       dt.datetime.now().month, dt.datetime.now().day,
                                                       dt.datetime.now().hour, dt.datetime.now().minute,
                                                       dt.datetime.now().second,
                                                       dt.datetime.now().microsecond / 100000)

    plotly.offline.plot(fig, filename=file_name_plot, auto_open=False)


def print_chart_MRSs(database, title, data_folder, filename, data_type, show_journey=1):

    """A function for printing a chart with multiple lines."""

    print('---> printing chart: %s ' % title)

    x = database[0]

    ar = []

    col_counter = 0
    for line_dat in database[1:]:

        gs1 = go.Scatter(
                x=x,
                y=line_dat,
                marker=dict(color='black', size=5),
                marker_symbol=data_type,
                mode='markers'
            )

        ar.append(gs1)

        col_counter += 1

    # here we find which of the data lines has the widest range so we use a red color - this is to show one agent's 'MRS journey' during a round
    if show_journey:

        counter = 0
        max_range = 0.0
        max_range_num = 0

        for line_dat in database[1:]:

            ag_range = np.max(line_dat) - np.min(line_dat)

            if ag_range >= max_range:
                max_range = copy.copy(ag_range)
                max_range_num = copy.copy(counter + 1)          # note it's +1 as 'counter' would have counted from database[1]

            counter += 1

        # only add a red colour journey if there is any variation by any of the agents
        if max_range > 0.0:

            gs_journey = go.Scatter(
                    x=x,
                    y=database[max_range_num],
                    marker=dict(color='red', size=8),
                    marker_symbol=data_type,
                    mode='markers'
                )

            ar.append(gs_journey)

    fig = go.Figure(ar)

    fig.update_layout(
                      xaxis_title="Moves During the Trading Phase",
                      yaxis_title="Marginal Rate of Substitution",
                      showlegend=False
                      )

    file_name_fig = "%s/%s %d - %d - %d - %d - %d - %d - %d.html" % (data_folder, filename, dt.datetime.now().year,
                                                   dt.datetime.now().month, dt.datetime.now().day,
                                                   dt.datetime.now().hour, dt.datetime.now().minute,
                                                   dt.datetime.now().second,
                                                   dt.datetime.now().microsecond / 100000)

    plotly.offline.plot(fig, filename=file_name_fig, auto_open=False)


def print_chart_prices(database, labels_array, title, y_axis_label, line_width, colors, data_folder, filename, data_type, dpi):

    """A function for printing a chart with multiple lines."""

    print('---> printing chart: %s ' % (title))

    colors=('black', 'blue')
    names = ('Actual', 'Optimal')

    x = database[0]

    ar = []

    col_counter = 0
    for line_dat in database[1:]:

        gs1 = go.Scatter(
                x=x,
                y=line_dat,
                line=dict(color=colors[col_counter], width=2),
                mode='lines',
                name=names[col_counter]
            )
        ar.append(gs1)

        col_counter += 1

    fig = go.Figure(ar)

    if len(database) > 2:
        show_ledge = True
    else:
        show_ledge = False

    fig.update_layout(
                      xaxis_title="Rounds",
                      yaxis_title="Price",
                      showlegend=show_ledge
                      )

    file_name_fig = "%s/%s %d - %d - %d - %d - %d - %d - %d.html" % (data_folder, filename, dt.datetime.now().year,
                                                   dt.datetime.now().month, dt.datetime.now().day,
                                                   dt.datetime.now().hour, dt.datetime.now().minute,
                                                   dt.datetime.now().second,
                                                   dt.datetime.now().microsecond / 100000)

    plotly.offline.plot(fig, filename=file_name_fig, auto_open=False)


    # old approach:
#     # we have to decide where we want to save the chart and its default name:
#     filepath = data_folder
#
#     # number of lines in the chart will probably be this (i.e. excluding the
#     # round data)
#     num_lines = len(database) - 1
#
#     # set up x_min and x_max (likely to be just start & end of rounds):
#     x_min = database[0][0]
#     x_max = database[0][-1]
#
#     # note 'y' corresponds to the line data:
#     y_min = np.min(database[1:])
#     y_max = np.max(database[1:])
#
# #    y_min = 0
# #    y_max = 3
#
#     # create 5% buffers above and below lines:
#     if y_min > 0:
#         y_min = y_min * 0.90
#     elif y_min <= 0:   # i.e. y_min is negative
#         y_min = y_min * 1.1
#
#     if math.isnan(y_min) or y_min == np.inf:
#         y_min = 0.0
#
#     if y_max > 0:
#         y_max = y_max * 1.1
#     elif y_max <= 0:   # i.e. y_max is negative
#         y_max = y_max * 0.90
#
#     if math.isnan(y_max) or y_max == np.inf:
#         y_max = 5.0
#
#     if math.isnan(y_min) or y_min == np.inf or math.isnan(y_max) or y_max == np.inf:
#
#         print('\n PROBLEM')
#         print('y_min =', y_min)
#         print(' y_min == np.nan', y_min == np.nan, ' y_min == np.inf', y_min == np.inf)
#         print('y_max =', y_max)
#         print(' y_max == np.nan', y_max == np.nan, ' y_max == np.inf', y_max == np.inf)
#         pause()
#
#     # Override y_min and y_max is it's this chart (sometimes we get a v high number, which is wrong)
#     if title == 'Turnover of Resources as a proportion of Gen Equ turnover':
#
#         y_min = 0.0
#         y_max = 1.5
#
#     # now build the chart:
#     plt.figure()
#     plt.title("%s" % (title))
#     plt.axis("auto")
#     plt.xlim(xmin=x_min, xmax=x_max)
#     plt.ylim(ymin=y_min, ymax=y_max)
#     plt.xlabel("Round")
#     plt.ylabel("%s" % (y_axis_label))
#
#     for i in range(num_lines):
#         plt.plot(database[0],
#                  database[i + 1], data_type,
#                  label="%s" % (labels_array[i]),
#                  color=colors[i],
#                  linewidth=line_width)
#
#     if dpi is not 'high':
#
#         plt.legend(loc=0, fontsize='small')
#
#     # if we want to show the chart:
#     # plt.show()
#
#     # if we want to save the chart:
#
#     if dpi == 'high':
#
#         plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
#                     % (filepath, filename, dt.datetime.now().year,
#                     dt.datetime.now().month, dt.datetime.now().day,
#                     dt.datetime.now().hour, dt.datetime.now().minute,
#                     dt.datetime.now().second,
#                     dt.datetime.now().microsecond / 100000), dpi=500)
#
#     else:
#
#         plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
#                     % (filepath, filename, dt.datetime.now().year,
#                     dt.datetime.now().month, dt.datetime.now().day,
#                     dt.datetime.now().hour, dt.datetime.now().minute,
#                     dt.datetime.now().second,
#                     dt.datetime.now().microsecond / 100000))
#
#     # if we want to close the chart:
#     plt.clf()
#     plt.close()
#     gc.collect()


def print_chart_cc(database, title, line_width, color, data_folder, filename, dimen, cluster_lag):

    """A function for printing a chart with multiple lines."""

    print('---> printing chart: %s ' % (title))

    # we have to decide where we want to save the chart and its default name:
    filepath = data_folder

    # set up x_min and x_max (likely to be just start & end of rounds):
    x_min = 0
    x_max = dimen

    # note 'y' corresponds to the line data:
    y_min = 0
    y_max = dimen

    for i in np.arange(cluster_lag):
        database[0][i] = None
        database[1][i] = None

    # now build the chart:
    plt.figure()
    plt.title("%s" % (title))
    plt.axis("auto")
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.xlabel("y_coord")
    plt.ylabel("x_coord")

    plt.plot(database[1],
             database[0], '-',
             color=color,
             linewidth=line_width)

    # if we want to show the chart:
    # plt.show()

    # if we want to save the chart:
    plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                % (filepath, filename, dt.datetime.now().year,
                dt.datetime.now().month, dt.datetime.now().day,
                dt.datetime.now().hour, dt.datetime.now().minute,
                dt.datetime.now().second,
                dt.datetime.now().microsecond / 100000))

    # if we want to close the chart:
    plt.clf()
    plt.close()
    gc.collect()


def print_SD_chart(supply_demand_array_1_res, title, data_folder, filename, labels=[], traded_data=[0.0, 0.0], show_old_curves=1, print_fine_dets=0):

    """A function for printing a Supply & Demand Curves chart with multiple lines."""

    print('---> printing chart: %s ' % (title))

    # print('\n supply_demand_array_1_res:\n\n', supply_demand_array_1_res)
    # print(' weighted_mean_traded_price =', weighted_mean_traded_price, 'net_supply =', net_supply)

    if len(traded_data) > 0:

        weighted_mean_traded_price, net_supply = traded_data

        # print(' initial weighted_mean_traded_price =', weighted_mean_traded_price)

        # invert price point
        weighted_mean_traded_price = 1 / float(weighted_mean_traded_price)

    # if show_old_curves:
    #     print('\n supply_demand_array_1_res :', supply_demand_array_1_res)

    if show_old_curves:
        x_old, y_1_old, y_2_old = supply_demand_array_1_res[0]          #[1] is historical data, [0] is current
        x, y_1, y_2 = supply_demand_array_1_res[1]

    else:
        x, y_1, y_2 = supply_demand_array_1_res

    ar = []

    gs1 = go.Scatter(
                     x=y_1,
                     y=x,
                     line=dict(color='blue', width=2),
                     mode='lines',
                     name = labels[1]
                     )
    ar.append(gs1)

    gs2 = go.Scatter(
                     x=y_2,
                     y=x,
                     line=dict(color='red', width=2),
                     mode='lines',
                     name=labels[0]
                     )
    ar.append(gs2)

    if show_old_curves:

        gs1_old = go.Scatter(
            x=y_1_old,
            y=x_old,
            line=dict(color='blue', width=2, dash='dash'),
            mode='lines',
            name=labels[3]
        )
        ar.append(gs1_old)

        gs2_old = go.Scatter(
            x=y_2_old,
            y=x_old,
            line=dict(color='red', width=2, dash='dash'),
            mode='lines',
            name=labels[2]
        )
        ar.append(gs2_old)

    if show_old_curves == 0 and len(traded_data) > 0 and net_supply > 0.0 and 0.2 < weighted_mean_traded_price < 50.0:

        gs3 = go.Scatter(
                x=[net_supply],
                y=[weighted_mean_traded_price],
                marker=dict(color='red', size=10),
                marker_symbol='star',
                mode='markers',
                name='Observed',
            )
        ar.append(gs3)

    layout_dict = dict() #dict(xaxis = {'zerolinecolor':'black'}, yaxis = {'zerolinecolor':'black'})

    fig = go.Figure(ar, layout=layout_dict)

    file_name_fig = "%s/%s %d - %d - %d - %d - %d - %d - %d.html" % (data_folder, filename, dt.datetime.now().year,
                                                   dt.datetime.now().month, dt.datetime.now().day,
                                                   dt.datetime.now().hour, dt.datetime.now().minute,
                                                   dt.datetime.now().second,
                                                   dt.datetime.now().microsecond / 100000)

    plotly.offline.plot(fig, filename=file_name_fig, auto_open=False)

    # old charting approach using matplotlib - below. this used up an enormous amount of RAM for reasons I couldn't discover - plotly is better anyway so have changed
    # pause()
#
#     # we have to decide where we want to save the chart and its default name:
#     filepath = data_folder
#
#     # number of lines in the chart will probably be this (i.e. excluding the
#     # round data)
#     num_lines = len(supply_demand_array_1_res) - 1
#
#     # set up x_min and x_max (likely to be just start & end of rounds):
#     x_min = np.min([np.min(supply_demand_array_1_res[1:]), -0.5])
#     x_max = np.max(supply_demand_array_1_res[1:]) * 1.05
#
#     # note 'y' corresponds to the line data:
#     y_min = supply_demand_array_1_res[0][0] * 0.99
#     y_max = supply_demand_array_1_res[0][-1] * 1.01
#
#     if print_fine_dets == 1:
#         print('pre y_max =', y_max)
#         print('pre y_min =', y_min)
#
#     # now build the chart:
#     plt.figure()
#     plt.title("%s" % (title))
#     plt.axis("auto")
#     plt.xlim(xmin=x_min, xmax=x_max)
#     plt.ylim(ymin=y_min, ymax=y_max)
#     plt.xlabel("Quantity Supplied or Demanded")
#     plt.ylabel("%s" % (y_axis_label))
#
#     plt.gca().invert_yaxis()
#
#     for i in range(num_lines):
#         plt.plot(supply_demand_array_1_res[i + 1],
#                  supply_demand_array_1_res[0], '-',
#                  label="%s" % (labels_array[i]),
#                  color=colors[i],
#                  linewidth=line_width)
#
#     if print_fine_dets == 1:
#         print('net_supply', net_supply, 'weighted_mean_traded_price', weighted_mean_traded_price)
#
#     # add actual transaction data
#     plt.scatter(net_supply, weighted_mean_traded_price, label='est net trans', marker='*', color='red', s=130)
#
# #    label_xcoord1 = x_min + (x_max - x_min) * 10 / 100.
# #    label_ycoord1 = y_min + (y_max - y_min) * 85 / 100.
# #    label_ycoord2 = y_min + (y_max - y_min) * 90 / 100.
# #
# #    if show_text == 1:
# #        plt.text(label_xcoord1, label_ycoord1, "Q trans = % 3.2f  |  Q equil = % 3.2f  |  percent =  % 2.1f" % (trans_Q, bilat_equ_data[1], bilat_equ_data[4]), fontsize=8)
# #        plt.text(label_xcoord1, label_ycoord2, "P trans = % 2.3f  |  P equil = % 2.3f" % (trans_p, bilat_equ_data[0]), fontsize=8)
#
#     plt.gca().invert_yaxis()
# #    plt.legend(loc='upper center', fontsize='small')
#
#     # if we want to show the chart:
#     # plt.show()
#
#     # if we want to save the chart:
#     if dpi == 'high':
#         plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
#                     % (filepath, filename, dt.datetime.now().year,
#                     dt.datetime.now().month, dt.datetime.now().day,
#                     dt.datetime.now().hour, dt.datetime.now().minute,
#                     dt.datetime.now().second,
#                     dt.datetime.now().microsecond / 100000), dpi=500)
#
#     else:
#         plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
#                     % (filepath, filename, dt.datetime.now().year,
#                     dt.datetime.now().month, dt.datetime.now().day,
#                     dt.datetime.now().hour, dt.datetime.now().minute,
#                     dt.datetime.now().second,
#                     dt.datetime.now().microsecond / 100000))
#
#     # if we want to close the chart:
#     plt.clf()
#     plt.close()
#     gc.collect()


def send_mail(send_from, send_to, subject, text, files=[], server="localhost", port=587, username='gregfisherhome@gmail.com', password='NewW0rld)rdergoogle', isTls=True):

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = send_to        # COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime = True)
    msg['Subject'] = subject

    msg.attach( MIMEText('%s\n\n\n' % (text)) )

    for f in files:
        part = MIMEBase('application', "octet-stream")
        part.set_payload( open(f,"rb").read() )
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="{0}"'.format(os.path.basename(f)))
        msg.attach(part)

    smtp = smtplib.SMTP(server, port)
    if isTls: smtp.starttls()
    smtp.login(username,password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.quit()


def pause():

    input("Press Enter to continue...")


def text_col(value, bands=[0, 0.02, 0.05, 0.1, 0.15], bkg_cols=1, num_dps=4):

    """This function adds text effects, colour, and background colour to text."""

    # text_lib = {'none'=0; 'bold'=1; 'underline'=2; 'negative1'=3; 'negative2'=5;
    #             'black'=30; 'red'=31; 'green'=32; 'yellow'=33; 'blue'=34; 'purple'=35; 'cyan'=36; 'white'=37;
    #             'BLACK'=40; 'RED'=41; 'GREEN'=42; 'YELLOW'=43; 'BLUE'=44; 'PURPLE'=45; 'CYAN'=46; 'GREY'=47; 'WHITE'=48}

    col_value = ''

    if bands[1] <= value or value <= (-1 * bands[1]):

        # start with initial text
        col_value = '\033[1;'

        # text colour:
        if bands[1] <= value:
            col_value += '30;'

        elif value < (-1 * bands[1]):
            col_value += '34;'

        # background colour:
        if bkg_cols:

            if (bands[2] <= value < bands[3]) or ((-1 * bands[3]) < value <= (-1 * bands[2])):
                col_value += '42m '

            elif (bands[3] <= value < bands[4]) or ((-1 * bands[4]) < value <= (-1 * bands[3])):
                col_value += '43m '

            elif bands[4] < value or value < (-1 * bands[4]):
                col_value += '41m '

            else:
                col_value += '48m '

        else:
            col_value += '48m '

    # add value:
    # first add num of decimal places
    dp_text = '1.%d' % num_dps

    if value > 0.0:
        col_value += '+%'
        col_value += dp_text
        col_value += 'f'
        col_value = col_value % value

    elif value < 0.0:
        col_value += '%'
        col_value += dp_text
        col_value += 'f'
        col_value = col_value % value

    else:
        col_value = '0.'
        for dp in range(num_dps + 1):
            col_value += '0'

    return col_value

    # # to test bands in a console:
    # for num in [0.01, -0.01, 0.025, -0.025, 0.06, -0.06, 0.11, -0.11, 0.16, -0.16]:
    #     print(num, '\t', text_col(num), '\033[0;30;48m')


def text_col_html(value, bands=[0, 0.02, 0.05, 0.1, 0.15]):

    """This function adds text effects, colour, and background colour to text."""



    # text_lib = {'none'=0; 'bold'=1; 'underline'=2; 'negative1'=3; 'negative2'=5;
    #             'black'=30; 'red'=31; 'green'=32; 'yellow'=33; 'blue'=34; 'purple'=35; 'cyan'=36; 'white'=37;
    #             'BLACK'=40; 'RED'=41; 'GREEN'=42; 'YELLOW'=43; 'BLUE'=44; 'PURPLE'=45; 'CYAN'=46; 'GREY'=47; 'WHITE'=48}  <span style="color: #ff0000; font-weight:bold; font-style:italic;">January 30, 2011</span>

    if bands[1] <= value or value <= (-1 * bands[1]):

        # start with initial text
        col_value = '<span style="'

        # text colour:
        if bands[1] <= value:
            col_value += 'font-weight: bold; '

        # if bands[1] < value < bands[2]:
        #     col_value += 'color: #ff0000; '

        if value < (-1 * bands[1]):
            col_value += 'color: #0000FF; font-weight: bold; '

        # background colour:
        if (bands[2] <= value < bands[3]) or ((-1 * bands[3]) < value <= (-1 * bands[2])):
            col_value += 'background-color: #FAEBD7; '

        elif (bands[3] <= value < bands[4]) or ((-1 * bands[4]) < value <= (-1 * bands[3])):
            col_value += 'background-color: #D2B48C;'

        elif bands[4] < value or value < (-1 * bands[4]):
            col_value += 'background-color: #FF0000'

        else:
            col_value += 'background-color: #FFFFFF;'

        # add value:
        if value > 0.0:
            col_value += '">+%1.4f</span>' % value

        elif value < 0.0:
            col_value += '">%1.4f</span>' % value

        else:
            col_value += '">0.00000</span>'

    else:

        if value > 0.0:
            col_value = '+%1.4f' % value
        elif value < 0.0:
            col_value = '%1.4f' % value
        else:
            col_value = '0.00000'

    return col_value

    # # to test bands in a console:
    # for num in [0.01, -0.01, 0.025, -0.025, 0.06, -0.06, 0.11, -0.11, 0.16, -0.16]:
    #     print(num, '\t', text_col(num), '\033[0;30;48m')


def fan_chart(data, colour='red', include_max_min=1, file_name='/Users/user/Desktop/test_chart.html', num_rounds=2000, median_data=None, bands=None):

    """This function takes a set of data and creates a fan chart using the mean of the data and two bands corresponding to the mean +/- 1 and 2 stds."""

    print(' printing fan chart ', file_name)

    # start with finding means and stds
    x = list(range(num_rounds))

    y = []
    y_upper = []
    y_lower = []
    y_upper_2 = []
    y_lower_2 = []
    y_upper_3 = []
    y_lower_3 = []

    if include_max_min:
        y_max = []
        y_min = []

    # print('\n file_name', file_name, 'data :\n')

    for day in range(num_rounds):

        # print('\n day', day, ': ', data[day])

        y_mean = np.mean(data[day])
        y_median = np.median(data[day])

        data_above_median = []
        data_below_median = []

        if len(data[day]) >= 2:

            for datum in data[day]:
                if datum - y_median >= 0.0:
                    data_above_median.append(datum - y_median)
                else:
                    data_below_median.append(datum - y_median)

            y_std_plus = np.std(data_above_median)
            y_std_minus = np.std(data_below_median)

        else:

            y_std_plus = 0.0
            y_std_minus = 0.0

        y.append(y_mean)
        y_upper.append(y_mean + (1.0 * y_std_plus))
        y_lower.append(y_mean - (1.0 * y_std_minus))
        y_upper_2.append(y_mean + (2.0 * y_std_plus))
        y_lower_2.append(y_mean - (2.0 * y_std_minus))
        y_upper_3.append(y_mean + (3.0 * y_std_plus))
        y_lower_3.append(y_mean - (3.0 * y_std_minus))

        if include_max_min:
            y_max.append(np.max(data[day]))
            y_min.append(np.min(data[day]))

    if colour == 'blue':
        fill_col = 'rgba(65, 131, 215, 0.2)'
        line_col = 'rgba(65, 131, 215, 0)'
    else:
        fill_col = 'rgba(255, 0, 0, 0.2)'
        line_col = 'rgba(255, 0, 0, 0)'

    gs1 = go.Scatter(
            x=x,
            y=y,
            line=dict(color='rgb(0, 0, 0, 1)', width=3),
            mode='lines'
        )

    gs2 = go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_upper + y_lower[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor=fill_col,
            line=dict(color=line_col),
            hoverinfo="skip",
            showlegend=False
        )

    gs3 = go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_upper_2 + y_lower_2[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor=fill_col,
            line=dict(color=line_col),
            hoverinfo="skip",
            showlegend=False
        )

    gs4 = go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_upper_3 + y_lower_3[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor=fill_col,
            line=dict(color=line_col),
            hoverinfo="skip",
            showlegend=False
        )

    ar = []
    ar.append(gs1)
    ar.append(gs2)
    ar.append(gs3)
    ar.append(gs4)

    # if median_data is not None:
    #     gs_median_data = go.Scatter(x=x, y=median_data, connectgaps=False, name='ps_median', line=dict(color='red', width=2))
    #     ar.append(gs_median_data)

    if include_max_min:

        gs_max = go.Scatter(
                            x=x,
                            y=y_max,
                            line=dict(color='rgb(0, 0, 0, 0.3)'),
                            mode='lines'
                            )

        gs_min = go.Scatter(
                            x=x,
                            y=y_min,
                            line=dict(color='rgb(0, 0, 0, 0.3)'),
                            mode='lines'
                            )

        ar.append(gs_max)
        ar.append(gs_min)

    fig = go.Figure(ar)

    fig.update_layout(
                      xaxis_title='Rounds',
                      yaxis_title='Propensity',
                      showlegend=False
                      )

    if bands:

        fig.add_shape(
            type='line',
            x0=0,
            y0=1,
            x1=num_rounds,
            y1=1,
            line=dict(
                color='green',
                width=3,
                dash='dash'
            )
        )

        fig.add_shape(
            type='line',
            # width=3,
            x0=0,
            y0=0,
            x1=num_rounds,
            y1=0,
            line=dict(
                color='green',
                width=3,
                dash='dash'
            )
        )

    plotly.offline.plot(fig, filename=file_name, auto_open=False)


def total_size(o, handlers={}, verbose=False):

    """ Excellent function found at https://code.activestate.com/recipes/577504/

    Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


##### Example call #####

# if __name__ == '__main__':
#     d = dict(a=1, b=2, c=3, d=[4,5,6,7], e='a string of chars')
#     print(total_size(d, verbose=True))


def create_plotly_2d_histogram(input_data, x_data, folder, filename, labels=None, x_axis_titles=None):

    if len(input_data.shape) > 1:               # i.e., it has more than 1d

        fig = make_subplots(rows=1, cols=len(input_data), shared_yaxes=True, subplot_titles=labels, y_title='Total Resources Held')

        max_value = np.max(input_data)

        counter = 0
        for line in input_data:

            fig.add_trace(go.Bar(y=line, x=x_data, name=labels[counter], marker_color='blue'), row=1, col=counter + 1)
            fig.update_xaxes(title_text=x_axis_titles[counter], row=1, col=counter + 1)

            counter += 1

        fig.update_yaxes(range=[0, max_value + 1])
        fig.update_xaxes({'showticklabels' : False})
        fig.update_layout({'showlegend' : False})

    else:

        # trace = go.Bar(y=input_data, x=x_data)
        # traces_array.append(trace)

        fig = go.Figure(go.Bar(y=input_data, x=x_data))


    file_name_fig = "%s/%s_%d - %d - %d - %d - %d - %d - %d.html" % (folder, filename, dt.datetime.now().year,
                                                                       dt.datetime.now().month, dt.datetime.now().day,
                                                                       dt.datetime.now().hour, dt.datetime.now().minute,
                                                                       dt.datetime.now().second,
                                                                       dt.datetime.now().microsecond / 100000)

    plotly.offline.plot(fig, filename=file_name_fig, auto_open=False)


def create_plotly_2_heatmaps(input_data_1, input_data_2, folder, filename, labels=['Fights', 'Transactions']):

    fig = make_subplots(rows=1, cols=2, subplot_titles=labels)

    print('\n input_data_1 =', input_data_1)
    print('\n input_data_2 =', input_data_2)

    # start with fights
    # layout_1 = go.Layout(yaxis={'scaleanchor': "x", 'scaleratio': 1, 'range': [0, 50], 'zeroline' : False, 'tickson':"boundaries"}, xaxis={'range': [0, 50]}, margin = {'l': 50, 'r': 50, 'b': 100, 't': 100, 'pad': 4}, width=1200, height=1200, autosize=False)
    plot_1 = go.Heatmap(z=input_data_1, colorscale='Reds')
    # margin_dict = {'l': 50, 'r': 50, 'b': 100, 't': 100, 'pad': 4}
    # fig.update_layout(margin=margin_dict, width=1200, height=1200, autosize=False)
    # plot_1.update_xaxes(range=[0, 50], zeroline=False, tickson='boundaries')
    # plot_1.update_yaxes(range=[0, 50], zeroline=False, tickson='boundaries')
    # trace_1 = go.Figure(data=plot_1, layout=layout_1)
    # fig.add_trace(data=plot_1, layout=layout_1)
    fig.add_trace(plot_1)

    # now transactions
    plot_2 = go.Heatmap(z=input_data_2, colorscale='Blues')
    # layout_2 = go.Layout(yaxis={'scaleanchor': "x", 'scaleratio': 1, 'range': [0, 50], 'zeroline' : False, 'tickson':"boundaries"}, xaxis={'range': [0, 50], 'margin' : {'l': 50, 'r': 50, 'b': 100, 't': 100, 'pad': 4}}, width=1200, height=1200, autosize=False)
    # trace_2 = go.Figure(data=plot_2, layout=layout_2)
    fig.add_trace(plot_2)

    # fig.update_layout(yaxis={'scaleanchor': "x", 'scaleratio': 1, 'range': [0, 50], 'zeroline' : False, 'tickson':"boundaries"}, xaxis={'range': [0, 50]}, width=1200, height=1200, autosize=False)

    file_name_fig = "%s/%s_%d - %d - %d - %d - %d - %d - %d.html" % (folder, filename, dt.datetime.now().year,
                                                                       dt.datetime.now().month, dt.datetime.now().day,
                                                                       dt.datetime.now().hour, dt.datetime.now().minute,
                                                                       dt.datetime.now().second,
                                                                       dt.datetime.now().microsecond / 100000)

    plotly.offline.plot(fig, filename=file_name_fig, auto_open=True)


def cap_floor_value(input, min, max):

    return np.min([np.max([input, min]), max])


def res_conc_test(num_agents=25, prob_intn=0.5, discern=0):

    agents_array = np.full(num_agents, 2, dtype=int)

    agent_numbers_array = np.arange(num_agents)
    # print('\n start agents_array =', agents_array)

    move_num = 0
    while np.max(agents_array) < (2 * num_agents):

        for agent_num in agent_numbers_array:

            # any one agent only interacts with this probability
            if random.random() < prob_intn:

                # print('\n agent_num', agent_num, ' agents_array =', agents_array)

                if discern == 0 or (np.max(agents_array) == np.min(agents_array)):

                    cp_agent_num = copy.copy(agent_num)

                    while cp_agent_num == agent_num:
                        cp_agent_num = random.choice(agent_numbers_array)

                    # print(' discern == 0 or move_num == 0 : cp_agent_num =', cp_agent_num)

                else:

                    max_res = np.max(agents_array)

                    # print('\n max_res = ', max_res, 'agents_array[agent_num]', agents_array[agent_num])

                    # in most situations, this will be true, which makes it easier to find agent with most resources
                    if max_res != agents_array[agent_num]:

                        for cp_ag in np.arange(num_agents):

                            if agents_array[cp_ag] == max_res:
                                cp_agent_num = cp_ag

                                # print(' found max_res agent: cp_agent_num =', cp_agent_num)

                    # there will always be at least one agent with the max number of resources
                    else:

                        agents_array_copy = copy.copy(agents_array)

                        # # in the first move_num, all the values are the same and agents_array_copy.sort() will return None
                        # if move_num > 0:
                        agents_array_copy = np.sort(agents_array_copy)

                        # print('\n agents_array_copy =', agents_array_copy)

                        # pause()

                        next_max = copy.copy(max_res)
                        counter = -1

                        while next_max == max_res:

                            next_max = agents_array_copy[counter]

                            # print('\n counter = ', counter, 'next_max =', next_max)

                            counter -= 1

                            # pause()

                        cp_agent_num = random.choice(agent_numbers_array)
                        while agents_array[cp_agent_num] != next_max:
                            cp_agent_num = random.choice(agent_numbers_array)

                        # for cp_ag in np.arange(num_agents):
                        #
                        #     if agents_array[agent_num] == next_max:
                        #         cp_agent_num = cp_ag

                        # print(' resulting cp_agent_num =', cp_agent_num)
                        # print(' agents_array =', agents_array)
                        # print(' agents_array[cp_agent_num] =', agents_array[cp_agent_num])

                        # pause()

                if random.random() < 0.5:
                    agents_array[agent_num] += agents_array[cp_agent_num]
                    agents_array[cp_agent_num] = 0

                else:
                    agents_array[cp_agent_num] += agents_array[agent_num]
                    agents_array[agent_num] = 0

        # print(' end of move_num', move_num, 'agents_array =', agents_array)

        move_num += 1

    # pause()

    # print(' final agents_array =', agents_array, 'move_num =', move_num)

    return move_num


def run_res_conc_tests(num_runs=100000, discern=1, prob_intn=0.5):

    results_array = np.zeros(num_runs)

    for run_num in np.arange(num_runs):

        results_array[run_num] = res_conc_test(prob_intn=prob_intn, discern=discern)

        if run_num % 1000 == 0:
            print('run_num =', run_num)

        # print(' result = ', result)

    print('\n discern =', discern, 'prob_intn =', prob_intn, 'mean number of moves for full concentration of resources = ', np.mean(results_array), 'SD =', np.std(results_array))


def create_3d_plotly_chart(x_data=0, y_data=0, z_data=0):

    ps_data = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]                # start ps
    lambda_data = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95]                 # lambdas
    z_data = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
              [0.0, 0.4, 0.6, 1.0, 1.0, 1.0],
              [0.0, 0.0, 0.2, 1.0, 1.0, 1.0],
              [0.0, 0.1, 0.6, 1.0, 1.0, 1.0],
              [0.0, 0.0, 0.3, 0.9, 1.0, 1.0]])

    fig = go.Figure()

    fig.add_surface(x=ps_data, y=lambda_data, z=z_data, colorscale=['yellow', 'blue'])




    # line_marker = dict(color='black', width=4)'Blues_r'
    # fig.add_scatter3d(x=ps_data, y=lambda_data, z=z_data + 0.03, mode='lines', line=line_marker, name='')

    # X = np.linspace(-1, 1, 6)
    # Y = np.linspace(-1, 3, 12)
    # # Define the first family of coordinate lines
    # X, Y = np.meshgrid(X, Y)
    # Z = X ** 3 - 3 * Y * X + 1
    # line_marker = dict(color='#101010', width=4)
    # for xx, yy, zz in zip(X, Y, Z + 0.03):
    #     fig.add_scatter3d(x=xx, y=yy, z=zz, mode='lines', line=line_marker, name='')
    # # Define the second family of coordinate lines
    # Y, X = np.meshgrid(Y, X)
    # Z = X ** 3 - 3 * Y * X + 1
    # line_marker = dict(color='#101010', width=4)
    # for xx, yy, zz in zip(X, Y, Z + 0.03):
    #     fig.add_scatter3d(x=xx, y=yy, z=zz, mode='lines', line=line_marker, name='')
    # fig.update_layout(width=700, height=700, showlegend=False)
    # fig.show()


    # fig = go.Figure(data=[go.Surface(z=z_data, x=ps_data, y=lambda_data)])

    fig.update_layout(xaxis=dict(title='Mean of Starting Propensities to Steal'), yaxis=dict(title='Bribe (as Proportion of Fine)'))
    fig.update_layout(scene=dict(
                                 # xaxis_title='Prop. Steal',
                                 # yaxis_title='Lambda',
                                 # zaxis_title='Proportion',
                                 yaxis=dict(title=dict(text='Mean Prop. Steal', font=dict(size=25)), tickfont=dict(size=15)),
                                 xaxis=dict(title=dict(text='Lambda', font=dict(size=25)), tickfont=dict(size=15)),
                                 zaxis=dict(title=dict(text='Success (Proportion)', font=dict(size=25)), tickfont=dict(size=15))),
                                 autosize=False,
                                 width=1250, height=1250)

    # fig.update_xaxes(tickfont=dict(size=30))

    save_folder = '/Users/user/Documents/SugarSync_Shared_Folders/ICSS_research/Research/Thesis/graphics/three_d_data_file'
    file_name = 'three_d_data'
    file_name_fig = "%s/%s_%d - %d - %d - %d - %d - %d - %d.html" % (save_folder, file_name, dt.datetime.now().year,
                                                                       dt.datetime.now().month, dt.datetime.now().day,
                                                                       dt.datetime.now().hour, dt.datetime.now().minute,
                                                                       dt.datetime.now().second,
                                                                       dt.datetime.now().microsecond / 100000)

    plotly.offline.plot(fig, filename=file_name_fig, auto_open=False)

    # fig.show()


def create_plotly_2d_scatter(x_data, y_data, title, x_axis_label, y_axis_label, dot_size, color, line_x=None, line_y=None, data_folder=None, filename=None, keep_name=0):

    data = []

    data.append(go.Scatter(
                      x=x_data,
                      y=y_data,
                      mode='markers',
                      marker_size=dot_size,
                      marker=dict(
                                  color=color
                                  )
                      )
    )

    if line_x is not None:
        data.append(go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            marker_size=2,
            marker=dict(
                color=color,
                line=dict(
                    color='blue',
                    width=2
                    )
            )
        ))

    fig = go.Figure(data=data)

    fig.update_layout(
                      title=title,
                      xaxis_title=x_axis_label,
                      yaxis_title=y_axis_label,
                      showlegend=False
                      )

    if keep_name:

        file_name_fig = "%s/%s.html" % (data_folder, filename)

    else:

        file_name_fig = "%s/%s_%d - %d - %d - %d - %d - %d - %d.html" % (data_folder, filename, dt.datetime.now().year,
                                                                         dt.datetime.now().month, dt.datetime.now().day,
                                                                         dt.datetime.now().hour, dt.datetime.now().minute,
                                                                         dt.datetime.now().second,
                                                                         dt.datetime.now().microsecond / 100000)



    plotly.offline.plot(fig, filename=file_name_fig, auto_open=False)


def add_time_stamp_string(years=1, months=1, days=1, hours=1, minutes=1, seconds=1, microseconds=1):

    time_stamp = ''

    if years:
        time_stamp += '%d - ' % dt.datetime.now().year

    if months:
        time_stamp += '%d - ' % dt.datetime.now().month

    if days:
        time_stamp += '%d - ' % dt.datetime.now().day

    if hours:
        time_stamp += '%d - ' % dt.datetime.now().hour

    if minutes:
        time_stamp += '%d - ' % dt.datetime.now().minute

    if seconds:
        time_stamp += '%d - ' % dt.datetime.now().second

    if microseconds:
        time_stamp += '%d' % int(dt.datetime.now().microsecond/100000)

    return time_stamp


# def add_time_stamp_difference_string(years=0, months=0, days=0, hours=0, minutes=0, seconds=0, microseconds=0):
#
#     time_stamp = ''
#
#     if years:
#         time_stamp += '%d - ' % dt.datetime.now().year
#
#     if months:
#         time_stamp += '%d - ' % dt.datetime.now().month
#
#     if days:
#         time_stamp += '%d - ' % dt.datetime.now().day
#
#     if hours:
#         time_stamp += '%d - ' % dt.datetime.now().hour
#
#     if minutes:
#         time_stamp += '%d - ' % dt.datetime.now().minute
#
#     if seconds:
#         time_stamp += '%d - ' % dt.datetime.now().second
#
#     if microseconds:
#         time_stamp += '%d' % int(dt.datetime.now().microsecond/100000)
#
#     return time_stamp


def props_speed_test(num_runs=10000000, lb=-0.5, ub=-0.4):

    print('\n test: lb =', lb, 'ub', ub)

    start_time = dt.datetime.now()

    for i in range(num_runs):

        p_s = random.uniform(lb, ub)

        # truncate p_s
        if p_s < 0.0:
            p_s = 0.0

        elif p_s > 1.0:
            p_s = 1.0

        # now process using random decimal number (between 0 and 1)
        if random.random() < p_s:
            agent_decision = 'steal'
        else:
            agent_decision = 'trade'

    end_time = dt.datetime.now()

    print(' first test total time taken =', end_time - start_time)

    first_total_time = end_time - start_time

    start_time = dt.datetime.now()

    for i in range(num_runs):

        p_s = random.uniform(lb, ub)

        # for negative of zero p_s (agent always traded)
        if p_s <= 0.0:
            agent_decision = 'trade'

        # for p_s >= 1 (agent always stole)
        elif p_s >= 1.0:
            agent_decision = 'steal'

        # for 0 < p_s < 1 (the agent might trade or steal)
        else:

            if random.random() < p_s:
                agent_decision = 'steal'
            else:
                agent_decision = 'trade'

    end_time = dt.datetime.now()

    second_total_time = end_time - start_time

    print(' second test total time taken =', end_time - start_time)
    print(' time reduced by % 2.3f' % float(100 * (first_total_time - second_total_time) / first_total_time), 'pct')


def run_speed_test(num_runs = 100000000):

    props_speed_test(num_runs=num_runs, lb=0.0, ub=1.0)
    props_speed_test(num_runs=num_runs, lb=1.0, ub=2.0)
    props_speed_test(num_runs=num_runs, lb=-2.0, ub=-1.0)


# def RW_speed_test(num_mem_entries=3):

def run_roulette_wheel(weights_dict):

    if len(weights_dict) == 0:

        target_location = 'random'

    else:

        total_weight = sum(weights_dict.values())

        random_weight = random.uniform(0, total_weight)

        location_not_found = 1
        loop_counter = 0
        aggr_weight = 0.0

        locations_in_memory = list(weights_dict.keys())

        while location_not_found:

            loc = locations_in_memory[loop_counter]

            aggr_weight += weights_dict[loc]

            if random_weight <= aggr_weight:

                location_not_found = 0
                target_location = loc

            else:
                loop_counter += 1

    return target_location


def create_memories(num_mem_entries):

    weights_dict = dict()

    for i in range(num_mem_entries):

        loc = '[%d, %d]' % (random.randint(0, 50), random.randint(0, 50))
        weight = random.uniform(0, 5.0)
        weights_dict[loc] = weight

    return weights_dict


def decision_code(tot_sims=10000000, num_mem_locs=2):

    weights_dict = create_memories(num_mem_locs)

    print('\n weights_dict =', weights_dict)

    start_time = dt.datetime.now()

    for i in range(tot_sims):

        # if len(weights_dict) == 0:
        #     target = 'random'
        #
        # elif len(weights_dict) == 1:
        #     target = list(weights_dict.keys())[0]
        #
        # else:
        #     target = run_roulette_wheel(weights_dict)

        target = run_roulette_wheel(weights_dict)

        # print(' target =', target)

    end_time = dt.datetime.now()

    total_time = end_time - start_time

    print('\n timings: num_mem_locs', num_mem_locs, 'tot_sims ', tot_sims, 'total_time =', total_time)



# def play():
#
#     x = np.linspace(-1, 1, 50)
#     y = np.linspace(-1, 3, 100)
#     x, y = np.meshgrid(x, y)
#
#     z = x ** 3 - 3 * y * x + 1  # the surface eqn
#
#     print('\n x =', x)
#     print('\n y =', y)
#     print('\n z =', z)
#
#     fig = go.Figure()
#     fig.add_surface(x=x, y=y, z=z, colorscale='Reds_r', colorbar_thickness=25, colorbar_len=0.75);
#
#     # fig.show()
#
#     X = np.linspace(-1, 1, 6)
#     Y = np.linspace(-1, 3, 12)
#
#     # Define the first family of coordinate lines
#     X, Y = np.meshgrid(X, Y)
#     Z = X ** 3 - 3 * Y * X + 1
#
#     print('\n X =', X)
#     print('\n Y =', Y)
#     print('\n Z =', Z)
#
#     line_marker = dict(color='#101010', width=4)
#     for xx, yy, zz in zip(X, Y, Z + 0.1):
#         fig.add_scatter3d(x=xx, y=yy, z=zz, mode='lines', line=line_marker, name='')
#
#     fig.show()
#
#     # Define the second family of coordinate lines
#     Y, X = np.meshgrid(Y, X)
#     Z = X ** 3 - 3 * Y * X + 1
#     line_marker = dict(color='#101010', width=4)
#     for xx, yy, zz in zip(X, Y, Z + 0.1):
#         fig.add_scatter3d(x=xx, y=yy, z=zz, mode='lines', line=line_marker, name='')
#     fig.update_layout(width=700, height=700, showlegend=False)
#     fig.show()
