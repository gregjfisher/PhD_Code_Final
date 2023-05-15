#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:51:47 2018

@author: Greg Fisher

This is Greg's library
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import smtplib
import os
import math
#from pandas import rolling_mean as pd_rolling_mean
import pandas
#from pandas import pandas.rolling as pd_rolling_mean
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate     # COMMASPACE  
from email import encoders


def create_supply_demand_charts(num_res_founts, supply_demand_array, fountain_population, agent_population, print_dets, print_fine_dets, run_folder, day, dbs, rounds, trade_moves, SD_charts_freq, daily_succ_trans, pop):
    """This function organises the printing of the supply and demand charts for all of the resource combinations."""

    #    if day > 20:
    #        print_fine_dets = 1

    if print_fine_dets == 1:
        print('\n\n supply_demand_array[0]:\n\n', supply_demand_array[0])
        print('\n\n supply_demand_array[1]:\n\n', supply_demand_array[1])

    # Now for Supply & Demand chart for res_0 versus res_1: we only do this if there are 2 resources
    res_1 = 0
    res_2 = 1

    # Now we're ready to send the data to print_chart
    labels_array = ['Supply', 'Demand']
    y_axis_label = 'Price'
    line_width = 2
    colors = ['blue', 'black', 'green', 'aqua', 'teal', 'navy', 'fuchsia', 'purple']

    if pop == 'all':

        weighted_mean_traded_price = dbs.mean_price_history[res_1][res_2][day]
        net_supply = dbs.net_net_transs_db[day][res_1]

        title = 'Day %s - Res %s vs Res %s: Supply and Demand' % (day, res_1, res_2)
        filename = 'Day %s - Supply & Demand: Res %s and Res %s' % (day, res_1, res_2)

    elif pop == 'sharks':

        weighted_mean_traded_price = dbs.mean_price_history_sharks[res_1][res_2][day]
        net_supply = dbs.net_net_transs_db_sharks[day][res_1]

        title = 'Day %s - Res %s vs Res %s: Supply and Demand (Sharks)' % (day, res_1, res_2)
        filename = 'Day %s - Supply & Demand (Sharks): Res %s and Res %s' % (day, res_1, res_2)

    elif pop == 'jets':

        weighted_mean_traded_price = dbs.mean_price_history_jets[res_1][res_2][day]
        net_supply = dbs.net_net_transs_db_jets[day][res_1]

        title = 'Day %s - Res %s vs Res %s: Supply and Demand (Jets)' % (day, res_1, res_2)
        filename = 'Day %s - Supply & Demand (Jets): Res %s and Res %s' % (day, res_1, res_2)

    print_SD_chart(supply_demand_array[res_1], [weighted_mean_traded_price, net_supply], labels_array, title, y_axis_label, line_width, colors, run_folder, filename, print_fine_dets, show_trans=0, show_text=0, dpi='low')

    if print_fine_dets == 1:
        print('\n\ndbs.optimal_bskt_turnover[day] =', dbs.optimal_bskt_turnover[day])
        print('dbs.net_net_transs_db[day] =', dbs.net_net_transs_db[day])

        print('dbs.net_turnover_prop[round] =', dbs.net_turnover_prop[day])

    #    # Here we print the daily scatter diagrams of agents' MRSs
    #    MRS_scatter_db_start = []
    #    MRS_scatter_db_end = []
    #
    #    for agent in agent_population.pop:
    #
    #        MRS_scatter_db_start.append(agent.MRS_history[day][res_1][res_2])
    #        MRS_scatter_db_end.append(agent.MRS_array[res_1][res_2])
    #
    #    labels = ['Start of round', 'End of round']
    #    title = 'Day %s - All Agent MRSs: Res %s vs Res %s' % (day, res_1, res_2)
    #    y_axis_label = 'MRS'
    #    line_width = 2
    #    colors = ['black']
    #    filename = 'Day %s - MRS_scatter_%s_vs_%s' % (day, res_1, res_2)
    #
    #    print_scatter_1d(MRS_scatter_db_start, MRS_scatter_db_end, rounds, agent_population, title, y_axis_label, line_width, colors, run_folder, filename, labels, dbs, dpi='low')

    # Here we print the daily scatter diagrams of agents' MRSs
    if pop == 'all' or pop == 'jets':  # ensures we do this just once

        database = dbs.MRS_moves_array[res_1][res_2]

        #    print('\n dbs.MRS_moves_array =\n\n', dbs.MRS_moves_array)

        #                print '\ndatabase =\n', database

        title = 'Day %s - Trading Agents MRSs During Trading Moves: Res %s vs Res %s' % (day, res_1, res_2)
        y_axis_label = 'MRS'
        line_width = 1
        colors = ['black']
        labels = ['Start of round', 'End of round']
        filename = 'Day %s - moves_MRS_scatter_%s_vs_%s' % (day, res_1, res_2)

        if day == rounds - 1:

            print_scatter_1d_MRS_moves(database, rounds, agent_population, '', y_axis_label, line_width, colors, run_folder, filename, labels, dbs, trade_moves, dpi='high')

        else:

            print_scatter_1d_MRS_moves(database, rounds, agent_population, title, y_axis_label, line_width, colors, run_folder, filename, labels, dbs, trade_moves, dpi='low')

        # create a histogram of the distribution of resources in agents' resource arrays

        colors = ['blue', 'red', 'green', 'black', 'aqua', 'teal', 'navy', 'fuchsia', 'purple']

        for res in np.arange(num_res_founts):

            reserves_array = []

            for agent in agent_population.pop:
                res_level = agent.agent_res_array[0][res]

                reserves_array.append(res_level)

            title = 'Day %s - Histogram of Resource %s Holdings' % (day, res)
            color = colors[res]
            filename = 'Day %s - histogram of Resource %s holdings (all agents)' % (day, res)

            print_histogram(reserves_array, title, run_folder, filename, color)


#    if print_fine_dets == 1:
#
#        input("Press Enter to continue...")


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
    plt.show()

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
    plt.show()

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

    print('---> printing chart: %s ' % (title))

#    print('\n database\n\n', database)

    x_axis = np.zeros(shape=(len(dbs.agent_list)))

    y_range = np.max(database) - np.min(database)

    max_y = np.max(database) + (y_range * 0.05)
    min_y = np.min(database) - (y_range * 0.05)

    min_x = -0.5
    max_x = trade_moves + 1 - 0.5

    # now build the chart:
    plt.figure()
    plt.title("%s" % (title))
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
    plt.show()

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
    plt.show()

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
    plt.close()


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
    #    print '\n chart function:'
    #    print '\nsearch_points_0_2 =\n', search_points_0_2
    #    print '\nsearch_points_1_2 =\n', search_points_1_2
    #    print '\n np.transpose(0_2_surface) =\n', np.transpose(surf_1)
    #    print '\n np.transpose(1_2_surface) =\n', np.transpose(surf_2)
    #    print '\n prices_array_0_2 =\n', prices_array_0_2
    #    print '\n prices_array_1_2 =\n', prices_array_1_2

    #    xGrid, yGrid = np.meshgrid(prices_array_0_2, prices_array_1_2)
    #
    #    print '\n xGrid =\n', xGrid
    #    print '\n yGrid =\n', yGrid

    #    # To zoom in on the square where we know both furface contain the zero contour, we must split the grid in to its price squares and find the four price points the double zero contour point is
    #
    #    find_best_box(surf_1, surf_2, prices_array_0_2, prices_array_1_2, print_fine_dets=0)

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
    #    print '\n chart function:\n\nsearch_points =\n', search_points
    #    print '\n np.transpose(three_d_data) =\n', np.transpose(three_d_data)
    #    print '\n prices_array_0_2 =\n', prices_array_0_2
    #    print '\n prices_array_1_2 =\n', prices_array_1_2

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


#    py.iplot([
#        dict(z=aggr_three_d_data, showscale=False, type='surface')],
#        filename='python_docs/SD_day_%d_res_%d_res_%d_day_%d - %d - %d - %d - %d - %d - %d'
#                        % (day, res_1, res_2, dt.datetime.now().year,
#                        dt.datetime.now().month, dt.datetime.now().day,
#                        dt.datetime.now().hour, dt.datetime.now().minute,
#                        dt.datetime.now().second,
#                        dt.datetime.now().microsecond / 100000))

#    raw_input("Press Enter to continue...")


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

        plt.show()

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

        plt.show()

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

            plt.show()

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

                plt.show()

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

    # we have to decide where we want to save the chart and its default name:
    filepath = data_folder

    plt.pcolor(database, cmap=color)        # , cmap=plt.cm.Blues         ,edgecolors='k'
    plt.colorbar()
    plt.title(title)
#    plt.invert_xaxis()
#    plt.invert_yaxis()

    plt.show()
    plt.axis([0, dimen, 0, dimen])
    plt.gca().invert_yaxis()

#    plt.gca().xaxis.tick_top()

    # if we want to save the chart:
    if dpi == 'low':

        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (filepath, filename, dt.datetime.now().year,
                    dt.datetime.now().month, dt.datetime.now().day,
                    dt.datetime.now().hour, dt.datetime.now().minute,
                    dt.datetime.now().second,
                    dt.datetime.now().microsecond / 100000))

    elif dpi == 'high':

        plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                    % (filepath, filename, dt.datetime.now().year,
                    dt.datetime.now().month, dt.datetime.now().day,
                    dt.datetime.now().hour, dt.datetime.now().minute,
                    dt.datetime.now().second,
                    dt.datetime.now().microsecond / 100000), dpi=500)

    plt.close()


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

    plt.show()
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

    plt.close()


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

    plt.close()


def within_striking_dist(wait_at_target_til_end, town_grid, location, wait_at_tgt_moves, agent_vision, poss_tgt_location, move, has_acted, print_dets):

    x_loc, y_loc = poss_tgt_location

    # We set wait_at_tgt_moves = 0 when the agent waits at its target until the end, otherwise we use whatever is given as a value
    if wait_at_target_til_end:
        wait_at_tgt_moves = 0

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

            print('\n town_grid.trade_moves ', town_grid.trade_moves, 'move ', move, 'wait_at_tgt_moves ', wait_at_tgt_moves, 'agent_vision', agent_vision)

            pause()

    if print_dets == 1:
        print('\nmax_movement =', max_movement)

    #    print('x_loc', x_loc, 'location', location)

    x_dist = math.fabs(x_loc - location[0])

    if x_dist > town_grid.dimen / 2.0:
        x_dist = town_grid.dimen - x_dist

    y_dist = math.fabs(y_loc - location[1])

    if y_dist > town_grid.dimen / 2.0:
        y_dist = town_grid.dimen - y_dist

    if print_dets == 1:
        print('\nx_dist =', x_dist)
        print('y_dist =', y_dist)

    if x_dist <= max_movement and y_dist <= max_movement:

        if print_dets == 1:
            print('\n=> location is within striking distance')
            print('\n--> within_striking_dist function ends <--\n')

        return 1

    else:

        if print_dets == 1:
            print('\n=> location is NOT within striking distance')
            print('\n--> within_striking_dist function ends <--\n')

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

        print('\nnet_turnover_trans =\n', net_turnover_trans)
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

    moving_averages_array = np.zeros(shape=(2, len(means_array[0])), dtype=np.float64)
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
    y_axis_label = 'Ratio'
    line_width = 2
    colors = ['blue', 'green', 'aqua', 'black', 'teal', 'navy', 'fuchsia', 'purple']
    filename = 'turnover_time_series_mean'

    print_chart(net_turnover_db, labels_array, title, y_axis_label, line_width, colors, data_folder, filename, data_type='-', dpi='low')

    print_chart(means_array, ['Mean'], '', y_axis_label, 3, ['black'], data_folder, filename, data_type='-', dpi='low')

    print_chart(moving_averages_array, [''], '', y_axis_label, 2, ['black'], data_folder, filename, data_type='-', dpi='high')



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

        labels_array = ['actual turnover', 'optimal turnover']

        title = 'Turnover of Resources: Actual and Optimal'
        y_axis_label = 'Num'
        line_width = 2
        colors = ['blue', 'green', 'aqua', 'black', 'teal', 'navy', 'fuchsia', 'purple']
        filename = 'turnover_res_time_series'

        print_chart(turnover_db, labels_array, title, y_axis_label, line_width, colors, data_folder, filename, data_type='-', dpi='low')


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
    plt.show()

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


def generate_MA_array(array, N):

    """This function takes a 1d array and returns a version of the array with moving averages (window of N).  It ensures the first data are the closest to the MA we want by calculating the MA for up to N."""

    # if we import a list, convert it to numpy array
    if type(array) == list:

        array = np.array(array)

    # .Series(x).rolling(window=2).mean()
    # MA_array = pandas.rolling_mean(array, N)
    # MA_array = pandas.Series(array).rolling(window=N).mean()

    MA_array = np.zeros(shape=(len(array)))

    for day in range(0, len(array)):

        start_point = np.max([0, day - N + 1])

        MA_array[day] = np.mean(array[start_point:day + 1])

    return MA_array


def length_of_time(delta):

    """"This function takes a time delata (from datetime) and returns values in years, days, hours, minutes, and seconds.  Note years are assumed to be 365 days and microseconds are added to seconds."""

    years = delta.days // 365
    days = delta.days % 365
    hours = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60
    seconds = (delta.seconds % 3600 % 60) + (delta.microseconds / 1000000.0)

    return years, days, hours, minutes, seconds


def print_chart(database, labels_array, title, y_axis_label, line_width, colors, data_folder, filename, data_type, dpi):

    """A function for printing a chart with multiple lines."""

    print('---> printing chart: %s ' % (title))

    # we have to decide where we want to save the chart and its default name:
    filepath = data_folder

    # number of lines in the chart will probably be this (i.e. excluding the
    # round data)
    num_lines = len(database) - 1

    # set up x_min and x_max (likely to be just start & end of rounds):
    x_min = database[0][0]
    x_max = database[0][-1]

#    print('\n database[1:] =\n', database[1:])

    # note 'y' corresponds to the line data:
    y_min = np.min(database[1:])
    y_max = np.max(database[1:])

    # create 5% buffers above and below lines:
    # create 5% buffers above and below lines:
    if y_min > 0:
        y_min = y_min * 0.90
    elif y_min <= 0:   # i.e. y_min is negative
        y_min = y_min * 1.1

    if math.isnan(y_min) or y_min == np.inf:
        y_min = 0.0

    if y_max > 0:
        y_max = y_max * 1.1
    elif y_max <= 0:   # i.e. y_max is negative
        y_max = y_max * 0.90

    if math.isnan(y_max) or y_max == np.inf:
        y_max = 5.0

    if labels_array[0] == 'Probability Threshold':

        y_min = 0
        y_max = 1.4

    # Override y_min and y_max is it's this chart (sometimes we get a v high number, which is wrong)
    if title == 'Turnover of Resources as a proportion of Gen Equ turnover':

        y_min = 0.0
        y_max = 1.5

    if math.isnan(y_min) or y_min == np.inf or math.isnan(y_max) or y_max == np.inf:

        print('\n PROBLEM')
        print('y_min =', y_min)
        print(' y_min == np.nan', y_min == np.nan, ' y_min == np.inf', y_min == np.inf)
        print('y_max =', y_max)
        print(' y_max == np.nan', y_max == np.nan, ' y_max == np.inf', y_max == np.inf)
        pause()

    # now build the chart:
    plt.figure()
    plt.title("%s" % (title))
    plt.axis("auto")
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.xlabel("Round")
    plt.ylabel("%s" % (y_axis_label))
        
    if len(labels_array) > 0:

        for i in range(num_lines):

            col_num = i % len(colors)
            if len(labels_array) > 0:

                label="%s" % (labels_array[i])

            else:

                label=''
                
            plt.plot(database[0],
                     database[i + 1], data_type,
                     color=colors[col_num],
                     label=label,
                     linewidth=line_width,
                     )

    else:

        for i in range(num_lines):
            plt.plot(database[0],
                     database[i + 1], data_type,
                     color=colors[i],
                     linewidth=line_width)

    if labels_array[0] == 'Probability Threshold':

#        print('\n y_min =', y_min)
#        print('\n y_max =', y_max)
        plt.legend(loc=0, fontsize='medium')

    else:

        plt.legend(loc=0, fontsize='small')

    # if we want to show the chart:
    plt.show()

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

    # if we want to close the chart:
    plt.close()


def print_chart_prices(database, labels_array, title, y_axis_label, line_width, colors, data_folder, filename, data_type, dpi):

    """A function for printing a chart with multiple lines."""

    print('---> printing chart: %s ' % (title))

    # we have to decide where we want to save the chart and its default name:
    filepath = data_folder

    # number of lines in the chart will probably be this (i.e. excluding the
    # round data)
    num_lines = len(database) - 1

    # set up x_min and x_max (likely to be just start & end of rounds):
    x_min = database[0][0]
    x_max = database[0][-1]

    # note 'y' corresponds to the line data:
    y_min = np.min(database[1:])
    y_max = np.max(database[1:])

#    y_min = 0
#    y_max = 3

    # create 5% buffers above and below lines:
    if y_min > 0:
        y_min = y_min * 0.90
    elif y_min <= 0:   # i.e. y_min is negative
        y_min = y_min * 1.1

    if math.isnan(y_min) or y_min == np.inf:
        y_min = 0.0

    if y_max > 0:
        y_max = y_max * 1.1
    elif y_max <= 0:   # i.e. y_max is negative
        y_max = y_max * 0.90

    if math.isnan(y_max) or y_max == np.inf:
        y_max = 5.0

    if math.isnan(y_min) or y_min == np.inf or math.isnan(y_max) or y_max == np.inf:

        print('\n PROBLEM')
        print('y_min =', y_min)
        print(' y_min == np.nan', y_min == np.nan, ' y_min == np.inf', y_min == np.inf)
        print('y_max =', y_max)
        print(' y_max == np.nan', y_max == np.nan, ' y_max == np.inf', y_max == np.inf)
        pause()

    # Override y_min and y_max is it's this chart (sometimes we get a v high number, which is wrong)
    if title == 'Turnover of Resources as a proportion of Gen Equ turnover':

        y_min = 0.0
        y_max = 1.5

    # now build the chart:
    plt.figure()
    plt.title("%s" % (title))
    plt.axis("auto")
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.xlabel("Round")
    plt.ylabel("%s" % (y_axis_label))

    for i in range(num_lines):
        plt.plot(database[0],
                 database[i + 1], data_type,
                 label="%s" % (labels_array[i]),
                 color=colors[i],
                 linewidth=line_width)

    if dpi != 'high':

        plt.legend(loc=0, fontsize='small')

    # if we want to show the chart:
    plt.show()

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

    # if we want to close the chart:
    plt.close()


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
    plt.show()

    # if we want to save the chart:
    plt.savefig("%s/%s %d - %d - %d - %d - %d - %d - %d.png"
                % (filepath, filename, dt.datetime.now().year,
                dt.datetime.now().month, dt.datetime.now().day,
                dt.datetime.now().hour, dt.datetime.now().minute,
                dt.datetime.now().second,
                dt.datetime.now().microsecond / 100000))

    # if we want to close the chart:
    plt.close()


def print_SD_chart(supply_demand_array_1_res, traded_data, labels_array, title, y_axis_label, line_width, colors, data_folder, filename, print_fine_dets, show_trans, show_text, dpi):

    """A function for printing a Supply & Demand Curves chart with multiple lines."""

    weighted_mean_traded_price, net_supply = traded_data

    print('---> printing chart: %s ' % (title))

#    print('\n supply_demand_array:\n\n', supply_demand_array_1_res)

    # we have to decide where we want to save the chart and its default name:
    filepath = data_folder

    # number of lines in the chart will probably be this (i.e. excluding the
    # round data)
    num_lines = len(supply_demand_array_1_res) - 1

    # set up x_min and x_max (likely to be just start & end of rounds):
    x_min = np.min([np.min(supply_demand_array_1_res[1:]), -0.5])
    x_max = np.max(supply_demand_array_1_res[1:]) * 1.05

    # note 'y' corresponds to the line data:
    y_min = supply_demand_array_1_res[0][0] * 0.99
    y_max = supply_demand_array_1_res[0][-1] * 1.01

    if print_fine_dets == 1:
        print('pre y_max =', y_max)
        print('pre y_min =', y_min)

    # now build the chart:
    plt.figure()
    plt.title("%s" % (title))
    plt.axis("auto")
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.xlabel("Quantity Supplied or Demanded")
    plt.ylabel("%s" % (y_axis_label))

    plt.gca().invert_yaxis()

    for i in range(num_lines):
        plt.plot(supply_demand_array_1_res[i + 1],
                 supply_demand_array_1_res[0], '-',
                 label="%s" % (labels_array[i]),
                 color=colors[i],
                 linewidth=line_width)

    if print_fine_dets == 1:
        print('net_supply', net_supply, 'weighted_mean_traded_price', weighted_mean_traded_price)

    # add actual transaction data
    plt.scatter(net_supply, weighted_mean_traded_price, label='est net trans', marker='*', color='red', s=130)

#    label_xcoord1 = x_min + (x_max - x_min) * 10 / 100.
#    label_ycoord1 = y_min + (y_max - y_min) * 85 / 100.
#    label_ycoord2 = y_min + (y_max - y_min) * 90 / 100.
#
#    if show_text == 1:
#        plt.text(label_xcoord1, label_ycoord1, "Q trans = % 3.2f  |  Q equil = % 3.2f  |  percent =  % 2.1f" % (trans_Q, bilat_equ_data[1], bilat_equ_data[4]), fontsize=8)
#        plt.text(label_xcoord1, label_ycoord2, "P trans = % 2.3f  |  P equil = % 2.3f" % (trans_p, bilat_equ_data[0]), fontsize=8)

    plt.gca().invert_yaxis()
#    plt.legend(loc='upper center', fontsize='small')

    # if we want to show the chart:
    plt.show()

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

    # if we want to close the chart:
    plt.close()


def send_mail(send_from, send_to, subject, text, files=[], server="localhost", port=587, username='', password='', isTls=True):

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
