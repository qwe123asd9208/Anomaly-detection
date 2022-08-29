# 专业：计算机科学与技术
# author: Yixian Luo

import os
import numpy as np
import pandas as pd
import sklearn
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import maxabs_scale
from statsmodels.tsa.api import SARIMAX, ExponentialSmoothing, SimpleExpSmoothing, Holt

import random

print(os.listdir("./input"))
W = np.asarray([2, 5, 10, 25, 50, 100, 200, 300, 400, 500])
delay = 7

def show_dataset(dataframe):
    print(dataframe.columns, '\n')
    print(dataframe.describe(), '\n')
    print(dataframe.head(), '\n')


def get_time_series_from_file(dataframe):
    # time series extraction
    ts_ids, ts_indexes, ts_point_counts = np.unique(dataframe['KPI ID'],
                                                    return_index=True,
                                                    return_counts=True)
    print('Extract are %d time series in the dataframe:' % (len(ts_ids)))

    # extract time series using ts_indexes
    ts_indexes.sort()
    ts_indexes = np.append(ts_indexes, len(dataframe))  # full ranges for extracting time series

    set_of_time_series = []
    set_of_time_series_label = []

    for i in np.arange(len(ts_indexes) - 1):
        print('Extracting %d th time series with index %d and %d (exclusive)'
              % (i, ts_indexes[i], ts_indexes[i + 1]))
        set_of_time_series.append(np.asarray(dataframe['value']
                                             [ts_indexes[i]:ts_indexes[i + 1]]))
        set_of_time_series_label.append(np.asarray(dataframe['label']
                                                   [ts_indexes[i]:ts_indexes[i + 1]]))

    return set_of_time_series, set_of_time_series_label


def plot_time_series_info(set_of_time_series, set_of_time_series_scaled, set_of_time_series_label, index):
    """
    plot a time series from the given set of time series using the given index
    """
    assert (len(set_of_time_series) == len(set_of_time_series_scaled) == len(set_of_time_series_label))

    index_revised = index % len(set_of_time_series)
    ts = set_of_time_series[index_revised]
    ts_scaled = set_of_time_series_scaled[index_revised]
    ts_label = set_of_time_series_label[index_revised]

    plt.subplot(3, 1, 1)
    plt.plot(ts, 'r')
    plt.subplot(3, 1, 2)
    plt.plot(ts_scaled, 'b')
    plt.subplot(3, 1, 3)
    plt.plot(ts_label, 'k')
    plt.show()

def get_feature_logs(time_series):
    return np.log(time_series + 1e-2)

def get_feature_SARIMA_residuals(time_series):
    predict = SARIMAX(time_series,
                      trend='n',
                      order=(5,1,1),
                      measurement_error=True).fit().get_prediction()
    return time_series - predict.predicted_mean

def get_feature_AddES_residuals(time_series):
    predict = ExponentialSmoothing(time_series, trend='add').fit(smoothing_level=1)
    return time_series - predict.fittedvalues

def get_feature_SimpleES_residuals(time_series):
    predict = SimpleExpSmoothing(time_series).fit(smoothing_level=1)
    return time_series - predict.fittedvalues

def get_feature_Holt_residuals(time_series):
    predict = Holt(time_series).fit(smoothing_level=1)
    return time_series - predict.fittedvalues


def get_features_and_labels_from_a_time_series(time_series, time_series_label, Windows, delay):
    """
    Input: time_series, time_series_label, Window, delay (for determining vital data)

    In a time series dataset, it maintains a list of values.
    We'll convert the list of values into a list of feature vectors,
    each feature vector corresponds to a time point in the time series.

    For example: a time series [1,2,3,4,5] --> a featured dataset [[1,2,3],[2,3,4],[3,4,5]] (use one window size 3)

    The labels for the feature vectors are remained and returned.

    time_series: a list of values, an array
    time_series_label: a list of labels, an array
    Windows: the window sizes for time series feature extraction, an array
    delay: the maximum delay for effectively detect an anomaly

    Output: features_for_the_timeseries (a list of arrays),
            labels_for_the_timeseries (a list of arrays),
            vital_labels_for_the_timeseries (a list of arrays)
    """
    data = []
    data_label = []
    data_label_vital = []

    start_point = 2 * max(Windows)
    start_accum = 0

    # features from tsa models
    time_series_SARIMA_residuals = get_feature_SARIMA_residuals(time_series)
    time_series_AddES_residuals = get_feature_AddES_residuals(time_series)
    time_series_SimpleES_residuals = get_feature_SimpleES_residuals(time_series)
    time_Series_Holt_residuals = get_feature_Holt_residuals(time_series)

    # features from tsa models for time series logarithm
    time_series_logs = get_feature_logs(time_series)

    for i in np.arange(start_point, len(time_series)):
        # the datum to put into the data pool
        datum = []
        datum_label = time_series_label[i]

        # fill the datum with f01-f09
        diff_plain = time_series[i] - time_series[i - 1]
        start_accum = start_accum + time_series[i]
        mean_accum = (start_accum) / (i - start_point + 1)

        # f01-f04: residuals
        datum.append(time_series_SARIMA_residuals[i])
        datum.append(time_series_AddES_residuals[i])
        datum.append(time_series_SimpleES_residuals[i])
        datum.append(time_Series_Holt_residuals[i])
        # f05: logarithm
        datum.append(time_series_logs[i])

        # f06: diff
        datum.append(diff_plain)
        # f07: diff percentage
        datum.append(diff_plain / (time_series[i - 1] + 1e-10))  # to avoid 0, plus 1e-10
        # f08: diff of diff - derivative
        datum.append(diff_plain - (time_series[i - 1] - time_series[i - 2]))
        # f09: diff of accumulated mean and current value
        datum.append(time_series[i] - mean_accum)

        # fill the datum with features related to windows
        # loop over different windows size to fill the datum
        for k in Windows:
            mean_w = np.mean(time_series[i - k:i + 1])
            var_w = np.mean((np.asarray(time_series[i - k:i + 1]) - mean_w) ** 2)
            # var_w = np.var(time_series[i-k:i+1])

            mean_w_and_1 = mean_w + (time_series[i - k - 1] - time_series[i]) / (k + 1)
            var_w_and_1 = np.mean((np.asarray(time_series[i - k - 1:i]) - mean_w_and_1) ** 2)
            # mean_w_and_1 = np.mean(time_series[i-k-1:i])
            # var_w_and_1 = np.var(time_series[i-k-1:i])

            mean_2w = np.mean(time_series[i - 2 * k:i - k + 1])
            var_2w = np.mean((np.asarray(time_series[i - 2 * k:i - k + 1]) - mean_2w) ** 2)
            # var_2w = np.var(time_series[i-2*k:i-k+1])

            # diff of sliding windows
            diff_mean_1 = mean_w - mean_w_and_1
            diff_var_1 = var_w - var_w_and_1

            # diff of jumping windows
            diff_mean_w = mean_w - mean_2w
            diff_var_w = var_w - var_2w

            # f1
            datum.append(mean_w)  # [0:2] is [0,1]
            # f2
            datum.append(var_w)
            # f3
            datum.append(diff_mean_1)
            # f4
            datum.append(diff_mean_1 / (mean_w_and_1 + 1e-10))
            # f5
            datum.append(diff_var_1)
            # f6
            datum.append(diff_var_1 / (var_w_and_1 + 1e-10))
            # f7
            datum.append(diff_mean_w)
            # f8
            datum.append(diff_mean_w / (mean_2w + 1e-10))
            # f9
            datum.append(diff_var_w)
            # f10
            datum.append(diff_var_w / (var_2w + 1e-10))

            # diff of sliding/jumping windows and current value
            # f11
            datum.append(time_series[i] - mean_w_and_1)
            # f12
            datum.append(time_series[i] - mean_2w)

        data.append(np.asarray(datum))
        data_label.append(np.asarray(datum_label))

        # an important step is to identify the start anomalous points which are said to be critical
        # if the anomaly is detected within delay window of the occurence of the first anomaly
        if datum_label == 1 and sum(time_series_label[i - delay:i]) < delay:
            data_label_vital.append(np.asarray(1))
        else:
            data_label_vital.append(np.asarray(0))

    return data, data_label, data_label_vital


def get_expanded_featuers_and_labels(data_pool, data_pool_label, data_pool_label_vital, oversample=0):
    assert (len(data_pool) == len(data_pool_label) == len(data_pool_label_vital))

    if oversample == 0:
        return data_pool, data_pool_label

    data_pool_len = len(data_pool)

    # the data points and labels to be appended into the data/label pool
    data_pool_plus = []
    data_pool_plus_label = []
    for i in np.arange(data_pool_len):
        if data_pool_label[i] == 1:  # anomalous point
            data_pool_plus.append(data_pool[i])
            data_pool_plus_label.append(data_pool_label[i])

    # the data points and labels to be appended into the data/label pool (critical ones)
    data_pool_vital = []
    data_pool_vital_label = []
    for i in np.arange(data_pool_len):
        if data_pool_label_vital[i] == 1:  # vital anomalous point
            data_pool_vital.append(data_pool[i])
            data_pool_vital_label.append(data_pool_label_vital[i])

    # oversample abnormal data instances and vital abnormal data instances to balance the dataset
    data_pool_complete = data_pool + \
                         oversample * data_pool_plus + \
                         oversample * data_pool_vital

    data_pool_complete_label = data_pool_label + \
                               oversample * data_pool_plus_label + \
                               oversample * data_pool_vital_label

    assert (len(data_pool_complete) == len(data_pool_complete_label))
    print('The augment size of the dataset: %d = %d + %d * %d + %d * %d' % (len(data_pool_complete),
                                                                            len(data_pool),
                                                                            oversample,
                                                                            len(data_pool_plus),
                                                                            oversample,
                                                                            len(data_pool_vital)))

    # data_pool_complete (X) and data_pool_complete_label (y) should be ready for training
    return data_pool_complete, data_pool_complete_label

def train_dataset_gene(train_time_series_dataset_size,train_time_series_dataset_scaled,train_time_series_dataset_label):
    # 1) feature engineering for training dataset
    # specify the set of window sizes
    # the maximum number is 125 means the start point to consider anomalies is 250, i.e., max(2W).
    W = np.asarray([2, 5, 10, 25, 50, 100, 200, 300, 400, 500])
    delay = 7

    # training: data pool for labeled data points (presented by 6n+2 features)
    train_data_pool = []
    train_data_pool_label = []
    train_data_pool_label_vital = []

    # loop over all the time series
    for i in np.arange(train_time_series_dataset_size):
        # loop over all the data points in each time series
        data, \
        data_label, \
        data_label_vital = get_features_and_labels_from_a_time_series(train_time_series_dataset_scaled[i],
                                                                      train_time_series_dataset_label[i],
                                                                      W, delay)
        train_data_pool = train_data_pool + list(scale(np.asarray(data)))
        # train_data_pool = train_data_pool + list(minmax_scale(abs(np.asarray(data))))
        # train_data_pool = train_data_pool + list(maxabs_scale(np.asarray(data)))

        train_data_pool_label = train_data_pool_label + data_label
        train_data_pool_label_vital = train_data_pool_label_vital + data_label_vital

    # 2) over sampling
    # the methodology to achieve over sampling is to pick samples from train_data according to train_data_label
    # data_pool + data_pool_plus + data_pool_vital, there are three datasets to be merged
    # data_pool_label + data_pool_plus_label + data_pool_vital_label, there are three label datasets to be merged
    train_data_pool_complete, \
    train_data_pool_complete_label = get_expanded_featuers_and_labels(train_data_pool,train_data_pool_label,
                                                                      train_data_pool_label_vital)
    return train_data_pool_complete,train_data_pool_complete_label,train_data_pool_label_vital

def test_dataset_gene(test_time_series_dataset_size,test_time_series_dataset_scaled,test_time_series_dataset_label):
    # feature engineering for testing dataset, a list of sequences

    test_data_pool = []

    # loop over all the time series
    for i in np.arange(test_time_series_dataset_size):
        # loop over all the data points in each time series
        data, \
        data_label, \
        data_label_vital = get_features_and_labels_from_a_time_series(test_time_series_dataset_scaled[i],
                                                                      test_time_series_dataset_label[i],W, delay)

        test_data_pool = test_data_pool + [list(scale(np.asarray(data)))]
        # test_data_pool = test_data_pool + [list(minmax_scale(abs(np.asarray(data))))]
        # test_data_pool = test_data_pool + [list(maxabs_scale(np.asarray(data)))]

    # due to the use of sliding windows, there should has (example):
    print('The length of a time series, e.g., %d, is %d longer than that of its feature vectors, e.g., %d.'
          % (len(test_time_series_dataset_scaled[0]), 2 * max(W), len(test_data_pool[0])))
    return test_data_pool


