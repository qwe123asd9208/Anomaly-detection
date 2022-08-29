# 专业：计算机科学与技术
# author: Yixian Luo

# import Keras libraries
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.regularizers import l2

import torch
import torch.nn as nn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
# fix random seed for reproducibility

np.random.seed(5)

def model_dnn(num_of_feature_with_window,W,num_of_feature_with_value):
    """
    The dropout machanism is expected to enhance the generalization capability of the model,
    but it takes more epochs to train and, if not trained for more epochs, may lead to degraded performance.
    # create ANN model
    model = Sequential()
    model.add(Dense(256, input_dim=num_of_feature_with_window*len(W) + num_of_feature_with_value, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    """

    # we can think of this chunk as the input layer
    model = Sequential()
    model.add(Dense(1024, input_dim=num_of_feature_with_window*len(W) + num_of_feature_with_value,
                   bias_regularizer=l2(0.01),
                   kernel_regularizer=l2(0.01)))
    #model.add(Dense(1024, input_dim=num_of_feature_with_window * len(W) + num_of_feature_with_value))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # we can think of this chunk as the hidden layer
    model.add(Dense(256,
                   bias_regularizer=l2(0.01),
                   kernel_regularizer=l2(0.01)))
    #model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # we can think of this chunk as the output layer
    # model.add(Dense(1,
    #                bias_regularizer=l2(0.01),
    #                kernel_regularizer=l2(0.01)))
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.summary()

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

num_of_feature_with_value = 9
num_of_feature_with_window = 12
W = np.asarray([2, 5, 10, 25, 50, 100, 200, 300, 400, 500])
model_dnn(num_of_feature_with_window,W,num_of_feature_with_value)
# input_dim=num_of_feature_with_window * len(W) + num_of_feature_with_value
#
# class Net1(nn.Module):
#     def __init__(self):
#         super(Net1, self).__init__()
#         self.dnn = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(1024, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 1),
#             nn.BatchNorm1d(1),
#
#             # nn.Conv1d(1, 32, kernel_size=8, padding=3, stride=1),
#             # nn.ReLU(),
#             # nn.BatchNorm1d(32),
#             # nn.Dropout(0.3),
#             # nn.MaxPool1d(2),
#             # nn.Conv1d(32, 64, kernel_size=4, padding=3, stride=1),
#             # nn.ReLU(),
#             # nn.BatchNorm1d(64),
#             # nn.Dropout(0.3),
#             # nn.MaxPool1d(2),
#             # nn.Conv1d(64, 128, kernel_size=3, padding=3, stride=1),
#             # nn.ReLU(),
#             # nn.BatchNorm1d(128),
#             # nn.MaxPool1d(2),
#             # nn.Conv1d(128, 64, kernel_size=2, padding=3, stride=1),
#             # nn.ReLU(),
#             # nn.BatchNorm1d(64),
#             # nn.MaxPool1d(2),
#             # nn.Flatten()
#         )
#         # self.rnn = nn.GRU(input_size=500, hidden_size=500, num_layers=1, batch_first=True, bidirectional=False, dropout=0)
#         # self.fc1 = nn.Linear(1408, 64)
#         # self.fc2 = nn.Linear(64,7)
#         #self.fc2 = nn.Linear(7)
#         #self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, input):
#         # for t in range(cfg.MAX_LENGTH):
#         #     if t == 0:
#         prev_out = self.dnn(input)
#         #prev_out = prev_out.view(-1,64)
#         # prev_out = self.fc1(prev_out)
#         # prev_out = self.fc2(prev_out)
#         prev_out = self.softmax(prev_out)
#         out = torch.clone(prev_out)
#         return out