import pandas as pd
import numpy as np
import re, os
from string import printable
from sklearn import model_selection
import tensorflow as tf
from keras.models import Sequential, Model, model_from_json, load_model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras.layers.convolutional import Conv1D
from matplotlib import pyplot as plt
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
from texttable import Texttable
from pathlib import Path
import json

import warnings
warnings.filterwarnings("ignore")

def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

#1D convolution with LSTM model
def cnn_lstm(maximum__data_length, embedding_dimension, maximum_vocabulary_length, lstm_output_size):

    model = Sequential()
    model.add(Embedding(input_dim=maximum_vocabulary_length, output_dim=embedding_dimension,
                        input_length=maximum__data_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(lstm_output_size))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',recall, precision, f1])
    print(model.summary())

    return model

def run():

    # Load data URL data

    file = "url_data.csv"

    url_data = pd.read_csv(file)

    # Data Preprocessing

    # Convert characters in url string (contained in printable) into integer
    url_int_tokens = [[printable.index(char) + 1 for char in url if char in printable] for url in url_data.url]

    # limit each url_int_tokens string at a maximum length and pad with zeros if shorter

    maximum__data_length = 100
    x = sequence.pad_sequences(url_int_tokens, maxlen=maximum__data_length)

    # extracting labels from dataframe to numpy array
    y = np.array(url_data.isMalicious)

    print('Matrix dimensions of X: ', x.shape, 'Vector dimension of target: ', y.shape)

    # Cross-Validation - split data set into training and test data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=33)

    #model parameters
    maximum_number_of_epoch = 1
    batch_size = 64
    embedding_dimension = 128
    maximum_vocabulary_length = 100
    lstm_output_size = 32

    model = cnn_lstm(maximum__data_length, embedding_dimension, maximum_vocabulary_length, lstm_output_size)

    model.fit(x_train, y_train, epochs=maximum_number_of_epoch, batch_size=batch_size)

    """--------------------------------------prepare plotting parameters-----------------------------------------"""

    prediction_lstm = model.predict(x_test).ravel()

    false_positive_lstm, true_positive_lstm, threshold_lstm = roc_curve(y_test, prediction_lstm)

    auc_lstm = auc(false_positive_lstm, true_positive_lstm)

    random_forest_classifier = RandomForestClassifier(max_depth=3, n_estimators=10)

    random_forest_classifier.fit(x_train,y_train)

    prediction_rf = random_forest_classifier.predict_log_proba(x_test)[:, 1]

    false_positive_rf, true_positive_rf, threshold_rf = roc_curve(y_test, prediction_rf)

    auc_rf = auc(false_positive_rf, true_positive_rf)

    """--------------------------------------graph plotting-----------------------------------------"""

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(false_positive_lstm, true_positive_lstm, label='CNN_LSTM (area = {:.3f})'.format(auc_lstm))
    plt.plot(false_positive_rf, true_positive_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(false_positive_lstm, true_positive_lstm, label='Keras (area = {:.3f})'.format(auc_lstm))
    plt.plot(false_positive_rf, true_positive_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()

    """--------------------------------------Model Evaluation-----------------------------------------"""

    # Final evaluation of the model
    print("final evaluation of the cnn_lstm model")
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    AUC = round(auc_lstm*100,2)
    Accuracy = round(scores[1]*100,2)
    Recall = round(scores[2]*100,2)
    Precision = round(scores[3]*100,2)
    F1score = round(scores[4]*100,2)

    #tabulating_accuracy_into_table form
    t = Texttable()
    t.add_rows([['AUC','Accuracy','Recall','Precision','F1-score'],[AUC,Accuracy,Recall, Precision,F1score]])
    print(t.draw())

    """--------------------------------------Save Model -----------------------------------------"""

    # serialize model to JSON
    model_json = model.to_json()
    with open("cnn_lstm_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("cnn_lstm_model.h5")
    print("Saved model to disk")

    """--------------------------------------Save Model -----------------------------------------

    #loading model
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    """
run()

