

import os

import numpy as np
from bayes_opt import BayesianOptimization
from syna.dataloader.ck_dataloader import load_CK_emotions
from syna.dataloader.openface_dataloader import load_OpenFace_features
from syna.syna_model import get_temporal_model
from keras.layers import (LSTM, Activation, BatchNormalization, Dense, Dropout, Input)
from keras.models import Sequential, Model
from keras.optimizers import Adam

import io_utils
import train_utils


def load_ck_data(openface_dir, emotion_dir, feature_type='AUs'):
    
    features = load_OpenFace_features(openface_dir, features=feature_type)
    labels = load_CK_emotions(emotion_dir)

    return train_utils.dicts2lists(features, labels)

def get_temporal_model(summary=False, layers=[100], lr=0.001, lr_decay=0.0,
                       input_shape=(None, 4096, )):
   
    input_features = Input(shape=input_shape, name='features')
    input_dropout = Dropout(rate=0.5)(input_features)
    lstm = LSTM(layers[-1], name='lsmt1')(input_dropout)
    output_dropout = Dropout(rate=0.5)(lstm)
    output = Dense(8, activation='softmax', name='fc')(output_dropout)

    model = Model(inputs=input_features, outputs=output)

    adam_opt = Adam(lr=lr, decay=lr_decay)
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

    if summary:
        print(model.summary())

    return model

def main():

    # Load the datasets
    features_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+norm")
    labels_dir = os.path.join(os.path.dirname(__file__), "..", "datasets/ck+")

    for feature_type in ['AUs', '2Dlandmarks']:
        print("Using " + feature_type)

        features, labels = load_ck_data(features_dir, labels_dir, feature_type=feature_type)
        features = train_utils.normalize(features)

        # Normalize length with zero-padding
        #maxlen = 71 # Maximum frames of a record from the Cohn-Kanade dataset
        #x_train = sequence.pad_sequences(x_train, maxlen=maxlen, dtype='float32')
        #x_test = sequence.pad_sequences(x_test, maxlen=maxlen, dtype='float32')

        # Bayesian Hyperparameter Optimization
        evaluator = train_utils.KFoldEvaluator(get_temporal_model, features, labels)
        hyper_opt = BayesianOptimization(evaluator.evaluate, {'neurons': (40, 200),
                                                              'epochs': (5, 100),
                                                              'lr': (0.0005, 0.005),
                                                              'lr_decay': (0.0, 1e-4)})
        hyper_opt.maximize()
        optimal = hyper_opt.res['max']

        print("Best hyperparameter settings: " + str(optimal))
        io_utils.kfold_report_metrics(get_temporal_model, optimal['max_params'], features, labels)

if __name__ == "__main__":
    main()
