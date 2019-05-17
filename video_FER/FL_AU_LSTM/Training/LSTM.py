
import numpy as np

from keras.layers import (LSTM, Activation, BatchNormalization, Dense, Dropout, Input)
from keras.models import Sequential, Model
from keras.optimizers import Adam

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
