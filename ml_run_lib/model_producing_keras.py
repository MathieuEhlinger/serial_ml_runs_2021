# -*- coding: utf-8 -*-
"""
@author: Mathieu Ehlinger
Part of the MetaExperiment Project
"""
from sklearn.model_selection import train_test_split
from keras import layers
from keras import Input
from keras.models import Model
import pandas as pd

X=  pd.DataFrame.from_csv('add_csv_path_1')
y=  pd.DataFrame.from_csv('add_csv_path_2')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

epochs= 80

input_tensor= Input(shape=(None,X.shape[1]))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(1, activation='sigmoid')(x)
model = Model(input_tensor,output_tensor)

model.compile(optimizer='rmsprop',
        loss='mse',
        metric=['mae','mse']
        )

history = model.fit(X_train.values, y_train.values,
            validation_data=(y_train.values,y_test.values),
            epochs=epochs, batch_size=1,verbose=2)
