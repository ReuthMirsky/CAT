import context
import tensorflow as tf
import numpy as np
from itertools import chain
def create_model(inputsize, hidden_layers=[128, 64, 32]):
    model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=inputsize)] + 
            list(chain.from_iterable((tf.keras.layers.Dense(hidden_layers[i], activation="relu"),
                tf.keras.layers.Dropout(0.2)) for i in range(len(hidden_layers)))) +
            [tf.keras.layers.Dense(1)])
    loss_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt,
            loss=loss_fn,
            metrics=['mse'])
    return model
            

def trainSupervised(model, data, numEpochs, updatesPerEpoch=1):
    train_data = []
    labels = []
    for g1, g2 in data:
        d = data[g1,g2]
        for i in range(len(d)):
            for j in range(len(d[i])):
                train_data.append([i, j] +  list(g1)+ list(g2))
                labels.append(d[i][j])
    train_data = np.array(train_data)
    labels = np.array(labels)
    model.fit(train_data, labels, epochs=numEpochs*updatesPerEpoch)

def trainMC(model, data, numEpochs, agentModel,  gridWidth=20, gridHeight=20, updatesPerEpoch=1):
    for _ in range(numEpochs):
        train_data = []
        labels = []
        g2 = np.random.randint([gridWidth, gridHeight])
        for g in data:
            count = 1
            d = data[g]
            for obs, action, obsp in reversed(d):
                train_data.append(list(obs)+list(g2) + list(g))
                if agentModel(obs, g2, action) > 0:
                    labels.append(count)
                    count += 1
                else:
                    count = 1
                    labels.append(count)
        model.fit(np.array(train_data), np.array(labels), epochs=updatesPerEpoch)



def trainTD(model, data, numEpochs, agentModel, gridWidth=20, gridHeight=20, updatesPerEpoch=1):
    for _ in range(numEpochs):
        train_data = []
        labels = []
        for g in data:
            d = data[g]
            g2 = np.random.randint([gridWidth, gridHeight])
            for obs, action, obsp in d:
                train_data.append(list(obs) + list(g2) + list(g))
                if agentModel(obs, g2, action) > 0 and obsp is not None:
                    target = 1 + model(np.array([list(obsp) + list(g2) + list(g)])).numpy()[0][0]
                    labels.append(target)
                else:
                    labels.append(1)
        model.fit(np.array(train_data), np.array(labels), epochs=updatesPerEpoch)
