#!/usr/bin/env python
# coding: utf-8

# In[1]:


from multiprocessing import Pool
import json
import sys
import os


# In[2]:


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sklearn
from balance import image_aug_balance


# In[ ]:





# In[3]:


def one_train(mapitems):
    from keras import backend as k
    import tensorflow as tf
    gpu = mapitems['gpu']
    dataset = mapitems['dataset']
    pathX = mapitems['x']
    pathY = mapitems['y']
    model = mapitems['model']

    X = np.load(pathX)
    Y = np.load(pathY)
    
    X,Y,indices = image_aug_balance(X,Y,0)
    indices = sklearn.utils.shuffle(np.arange(Y.shape[0]))
    X = X[indices]
    Y = Y[indices]
    
    with tf.device('/gpu:' + str(gpu) if gpu != "" else "/cpu"):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        try:
            k.set_session(sess)
        except:
            print('No session available')

        train_model, base_model = model['model'](224, 224, 2)
        countbase = len(base_model.layers)

        path = os.path.join("../best-models", dataset, model["name"])
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "weights.{epoch:02d}-{val_loss:.2f}.hdf5") 

        callbacks = [
            ReduceLROnPlateau(monitor="loss", factor=0.2, mode="min", verbose=1),
            ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        ]

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        for train, test in kfold.split(X, Y):
            trainx = X[train]
            trainy = Y[train]
            testx = X[test]
            testy = Y[test]

            for layer in train_model.layers[:countbase]:
                layer.trainable = False
            train_model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=['accuracy'])
            train_model.fit(trainx, trainy, epochs=20, batch_size=8, validation_data=(testx, testy), callbacks=callbacks, verbose=2)

            for layer in train_model.layers[:countbase]:
                layer.trainable = True
            train_model.compile(optimizer="sgd", loss='binary_crossentropy', metrics=['accuracy'])
            train_model.fit(trainx, trainy, epochs=150, batch_size=8, validation_data=(testx, testy), callbacks=callbacks, verbose=2)

            break


# In[4]:


from import_all_models import DenseNet121_imagenet, DenseNet121_sinpesos
from import_all_models import DenseNet169_imagenet, DenseNet169_sinpesos
from import_all_models import DenseNet201_imagenet, DenseNet201_sinpesos
from import_all_models import InceptionResNetV2_imagenet, InceptionResNetV2_sinpesos
from import_all_models import InceptionV3_imagenet, InceptionV3_sinpesos
from import_all_models import MobileNet_imagenet, MobileNet_sinpesos
from import_all_models import ResNet50_sinpesos, ResNet50_imagenet
from import_all_models import VGG16_imagenet, VGG16_sinpesos
from import_all_models import VGG19_imagenet, VGG19_sinpesos
from import_all_models import Xception_imagenet, Xception_sinpesos
from import_all_models import InceptionV4_imagenet, InceptionV4_sinpesos

from import_all_models import NASNetMobile_imagenet, NASNetMobile_sinpesos

models = [
    MobileNet_imagenet(),
    #NASNetMobile_imagenet(),
    #DenseNet121_imagenet(),
    #DenseNet169_imagenet(),
    #DenseNet201_imagenet(),
    #InceptionV3_imagenet(),
    #Xception_imagenet(),
    #VGG16_imagenet(),
    #VGG19_imagenet(),
    #ResNet50_imagenet(),
    #InceptionResNetV2_imagenet(),
    #InceptionV4_imagenet
]

#config_file = sys.argv[1]
config_file = "./fase.json"


# In[5]:


with open(config_file) as json_data:
    configuration = json.load(json_data)
    
    for dataset in configuration['datasets']:
        for model in models:
            with Pool(processes=1) as pool:
                pool.map(
                    one_train, 
                    [{
                        "gpu": configuration["gpu"],
                        "dataset": dataset["name"],
                        "x": dataset["x"],
                        "y": dataset["y"],
                        "model": model
                    }]
                )


# In[ ]:




