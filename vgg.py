import os

# let's not pollute this blog post with warnings
from warnings import filterwarnings
filterwarnings('ignore')

import keras
import numpy as np
import pandas as pd
import skvideo.io as skv
from tqdm import tqdm

# load the data
run = '-VGG-micro'
labelpath = os.path.join('train_labels.csv')
train_labels = pd.read_csv(labelpath, index_col='filename')

train_labels.info()
train_labels.sum(axis=0).sort_values(ascending=False)
(train_labels.sum(axis=1) > 1).sum()

from primatrix_dataset_utils import Dataset
datapath = os.path.join('.')
whichset = run.split('-')[-1]
redframes = whichset == 'nano'
data = Dataset(datapath=datapath, 
               dataset_type=whichset,
               reduce_frames=redframes, 
               batch_size=32, 
               test=False)

data.num_samples
data.y_val.shape[0]

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())

from keras.models import Sequential
from keras.layers import TimeDistributed, GlobalAveragePooling2D, Activation, Dense, Input
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dropout, Flatten
from keras.layers import concatenate
from keras import regularizers
from keras import initializers
from keras import constraints
from keras.models import Model
# Backend
from keras import backend as K
# Utils
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file

def Conv2D_bn(x,
              filters,
              filter_size,
              padding='same',
              strides=(1, 1),
              name=None,
             activation='relu'):
    num_row, num_col = filter_size
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    bn_axis = -1
    x = TimeDistributed(Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name,
        kernel_initializer=initializers.he_normal()))(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis, scale=False, name=bn_name))(x)
    x = TimeDistributed(Activation('relu', name=name))(x)
    return x


def VGG19(include_top=True,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=24):
    channel_axis = -1
    inputs = Input((data.num_frames, data.width, data.height, 3))
    x = BatchNormalization(axis=-1)(inputs)
    
    x = Conv2D_bn(x, 32, (3, 3), activation='relu', padding='same', name='block1_conv1')
    x = Conv2D_bn(x, 32, (3, 3), activation='relu', padding='same', name='block1_conv2')
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = Conv2D_bn(x, 64, (3, 3), activation='relu', padding='same', name='block2_conv1')
    x = Conv2D_bn(x, 64, (3, 3), activation='relu', padding='same', name='block2_conv2')
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = Conv2D_bn(x, 128, (3, 3), activation='relu', padding='same', name='block3_conv1')
    x = Conv2D_bn(x, 128, (3, 3), activation='relu', padding='same', name='block3_conv2')
    x = Conv2D_bn(x, 128, (3, 3), activation='relu', padding='same', name='block3_conv3')
    x = Conv2D_bn(x, 128, (3, 3), activation='relu', padding='same', name='block3_conv4')
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = Conv2D_bn(x, 256, (3, 3), activation='relu', padding='same', name='block4_conv1')
    x = Conv2D_bn(x, 256, (3, 3), activation='relu', padding='same', name='block4_conv2')
    x = Conv2D_bn(x, 256, (3, 3), activation='relu', padding='same', name='block4_conv3')
    x = Conv2D_bn(x, 256, (3, 3), activation='relu', padding='same', name='block4_conv4')
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = Conv2D_bn(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv1')
    x = Conv2D_bn(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv2')
    x = Conv2D_bn(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv3')
    x = Conv2D_bn(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv4')
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)

    if include_top:
        # Classification block
        x = TimeDistributed(Flatten(name='flatten'))(x)
        x = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(x)
        x = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(x)   
        x = LSTM(256, return_sequences=False)(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, x, name='vgg16')

    # load weights
    return model

# instantiate model
model = VGG19()


# classifier with sigmoid activation for multilabel

adam = keras.optimizers.Adam(lr=0.001, decay=0.0, clipnorm=5.)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=20, min_lr=0.00001, verbose=True)


# compile the model with binary_crossentropy loss for multilabel
model.compile(optimizer=adam, loss='binary_crossentropy')

model_name = 'model' + run + '.h5'
checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint, reduce_lr]


# look at the params before training
model.summary()

model.fit_generator(
    data.batches(), 
    steps_per_epoch=int(data.num_batches/20),                  # data.num_batches to train on full set 
    epochs=6000, 
    validation_data=data.val_batches(), 
    validation_steps=int(data.num_val_batches/20),                  # data.num_val_batches to validate on full set
    callbacks=callbacks_list
)

# load model
from keras.models import load_model

trained_model = load_model(model_name)

# generate predictions
for batch_num in tqdm(range(data.num_test_batches), total=data.num_test_batches):

    # make predictions on batch
    results = trained_model.predict(next(data.test_batches()), 
                                          batch_size=data.batch_size, 
                                          verbose=0)

    # update submission format dataframe stored in dataset object
    data.update_predictions(results)          
    