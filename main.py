import pickle

import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Input, BatchNormalization, Activation, MaxPooling2D, \
    add, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

with open("./data/X_train", 'rb') as fh:
    X_train = np.array(pickle.load(fh))

with open("./data/y_train", 'rb') as fh:
    y_train = np.array(pickle.load(fh))


def resnet_block(inputs, num_filters, kernel_size, strides, activation='relu'):
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(inputs)
    x = BatchNormalization()(x)
    if (activation):
        x = Activation('relu')(x)
    return x


inputs = Input((100, 150, 3))

# conv1
x = resnet_block(inputs, 64, [7, 7], 1)

# conv2
x = MaxPooling2D([3, 3], 1, 'same')(x)
for i in range(2):
    a = resnet_block(x, 64, [3, 3], 1)
    b = resnet_block(a, 64, [3, 3], 1, activation=None)
    x = add([x, b])
    x = Activation('relu')(x)

# conv3
a = resnet_block(x, 128, [1, 1], 2)
b = resnet_block(a, 128, [3, 3], 1, activation=None)
x = Conv2D(128, kernel_size=[1, 1], strides=2, padding='same', kernel_initializer='he_normal',
           kernel_regularizer=l2(1e-3))(x)
x = add([x, b])
x = Activation('relu')(x)

a = resnet_block(x, 128, [3, 3], 1)
b = resnet_block(a, 128, [3, 3], 1, activation=None)
x = add([x, b])
x = Activation('relu')(x)

a = resnet_block(x, 256, [1, 1], 2)
b = resnet_block(a, 256, [3, 3], 1, activation=None)
x = Conv2D(256, kernel_size=[1, 1], strides=2, padding='same', kernel_initializer='he_normal',
           kernel_regularizer=l2(1e-3))(x)
x = add([x, b])
x = Activation('relu')(x)
y = GlobalAveragePooling2D(data_format="channels_last")(x)

# out:512
y = Dense(500, kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(y)
y = Dense(4, kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(y)
outputs = Activation('softmax')(y)

model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()


def lr_sch(epoch):
    # 200 total
    if epoch < 10:
        return 1e-3
    if 10 <= epoch < 20:
        return 1e-4
    if epoch >= 20:
        return 1e-5


lr_scheduler = LearningRateScheduler(lr_sch)
lr_reducer = ReduceLROnPlateau(monitor='categorical_crossentropy', factor=0.2, patience=5, mode='min', min_lr=1e-3)

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

model_details = model.fit(X_train, y_train, batch_size=128, epochs=25, shuffle=True, validation_split=0.15,
                          callbacks=[lr_scheduler, lr_reducer, checkpoint], verbose=1)

model.save('model.h5')
