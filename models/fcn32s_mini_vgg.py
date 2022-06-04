import tensorflow as tf

from tensorflow.keras.layers import (
    Conv2D, Activation, MaxPool2D,
    Conv2DTranspose
)
from tensorflow.keras import Model

def fcn32s_mini_vgg(num_classes, input_shape):
    inputs = tf.keras.layers.Input(input_shape)

    # Convolution encoder block 1.
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # Convolution encoder block 2.
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # Convolution encoder block 3.
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # Convolution encoder block 4.
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # Convolution encoder block 5.
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(num_classes, (1, 1), padding='same')(x)

    # Upsampling using Transposed Convolution.
    outputs = Conv2DTranspose(
        num_classes , 
        kernel_size=(32, 32), 
        strides=(32, 32),  
        padding='same'
    )(x)
    outputs = Activation('softmax')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model

if __name__ == '__main__':
    input_shape = (224, 224, 3)
    model = fcn32s_mini_vgg(num_classes=3, input_shape=input_shape)
    print(model.summary())