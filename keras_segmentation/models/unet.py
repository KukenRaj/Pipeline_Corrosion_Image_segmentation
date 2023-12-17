from keras.models import *
from keras.layers import *
from keras.losses import *

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder

# from keras.layers import Reshape, Dense, Multiply, Add
import tensorflow as tf

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    # Define binary cross-entropy
    bce = binary_crossentropy(y_true, y_pred)

    # Calculate the modulating factor
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    modulating_factor = tf.pow((1 - p_t), gamma)

    # Calculate the focal loss
    focal_loss = alpha * modulating_factor * bce

    return focal_loss

def SE_block(inputs, ratio=16):
    channels = inputs.shape[-1]
    se = GlobalAveragePooling2D()(inputs)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    se = Reshape((1, 1, channels))(se)
    output = Multiply()([inputs, se])
    return output

def FRAM_block(level, filters, kernel_size):
    # Attention mechanism (Squeeze-and-Excitation block)
    attention = SE_block(level)
    refined_features = Multiply()([level, attention])
    # Feature refinement
    x = Conv2D(filters, kernel_size, padding='same')(refined_features)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Merge original and refined features
    output = Add()([level, x])

    return output

if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1


def unet_mini(n_classes, input_height=360, input_width=480, channels=3):

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(channels, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, channels))

    conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv1)

    conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), data_format=IMAGE_ORDERING)(conv2)

    conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
        conv3), conv2], axis=MERGE_AXIS)
    conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(
        conv4), conv1], axis=MERGE_AXIS)
    conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), data_format=IMAGE_ORDERING,
                   activation='relu', padding='same' , name="seg_feats")(conv5)

    o = Conv2D(n_classes, (1, 1), data_format=IMAGE_ORDERING,
               padding='same')(conv5)

    model = get_segmentation_model(img_input, o)
    model.model_name = "unet_mini"
    return model


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608, channels=3, pipe=False):

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels
    
    if pipe==True:
        f3 = FRAM_block(f3, 256, (3, 3))
        f2 = FRAM_block(f2, 256, (3, 3))
        f1 = FRMA_block(f1, 256, (3, 3))
    o = f4

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING, name="seg_feats"))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same',
               data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)
    if pipe==True:
        model.compile(optimizer='adam', loss=focal_loss)

    return model


def unet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):

    model = _unet(n_classes, vanilla_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "unet"
    return model


def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):

    model = _unet(n_classes, get_vgg_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "vgg_unet"
    return model


def resnet50_unet(n_classes, input_height=416, input_width=608,
                  encoder_level=3, channels=3):

    model = _unet(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width, channels=channels, pipe=False)
    model.model_name = "resnet50_unet"
    return model

def resnet50_pipeunet(n_classes, input_height=416, input_width=608,
                  encoder_level=3, channels=3):

    model = _unet(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width, channels=channels, pipe=True)
    model.model_name = "resnet50_unet"
    return model

def resnet50_pipeunet(n_classes, input_height=416, input_width=608,
                  encoder_level=3, channels=3):

    model = _unet(n_classes, get_resnet50_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "resnet50_pipeunet"
    return model


def mobilenet_unet(n_classes, input_height=224, input_width=224,
                   encoder_level=3, channels=3):

    model = _unet(n_classes, get_mobilenet_encoder,
                  input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "mobilenet_unet"
    return model


if __name__ == '__main__':
    m = unet_mini(101)
    m = _unet(101, vanilla_encoder)
    # m = _unet( 101 , get_mobilenet_encoder ,True , 224 , 224  )
    m = _unet(101, get_vgg_encoder)
    m = _unet(101, get_resnet50_encoder)
