from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, Average
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Train_Unet import dice_loss, dice_coef
from tensorflow.keras.metrics import MeanMetricWrapper

def conv_block(x, filters, dropout_rate=0.1):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    return x

def build_unetpp(input_shape=(320, 320, 1), base_filters=32, dropout_rate=0.1, learning_rate=1e-3, deep_supervision=True):
    inputs = Input(shape=input_shape)

    # Encoder path
    x00 = conv_block(inputs, base_filters, dropout_rate)
    x10 = conv_block(MaxPooling2D()(x00), base_filters * 2, dropout_rate)
    x20 = conv_block(MaxPooling2D()(x10), base_filters * 4, dropout_rate)
    x30 = conv_block(MaxPooling2D()(x20), base_filters * 8, dropout_rate)
    x40 = conv_block(MaxPooling2D()(x30), base_filters * 16, dropout_rate)

    # Decoder path (nested dense skip connections)
    x01 = conv_block(concatenate([x00, UpSampling2D()(x10)]), base_filters, dropout_rate)
    x11 = conv_block(concatenate([x10, UpSampling2D()(x20)]), base_filters * 2, dropout_rate)
    x02 = conv_block(concatenate([x00, x01, UpSampling2D()(x11)]), base_filters, dropout_rate)
    x21 = conv_block(concatenate([x20, UpSampling2D()(x30)]), base_filters * 4, dropout_rate)
    x12 = conv_block(concatenate([x10, x11, UpSampling2D()(x21)]), base_filters * 2, dropout_rate)
    x03 = conv_block(concatenate([x00, x01, x02, UpSampling2D()(x12)]), base_filters, dropout_rate)
    x31 = conv_block(concatenate([x30, UpSampling2D()(x40)]), base_filters * 8, dropout_rate)
    x22 = conv_block(concatenate([x20, x21, UpSampling2D()(x31)]), base_filters * 4, dropout_rate)
    x13 = conv_block(concatenate([x10, x11, x12, UpSampling2D()(x22)]), base_filters * 2, dropout_rate)
    x04 = conv_block(concatenate([x00, x01, x02, x03, UpSampling2D()(x13)]), base_filters, dropout_rate)

    if deep_supervision:
        out1 = Conv2D(1, 1, activation='sigmoid', name='out1')(x01)
        out2 = Conv2D(1, 1, activation='sigmoid', name='out2')(x02)
        out3 = Conv2D(1, 1, activation='sigmoid', name='out3')(x03)
        out4 = Conv2D(1, 1, activation='sigmoid', name='out4')(x04)

        model = Model(inputs, [out1, out2, out3, out4])
        model.compile(
            optimizer=Adam(learning_rate),
            loss=[dice_loss] * 4,
            loss_weights=[0.25] * 4,
            metrics={
                'out1': [MeanMetricWrapper(dice_coef, name='out1_dice')],
                'out2': [MeanMetricWrapper(dice_coef, name='out2_dice')],
                'out3': [MeanMetricWrapper(dice_coef, name='out3_dice')],
                'out4': [MeanMetricWrapper(dice_coef, name='out4_dice')],
            }
        )
    else:
        out = Conv2D(1, 1, activation='sigmoid', name='out')(x04)
        model = Model(inputs, out)
        model.compile(
            optimizer=Adam(learning_rate),
            loss=dice_loss,
            metrics=[dice_coef]
        )

    return model
