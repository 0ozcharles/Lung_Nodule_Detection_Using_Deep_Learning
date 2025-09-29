# simplified_unetpp.py
"""
简化但有效的UNet++实现
- 保留原始UNet++的核心结构
- 适度添加正则化
- 使用更稳定的训练策略
"""
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D,
                                     concatenate, BatchNormalization, Activation,
                                     Dropout, GlobalAveragePooling2D, Dense, Reshape, multiply)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class SimpleUNetPP:
    def __init__(self, input_size=(320, 320, 1), num_classes=1):
        self.input_size = input_size
        self.num_classes = num_classes

    def conv_block(self, x, filters, use_dropout=False, dropout_rate=0.1):
        """标准卷积块，可选dropout"""
        # 第一个卷积
        x = Conv2D(filters, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # 第二个卷积
        x = Conv2D(filters, (3, 3), padding="same", kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # 只在深层使用dropout
        if use_dropout:
            x = Dropout(dropout_rate)(x)

        return x

    def squeeze_excite_block(self, x, filters, ratio=16):
        """轻量级注意力机制 - Squeeze and Excitation"""
        # Squeeze
        se = GlobalAveragePooling2D()(x)

        # Excitation
        se = Dense(filters // ratio, activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)

        # Scale
        se = Reshape((1, 1, filters))(se)
        x = multiply([x, se])

        return x

    def build(self):
        inputs = Input(self.input_size)

        # ========== 编码器路径 ==========
        # Level 0
        x0_0 = self.conv_block(inputs, 32)
        pool1 = MaxPooling2D(pool_size=(2, 2))(x0_0)

        # Level 1
        x1_0 = self.conv_block(pool1, 64)
        pool2 = MaxPooling2D(pool_size=(2, 2))(x1_0)

        # Level 2
        x2_0 = self.conv_block(pool2, 128, use_dropout=True, dropout_rate=0.1)
        pool3 = MaxPooling2D(pool_size=(2, 2))(x2_0)

        # Level 3
        x3_0 = self.conv_block(pool3, 256, use_dropout=True, dropout_rate=0.2)
        pool4 = MaxPooling2D(pool_size=(2, 2))(x3_0)

        # Level 4 - 瓶颈层
        x4_0 = self.conv_block(pool4, 512, use_dropout=True, dropout_rate=0.3)

        # ========== 解码器路径 - 第1列 ==========
        # 使用SE block增强特征
        x3_1 = self.conv_block(
            concatenate([x3_0, UpSampling2D(size=(2, 2))(x4_0)], axis=3),
            256, use_dropout=True, dropout_rate=0.2
        )
        x3_1 = self.squeeze_excite_block(x3_1, 256)

        x2_1 = self.conv_block(
            concatenate([x2_0, UpSampling2D(size=(2, 2))(x3_1)], axis=3),
            128, use_dropout=True, dropout_rate=0.1
        )

        x1_1 = self.conv_block(
            concatenate([x1_0, UpSampling2D(size=(2, 2))(x2_1)], axis=3),
            64
        )

        x0_1 = self.conv_block(
            concatenate([x0_0, UpSampling2D(size=(2, 2))(x1_1)], axis=3),
            32
        )

        # ========== 解码器路径 - 第2列 ==========
        x2_2 = self.conv_block(
            concatenate([x2_0, x2_1, UpSampling2D(size=(2, 2))(x3_1)], axis=3),
            128
        )

        x1_2 = self.conv_block(
            concatenate([x1_0, x1_1, UpSampling2D(size=(2, 2))(x2_2)], axis=3),
            64
        )

        x0_2 = self.conv_block(
            concatenate([x0_0, x0_1, UpSampling2D(size=(2, 2))(x1_2)], axis=3),
            32
        )

        # ========== 解码器路径 - 第3列 ==========
        x1_3 = self.conv_block(
            concatenate([x1_0, x1_1, x1_2, UpSampling2D(size=(2, 2))(x2_2)], axis=3),
            64
        )

        x0_3 = self.conv_block(
            concatenate([x0_0, x0_1, x0_2, UpSampling2D(size=(2, 2))(x1_3)], axis=3),
            32
        )

        # ========== 解码器路径 - 第4列 ==========
        x0_4 = self.conv_block(
            concatenate([x0_0, x0_1, x0_2, x0_3, UpSampling2D(size=(2, 2))(x1_3)], axis=3),
            32
        )

        # ========== 输出层 ==========
        output = Conv2D(self.num_classes, (1, 1), activation='sigmoid')(x0_4)

        model = Model(inputs=inputs, outputs=output)
        return model


# 更稳定的损失函数组合
def stable_dice_loss(y_true, y_pred, smooth=1.0):
    """更稳定的Dice Loss"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def stable_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """稳定的Focal Loss实现"""
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # 计算交叉熵
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    # focal loss
    loss = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
           - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return loss


def stable_combined_loss(y_true, y_pred):
    """更稳定的组合损失"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = stable_dice_loss(y_true, y_pred)
    focal = stable_focal_loss(y_true, y_pred)

    # 调整权重，让BCE占主导（更稳定）
    return 0.5 * bce + 0.3 * dice + 0.2 * focal