import tensorflow as tf
from tensorflow.keras import layers, Model


# SqueezeExcitationBlock
class SqueezeExcitationBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.squeeze = layers.GlobalAveragePooling2D(data_format='channels_first')
        self.excitation = tf.keras.Sequential([
            layers.Dense(units=in_channels // reduction, activation='relu', use_bias=False),
            layers.Dense(units=in_channels, activation='sigmoid', use_bias=False),
            layers.Reshape((in_channels, 1, 1))
        ])

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.excitation(x)
        return inputs * x


# RDB_Conv
class RDB_Conv(tf.keras.layers.Layer):
    def __init__(self, in_channels, grow_rate, k_size=3):
        super(RDB_Conv, self).__init__()
        padding = (k_size - 1) // 2
        self.conv = layers.Conv2D(grow_rate, kernel_size=k_size, padding='same',
                                  data_format='channels_first', activation="relu")

    def call(self, x):
        out = self.conv(x)
        return tf.concat([x, out], axis=1)


# RDB
class RDB(tf.keras.layers.Layer):
    def __init__(self, grow_rate0, grow_rate, n_conv_layers, k_size=3):
        super(RDB, self).__init__()
        self.convs = [RDB_Conv(grow_rate0 + c * grow_rate, grow_rate, k_size) for c in range(n_conv_layers)]
        self.lff = layers.Conv2D(grow_rate0, kernel_size=1, data_format='channels_first')
        self.se_block = SqueezeExcitationBlock(grow_rate0)

    def call(self, x):
        combined = x
        for conv in self.convs:
            x = conv(x)
        x = self.se_block(self.lff(x))
        return x + combined


# Assumed VGGEncoder structure
class VGGEncoder(tf.keras.layers.Layer):
    def __init__(self, in_channels, num_features):
        super(VGGEncoder, self).__init__()
        # Assumed layers for a simple VGG-style encoder
        self.conv1 = layers.Conv2D(num_features, kernel_size=3, padding="same", data_format='channels_first',
                                   activation="relu")
        self.conv2 = layers.Conv2D(num_features, kernel_size=3, padding="same", data_format='channels_first',
                                   activation="relu")
        self.pool = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        return x


# Assumed UpscaleBlock structure
class UpscaleBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpscaleBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size=3, padding="same", data_format='channels_first',
                                  activation="relu")
        self.upscale = layers.UpSampling2D(size=scale_factor, data_format='channels_first')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.upscale(x)
        return x


# VGGSuperResEncoder
class VGGSuperResEncoder(tf.keras.layers.Layer):
    def __init__(self, in_channels=3, num_rdb_blocks=12, grow_rate0=64, grow_rate=64, n_conv_layers=8, scale_factor=2):
        super(VGGSuperResEncoder, self).__init__()

        self.vgg_encoder = VGGEncoder(in_channels=in_channels, num_features=grow_rate0)
        self.rdb_blocks = [RDB(grow_rate0, grow_rate, n_conv_layers) for _ in range(num_rdb_blocks)]

        self.gff = tf.keras.Sequential([
            layers.Conv2D(num_rdb_blocks * grow_rate0, kernel_size=1, data_format='channels_first'),
            layers.Conv2D(grow_rate0, kernel_size=3, padding="same", data_format='channels_first'),
            SqueezeExcitationBlock(grow_rate0)
        ])

        self.upscale = UpscaleBlock(grow_rate0, grow_rate0, scale_factor)

    def call(self, inputs):
        x = self.vgg_encoder(inputs)
        rdb_outputs = [x]

        for rdb_block in self.rdb_blocks:
            x = rdb_block(x)
            rdb_outputs.append(x)

        x = self.gff(tf.concat(rdb_outputs, axis=1)) + x
        x = self.upscale(x)
        return x
