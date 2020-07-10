import tensorflow as tf
from classifier.tf_models.mbconv import MBConvBlock

class ClassificationLayer(tf.keras.layers.Layer):
    def __init__(self, filters=40):
        super(ClassificationLayer, self).__init__()
        self.classification_layers = [tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), padding='same'),
                                      tf.keras.layers.BatchNormalization(),
                                      tf.keras.layers.GlobalAveragePooling2D(),
                                      tf.keras.layers.Dense(2, activation='softmax')]

    def call(self, x, training_classification_layers=True):
        for layer in self.classification_layers:
          x = layer(x, training=training_classification_layers)
        return x

class ImageClassifier(tf.keras.Model):
    def __init__(self, first_conv_filters=12, first_conv_kernel=(7,7), first_conv_stride=(2,2),
                 classifier_filters=40, depth_scale=1.2, width_scale=1.05, phi_scaling_factor=0, **kwargs):

        super(ImageClassifier, self).__init__()
        depth_scale = depth_scale ** phi_scaling_factor
        width_scale = width_scale ** phi_scaling_factor
        self.inp_conv = tf.keras.layers.Conv2D(filters=first_conv_filters, kernel_size=first_conv_kernel, strides=first_conv_stride, padding='same')
        self.pool = tf.keras.layers.MaxPool2D()
        self.mb_conv_layers = self._init_mb_conv_layers(width_scale, depth_scale, **kwargs)
        self.classify_layer = ClassificationLayer(classifier_filters)

    @staticmethod
    def _init_mb_conv_layers(width_scale=1.0, depth_scale=1.0, base_strides=None, base_channels=None, base_layers=None,
                             base_t_expansions=None, base_kernel_size=None, **kwargs):
        if base_strides is None:
            base_strides = [(2, 2), (1, 1), (1, 1), (2, 2)]
        if base_channels is None:
            base_channels = [16, 22, 26, 32]
        if base_layers is None:
            base_layers = [1, 1, 2, 1]
        if base_t_expansions:
            base_t_expansions = [1, 6, 6, 6]
        if base_kernel_size is None:
            base_kernel_size = [(3, 3), (5, 5), (3, 3), (3, 3)]

        mb_conv_layers = [MBConvBlock(start_stride=base_strides[i],
                                           channels=round(base_channels[i] * width_scale),
                                           t=base_t_expansions[i],
                                           layers=round(base_layers[i] * depth_scale),
                                           kernel_size=base_kernel_size[i])
                               for i in range(len(base_channels))]

        return mb_conv_layers


    def call(self, x, training_mbconv=True, training_classification_layers=True):
        x = self.inp_conv(x)
        x = self.pool(x)
        for layer in self.mb_conv_layers:
            x = layer(x, training=training_mbconv)

        x = self.classify_layer(x, training_classification_layers)

        return x