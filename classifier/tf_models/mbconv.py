import tensorflow as tf

class MBConvLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(3,3), stride=(1,1), t=6, channels=32):
        super(MBConvLayer, self).__init__()
        self.bn = [tf.keras.layers.BatchNormalization(), tf.keras.layers.BatchNormalization(), tf.keras.layers.BatchNormalization()]
        self.conv = [tf.keras.layers.Conv2D(filters=channels * t, kernel_size=(1,1), padding='same', strides=stride), tf.keras.layers.Conv2D(filters=channels, kernel_size=(1,1), padding='same')]
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same')
        self.inp_reshape = tf.keras.layers.Conv2D(filters=channels, kernel_size=(1,1), padding='same', strides=stride)

    def call(self, x, training=True):
        x_next = self.conv[0](x)
        x_next = self.bn[0](x_next, training=training)
        x_next = tf.nn.relu6(x_next)
        x_next = self.depthwise_conv(x_next)
        x_next = self.bn[1](x_next, training=training)
        x_next = tf.nn.relu6(x_next)
        x_next = self.conv[1](x_next)
        x_next = self.bn[2](x_next, training=training)
        if not tf.reduce_all(tf.math.equal(tf.shape(x), tf.shape(x_next))):
          x = self.inp_reshape(x)
        return x + x_next

class MBConvBlock(tf.keras.layers.Layer):
    def __init__(self, layers=1, kernel_size=(3,3), start_stride=(2,2), t=6, channels=32):
        super(MBConvBlock, self).__init__()
        self.stride = start_stride
        self.channels = channels
        self.layers = [MBConvLayer(stride=start_stride, t=t, channels=channels, kernel_size=kernel_size)] + [MBConvLayer(t=t, channels=channels, kernel_size=kernel_size) for i in range(1, layers)]

    def call(self, x, training=True):
        for layer in self.layers:
          x = layer(x, training=training)
        return x