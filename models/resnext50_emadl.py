import mxnet as mx
import numpy as np
import math
from mxnet import gluon


class ZScoreNormalization(gluon.HybridBlock):
    def __init__(self, data_mean, data_std, **kwargs):
        super(ZScoreNormalization, self).__init__(**kwargs)
        with self.name_scope():
            self.data_mean = self.params.get('data_mean', shape=data_mean.shape,
                init=mx.init.Constant(data_mean.asnumpy().tolist()), differentiable=False)
            self.data_std = self.params.get('data_std', shape=data_mean.shape,
                init=mx.init.Constant(data_std.asnumpy().tolist()), differentiable=False)

    def hybrid_forward(self, F, x, data_mean, data_std):
        x = F.broadcast_sub(x, data_mean)
        x = F.broadcast_div(x, data_std)
        return x


class Padding(gluon.HybridBlock):
    def __init__(self, padding, **kwargs):
        super(Padding, self).__init__(**kwargs)
        with self.name_scope():
            self.pad_width = padding

    def hybrid_forward(self, F, x):
        x = F.pad(data=x,
            mode='constant',
            pad_width=self.pad_width,
            constant_value=0)
        return x


class NoNormalization(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(NoNormalization, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return x


class Reshape(gluon.HybridBlock):
    def __init__(self, shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        with self.name_scope():
            self.shape = shape

    def hybrid_forward(self, F, x):
        return F.reshape(data=x, shape=self.shape)


class CustomRNN(gluon.HybridBlock):
    def __init__(self, hidden_size, num_layers, dropout, bidirectional, **kwargs):
        super(CustomRNN, self).__init__(**kwargs)
        with self.name_scope():
            self.rnn = gluon.rnn.RNN(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                                     bidirectional=bidirectional, activation='tanh', layout='NTC')

    def hybrid_forward(self, F, data, state0):
        output, [state0] = self.rnn(data, [F.swapaxes(state0, 0, 1)])
        return output, F.swapaxes(state0, 0, 1)


class CustomLSTM(gluon.HybridBlock):
    def __init__(self, hidden_size, num_layers, dropout, bidirectional, **kwargs):
        super(CustomLSTM, self).__init__(**kwargs)
        with self.name_scope():
            self.lstm = gluon.rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                                       bidirectional=bidirectional, layout='NTC')

    def hybrid_forward(self, F, data, state0, state1):
        output, [state0, state1] = self.lstm(data, [F.swapaxes(state0, 0, 1), F.swapaxes(state1, 0, 1)])
        return output, F.swapaxes(state0, 0, 1), F.swapaxes(state1, 0, 1)


class CustomGRU(gluon.HybridBlock):
    def __init__(self, hidden_size, num_layers, dropout, bidirectional, **kwargs):
        super(CustomGRU, self).__init__(**kwargs)
        with self.name_scope():
            self.gru = gluon.rnn.GRU(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                                     bidirectional=bidirectional, layout='NTC')

    def hybrid_forward(self, F, data, state0):
        output, [state0] = self.gru(data, [F.swapaxes(state0, 0, 1)])
        return output, F.swapaxes(state0, 0, 1)


class ResNeXT50(gluon.HybridBlock):
    def __init__(self, data_mean=None, data_std=None, **kwargs):
        super(ResNeXT50, self).__init__(**kwargs)
        with self.name_scope():
            if data_mean and data_std:
                assert(data_std)
                self.input_normalization_data_ = ZScoreNormalization(data_mean=data_mean['data_'],
                                                                               data_std=data_std['data_'])
            else:
                self.input_normalization_data_ = NoNormalization()

            self.conv1_padding = Padding(padding=(0,0,0,0,3,2,3,2))
            self.conv1_ = gluon.nn.Conv2D(channels=64,
                kernel_size=(7,7),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv1_, output shape: {[64,240,240]}

            self.batchnorm1_ = gluon.nn.BatchNorm()
            # batchnorm1_, output shape: {[64,240,240]}

            self.relu1_ = gluon.nn.Activation(activation='relu')
            self.pool1_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.pool1_ = gluon.nn.MaxPool2D(
                pool_size=(3,3),
                strides=(2,2))
            # pool1_, output shape: {[64,120,120]}

            self.conv3_1_1_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_1_, output shape: {[4,120,120]}

            self.batchnorm3_1_1_ = gluon.nn.BatchNorm()
            # batchnorm3_1_1_, output shape: {[4,120,120]}

            self.relu3_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_1_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_1_, output shape: {[4,120,120]}

            self.batchnorm4_1_1_ = gluon.nn.BatchNorm()
            # batchnorm4_1_1_, output shape: {[4,120,120]}

            self.relu4_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_1_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_1_, output shape: {[256,120,120]}

            self.batchnorm5_1_1_ = gluon.nn.BatchNorm()
            # batchnorm5_1_1_, output shape: {[256,120,120]}

            self.conv3_1_2_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_2_, output shape: {[4,120,120]}

            self.batchnorm3_1_2_ = gluon.nn.BatchNorm()
            # batchnorm3_1_2_, output shape: {[4,120,120]}

            self.relu3_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_2_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_2_, output shape: {[4,120,120]}

            self.batchnorm4_1_2_ = gluon.nn.BatchNorm()
            # batchnorm4_1_2_, output shape: {[4,120,120]}

            self.relu4_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_2_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_2_, output shape: {[256,120,120]}

            self.batchnorm5_1_2_ = gluon.nn.BatchNorm()
            # batchnorm5_1_2_, output shape: {[256,120,120]}

            self.conv3_1_3_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_3_, output shape: {[4,120,120]}

            self.batchnorm3_1_3_ = gluon.nn.BatchNorm()
            # batchnorm3_1_3_, output shape: {[4,120,120]}

            self.relu3_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_3_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_3_, output shape: {[4,120,120]}

            self.batchnorm4_1_3_ = gluon.nn.BatchNorm()
            # batchnorm4_1_3_, output shape: {[4,120,120]}

            self.relu4_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_3_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_3_, output shape: {[256,120,120]}

            self.batchnorm5_1_3_ = gluon.nn.BatchNorm()
            # batchnorm5_1_3_, output shape: {[256,120,120]}

            self.conv3_1_4_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_4_, output shape: {[4,120,120]}

            self.batchnorm3_1_4_ = gluon.nn.BatchNorm()
            # batchnorm3_1_4_, output shape: {[4,120,120]}

            self.relu3_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_4_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_4_, output shape: {[4,120,120]}

            self.batchnorm4_1_4_ = gluon.nn.BatchNorm()
            # batchnorm4_1_4_, output shape: {[4,120,120]}

            self.relu4_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_4_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_4_, output shape: {[256,120,120]}

            self.batchnorm5_1_4_ = gluon.nn.BatchNorm()
            # batchnorm5_1_4_, output shape: {[256,120,120]}

            self.conv3_1_5_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_5_, output shape: {[4,120,120]}

            self.batchnorm3_1_5_ = gluon.nn.BatchNorm()
            # batchnorm3_1_5_, output shape: {[4,120,120]}

            self.relu3_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_5_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_5_, output shape: {[4,120,120]}

            self.batchnorm4_1_5_ = gluon.nn.BatchNorm()
            # batchnorm4_1_5_, output shape: {[4,120,120]}

            self.relu4_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_5_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_5_, output shape: {[256,120,120]}

            self.batchnorm5_1_5_ = gluon.nn.BatchNorm()
            # batchnorm5_1_5_, output shape: {[256,120,120]}

            self.conv3_1_6_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_6_, output shape: {[4,120,120]}

            self.batchnorm3_1_6_ = gluon.nn.BatchNorm()
            # batchnorm3_1_6_, output shape: {[4,120,120]}

            self.relu3_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_6_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_6_, output shape: {[4,120,120]}

            self.batchnorm4_1_6_ = gluon.nn.BatchNorm()
            # batchnorm4_1_6_, output shape: {[4,120,120]}

            self.relu4_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_6_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_6_, output shape: {[256,120,120]}

            self.batchnorm5_1_6_ = gluon.nn.BatchNorm()
            # batchnorm5_1_6_, output shape: {[256,120,120]}

            self.conv3_1_7_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_7_, output shape: {[4,120,120]}

            self.batchnorm3_1_7_ = gluon.nn.BatchNorm()
            # batchnorm3_1_7_, output shape: {[4,120,120]}

            self.relu3_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_7_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_7_, output shape: {[4,120,120]}

            self.batchnorm4_1_7_ = gluon.nn.BatchNorm()
            # batchnorm4_1_7_, output shape: {[4,120,120]}

            self.relu4_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_7_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_7_, output shape: {[256,120,120]}

            self.batchnorm5_1_7_ = gluon.nn.BatchNorm()
            # batchnorm5_1_7_, output shape: {[256,120,120]}

            self.conv3_1_8_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_8_, output shape: {[4,120,120]}

            self.batchnorm3_1_8_ = gluon.nn.BatchNorm()
            # batchnorm3_1_8_, output shape: {[4,120,120]}

            self.relu3_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_8_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_8_, output shape: {[4,120,120]}

            self.batchnorm4_1_8_ = gluon.nn.BatchNorm()
            # batchnorm4_1_8_, output shape: {[4,120,120]}

            self.relu4_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_8_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_8_, output shape: {[256,120,120]}

            self.batchnorm5_1_8_ = gluon.nn.BatchNorm()
            # batchnorm5_1_8_, output shape: {[256,120,120]}

            self.conv3_1_9_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_9_, output shape: {[4,120,120]}

            self.batchnorm3_1_9_ = gluon.nn.BatchNorm()
            # batchnorm3_1_9_, output shape: {[4,120,120]}

            self.relu3_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_9_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_9_, output shape: {[4,120,120]}

            self.batchnorm4_1_9_ = gluon.nn.BatchNorm()
            # batchnorm4_1_9_, output shape: {[4,120,120]}

            self.relu4_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_9_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_9_, output shape: {[256,120,120]}

            self.batchnorm5_1_9_ = gluon.nn.BatchNorm()
            # batchnorm5_1_9_, output shape: {[256,120,120]}

            self.conv3_1_10_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_10_, output shape: {[4,120,120]}

            self.batchnorm3_1_10_ = gluon.nn.BatchNorm()
            # batchnorm3_1_10_, output shape: {[4,120,120]}

            self.relu3_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_10_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_10_, output shape: {[4,120,120]}

            self.batchnorm4_1_10_ = gluon.nn.BatchNorm()
            # batchnorm4_1_10_, output shape: {[4,120,120]}

            self.relu4_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_10_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_10_, output shape: {[256,120,120]}

            self.batchnorm5_1_10_ = gluon.nn.BatchNorm()
            # batchnorm5_1_10_, output shape: {[256,120,120]}

            self.conv3_1_11_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_11_, output shape: {[4,120,120]}

            self.batchnorm3_1_11_ = gluon.nn.BatchNorm()
            # batchnorm3_1_11_, output shape: {[4,120,120]}

            self.relu3_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_11_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_11_, output shape: {[4,120,120]}

            self.batchnorm4_1_11_ = gluon.nn.BatchNorm()
            # batchnorm4_1_11_, output shape: {[4,120,120]}

            self.relu4_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_11_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_11_, output shape: {[256,120,120]}

            self.batchnorm5_1_11_ = gluon.nn.BatchNorm()
            # batchnorm5_1_11_, output shape: {[256,120,120]}

            self.conv3_1_12_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_12_, output shape: {[4,120,120]}

            self.batchnorm3_1_12_ = gluon.nn.BatchNorm()
            # batchnorm3_1_12_, output shape: {[4,120,120]}

            self.relu3_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_12_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_12_, output shape: {[4,120,120]}

            self.batchnorm4_1_12_ = gluon.nn.BatchNorm()
            # batchnorm4_1_12_, output shape: {[4,120,120]}

            self.relu4_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_12_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_12_, output shape: {[256,120,120]}

            self.batchnorm5_1_12_ = gluon.nn.BatchNorm()
            # batchnorm5_1_12_, output shape: {[256,120,120]}

            self.conv3_1_13_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_13_, output shape: {[4,120,120]}

            self.batchnorm3_1_13_ = gluon.nn.BatchNorm()
            # batchnorm3_1_13_, output shape: {[4,120,120]}

            self.relu3_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_13_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_13_, output shape: {[4,120,120]}

            self.batchnorm4_1_13_ = gluon.nn.BatchNorm()
            # batchnorm4_1_13_, output shape: {[4,120,120]}

            self.relu4_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_13_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_13_, output shape: {[256,120,120]}

            self.batchnorm5_1_13_ = gluon.nn.BatchNorm()
            # batchnorm5_1_13_, output shape: {[256,120,120]}

            self.conv3_1_14_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_14_, output shape: {[4,120,120]}

            self.batchnorm3_1_14_ = gluon.nn.BatchNorm()
            # batchnorm3_1_14_, output shape: {[4,120,120]}

            self.relu3_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_14_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_14_, output shape: {[4,120,120]}

            self.batchnorm4_1_14_ = gluon.nn.BatchNorm()
            # batchnorm4_1_14_, output shape: {[4,120,120]}

            self.relu4_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_14_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_14_, output shape: {[256,120,120]}

            self.batchnorm5_1_14_ = gluon.nn.BatchNorm()
            # batchnorm5_1_14_, output shape: {[256,120,120]}

            self.conv3_1_15_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_15_, output shape: {[4,120,120]}

            self.batchnorm3_1_15_ = gluon.nn.BatchNorm()
            # batchnorm3_1_15_, output shape: {[4,120,120]}

            self.relu3_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_15_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_15_, output shape: {[4,120,120]}

            self.batchnorm4_1_15_ = gluon.nn.BatchNorm()
            # batchnorm4_1_15_, output shape: {[4,120,120]}

            self.relu4_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_15_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_15_, output shape: {[256,120,120]}

            self.batchnorm5_1_15_ = gluon.nn.BatchNorm()
            # batchnorm5_1_15_, output shape: {[256,120,120]}

            self.conv3_1_16_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_16_, output shape: {[4,120,120]}

            self.batchnorm3_1_16_ = gluon.nn.BatchNorm()
            # batchnorm3_1_16_, output shape: {[4,120,120]}

            self.relu3_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_16_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_16_, output shape: {[4,120,120]}

            self.batchnorm4_1_16_ = gluon.nn.BatchNorm()
            # batchnorm4_1_16_, output shape: {[4,120,120]}

            self.relu4_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_16_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_16_, output shape: {[256,120,120]}

            self.batchnorm5_1_16_ = gluon.nn.BatchNorm()
            # batchnorm5_1_16_, output shape: {[256,120,120]}

            self.conv3_1_17_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_17_, output shape: {[4,120,120]}

            self.batchnorm3_1_17_ = gluon.nn.BatchNorm()
            # batchnorm3_1_17_, output shape: {[4,120,120]}

            self.relu3_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_17_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_17_, output shape: {[4,120,120]}

            self.batchnorm4_1_17_ = gluon.nn.BatchNorm()
            # batchnorm4_1_17_, output shape: {[4,120,120]}

            self.relu4_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_17_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_17_, output shape: {[256,120,120]}

            self.batchnorm5_1_17_ = gluon.nn.BatchNorm()
            # batchnorm5_1_17_, output shape: {[256,120,120]}

            self.conv3_1_18_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_18_, output shape: {[4,120,120]}

            self.batchnorm3_1_18_ = gluon.nn.BatchNorm()
            # batchnorm3_1_18_, output shape: {[4,120,120]}

            self.relu3_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_18_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_18_, output shape: {[4,120,120]}

            self.batchnorm4_1_18_ = gluon.nn.BatchNorm()
            # batchnorm4_1_18_, output shape: {[4,120,120]}

            self.relu4_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_18_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_18_, output shape: {[256,120,120]}

            self.batchnorm5_1_18_ = gluon.nn.BatchNorm()
            # batchnorm5_1_18_, output shape: {[256,120,120]}

            self.conv3_1_19_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_19_, output shape: {[4,120,120]}

            self.batchnorm3_1_19_ = gluon.nn.BatchNorm()
            # batchnorm3_1_19_, output shape: {[4,120,120]}

            self.relu3_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_19_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_19_, output shape: {[4,120,120]}

            self.batchnorm4_1_19_ = gluon.nn.BatchNorm()
            # batchnorm4_1_19_, output shape: {[4,120,120]}

            self.relu4_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_19_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_19_, output shape: {[256,120,120]}

            self.batchnorm5_1_19_ = gluon.nn.BatchNorm()
            # batchnorm5_1_19_, output shape: {[256,120,120]}

            self.conv3_1_20_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_20_, output shape: {[4,120,120]}

            self.batchnorm3_1_20_ = gluon.nn.BatchNorm()
            # batchnorm3_1_20_, output shape: {[4,120,120]}

            self.relu3_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_20_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_20_, output shape: {[4,120,120]}

            self.batchnorm4_1_20_ = gluon.nn.BatchNorm()
            # batchnorm4_1_20_, output shape: {[4,120,120]}

            self.relu4_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_20_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_20_, output shape: {[256,120,120]}

            self.batchnorm5_1_20_ = gluon.nn.BatchNorm()
            # batchnorm5_1_20_, output shape: {[256,120,120]}

            self.conv3_1_21_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_21_, output shape: {[4,120,120]}

            self.batchnorm3_1_21_ = gluon.nn.BatchNorm()
            # batchnorm3_1_21_, output shape: {[4,120,120]}

            self.relu3_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_21_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_21_, output shape: {[4,120,120]}

            self.batchnorm4_1_21_ = gluon.nn.BatchNorm()
            # batchnorm4_1_21_, output shape: {[4,120,120]}

            self.relu4_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_21_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_21_, output shape: {[256,120,120]}

            self.batchnorm5_1_21_ = gluon.nn.BatchNorm()
            # batchnorm5_1_21_, output shape: {[256,120,120]}

            self.conv3_1_22_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_22_, output shape: {[4,120,120]}

            self.batchnorm3_1_22_ = gluon.nn.BatchNorm()
            # batchnorm3_1_22_, output shape: {[4,120,120]}

            self.relu3_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_22_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_22_, output shape: {[4,120,120]}

            self.batchnorm4_1_22_ = gluon.nn.BatchNorm()
            # batchnorm4_1_22_, output shape: {[4,120,120]}

            self.relu4_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_22_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_22_, output shape: {[256,120,120]}

            self.batchnorm5_1_22_ = gluon.nn.BatchNorm()
            # batchnorm5_1_22_, output shape: {[256,120,120]}

            self.conv3_1_23_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_23_, output shape: {[4,120,120]}

            self.batchnorm3_1_23_ = gluon.nn.BatchNorm()
            # batchnorm3_1_23_, output shape: {[4,120,120]}

            self.relu3_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_23_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_23_, output shape: {[4,120,120]}

            self.batchnorm4_1_23_ = gluon.nn.BatchNorm()
            # batchnorm4_1_23_, output shape: {[4,120,120]}

            self.relu4_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_23_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_23_, output shape: {[256,120,120]}

            self.batchnorm5_1_23_ = gluon.nn.BatchNorm()
            # batchnorm5_1_23_, output shape: {[256,120,120]}

            self.conv3_1_24_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_24_, output shape: {[4,120,120]}

            self.batchnorm3_1_24_ = gluon.nn.BatchNorm()
            # batchnorm3_1_24_, output shape: {[4,120,120]}

            self.relu3_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_24_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_24_, output shape: {[4,120,120]}

            self.batchnorm4_1_24_ = gluon.nn.BatchNorm()
            # batchnorm4_1_24_, output shape: {[4,120,120]}

            self.relu4_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_24_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_24_, output shape: {[256,120,120]}

            self.batchnorm5_1_24_ = gluon.nn.BatchNorm()
            # batchnorm5_1_24_, output shape: {[256,120,120]}

            self.conv3_1_25_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_25_, output shape: {[4,120,120]}

            self.batchnorm3_1_25_ = gluon.nn.BatchNorm()
            # batchnorm3_1_25_, output shape: {[4,120,120]}

            self.relu3_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_25_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_25_, output shape: {[4,120,120]}

            self.batchnorm4_1_25_ = gluon.nn.BatchNorm()
            # batchnorm4_1_25_, output shape: {[4,120,120]}

            self.relu4_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_25_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_25_, output shape: {[256,120,120]}

            self.batchnorm5_1_25_ = gluon.nn.BatchNorm()
            # batchnorm5_1_25_, output shape: {[256,120,120]}

            self.conv3_1_26_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_26_, output shape: {[4,120,120]}

            self.batchnorm3_1_26_ = gluon.nn.BatchNorm()
            # batchnorm3_1_26_, output shape: {[4,120,120]}

            self.relu3_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_26_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_26_, output shape: {[4,120,120]}

            self.batchnorm4_1_26_ = gluon.nn.BatchNorm()
            # batchnorm4_1_26_, output shape: {[4,120,120]}

            self.relu4_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_26_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_26_, output shape: {[256,120,120]}

            self.batchnorm5_1_26_ = gluon.nn.BatchNorm()
            # batchnorm5_1_26_, output shape: {[256,120,120]}

            self.conv3_1_27_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_27_, output shape: {[4,120,120]}

            self.batchnorm3_1_27_ = gluon.nn.BatchNorm()
            # batchnorm3_1_27_, output shape: {[4,120,120]}

            self.relu3_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_27_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_27_, output shape: {[4,120,120]}

            self.batchnorm4_1_27_ = gluon.nn.BatchNorm()
            # batchnorm4_1_27_, output shape: {[4,120,120]}

            self.relu4_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_27_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_27_, output shape: {[256,120,120]}

            self.batchnorm5_1_27_ = gluon.nn.BatchNorm()
            # batchnorm5_1_27_, output shape: {[256,120,120]}

            self.conv3_1_28_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_28_, output shape: {[4,120,120]}

            self.batchnorm3_1_28_ = gluon.nn.BatchNorm()
            # batchnorm3_1_28_, output shape: {[4,120,120]}

            self.relu3_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_28_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_28_, output shape: {[4,120,120]}

            self.batchnorm4_1_28_ = gluon.nn.BatchNorm()
            # batchnorm4_1_28_, output shape: {[4,120,120]}

            self.relu4_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_28_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_28_, output shape: {[256,120,120]}

            self.batchnorm5_1_28_ = gluon.nn.BatchNorm()
            # batchnorm5_1_28_, output shape: {[256,120,120]}

            self.conv3_1_29_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_29_, output shape: {[4,120,120]}

            self.batchnorm3_1_29_ = gluon.nn.BatchNorm()
            # batchnorm3_1_29_, output shape: {[4,120,120]}

            self.relu3_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_29_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_29_, output shape: {[4,120,120]}

            self.batchnorm4_1_29_ = gluon.nn.BatchNorm()
            # batchnorm4_1_29_, output shape: {[4,120,120]}

            self.relu4_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_29_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_29_, output shape: {[256,120,120]}

            self.batchnorm5_1_29_ = gluon.nn.BatchNorm()
            # batchnorm5_1_29_, output shape: {[256,120,120]}

            self.conv3_1_30_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_30_, output shape: {[4,120,120]}

            self.batchnorm3_1_30_ = gluon.nn.BatchNorm()
            # batchnorm3_1_30_, output shape: {[4,120,120]}

            self.relu3_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_30_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_30_, output shape: {[4,120,120]}

            self.batchnorm4_1_30_ = gluon.nn.BatchNorm()
            # batchnorm4_1_30_, output shape: {[4,120,120]}

            self.relu4_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_30_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_30_, output shape: {[256,120,120]}

            self.batchnorm5_1_30_ = gluon.nn.BatchNorm()
            # batchnorm5_1_30_, output shape: {[256,120,120]}

            self.conv3_1_31_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_31_, output shape: {[4,120,120]}

            self.batchnorm3_1_31_ = gluon.nn.BatchNorm()
            # batchnorm3_1_31_, output shape: {[4,120,120]}

            self.relu3_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_31_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_31_, output shape: {[4,120,120]}

            self.batchnorm4_1_31_ = gluon.nn.BatchNorm()
            # batchnorm4_1_31_, output shape: {[4,120,120]}

            self.relu4_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_31_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_31_, output shape: {[256,120,120]}

            self.batchnorm5_1_31_ = gluon.nn.BatchNorm()
            # batchnorm5_1_31_, output shape: {[256,120,120]}

            self.conv3_1_32_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv3_1_32_, output shape: {[4,120,120]}

            self.batchnorm3_1_32_ = gluon.nn.BatchNorm()
            # batchnorm3_1_32_, output shape: {[4,120,120]}

            self.relu3_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_32_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv4_1_32_, output shape: {[4,120,120]}

            self.batchnorm4_1_32_ = gluon.nn.BatchNorm()
            # batchnorm4_1_32_, output shape: {[4,120,120]}

            self.relu4_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv5_1_32_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv5_1_32_, output shape: {[256,120,120]}

            self.batchnorm5_1_32_ = gluon.nn.BatchNorm()
            # batchnorm5_1_32_, output shape: {[256,120,120]}

            self.conv2_2_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv2_2_, output shape: {[256,120,120]}

            self.batchnorm2_2_ = gluon.nn.BatchNorm()
            # batchnorm2_2_, output shape: {[256,120,120]}

            self.relu7_ = gluon.nn.Activation(activation='relu')
            self.conv9_1_1_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_1_, output shape: {[4,120,120]}

            self.batchnorm9_1_1_ = gluon.nn.BatchNorm()
            # batchnorm9_1_1_, output shape: {[4,120,120]}

            self.relu9_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_1_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_1_, output shape: {[4,120,120]}

            self.batchnorm10_1_1_ = gluon.nn.BatchNorm()
            # batchnorm10_1_1_, output shape: {[4,120,120]}

            self.relu10_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_1_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_1_, output shape: {[256,120,120]}

            self.batchnorm11_1_1_ = gluon.nn.BatchNorm()
            # batchnorm11_1_1_, output shape: {[256,120,120]}

            self.conv9_1_2_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_2_, output shape: {[4,120,120]}

            self.batchnorm9_1_2_ = gluon.nn.BatchNorm()
            # batchnorm9_1_2_, output shape: {[4,120,120]}

            self.relu9_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_2_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_2_, output shape: {[4,120,120]}

            self.batchnorm10_1_2_ = gluon.nn.BatchNorm()
            # batchnorm10_1_2_, output shape: {[4,120,120]}

            self.relu10_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_2_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_2_, output shape: {[256,120,120]}

            self.batchnorm11_1_2_ = gluon.nn.BatchNorm()
            # batchnorm11_1_2_, output shape: {[256,120,120]}

            self.conv9_1_3_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_3_, output shape: {[4,120,120]}

            self.batchnorm9_1_3_ = gluon.nn.BatchNorm()
            # batchnorm9_1_3_, output shape: {[4,120,120]}

            self.relu9_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_3_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_3_, output shape: {[4,120,120]}

            self.batchnorm10_1_3_ = gluon.nn.BatchNorm()
            # batchnorm10_1_3_, output shape: {[4,120,120]}

            self.relu10_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_3_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_3_, output shape: {[256,120,120]}

            self.batchnorm11_1_3_ = gluon.nn.BatchNorm()
            # batchnorm11_1_3_, output shape: {[256,120,120]}

            self.conv9_1_4_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_4_, output shape: {[4,120,120]}

            self.batchnorm9_1_4_ = gluon.nn.BatchNorm()
            # batchnorm9_1_4_, output shape: {[4,120,120]}

            self.relu9_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_4_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_4_, output shape: {[4,120,120]}

            self.batchnorm10_1_4_ = gluon.nn.BatchNorm()
            # batchnorm10_1_4_, output shape: {[4,120,120]}

            self.relu10_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_4_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_4_, output shape: {[256,120,120]}

            self.batchnorm11_1_4_ = gluon.nn.BatchNorm()
            # batchnorm11_1_4_, output shape: {[256,120,120]}

            self.conv9_1_5_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_5_, output shape: {[4,120,120]}

            self.batchnorm9_1_5_ = gluon.nn.BatchNorm()
            # batchnorm9_1_5_, output shape: {[4,120,120]}

            self.relu9_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_5_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_5_, output shape: {[4,120,120]}

            self.batchnorm10_1_5_ = gluon.nn.BatchNorm()
            # batchnorm10_1_5_, output shape: {[4,120,120]}

            self.relu10_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_5_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_5_, output shape: {[256,120,120]}

            self.batchnorm11_1_5_ = gluon.nn.BatchNorm()
            # batchnorm11_1_5_, output shape: {[256,120,120]}

            self.conv9_1_6_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_6_, output shape: {[4,120,120]}

            self.batchnorm9_1_6_ = gluon.nn.BatchNorm()
            # batchnorm9_1_6_, output shape: {[4,120,120]}

            self.relu9_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_6_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_6_, output shape: {[4,120,120]}

            self.batchnorm10_1_6_ = gluon.nn.BatchNorm()
            # batchnorm10_1_6_, output shape: {[4,120,120]}

            self.relu10_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_6_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_6_, output shape: {[256,120,120]}

            self.batchnorm11_1_6_ = gluon.nn.BatchNorm()
            # batchnorm11_1_6_, output shape: {[256,120,120]}

            self.conv9_1_7_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_7_, output shape: {[4,120,120]}

            self.batchnorm9_1_7_ = gluon.nn.BatchNorm()
            # batchnorm9_1_7_, output shape: {[4,120,120]}

            self.relu9_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_7_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_7_, output shape: {[4,120,120]}

            self.batchnorm10_1_7_ = gluon.nn.BatchNorm()
            # batchnorm10_1_7_, output shape: {[4,120,120]}

            self.relu10_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_7_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_7_, output shape: {[256,120,120]}

            self.batchnorm11_1_7_ = gluon.nn.BatchNorm()
            # batchnorm11_1_7_, output shape: {[256,120,120]}

            self.conv9_1_8_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_8_, output shape: {[4,120,120]}

            self.batchnorm9_1_8_ = gluon.nn.BatchNorm()
            # batchnorm9_1_8_, output shape: {[4,120,120]}

            self.relu9_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_8_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_8_, output shape: {[4,120,120]}

            self.batchnorm10_1_8_ = gluon.nn.BatchNorm()
            # batchnorm10_1_8_, output shape: {[4,120,120]}

            self.relu10_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_8_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_8_, output shape: {[256,120,120]}

            self.batchnorm11_1_8_ = gluon.nn.BatchNorm()
            # batchnorm11_1_8_, output shape: {[256,120,120]}

            self.conv9_1_9_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_9_, output shape: {[4,120,120]}

            self.batchnorm9_1_9_ = gluon.nn.BatchNorm()
            # batchnorm9_1_9_, output shape: {[4,120,120]}

            self.relu9_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_9_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_9_, output shape: {[4,120,120]}

            self.batchnorm10_1_9_ = gluon.nn.BatchNorm()
            # batchnorm10_1_9_, output shape: {[4,120,120]}

            self.relu10_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_9_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_9_, output shape: {[256,120,120]}

            self.batchnorm11_1_9_ = gluon.nn.BatchNorm()
            # batchnorm11_1_9_, output shape: {[256,120,120]}

            self.conv9_1_10_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_10_, output shape: {[4,120,120]}

            self.batchnorm9_1_10_ = gluon.nn.BatchNorm()
            # batchnorm9_1_10_, output shape: {[4,120,120]}

            self.relu9_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_10_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_10_, output shape: {[4,120,120]}

            self.batchnorm10_1_10_ = gluon.nn.BatchNorm()
            # batchnorm10_1_10_, output shape: {[4,120,120]}

            self.relu10_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_10_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_10_, output shape: {[256,120,120]}

            self.batchnorm11_1_10_ = gluon.nn.BatchNorm()
            # batchnorm11_1_10_, output shape: {[256,120,120]}

            self.conv9_1_11_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_11_, output shape: {[4,120,120]}

            self.batchnorm9_1_11_ = gluon.nn.BatchNorm()
            # batchnorm9_1_11_, output shape: {[4,120,120]}

            self.relu9_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_11_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_11_, output shape: {[4,120,120]}

            self.batchnorm10_1_11_ = gluon.nn.BatchNorm()
            # batchnorm10_1_11_, output shape: {[4,120,120]}

            self.relu10_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_11_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_11_, output shape: {[256,120,120]}

            self.batchnorm11_1_11_ = gluon.nn.BatchNorm()
            # batchnorm11_1_11_, output shape: {[256,120,120]}

            self.conv9_1_12_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_12_, output shape: {[4,120,120]}

            self.batchnorm9_1_12_ = gluon.nn.BatchNorm()
            # batchnorm9_1_12_, output shape: {[4,120,120]}

            self.relu9_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_12_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_12_, output shape: {[4,120,120]}

            self.batchnorm10_1_12_ = gluon.nn.BatchNorm()
            # batchnorm10_1_12_, output shape: {[4,120,120]}

            self.relu10_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_12_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_12_, output shape: {[256,120,120]}

            self.batchnorm11_1_12_ = gluon.nn.BatchNorm()
            # batchnorm11_1_12_, output shape: {[256,120,120]}

            self.conv9_1_13_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_13_, output shape: {[4,120,120]}

            self.batchnorm9_1_13_ = gluon.nn.BatchNorm()
            # batchnorm9_1_13_, output shape: {[4,120,120]}

            self.relu9_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_13_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_13_, output shape: {[4,120,120]}

            self.batchnorm10_1_13_ = gluon.nn.BatchNorm()
            # batchnorm10_1_13_, output shape: {[4,120,120]}

            self.relu10_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_13_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_13_, output shape: {[256,120,120]}

            self.batchnorm11_1_13_ = gluon.nn.BatchNorm()
            # batchnorm11_1_13_, output shape: {[256,120,120]}

            self.conv9_1_14_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_14_, output shape: {[4,120,120]}

            self.batchnorm9_1_14_ = gluon.nn.BatchNorm()
            # batchnorm9_1_14_, output shape: {[4,120,120]}

            self.relu9_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_14_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_14_, output shape: {[4,120,120]}

            self.batchnorm10_1_14_ = gluon.nn.BatchNorm()
            # batchnorm10_1_14_, output shape: {[4,120,120]}

            self.relu10_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_14_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_14_, output shape: {[256,120,120]}

            self.batchnorm11_1_14_ = gluon.nn.BatchNorm()
            # batchnorm11_1_14_, output shape: {[256,120,120]}

            self.conv9_1_15_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_15_, output shape: {[4,120,120]}

            self.batchnorm9_1_15_ = gluon.nn.BatchNorm()
            # batchnorm9_1_15_, output shape: {[4,120,120]}

            self.relu9_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_15_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_15_, output shape: {[4,120,120]}

            self.batchnorm10_1_15_ = gluon.nn.BatchNorm()
            # batchnorm10_1_15_, output shape: {[4,120,120]}

            self.relu10_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_15_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_15_, output shape: {[256,120,120]}

            self.batchnorm11_1_15_ = gluon.nn.BatchNorm()
            # batchnorm11_1_15_, output shape: {[256,120,120]}

            self.conv9_1_16_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_16_, output shape: {[4,120,120]}

            self.batchnorm9_1_16_ = gluon.nn.BatchNorm()
            # batchnorm9_1_16_, output shape: {[4,120,120]}

            self.relu9_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_16_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_16_, output shape: {[4,120,120]}

            self.batchnorm10_1_16_ = gluon.nn.BatchNorm()
            # batchnorm10_1_16_, output shape: {[4,120,120]}

            self.relu10_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_16_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_16_, output shape: {[256,120,120]}

            self.batchnorm11_1_16_ = gluon.nn.BatchNorm()
            # batchnorm11_1_16_, output shape: {[256,120,120]}

            self.conv9_1_17_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_17_, output shape: {[4,120,120]}

            self.batchnorm9_1_17_ = gluon.nn.BatchNorm()
            # batchnorm9_1_17_, output shape: {[4,120,120]}

            self.relu9_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_17_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_17_, output shape: {[4,120,120]}

            self.batchnorm10_1_17_ = gluon.nn.BatchNorm()
            # batchnorm10_1_17_, output shape: {[4,120,120]}

            self.relu10_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_17_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_17_, output shape: {[256,120,120]}

            self.batchnorm11_1_17_ = gluon.nn.BatchNorm()
            # batchnorm11_1_17_, output shape: {[256,120,120]}

            self.conv9_1_18_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_18_, output shape: {[4,120,120]}

            self.batchnorm9_1_18_ = gluon.nn.BatchNorm()
            # batchnorm9_1_18_, output shape: {[4,120,120]}

            self.relu9_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_18_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_18_, output shape: {[4,120,120]}

            self.batchnorm10_1_18_ = gluon.nn.BatchNorm()
            # batchnorm10_1_18_, output shape: {[4,120,120]}

            self.relu10_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_18_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_18_, output shape: {[256,120,120]}

            self.batchnorm11_1_18_ = gluon.nn.BatchNorm()
            # batchnorm11_1_18_, output shape: {[256,120,120]}

            self.conv9_1_19_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_19_, output shape: {[4,120,120]}

            self.batchnorm9_1_19_ = gluon.nn.BatchNorm()
            # batchnorm9_1_19_, output shape: {[4,120,120]}

            self.relu9_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_19_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_19_, output shape: {[4,120,120]}

            self.batchnorm10_1_19_ = gluon.nn.BatchNorm()
            # batchnorm10_1_19_, output shape: {[4,120,120]}

            self.relu10_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_19_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_19_, output shape: {[256,120,120]}

            self.batchnorm11_1_19_ = gluon.nn.BatchNorm()
            # batchnorm11_1_19_, output shape: {[256,120,120]}

            self.conv9_1_20_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_20_, output shape: {[4,120,120]}

            self.batchnorm9_1_20_ = gluon.nn.BatchNorm()
            # batchnorm9_1_20_, output shape: {[4,120,120]}

            self.relu9_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_20_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_20_, output shape: {[4,120,120]}

            self.batchnorm10_1_20_ = gluon.nn.BatchNorm()
            # batchnorm10_1_20_, output shape: {[4,120,120]}

            self.relu10_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_20_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_20_, output shape: {[256,120,120]}

            self.batchnorm11_1_20_ = gluon.nn.BatchNorm()
            # batchnorm11_1_20_, output shape: {[256,120,120]}

            self.conv9_1_21_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_21_, output shape: {[4,120,120]}

            self.batchnorm9_1_21_ = gluon.nn.BatchNorm()
            # batchnorm9_1_21_, output shape: {[4,120,120]}

            self.relu9_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_21_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_21_, output shape: {[4,120,120]}

            self.batchnorm10_1_21_ = gluon.nn.BatchNorm()
            # batchnorm10_1_21_, output shape: {[4,120,120]}

            self.relu10_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_21_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_21_, output shape: {[256,120,120]}

            self.batchnorm11_1_21_ = gluon.nn.BatchNorm()
            # batchnorm11_1_21_, output shape: {[256,120,120]}

            self.conv9_1_22_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_22_, output shape: {[4,120,120]}

            self.batchnorm9_1_22_ = gluon.nn.BatchNorm()
            # batchnorm9_1_22_, output shape: {[4,120,120]}

            self.relu9_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_22_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_22_, output shape: {[4,120,120]}

            self.batchnorm10_1_22_ = gluon.nn.BatchNorm()
            # batchnorm10_1_22_, output shape: {[4,120,120]}

            self.relu10_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_22_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_22_, output shape: {[256,120,120]}

            self.batchnorm11_1_22_ = gluon.nn.BatchNorm()
            # batchnorm11_1_22_, output shape: {[256,120,120]}

            self.conv9_1_23_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_23_, output shape: {[4,120,120]}

            self.batchnorm9_1_23_ = gluon.nn.BatchNorm()
            # batchnorm9_1_23_, output shape: {[4,120,120]}

            self.relu9_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_23_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_23_, output shape: {[4,120,120]}

            self.batchnorm10_1_23_ = gluon.nn.BatchNorm()
            # batchnorm10_1_23_, output shape: {[4,120,120]}

            self.relu10_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_23_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_23_, output shape: {[256,120,120]}

            self.batchnorm11_1_23_ = gluon.nn.BatchNorm()
            # batchnorm11_1_23_, output shape: {[256,120,120]}

            self.conv9_1_24_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_24_, output shape: {[4,120,120]}

            self.batchnorm9_1_24_ = gluon.nn.BatchNorm()
            # batchnorm9_1_24_, output shape: {[4,120,120]}

            self.relu9_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_24_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_24_, output shape: {[4,120,120]}

            self.batchnorm10_1_24_ = gluon.nn.BatchNorm()
            # batchnorm10_1_24_, output shape: {[4,120,120]}

            self.relu10_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_24_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_24_, output shape: {[256,120,120]}

            self.batchnorm11_1_24_ = gluon.nn.BatchNorm()
            # batchnorm11_1_24_, output shape: {[256,120,120]}

            self.conv9_1_25_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_25_, output shape: {[4,120,120]}

            self.batchnorm9_1_25_ = gluon.nn.BatchNorm()
            # batchnorm9_1_25_, output shape: {[4,120,120]}

            self.relu9_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_25_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_25_, output shape: {[4,120,120]}

            self.batchnorm10_1_25_ = gluon.nn.BatchNorm()
            # batchnorm10_1_25_, output shape: {[4,120,120]}

            self.relu10_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_25_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_25_, output shape: {[256,120,120]}

            self.batchnorm11_1_25_ = gluon.nn.BatchNorm()
            # batchnorm11_1_25_, output shape: {[256,120,120]}

            self.conv9_1_26_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_26_, output shape: {[4,120,120]}

            self.batchnorm9_1_26_ = gluon.nn.BatchNorm()
            # batchnorm9_1_26_, output shape: {[4,120,120]}

            self.relu9_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_26_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_26_, output shape: {[4,120,120]}

            self.batchnorm10_1_26_ = gluon.nn.BatchNorm()
            # batchnorm10_1_26_, output shape: {[4,120,120]}

            self.relu10_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_26_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_26_, output shape: {[256,120,120]}

            self.batchnorm11_1_26_ = gluon.nn.BatchNorm()
            # batchnorm11_1_26_, output shape: {[256,120,120]}

            self.conv9_1_27_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_27_, output shape: {[4,120,120]}

            self.batchnorm9_1_27_ = gluon.nn.BatchNorm()
            # batchnorm9_1_27_, output shape: {[4,120,120]}

            self.relu9_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_27_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_27_, output shape: {[4,120,120]}

            self.batchnorm10_1_27_ = gluon.nn.BatchNorm()
            # batchnorm10_1_27_, output shape: {[4,120,120]}

            self.relu10_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_27_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_27_, output shape: {[256,120,120]}

            self.batchnorm11_1_27_ = gluon.nn.BatchNorm()
            # batchnorm11_1_27_, output shape: {[256,120,120]}

            self.conv9_1_28_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_28_, output shape: {[4,120,120]}

            self.batchnorm9_1_28_ = gluon.nn.BatchNorm()
            # batchnorm9_1_28_, output shape: {[4,120,120]}

            self.relu9_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_28_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_28_, output shape: {[4,120,120]}

            self.batchnorm10_1_28_ = gluon.nn.BatchNorm()
            # batchnorm10_1_28_, output shape: {[4,120,120]}

            self.relu10_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_28_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_28_, output shape: {[256,120,120]}

            self.batchnorm11_1_28_ = gluon.nn.BatchNorm()
            # batchnorm11_1_28_, output shape: {[256,120,120]}

            self.conv9_1_29_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_29_, output shape: {[4,120,120]}

            self.batchnorm9_1_29_ = gluon.nn.BatchNorm()
            # batchnorm9_1_29_, output shape: {[4,120,120]}

            self.relu9_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_29_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_29_, output shape: {[4,120,120]}

            self.batchnorm10_1_29_ = gluon.nn.BatchNorm()
            # batchnorm10_1_29_, output shape: {[4,120,120]}

            self.relu10_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_29_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_29_, output shape: {[256,120,120]}

            self.batchnorm11_1_29_ = gluon.nn.BatchNorm()
            # batchnorm11_1_29_, output shape: {[256,120,120]}

            self.conv9_1_30_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_30_, output shape: {[4,120,120]}

            self.batchnorm9_1_30_ = gluon.nn.BatchNorm()
            # batchnorm9_1_30_, output shape: {[4,120,120]}

            self.relu9_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_30_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_30_, output shape: {[4,120,120]}

            self.batchnorm10_1_30_ = gluon.nn.BatchNorm()
            # batchnorm10_1_30_, output shape: {[4,120,120]}

            self.relu10_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_30_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_30_, output shape: {[256,120,120]}

            self.batchnorm11_1_30_ = gluon.nn.BatchNorm()
            # batchnorm11_1_30_, output shape: {[256,120,120]}

            self.conv9_1_31_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_31_, output shape: {[4,120,120]}

            self.batchnorm9_1_31_ = gluon.nn.BatchNorm()
            # batchnorm9_1_31_, output shape: {[4,120,120]}

            self.relu9_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_31_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_31_, output shape: {[4,120,120]}

            self.batchnorm10_1_31_ = gluon.nn.BatchNorm()
            # batchnorm10_1_31_, output shape: {[4,120,120]}

            self.relu10_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_31_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_31_, output shape: {[256,120,120]}

            self.batchnorm11_1_31_ = gluon.nn.BatchNorm()
            # batchnorm11_1_31_, output shape: {[256,120,120]}

            self.conv9_1_32_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv9_1_32_, output shape: {[4,120,120]}

            self.batchnorm9_1_32_ = gluon.nn.BatchNorm()
            # batchnorm9_1_32_, output shape: {[4,120,120]}

            self.relu9_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_32_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv10_1_32_, output shape: {[4,120,120]}

            self.batchnorm10_1_32_ = gluon.nn.BatchNorm()
            # batchnorm10_1_32_, output shape: {[4,120,120]}

            self.relu10_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv11_1_32_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv11_1_32_, output shape: {[256,120,120]}

            self.batchnorm11_1_32_ = gluon.nn.BatchNorm()
            # batchnorm11_1_32_, output shape: {[256,120,120]}

            self.conv8_2_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv8_2_, output shape: {[256,120,120]}

            self.batchnorm8_2_ = gluon.nn.BatchNorm()
            # batchnorm8_2_, output shape: {[256,120,120]}

            self.relu13_ = gluon.nn.Activation(activation='relu')
            self.conv15_1_1_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_1_, output shape: {[4,120,120]}

            self.batchnorm15_1_1_ = gluon.nn.BatchNorm()
            # batchnorm15_1_1_, output shape: {[4,120,120]}

            self.relu15_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_1_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_1_, output shape: {[4,120,120]}

            self.batchnorm16_1_1_ = gluon.nn.BatchNorm()
            # batchnorm16_1_1_, output shape: {[4,120,120]}

            self.relu16_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_1_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_1_, output shape: {[256,120,120]}

            self.batchnorm17_1_1_ = gluon.nn.BatchNorm()
            # batchnorm17_1_1_, output shape: {[256,120,120]}

            self.conv15_1_2_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_2_, output shape: {[4,120,120]}

            self.batchnorm15_1_2_ = gluon.nn.BatchNorm()
            # batchnorm15_1_2_, output shape: {[4,120,120]}

            self.relu15_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_2_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_2_, output shape: {[4,120,120]}

            self.batchnorm16_1_2_ = gluon.nn.BatchNorm()
            # batchnorm16_1_2_, output shape: {[4,120,120]}

            self.relu16_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_2_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_2_, output shape: {[256,120,120]}

            self.batchnorm17_1_2_ = gluon.nn.BatchNorm()
            # batchnorm17_1_2_, output shape: {[256,120,120]}

            self.conv15_1_3_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_3_, output shape: {[4,120,120]}

            self.batchnorm15_1_3_ = gluon.nn.BatchNorm()
            # batchnorm15_1_3_, output shape: {[4,120,120]}

            self.relu15_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_3_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_3_, output shape: {[4,120,120]}

            self.batchnorm16_1_3_ = gluon.nn.BatchNorm()
            # batchnorm16_1_3_, output shape: {[4,120,120]}

            self.relu16_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_3_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_3_, output shape: {[256,120,120]}

            self.batchnorm17_1_3_ = gluon.nn.BatchNorm()
            # batchnorm17_1_3_, output shape: {[256,120,120]}

            self.conv15_1_4_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_4_, output shape: {[4,120,120]}

            self.batchnorm15_1_4_ = gluon.nn.BatchNorm()
            # batchnorm15_1_4_, output shape: {[4,120,120]}

            self.relu15_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_4_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_4_, output shape: {[4,120,120]}

            self.batchnorm16_1_4_ = gluon.nn.BatchNorm()
            # batchnorm16_1_4_, output shape: {[4,120,120]}

            self.relu16_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_4_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_4_, output shape: {[256,120,120]}

            self.batchnorm17_1_4_ = gluon.nn.BatchNorm()
            # batchnorm17_1_4_, output shape: {[256,120,120]}

            self.conv15_1_5_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_5_, output shape: {[4,120,120]}

            self.batchnorm15_1_5_ = gluon.nn.BatchNorm()
            # batchnorm15_1_5_, output shape: {[4,120,120]}

            self.relu15_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_5_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_5_, output shape: {[4,120,120]}

            self.batchnorm16_1_5_ = gluon.nn.BatchNorm()
            # batchnorm16_1_5_, output shape: {[4,120,120]}

            self.relu16_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_5_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_5_, output shape: {[256,120,120]}

            self.batchnorm17_1_5_ = gluon.nn.BatchNorm()
            # batchnorm17_1_5_, output shape: {[256,120,120]}

            self.conv15_1_6_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_6_, output shape: {[4,120,120]}

            self.batchnorm15_1_6_ = gluon.nn.BatchNorm()
            # batchnorm15_1_6_, output shape: {[4,120,120]}

            self.relu15_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_6_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_6_, output shape: {[4,120,120]}

            self.batchnorm16_1_6_ = gluon.nn.BatchNorm()
            # batchnorm16_1_6_, output shape: {[4,120,120]}

            self.relu16_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_6_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_6_, output shape: {[256,120,120]}

            self.batchnorm17_1_6_ = gluon.nn.BatchNorm()
            # batchnorm17_1_6_, output shape: {[256,120,120]}

            self.conv15_1_7_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_7_, output shape: {[4,120,120]}

            self.batchnorm15_1_7_ = gluon.nn.BatchNorm()
            # batchnorm15_1_7_, output shape: {[4,120,120]}

            self.relu15_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_7_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_7_, output shape: {[4,120,120]}

            self.batchnorm16_1_7_ = gluon.nn.BatchNorm()
            # batchnorm16_1_7_, output shape: {[4,120,120]}

            self.relu16_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_7_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_7_, output shape: {[256,120,120]}

            self.batchnorm17_1_7_ = gluon.nn.BatchNorm()
            # batchnorm17_1_7_, output shape: {[256,120,120]}

            self.conv15_1_8_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_8_, output shape: {[4,120,120]}

            self.batchnorm15_1_8_ = gluon.nn.BatchNorm()
            # batchnorm15_1_8_, output shape: {[4,120,120]}

            self.relu15_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_8_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_8_, output shape: {[4,120,120]}

            self.batchnorm16_1_8_ = gluon.nn.BatchNorm()
            # batchnorm16_1_8_, output shape: {[4,120,120]}

            self.relu16_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_8_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_8_, output shape: {[256,120,120]}

            self.batchnorm17_1_8_ = gluon.nn.BatchNorm()
            # batchnorm17_1_8_, output shape: {[256,120,120]}

            self.conv15_1_9_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_9_, output shape: {[4,120,120]}

            self.batchnorm15_1_9_ = gluon.nn.BatchNorm()
            # batchnorm15_1_9_, output shape: {[4,120,120]}

            self.relu15_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_9_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_9_, output shape: {[4,120,120]}

            self.batchnorm16_1_9_ = gluon.nn.BatchNorm()
            # batchnorm16_1_9_, output shape: {[4,120,120]}

            self.relu16_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_9_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_9_, output shape: {[256,120,120]}

            self.batchnorm17_1_9_ = gluon.nn.BatchNorm()
            # batchnorm17_1_9_, output shape: {[256,120,120]}

            self.conv15_1_10_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_10_, output shape: {[4,120,120]}

            self.batchnorm15_1_10_ = gluon.nn.BatchNorm()
            # batchnorm15_1_10_, output shape: {[4,120,120]}

            self.relu15_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_10_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_10_, output shape: {[4,120,120]}

            self.batchnorm16_1_10_ = gluon.nn.BatchNorm()
            # batchnorm16_1_10_, output shape: {[4,120,120]}

            self.relu16_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_10_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_10_, output shape: {[256,120,120]}

            self.batchnorm17_1_10_ = gluon.nn.BatchNorm()
            # batchnorm17_1_10_, output shape: {[256,120,120]}

            self.conv15_1_11_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_11_, output shape: {[4,120,120]}

            self.batchnorm15_1_11_ = gluon.nn.BatchNorm()
            # batchnorm15_1_11_, output shape: {[4,120,120]}

            self.relu15_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_11_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_11_, output shape: {[4,120,120]}

            self.batchnorm16_1_11_ = gluon.nn.BatchNorm()
            # batchnorm16_1_11_, output shape: {[4,120,120]}

            self.relu16_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_11_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_11_, output shape: {[256,120,120]}

            self.batchnorm17_1_11_ = gluon.nn.BatchNorm()
            # batchnorm17_1_11_, output shape: {[256,120,120]}

            self.conv15_1_12_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_12_, output shape: {[4,120,120]}

            self.batchnorm15_1_12_ = gluon.nn.BatchNorm()
            # batchnorm15_1_12_, output shape: {[4,120,120]}

            self.relu15_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_12_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_12_, output shape: {[4,120,120]}

            self.batchnorm16_1_12_ = gluon.nn.BatchNorm()
            # batchnorm16_1_12_, output shape: {[4,120,120]}

            self.relu16_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_12_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_12_, output shape: {[256,120,120]}

            self.batchnorm17_1_12_ = gluon.nn.BatchNorm()
            # batchnorm17_1_12_, output shape: {[256,120,120]}

            self.conv15_1_13_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_13_, output shape: {[4,120,120]}

            self.batchnorm15_1_13_ = gluon.nn.BatchNorm()
            # batchnorm15_1_13_, output shape: {[4,120,120]}

            self.relu15_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_13_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_13_, output shape: {[4,120,120]}

            self.batchnorm16_1_13_ = gluon.nn.BatchNorm()
            # batchnorm16_1_13_, output shape: {[4,120,120]}

            self.relu16_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_13_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_13_, output shape: {[256,120,120]}

            self.batchnorm17_1_13_ = gluon.nn.BatchNorm()
            # batchnorm17_1_13_, output shape: {[256,120,120]}

            self.conv15_1_14_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_14_, output shape: {[4,120,120]}

            self.batchnorm15_1_14_ = gluon.nn.BatchNorm()
            # batchnorm15_1_14_, output shape: {[4,120,120]}

            self.relu15_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_14_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_14_, output shape: {[4,120,120]}

            self.batchnorm16_1_14_ = gluon.nn.BatchNorm()
            # batchnorm16_1_14_, output shape: {[4,120,120]}

            self.relu16_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_14_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_14_, output shape: {[256,120,120]}

            self.batchnorm17_1_14_ = gluon.nn.BatchNorm()
            # batchnorm17_1_14_, output shape: {[256,120,120]}

            self.conv15_1_15_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_15_, output shape: {[4,120,120]}

            self.batchnorm15_1_15_ = gluon.nn.BatchNorm()
            # batchnorm15_1_15_, output shape: {[4,120,120]}

            self.relu15_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_15_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_15_, output shape: {[4,120,120]}

            self.batchnorm16_1_15_ = gluon.nn.BatchNorm()
            # batchnorm16_1_15_, output shape: {[4,120,120]}

            self.relu16_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_15_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_15_, output shape: {[256,120,120]}

            self.batchnorm17_1_15_ = gluon.nn.BatchNorm()
            # batchnorm17_1_15_, output shape: {[256,120,120]}

            self.conv15_1_16_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_16_, output shape: {[4,120,120]}

            self.batchnorm15_1_16_ = gluon.nn.BatchNorm()
            # batchnorm15_1_16_, output shape: {[4,120,120]}

            self.relu15_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_16_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_16_, output shape: {[4,120,120]}

            self.batchnorm16_1_16_ = gluon.nn.BatchNorm()
            # batchnorm16_1_16_, output shape: {[4,120,120]}

            self.relu16_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_16_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_16_, output shape: {[256,120,120]}

            self.batchnorm17_1_16_ = gluon.nn.BatchNorm()
            # batchnorm17_1_16_, output shape: {[256,120,120]}

            self.conv15_1_17_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_17_, output shape: {[4,120,120]}

            self.batchnorm15_1_17_ = gluon.nn.BatchNorm()
            # batchnorm15_1_17_, output shape: {[4,120,120]}

            self.relu15_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_17_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_17_, output shape: {[4,120,120]}

            self.batchnorm16_1_17_ = gluon.nn.BatchNorm()
            # batchnorm16_1_17_, output shape: {[4,120,120]}

            self.relu16_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_17_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_17_, output shape: {[256,120,120]}

            self.batchnorm17_1_17_ = gluon.nn.BatchNorm()
            # batchnorm17_1_17_, output shape: {[256,120,120]}

            self.conv15_1_18_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_18_, output shape: {[4,120,120]}

            self.batchnorm15_1_18_ = gluon.nn.BatchNorm()
            # batchnorm15_1_18_, output shape: {[4,120,120]}

            self.relu15_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_18_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_18_, output shape: {[4,120,120]}

            self.batchnorm16_1_18_ = gluon.nn.BatchNorm()
            # batchnorm16_1_18_, output shape: {[4,120,120]}

            self.relu16_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_18_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_18_, output shape: {[256,120,120]}

            self.batchnorm17_1_18_ = gluon.nn.BatchNorm()
            # batchnorm17_1_18_, output shape: {[256,120,120]}

            self.conv15_1_19_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_19_, output shape: {[4,120,120]}

            self.batchnorm15_1_19_ = gluon.nn.BatchNorm()
            # batchnorm15_1_19_, output shape: {[4,120,120]}

            self.relu15_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_19_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_19_, output shape: {[4,120,120]}

            self.batchnorm16_1_19_ = gluon.nn.BatchNorm()
            # batchnorm16_1_19_, output shape: {[4,120,120]}

            self.relu16_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_19_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_19_, output shape: {[256,120,120]}

            self.batchnorm17_1_19_ = gluon.nn.BatchNorm()
            # batchnorm17_1_19_, output shape: {[256,120,120]}

            self.conv15_1_20_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_20_, output shape: {[4,120,120]}

            self.batchnorm15_1_20_ = gluon.nn.BatchNorm()
            # batchnorm15_1_20_, output shape: {[4,120,120]}

            self.relu15_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_20_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_20_, output shape: {[4,120,120]}

            self.batchnorm16_1_20_ = gluon.nn.BatchNorm()
            # batchnorm16_1_20_, output shape: {[4,120,120]}

            self.relu16_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_20_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_20_, output shape: {[256,120,120]}

            self.batchnorm17_1_20_ = gluon.nn.BatchNorm()
            # batchnorm17_1_20_, output shape: {[256,120,120]}

            self.conv15_1_21_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_21_, output shape: {[4,120,120]}

            self.batchnorm15_1_21_ = gluon.nn.BatchNorm()
            # batchnorm15_1_21_, output shape: {[4,120,120]}

            self.relu15_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_21_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_21_, output shape: {[4,120,120]}

            self.batchnorm16_1_21_ = gluon.nn.BatchNorm()
            # batchnorm16_1_21_, output shape: {[4,120,120]}

            self.relu16_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_21_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_21_, output shape: {[256,120,120]}

            self.batchnorm17_1_21_ = gluon.nn.BatchNorm()
            # batchnorm17_1_21_, output shape: {[256,120,120]}

            self.conv15_1_22_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_22_, output shape: {[4,120,120]}

            self.batchnorm15_1_22_ = gluon.nn.BatchNorm()
            # batchnorm15_1_22_, output shape: {[4,120,120]}

            self.relu15_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_22_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_22_, output shape: {[4,120,120]}

            self.batchnorm16_1_22_ = gluon.nn.BatchNorm()
            # batchnorm16_1_22_, output shape: {[4,120,120]}

            self.relu16_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_22_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_22_, output shape: {[256,120,120]}

            self.batchnorm17_1_22_ = gluon.nn.BatchNorm()
            # batchnorm17_1_22_, output shape: {[256,120,120]}

            self.conv15_1_23_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_23_, output shape: {[4,120,120]}

            self.batchnorm15_1_23_ = gluon.nn.BatchNorm()
            # batchnorm15_1_23_, output shape: {[4,120,120]}

            self.relu15_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_23_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_23_, output shape: {[4,120,120]}

            self.batchnorm16_1_23_ = gluon.nn.BatchNorm()
            # batchnorm16_1_23_, output shape: {[4,120,120]}

            self.relu16_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_23_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_23_, output shape: {[256,120,120]}

            self.batchnorm17_1_23_ = gluon.nn.BatchNorm()
            # batchnorm17_1_23_, output shape: {[256,120,120]}

            self.conv15_1_24_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_24_, output shape: {[4,120,120]}

            self.batchnorm15_1_24_ = gluon.nn.BatchNorm()
            # batchnorm15_1_24_, output shape: {[4,120,120]}

            self.relu15_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_24_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_24_, output shape: {[4,120,120]}

            self.batchnorm16_1_24_ = gluon.nn.BatchNorm()
            # batchnorm16_1_24_, output shape: {[4,120,120]}

            self.relu16_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_24_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_24_, output shape: {[256,120,120]}

            self.batchnorm17_1_24_ = gluon.nn.BatchNorm()
            # batchnorm17_1_24_, output shape: {[256,120,120]}

            self.conv15_1_25_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_25_, output shape: {[4,120,120]}

            self.batchnorm15_1_25_ = gluon.nn.BatchNorm()
            # batchnorm15_1_25_, output shape: {[4,120,120]}

            self.relu15_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_25_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_25_, output shape: {[4,120,120]}

            self.batchnorm16_1_25_ = gluon.nn.BatchNorm()
            # batchnorm16_1_25_, output shape: {[4,120,120]}

            self.relu16_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_25_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_25_, output shape: {[256,120,120]}

            self.batchnorm17_1_25_ = gluon.nn.BatchNorm()
            # batchnorm17_1_25_, output shape: {[256,120,120]}

            self.conv15_1_26_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_26_, output shape: {[4,120,120]}

            self.batchnorm15_1_26_ = gluon.nn.BatchNorm()
            # batchnorm15_1_26_, output shape: {[4,120,120]}

            self.relu15_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_26_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_26_, output shape: {[4,120,120]}

            self.batchnorm16_1_26_ = gluon.nn.BatchNorm()
            # batchnorm16_1_26_, output shape: {[4,120,120]}

            self.relu16_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_26_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_26_, output shape: {[256,120,120]}

            self.batchnorm17_1_26_ = gluon.nn.BatchNorm()
            # batchnorm17_1_26_, output shape: {[256,120,120]}

            self.conv15_1_27_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_27_, output shape: {[4,120,120]}

            self.batchnorm15_1_27_ = gluon.nn.BatchNorm()
            # batchnorm15_1_27_, output shape: {[4,120,120]}

            self.relu15_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_27_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_27_, output shape: {[4,120,120]}

            self.batchnorm16_1_27_ = gluon.nn.BatchNorm()
            # batchnorm16_1_27_, output shape: {[4,120,120]}

            self.relu16_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_27_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_27_, output shape: {[256,120,120]}

            self.batchnorm17_1_27_ = gluon.nn.BatchNorm()
            # batchnorm17_1_27_, output shape: {[256,120,120]}

            self.conv15_1_28_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_28_, output shape: {[4,120,120]}

            self.batchnorm15_1_28_ = gluon.nn.BatchNorm()
            # batchnorm15_1_28_, output shape: {[4,120,120]}

            self.relu15_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_28_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_28_, output shape: {[4,120,120]}

            self.batchnorm16_1_28_ = gluon.nn.BatchNorm()
            # batchnorm16_1_28_, output shape: {[4,120,120]}

            self.relu16_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_28_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_28_, output shape: {[256,120,120]}

            self.batchnorm17_1_28_ = gluon.nn.BatchNorm()
            # batchnorm17_1_28_, output shape: {[256,120,120]}

            self.conv15_1_29_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_29_, output shape: {[4,120,120]}

            self.batchnorm15_1_29_ = gluon.nn.BatchNorm()
            # batchnorm15_1_29_, output shape: {[4,120,120]}

            self.relu15_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_29_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_29_, output shape: {[4,120,120]}

            self.batchnorm16_1_29_ = gluon.nn.BatchNorm()
            # batchnorm16_1_29_, output shape: {[4,120,120]}

            self.relu16_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_29_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_29_, output shape: {[256,120,120]}

            self.batchnorm17_1_29_ = gluon.nn.BatchNorm()
            # batchnorm17_1_29_, output shape: {[256,120,120]}

            self.conv15_1_30_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_30_, output shape: {[4,120,120]}

            self.batchnorm15_1_30_ = gluon.nn.BatchNorm()
            # batchnorm15_1_30_, output shape: {[4,120,120]}

            self.relu15_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_30_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_30_, output shape: {[4,120,120]}

            self.batchnorm16_1_30_ = gluon.nn.BatchNorm()
            # batchnorm16_1_30_, output shape: {[4,120,120]}

            self.relu16_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_30_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_30_, output shape: {[256,120,120]}

            self.batchnorm17_1_30_ = gluon.nn.BatchNorm()
            # batchnorm17_1_30_, output shape: {[256,120,120]}

            self.conv15_1_31_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_31_, output shape: {[4,120,120]}

            self.batchnorm15_1_31_ = gluon.nn.BatchNorm()
            # batchnorm15_1_31_, output shape: {[4,120,120]}

            self.relu15_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_31_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_31_, output shape: {[4,120,120]}

            self.batchnorm16_1_31_ = gluon.nn.BatchNorm()
            # batchnorm16_1_31_, output shape: {[4,120,120]}

            self.relu16_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_31_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_31_, output shape: {[256,120,120]}

            self.batchnorm17_1_31_ = gluon.nn.BatchNorm()
            # batchnorm17_1_31_, output shape: {[256,120,120]}

            self.conv15_1_32_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv15_1_32_, output shape: {[4,120,120]}

            self.batchnorm15_1_32_ = gluon.nn.BatchNorm()
            # batchnorm15_1_32_, output shape: {[4,120,120]}

            self.relu15_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_32_ = gluon.nn.Conv2D(channels=4,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv16_1_32_, output shape: {[4,120,120]}

            self.batchnorm16_1_32_ = gluon.nn.BatchNorm()
            # batchnorm16_1_32_, output shape: {[4,120,120]}

            self.relu16_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv17_1_32_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv17_1_32_, output shape: {[256,120,120]}

            self.batchnorm17_1_32_ = gluon.nn.BatchNorm()
            # batchnorm17_1_32_, output shape: {[256,120,120]}

            self.conv14_2_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv14_2_, output shape: {[256,120,120]}

            self.batchnorm14_2_ = gluon.nn.BatchNorm()
            # batchnorm14_2_, output shape: {[256,120,120]}

            self.relu19_ = gluon.nn.Activation(activation='relu')
            self.conv21_1_1_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_1_, output shape: {[8,120,120]}

            self.batchnorm21_1_1_ = gluon.nn.BatchNorm()
            # batchnorm21_1_1_, output shape: {[8,120,120]}

            self.relu21_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_1_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_1_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_1_, output shape: {[8,60,60]}

            self.batchnorm22_1_1_ = gluon.nn.BatchNorm()
            # batchnorm22_1_1_, output shape: {[8,60,60]}

            self.relu22_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_1_, output shape: {[512,60,60]}

            self.batchnorm23_1_1_ = gluon.nn.BatchNorm()
            # batchnorm23_1_1_, output shape: {[512,60,60]}

            self.conv21_1_2_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_2_, output shape: {[8,120,120]}

            self.batchnorm21_1_2_ = gluon.nn.BatchNorm()
            # batchnorm21_1_2_, output shape: {[8,120,120]}

            self.relu21_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_2_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_2_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_2_, output shape: {[8,60,60]}

            self.batchnorm22_1_2_ = gluon.nn.BatchNorm()
            # batchnorm22_1_2_, output shape: {[8,60,60]}

            self.relu22_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_2_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_2_, output shape: {[512,60,60]}

            self.batchnorm23_1_2_ = gluon.nn.BatchNorm()
            # batchnorm23_1_2_, output shape: {[512,60,60]}

            self.conv21_1_3_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_3_, output shape: {[8,120,120]}

            self.batchnorm21_1_3_ = gluon.nn.BatchNorm()
            # batchnorm21_1_3_, output shape: {[8,120,120]}

            self.relu21_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_3_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_3_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_3_, output shape: {[8,60,60]}

            self.batchnorm22_1_3_ = gluon.nn.BatchNorm()
            # batchnorm22_1_3_, output shape: {[8,60,60]}

            self.relu22_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_3_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_3_, output shape: {[512,60,60]}

            self.batchnorm23_1_3_ = gluon.nn.BatchNorm()
            # batchnorm23_1_3_, output shape: {[512,60,60]}

            self.conv21_1_4_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_4_, output shape: {[8,120,120]}

            self.batchnorm21_1_4_ = gluon.nn.BatchNorm()
            # batchnorm21_1_4_, output shape: {[8,120,120]}

            self.relu21_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_4_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_4_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_4_, output shape: {[8,60,60]}

            self.batchnorm22_1_4_ = gluon.nn.BatchNorm()
            # batchnorm22_1_4_, output shape: {[8,60,60]}

            self.relu22_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_4_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_4_, output shape: {[512,60,60]}

            self.batchnorm23_1_4_ = gluon.nn.BatchNorm()
            # batchnorm23_1_4_, output shape: {[512,60,60]}

            self.conv21_1_5_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_5_, output shape: {[8,120,120]}

            self.batchnorm21_1_5_ = gluon.nn.BatchNorm()
            # batchnorm21_1_5_, output shape: {[8,120,120]}

            self.relu21_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_5_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_5_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_5_, output shape: {[8,60,60]}

            self.batchnorm22_1_5_ = gluon.nn.BatchNorm()
            # batchnorm22_1_5_, output shape: {[8,60,60]}

            self.relu22_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_5_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_5_, output shape: {[512,60,60]}

            self.batchnorm23_1_5_ = gluon.nn.BatchNorm()
            # batchnorm23_1_5_, output shape: {[512,60,60]}

            self.conv21_1_6_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_6_, output shape: {[8,120,120]}

            self.batchnorm21_1_6_ = gluon.nn.BatchNorm()
            # batchnorm21_1_6_, output shape: {[8,120,120]}

            self.relu21_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_6_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_6_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_6_, output shape: {[8,60,60]}

            self.batchnorm22_1_6_ = gluon.nn.BatchNorm()
            # batchnorm22_1_6_, output shape: {[8,60,60]}

            self.relu22_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_6_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_6_, output shape: {[512,60,60]}

            self.batchnorm23_1_6_ = gluon.nn.BatchNorm()
            # batchnorm23_1_6_, output shape: {[512,60,60]}

            self.conv21_1_7_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_7_, output shape: {[8,120,120]}

            self.batchnorm21_1_7_ = gluon.nn.BatchNorm()
            # batchnorm21_1_7_, output shape: {[8,120,120]}

            self.relu21_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_7_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_7_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_7_, output shape: {[8,60,60]}

            self.batchnorm22_1_7_ = gluon.nn.BatchNorm()
            # batchnorm22_1_7_, output shape: {[8,60,60]}

            self.relu22_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_7_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_7_, output shape: {[512,60,60]}

            self.batchnorm23_1_7_ = gluon.nn.BatchNorm()
            # batchnorm23_1_7_, output shape: {[512,60,60]}

            self.conv21_1_8_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_8_, output shape: {[8,120,120]}

            self.batchnorm21_1_8_ = gluon.nn.BatchNorm()
            # batchnorm21_1_8_, output shape: {[8,120,120]}

            self.relu21_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_8_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_8_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_8_, output shape: {[8,60,60]}

            self.batchnorm22_1_8_ = gluon.nn.BatchNorm()
            # batchnorm22_1_8_, output shape: {[8,60,60]}

            self.relu22_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_8_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_8_, output shape: {[512,60,60]}

            self.batchnorm23_1_8_ = gluon.nn.BatchNorm()
            # batchnorm23_1_8_, output shape: {[512,60,60]}

            self.conv21_1_9_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_9_, output shape: {[8,120,120]}

            self.batchnorm21_1_9_ = gluon.nn.BatchNorm()
            # batchnorm21_1_9_, output shape: {[8,120,120]}

            self.relu21_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_9_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_9_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_9_, output shape: {[8,60,60]}

            self.batchnorm22_1_9_ = gluon.nn.BatchNorm()
            # batchnorm22_1_9_, output shape: {[8,60,60]}

            self.relu22_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_9_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_9_, output shape: {[512,60,60]}

            self.batchnorm23_1_9_ = gluon.nn.BatchNorm()
            # batchnorm23_1_9_, output shape: {[512,60,60]}

            self.conv21_1_10_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_10_, output shape: {[8,120,120]}

            self.batchnorm21_1_10_ = gluon.nn.BatchNorm()
            # batchnorm21_1_10_, output shape: {[8,120,120]}

            self.relu21_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_10_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_10_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_10_, output shape: {[8,60,60]}

            self.batchnorm22_1_10_ = gluon.nn.BatchNorm()
            # batchnorm22_1_10_, output shape: {[8,60,60]}

            self.relu22_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_10_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_10_, output shape: {[512,60,60]}

            self.batchnorm23_1_10_ = gluon.nn.BatchNorm()
            # batchnorm23_1_10_, output shape: {[512,60,60]}

            self.conv21_1_11_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_11_, output shape: {[8,120,120]}

            self.batchnorm21_1_11_ = gluon.nn.BatchNorm()
            # batchnorm21_1_11_, output shape: {[8,120,120]}

            self.relu21_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_11_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_11_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_11_, output shape: {[8,60,60]}

            self.batchnorm22_1_11_ = gluon.nn.BatchNorm()
            # batchnorm22_1_11_, output shape: {[8,60,60]}

            self.relu22_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_11_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_11_, output shape: {[512,60,60]}

            self.batchnorm23_1_11_ = gluon.nn.BatchNorm()
            # batchnorm23_1_11_, output shape: {[512,60,60]}

            self.conv21_1_12_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_12_, output shape: {[8,120,120]}

            self.batchnorm21_1_12_ = gluon.nn.BatchNorm()
            # batchnorm21_1_12_, output shape: {[8,120,120]}

            self.relu21_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_12_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_12_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_12_, output shape: {[8,60,60]}

            self.batchnorm22_1_12_ = gluon.nn.BatchNorm()
            # batchnorm22_1_12_, output shape: {[8,60,60]}

            self.relu22_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_12_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_12_, output shape: {[512,60,60]}

            self.batchnorm23_1_12_ = gluon.nn.BatchNorm()
            # batchnorm23_1_12_, output shape: {[512,60,60]}

            self.conv21_1_13_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_13_, output shape: {[8,120,120]}

            self.batchnorm21_1_13_ = gluon.nn.BatchNorm()
            # batchnorm21_1_13_, output shape: {[8,120,120]}

            self.relu21_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_13_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_13_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_13_, output shape: {[8,60,60]}

            self.batchnorm22_1_13_ = gluon.nn.BatchNorm()
            # batchnorm22_1_13_, output shape: {[8,60,60]}

            self.relu22_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_13_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_13_, output shape: {[512,60,60]}

            self.batchnorm23_1_13_ = gluon.nn.BatchNorm()
            # batchnorm23_1_13_, output shape: {[512,60,60]}

            self.conv21_1_14_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_14_, output shape: {[8,120,120]}

            self.batchnorm21_1_14_ = gluon.nn.BatchNorm()
            # batchnorm21_1_14_, output shape: {[8,120,120]}

            self.relu21_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_14_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_14_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_14_, output shape: {[8,60,60]}

            self.batchnorm22_1_14_ = gluon.nn.BatchNorm()
            # batchnorm22_1_14_, output shape: {[8,60,60]}

            self.relu22_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_14_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_14_, output shape: {[512,60,60]}

            self.batchnorm23_1_14_ = gluon.nn.BatchNorm()
            # batchnorm23_1_14_, output shape: {[512,60,60]}

            self.conv21_1_15_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_15_, output shape: {[8,120,120]}

            self.batchnorm21_1_15_ = gluon.nn.BatchNorm()
            # batchnorm21_1_15_, output shape: {[8,120,120]}

            self.relu21_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_15_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_15_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_15_, output shape: {[8,60,60]}

            self.batchnorm22_1_15_ = gluon.nn.BatchNorm()
            # batchnorm22_1_15_, output shape: {[8,60,60]}

            self.relu22_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_15_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_15_, output shape: {[512,60,60]}

            self.batchnorm23_1_15_ = gluon.nn.BatchNorm()
            # batchnorm23_1_15_, output shape: {[512,60,60]}

            self.conv21_1_16_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_16_, output shape: {[8,120,120]}

            self.batchnorm21_1_16_ = gluon.nn.BatchNorm()
            # batchnorm21_1_16_, output shape: {[8,120,120]}

            self.relu21_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_16_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_16_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_16_, output shape: {[8,60,60]}

            self.batchnorm22_1_16_ = gluon.nn.BatchNorm()
            # batchnorm22_1_16_, output shape: {[8,60,60]}

            self.relu22_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_16_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_16_, output shape: {[512,60,60]}

            self.batchnorm23_1_16_ = gluon.nn.BatchNorm()
            # batchnorm23_1_16_, output shape: {[512,60,60]}

            self.conv21_1_17_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_17_, output shape: {[8,120,120]}

            self.batchnorm21_1_17_ = gluon.nn.BatchNorm()
            # batchnorm21_1_17_, output shape: {[8,120,120]}

            self.relu21_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_17_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_17_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_17_, output shape: {[8,60,60]}

            self.batchnorm22_1_17_ = gluon.nn.BatchNorm()
            # batchnorm22_1_17_, output shape: {[8,60,60]}

            self.relu22_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_17_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_17_, output shape: {[512,60,60]}

            self.batchnorm23_1_17_ = gluon.nn.BatchNorm()
            # batchnorm23_1_17_, output shape: {[512,60,60]}

            self.conv21_1_18_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_18_, output shape: {[8,120,120]}

            self.batchnorm21_1_18_ = gluon.nn.BatchNorm()
            # batchnorm21_1_18_, output shape: {[8,120,120]}

            self.relu21_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_18_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_18_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_18_, output shape: {[8,60,60]}

            self.batchnorm22_1_18_ = gluon.nn.BatchNorm()
            # batchnorm22_1_18_, output shape: {[8,60,60]}

            self.relu22_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_18_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_18_, output shape: {[512,60,60]}

            self.batchnorm23_1_18_ = gluon.nn.BatchNorm()
            # batchnorm23_1_18_, output shape: {[512,60,60]}

            self.conv21_1_19_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_19_, output shape: {[8,120,120]}

            self.batchnorm21_1_19_ = gluon.nn.BatchNorm()
            # batchnorm21_1_19_, output shape: {[8,120,120]}

            self.relu21_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_19_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_19_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_19_, output shape: {[8,60,60]}

            self.batchnorm22_1_19_ = gluon.nn.BatchNorm()
            # batchnorm22_1_19_, output shape: {[8,60,60]}

            self.relu22_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_19_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_19_, output shape: {[512,60,60]}

            self.batchnorm23_1_19_ = gluon.nn.BatchNorm()
            # batchnorm23_1_19_, output shape: {[512,60,60]}

            self.conv21_1_20_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_20_, output shape: {[8,120,120]}

            self.batchnorm21_1_20_ = gluon.nn.BatchNorm()
            # batchnorm21_1_20_, output shape: {[8,120,120]}

            self.relu21_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_20_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_20_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_20_, output shape: {[8,60,60]}

            self.batchnorm22_1_20_ = gluon.nn.BatchNorm()
            # batchnorm22_1_20_, output shape: {[8,60,60]}

            self.relu22_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_20_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_20_, output shape: {[512,60,60]}

            self.batchnorm23_1_20_ = gluon.nn.BatchNorm()
            # batchnorm23_1_20_, output shape: {[512,60,60]}

            self.conv21_1_21_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_21_, output shape: {[8,120,120]}

            self.batchnorm21_1_21_ = gluon.nn.BatchNorm()
            # batchnorm21_1_21_, output shape: {[8,120,120]}

            self.relu21_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_21_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_21_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_21_, output shape: {[8,60,60]}

            self.batchnorm22_1_21_ = gluon.nn.BatchNorm()
            # batchnorm22_1_21_, output shape: {[8,60,60]}

            self.relu22_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_21_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_21_, output shape: {[512,60,60]}

            self.batchnorm23_1_21_ = gluon.nn.BatchNorm()
            # batchnorm23_1_21_, output shape: {[512,60,60]}

            self.conv21_1_22_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_22_, output shape: {[8,120,120]}

            self.batchnorm21_1_22_ = gluon.nn.BatchNorm()
            # batchnorm21_1_22_, output shape: {[8,120,120]}

            self.relu21_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_22_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_22_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_22_, output shape: {[8,60,60]}

            self.batchnorm22_1_22_ = gluon.nn.BatchNorm()
            # batchnorm22_1_22_, output shape: {[8,60,60]}

            self.relu22_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_22_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_22_, output shape: {[512,60,60]}

            self.batchnorm23_1_22_ = gluon.nn.BatchNorm()
            # batchnorm23_1_22_, output shape: {[512,60,60]}

            self.conv21_1_23_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_23_, output shape: {[8,120,120]}

            self.batchnorm21_1_23_ = gluon.nn.BatchNorm()
            # batchnorm21_1_23_, output shape: {[8,120,120]}

            self.relu21_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_23_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_23_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_23_, output shape: {[8,60,60]}

            self.batchnorm22_1_23_ = gluon.nn.BatchNorm()
            # batchnorm22_1_23_, output shape: {[8,60,60]}

            self.relu22_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_23_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_23_, output shape: {[512,60,60]}

            self.batchnorm23_1_23_ = gluon.nn.BatchNorm()
            # batchnorm23_1_23_, output shape: {[512,60,60]}

            self.conv21_1_24_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_24_, output shape: {[8,120,120]}

            self.batchnorm21_1_24_ = gluon.nn.BatchNorm()
            # batchnorm21_1_24_, output shape: {[8,120,120]}

            self.relu21_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_24_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_24_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_24_, output shape: {[8,60,60]}

            self.batchnorm22_1_24_ = gluon.nn.BatchNorm()
            # batchnorm22_1_24_, output shape: {[8,60,60]}

            self.relu22_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_24_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_24_, output shape: {[512,60,60]}

            self.batchnorm23_1_24_ = gluon.nn.BatchNorm()
            # batchnorm23_1_24_, output shape: {[512,60,60]}

            self.conv21_1_25_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_25_, output shape: {[8,120,120]}

            self.batchnorm21_1_25_ = gluon.nn.BatchNorm()
            # batchnorm21_1_25_, output shape: {[8,120,120]}

            self.relu21_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_25_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_25_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_25_, output shape: {[8,60,60]}

            self.batchnorm22_1_25_ = gluon.nn.BatchNorm()
            # batchnorm22_1_25_, output shape: {[8,60,60]}

            self.relu22_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_25_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_25_, output shape: {[512,60,60]}

            self.batchnorm23_1_25_ = gluon.nn.BatchNorm()
            # batchnorm23_1_25_, output shape: {[512,60,60]}

            self.conv21_1_26_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_26_, output shape: {[8,120,120]}

            self.batchnorm21_1_26_ = gluon.nn.BatchNorm()
            # batchnorm21_1_26_, output shape: {[8,120,120]}

            self.relu21_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_26_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_26_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_26_, output shape: {[8,60,60]}

            self.batchnorm22_1_26_ = gluon.nn.BatchNorm()
            # batchnorm22_1_26_, output shape: {[8,60,60]}

            self.relu22_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_26_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_26_, output shape: {[512,60,60]}

            self.batchnorm23_1_26_ = gluon.nn.BatchNorm()
            # batchnorm23_1_26_, output shape: {[512,60,60]}

            self.conv21_1_27_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_27_, output shape: {[8,120,120]}

            self.batchnorm21_1_27_ = gluon.nn.BatchNorm()
            # batchnorm21_1_27_, output shape: {[8,120,120]}

            self.relu21_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_27_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_27_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_27_, output shape: {[8,60,60]}

            self.batchnorm22_1_27_ = gluon.nn.BatchNorm()
            # batchnorm22_1_27_, output shape: {[8,60,60]}

            self.relu22_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_27_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_27_, output shape: {[512,60,60]}

            self.batchnorm23_1_27_ = gluon.nn.BatchNorm()
            # batchnorm23_1_27_, output shape: {[512,60,60]}

            self.conv21_1_28_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_28_, output shape: {[8,120,120]}

            self.batchnorm21_1_28_ = gluon.nn.BatchNorm()
            # batchnorm21_1_28_, output shape: {[8,120,120]}

            self.relu21_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_28_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_28_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_28_, output shape: {[8,60,60]}

            self.batchnorm22_1_28_ = gluon.nn.BatchNorm()
            # batchnorm22_1_28_, output shape: {[8,60,60]}

            self.relu22_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_28_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_28_, output shape: {[512,60,60]}

            self.batchnorm23_1_28_ = gluon.nn.BatchNorm()
            # batchnorm23_1_28_, output shape: {[512,60,60]}

            self.conv21_1_29_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_29_, output shape: {[8,120,120]}

            self.batchnorm21_1_29_ = gluon.nn.BatchNorm()
            # batchnorm21_1_29_, output shape: {[8,120,120]}

            self.relu21_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_29_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_29_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_29_, output shape: {[8,60,60]}

            self.batchnorm22_1_29_ = gluon.nn.BatchNorm()
            # batchnorm22_1_29_, output shape: {[8,60,60]}

            self.relu22_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_29_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_29_, output shape: {[512,60,60]}

            self.batchnorm23_1_29_ = gluon.nn.BatchNorm()
            # batchnorm23_1_29_, output shape: {[512,60,60]}

            self.conv21_1_30_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_30_, output shape: {[8,120,120]}

            self.batchnorm21_1_30_ = gluon.nn.BatchNorm()
            # batchnorm21_1_30_, output shape: {[8,120,120]}

            self.relu21_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_30_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_30_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_30_, output shape: {[8,60,60]}

            self.batchnorm22_1_30_ = gluon.nn.BatchNorm()
            # batchnorm22_1_30_, output shape: {[8,60,60]}

            self.relu22_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_30_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_30_, output shape: {[512,60,60]}

            self.batchnorm23_1_30_ = gluon.nn.BatchNorm()
            # batchnorm23_1_30_, output shape: {[512,60,60]}

            self.conv21_1_31_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_31_, output shape: {[8,120,120]}

            self.batchnorm21_1_31_ = gluon.nn.BatchNorm()
            # batchnorm21_1_31_, output shape: {[8,120,120]}

            self.relu21_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_31_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_31_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_31_, output shape: {[8,60,60]}

            self.batchnorm22_1_31_ = gluon.nn.BatchNorm()
            # batchnorm22_1_31_, output shape: {[8,60,60]}

            self.relu22_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_31_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_31_, output shape: {[512,60,60]}

            self.batchnorm23_1_31_ = gluon.nn.BatchNorm()
            # batchnorm23_1_31_, output shape: {[512,60,60]}

            self.conv21_1_32_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv21_1_32_, output shape: {[8,120,120]}

            self.batchnorm21_1_32_ = gluon.nn.BatchNorm()
            # batchnorm21_1_32_, output shape: {[8,120,120]}

            self.relu21_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv22_1_32_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv22_1_32_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv22_1_32_, output shape: {[8,60,60]}

            self.batchnorm22_1_32_ = gluon.nn.BatchNorm()
            # batchnorm22_1_32_, output shape: {[8,60,60]}

            self.relu22_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv23_1_32_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv23_1_32_, output shape: {[512,60,60]}

            self.batchnorm23_1_32_ = gluon.nn.BatchNorm()
            # batchnorm23_1_32_, output shape: {[512,60,60]}

            self.conv20_2_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv20_2_, output shape: {[512,60,60]}

            self.batchnorm20_2_ = gluon.nn.BatchNorm()
            # batchnorm20_2_, output shape: {[512,60,60]}

            self.relu25_ = gluon.nn.Activation(activation='relu')
            self.conv27_1_1_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_1_, output shape: {[8,60,60]}

            self.batchnorm27_1_1_ = gluon.nn.BatchNorm()
            # batchnorm27_1_1_, output shape: {[8,60,60]}

            self.relu27_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_1_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_1_, output shape: {[8,60,60]}

            self.batchnorm28_1_1_ = gluon.nn.BatchNorm()
            # batchnorm28_1_1_, output shape: {[8,60,60]}

            self.relu28_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_1_, output shape: {[512,60,60]}

            self.batchnorm29_1_1_ = gluon.nn.BatchNorm()
            # batchnorm29_1_1_, output shape: {[512,60,60]}

            self.conv27_1_2_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_2_, output shape: {[8,60,60]}

            self.batchnorm27_1_2_ = gluon.nn.BatchNorm()
            # batchnorm27_1_2_, output shape: {[8,60,60]}

            self.relu27_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_2_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_2_, output shape: {[8,60,60]}

            self.batchnorm28_1_2_ = gluon.nn.BatchNorm()
            # batchnorm28_1_2_, output shape: {[8,60,60]}

            self.relu28_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_2_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_2_, output shape: {[512,60,60]}

            self.batchnorm29_1_2_ = gluon.nn.BatchNorm()
            # batchnorm29_1_2_, output shape: {[512,60,60]}

            self.conv27_1_3_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_3_, output shape: {[8,60,60]}

            self.batchnorm27_1_3_ = gluon.nn.BatchNorm()
            # batchnorm27_1_3_, output shape: {[8,60,60]}

            self.relu27_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_3_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_3_, output shape: {[8,60,60]}

            self.batchnorm28_1_3_ = gluon.nn.BatchNorm()
            # batchnorm28_1_3_, output shape: {[8,60,60]}

            self.relu28_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_3_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_3_, output shape: {[512,60,60]}

            self.batchnorm29_1_3_ = gluon.nn.BatchNorm()
            # batchnorm29_1_3_, output shape: {[512,60,60]}

            self.conv27_1_4_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_4_, output shape: {[8,60,60]}

            self.batchnorm27_1_4_ = gluon.nn.BatchNorm()
            # batchnorm27_1_4_, output shape: {[8,60,60]}

            self.relu27_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_4_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_4_, output shape: {[8,60,60]}

            self.batchnorm28_1_4_ = gluon.nn.BatchNorm()
            # batchnorm28_1_4_, output shape: {[8,60,60]}

            self.relu28_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_4_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_4_, output shape: {[512,60,60]}

            self.batchnorm29_1_4_ = gluon.nn.BatchNorm()
            # batchnorm29_1_4_, output shape: {[512,60,60]}

            self.conv27_1_5_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_5_, output shape: {[8,60,60]}

            self.batchnorm27_1_5_ = gluon.nn.BatchNorm()
            # batchnorm27_1_5_, output shape: {[8,60,60]}

            self.relu27_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_5_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_5_, output shape: {[8,60,60]}

            self.batchnorm28_1_5_ = gluon.nn.BatchNorm()
            # batchnorm28_1_5_, output shape: {[8,60,60]}

            self.relu28_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_5_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_5_, output shape: {[512,60,60]}

            self.batchnorm29_1_5_ = gluon.nn.BatchNorm()
            # batchnorm29_1_5_, output shape: {[512,60,60]}

            self.conv27_1_6_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_6_, output shape: {[8,60,60]}

            self.batchnorm27_1_6_ = gluon.nn.BatchNorm()
            # batchnorm27_1_6_, output shape: {[8,60,60]}

            self.relu27_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_6_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_6_, output shape: {[8,60,60]}

            self.batchnorm28_1_6_ = gluon.nn.BatchNorm()
            # batchnorm28_1_6_, output shape: {[8,60,60]}

            self.relu28_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_6_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_6_, output shape: {[512,60,60]}

            self.batchnorm29_1_6_ = gluon.nn.BatchNorm()
            # batchnorm29_1_6_, output shape: {[512,60,60]}

            self.conv27_1_7_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_7_, output shape: {[8,60,60]}

            self.batchnorm27_1_7_ = gluon.nn.BatchNorm()
            # batchnorm27_1_7_, output shape: {[8,60,60]}

            self.relu27_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_7_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_7_, output shape: {[8,60,60]}

            self.batchnorm28_1_7_ = gluon.nn.BatchNorm()
            # batchnorm28_1_7_, output shape: {[8,60,60]}

            self.relu28_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_7_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_7_, output shape: {[512,60,60]}

            self.batchnorm29_1_7_ = gluon.nn.BatchNorm()
            # batchnorm29_1_7_, output shape: {[512,60,60]}

            self.conv27_1_8_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_8_, output shape: {[8,60,60]}

            self.batchnorm27_1_8_ = gluon.nn.BatchNorm()
            # batchnorm27_1_8_, output shape: {[8,60,60]}

            self.relu27_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_8_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_8_, output shape: {[8,60,60]}

            self.batchnorm28_1_8_ = gluon.nn.BatchNorm()
            # batchnorm28_1_8_, output shape: {[8,60,60]}

            self.relu28_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_8_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_8_, output shape: {[512,60,60]}

            self.batchnorm29_1_8_ = gluon.nn.BatchNorm()
            # batchnorm29_1_8_, output shape: {[512,60,60]}

            self.conv27_1_9_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_9_, output shape: {[8,60,60]}

            self.batchnorm27_1_9_ = gluon.nn.BatchNorm()
            # batchnorm27_1_9_, output shape: {[8,60,60]}

            self.relu27_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_9_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_9_, output shape: {[8,60,60]}

            self.batchnorm28_1_9_ = gluon.nn.BatchNorm()
            # batchnorm28_1_9_, output shape: {[8,60,60]}

            self.relu28_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_9_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_9_, output shape: {[512,60,60]}

            self.batchnorm29_1_9_ = gluon.nn.BatchNorm()
            # batchnorm29_1_9_, output shape: {[512,60,60]}

            self.conv27_1_10_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_10_, output shape: {[8,60,60]}

            self.batchnorm27_1_10_ = gluon.nn.BatchNorm()
            # batchnorm27_1_10_, output shape: {[8,60,60]}

            self.relu27_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_10_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_10_, output shape: {[8,60,60]}

            self.batchnorm28_1_10_ = gluon.nn.BatchNorm()
            # batchnorm28_1_10_, output shape: {[8,60,60]}

            self.relu28_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_10_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_10_, output shape: {[512,60,60]}

            self.batchnorm29_1_10_ = gluon.nn.BatchNorm()
            # batchnorm29_1_10_, output shape: {[512,60,60]}

            self.conv27_1_11_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_11_, output shape: {[8,60,60]}

            self.batchnorm27_1_11_ = gluon.nn.BatchNorm()
            # batchnorm27_1_11_, output shape: {[8,60,60]}

            self.relu27_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_11_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_11_, output shape: {[8,60,60]}

            self.batchnorm28_1_11_ = gluon.nn.BatchNorm()
            # batchnorm28_1_11_, output shape: {[8,60,60]}

            self.relu28_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_11_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_11_, output shape: {[512,60,60]}

            self.batchnorm29_1_11_ = gluon.nn.BatchNorm()
            # batchnorm29_1_11_, output shape: {[512,60,60]}

            self.conv27_1_12_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_12_, output shape: {[8,60,60]}

            self.batchnorm27_1_12_ = gluon.nn.BatchNorm()
            # batchnorm27_1_12_, output shape: {[8,60,60]}

            self.relu27_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_12_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_12_, output shape: {[8,60,60]}

            self.batchnorm28_1_12_ = gluon.nn.BatchNorm()
            # batchnorm28_1_12_, output shape: {[8,60,60]}

            self.relu28_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_12_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_12_, output shape: {[512,60,60]}

            self.batchnorm29_1_12_ = gluon.nn.BatchNorm()
            # batchnorm29_1_12_, output shape: {[512,60,60]}

            self.conv27_1_13_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_13_, output shape: {[8,60,60]}

            self.batchnorm27_1_13_ = gluon.nn.BatchNorm()
            # batchnorm27_1_13_, output shape: {[8,60,60]}

            self.relu27_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_13_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_13_, output shape: {[8,60,60]}

            self.batchnorm28_1_13_ = gluon.nn.BatchNorm()
            # batchnorm28_1_13_, output shape: {[8,60,60]}

            self.relu28_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_13_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_13_, output shape: {[512,60,60]}

            self.batchnorm29_1_13_ = gluon.nn.BatchNorm()
            # batchnorm29_1_13_, output shape: {[512,60,60]}

            self.conv27_1_14_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_14_, output shape: {[8,60,60]}

            self.batchnorm27_1_14_ = gluon.nn.BatchNorm()
            # batchnorm27_1_14_, output shape: {[8,60,60]}

            self.relu27_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_14_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_14_, output shape: {[8,60,60]}

            self.batchnorm28_1_14_ = gluon.nn.BatchNorm()
            # batchnorm28_1_14_, output shape: {[8,60,60]}

            self.relu28_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_14_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_14_, output shape: {[512,60,60]}

            self.batchnorm29_1_14_ = gluon.nn.BatchNorm()
            # batchnorm29_1_14_, output shape: {[512,60,60]}

            self.conv27_1_15_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_15_, output shape: {[8,60,60]}

            self.batchnorm27_1_15_ = gluon.nn.BatchNorm()
            # batchnorm27_1_15_, output shape: {[8,60,60]}

            self.relu27_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_15_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_15_, output shape: {[8,60,60]}

            self.batchnorm28_1_15_ = gluon.nn.BatchNorm()
            # batchnorm28_1_15_, output shape: {[8,60,60]}

            self.relu28_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_15_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_15_, output shape: {[512,60,60]}

            self.batchnorm29_1_15_ = gluon.nn.BatchNorm()
            # batchnorm29_1_15_, output shape: {[512,60,60]}

            self.conv27_1_16_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_16_, output shape: {[8,60,60]}

            self.batchnorm27_1_16_ = gluon.nn.BatchNorm()
            # batchnorm27_1_16_, output shape: {[8,60,60]}

            self.relu27_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_16_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_16_, output shape: {[8,60,60]}

            self.batchnorm28_1_16_ = gluon.nn.BatchNorm()
            # batchnorm28_1_16_, output shape: {[8,60,60]}

            self.relu28_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_16_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_16_, output shape: {[512,60,60]}

            self.batchnorm29_1_16_ = gluon.nn.BatchNorm()
            # batchnorm29_1_16_, output shape: {[512,60,60]}

            self.conv27_1_17_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_17_, output shape: {[8,60,60]}

            self.batchnorm27_1_17_ = gluon.nn.BatchNorm()
            # batchnorm27_1_17_, output shape: {[8,60,60]}

            self.relu27_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_17_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_17_, output shape: {[8,60,60]}

            self.batchnorm28_1_17_ = gluon.nn.BatchNorm()
            # batchnorm28_1_17_, output shape: {[8,60,60]}

            self.relu28_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_17_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_17_, output shape: {[512,60,60]}

            self.batchnorm29_1_17_ = gluon.nn.BatchNorm()
            # batchnorm29_1_17_, output shape: {[512,60,60]}

            self.conv27_1_18_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_18_, output shape: {[8,60,60]}

            self.batchnorm27_1_18_ = gluon.nn.BatchNorm()
            # batchnorm27_1_18_, output shape: {[8,60,60]}

            self.relu27_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_18_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_18_, output shape: {[8,60,60]}

            self.batchnorm28_1_18_ = gluon.nn.BatchNorm()
            # batchnorm28_1_18_, output shape: {[8,60,60]}

            self.relu28_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_18_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_18_, output shape: {[512,60,60]}

            self.batchnorm29_1_18_ = gluon.nn.BatchNorm()
            # batchnorm29_1_18_, output shape: {[512,60,60]}

            self.conv27_1_19_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_19_, output shape: {[8,60,60]}

            self.batchnorm27_1_19_ = gluon.nn.BatchNorm()
            # batchnorm27_1_19_, output shape: {[8,60,60]}

            self.relu27_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_19_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_19_, output shape: {[8,60,60]}

            self.batchnorm28_1_19_ = gluon.nn.BatchNorm()
            # batchnorm28_1_19_, output shape: {[8,60,60]}

            self.relu28_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_19_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_19_, output shape: {[512,60,60]}

            self.batchnorm29_1_19_ = gluon.nn.BatchNorm()
            # batchnorm29_1_19_, output shape: {[512,60,60]}

            self.conv27_1_20_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_20_, output shape: {[8,60,60]}

            self.batchnorm27_1_20_ = gluon.nn.BatchNorm()
            # batchnorm27_1_20_, output shape: {[8,60,60]}

            self.relu27_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_20_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_20_, output shape: {[8,60,60]}

            self.batchnorm28_1_20_ = gluon.nn.BatchNorm()
            # batchnorm28_1_20_, output shape: {[8,60,60]}

            self.relu28_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_20_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_20_, output shape: {[512,60,60]}

            self.batchnorm29_1_20_ = gluon.nn.BatchNorm()
            # batchnorm29_1_20_, output shape: {[512,60,60]}

            self.conv27_1_21_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_21_, output shape: {[8,60,60]}

            self.batchnorm27_1_21_ = gluon.nn.BatchNorm()
            # batchnorm27_1_21_, output shape: {[8,60,60]}

            self.relu27_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_21_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_21_, output shape: {[8,60,60]}

            self.batchnorm28_1_21_ = gluon.nn.BatchNorm()
            # batchnorm28_1_21_, output shape: {[8,60,60]}

            self.relu28_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_21_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_21_, output shape: {[512,60,60]}

            self.batchnorm29_1_21_ = gluon.nn.BatchNorm()
            # batchnorm29_1_21_, output shape: {[512,60,60]}

            self.conv27_1_22_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_22_, output shape: {[8,60,60]}

            self.batchnorm27_1_22_ = gluon.nn.BatchNorm()
            # batchnorm27_1_22_, output shape: {[8,60,60]}

            self.relu27_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_22_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_22_, output shape: {[8,60,60]}

            self.batchnorm28_1_22_ = gluon.nn.BatchNorm()
            # batchnorm28_1_22_, output shape: {[8,60,60]}

            self.relu28_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_22_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_22_, output shape: {[512,60,60]}

            self.batchnorm29_1_22_ = gluon.nn.BatchNorm()
            # batchnorm29_1_22_, output shape: {[512,60,60]}

            self.conv27_1_23_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_23_, output shape: {[8,60,60]}

            self.batchnorm27_1_23_ = gluon.nn.BatchNorm()
            # batchnorm27_1_23_, output shape: {[8,60,60]}

            self.relu27_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_23_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_23_, output shape: {[8,60,60]}

            self.batchnorm28_1_23_ = gluon.nn.BatchNorm()
            # batchnorm28_1_23_, output shape: {[8,60,60]}

            self.relu28_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_23_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_23_, output shape: {[512,60,60]}

            self.batchnorm29_1_23_ = gluon.nn.BatchNorm()
            # batchnorm29_1_23_, output shape: {[512,60,60]}

            self.conv27_1_24_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_24_, output shape: {[8,60,60]}

            self.batchnorm27_1_24_ = gluon.nn.BatchNorm()
            # batchnorm27_1_24_, output shape: {[8,60,60]}

            self.relu27_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_24_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_24_, output shape: {[8,60,60]}

            self.batchnorm28_1_24_ = gluon.nn.BatchNorm()
            # batchnorm28_1_24_, output shape: {[8,60,60]}

            self.relu28_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_24_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_24_, output shape: {[512,60,60]}

            self.batchnorm29_1_24_ = gluon.nn.BatchNorm()
            # batchnorm29_1_24_, output shape: {[512,60,60]}

            self.conv27_1_25_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_25_, output shape: {[8,60,60]}

            self.batchnorm27_1_25_ = gluon.nn.BatchNorm()
            # batchnorm27_1_25_, output shape: {[8,60,60]}

            self.relu27_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_25_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_25_, output shape: {[8,60,60]}

            self.batchnorm28_1_25_ = gluon.nn.BatchNorm()
            # batchnorm28_1_25_, output shape: {[8,60,60]}

            self.relu28_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_25_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_25_, output shape: {[512,60,60]}

            self.batchnorm29_1_25_ = gluon.nn.BatchNorm()
            # batchnorm29_1_25_, output shape: {[512,60,60]}

            self.conv27_1_26_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_26_, output shape: {[8,60,60]}

            self.batchnorm27_1_26_ = gluon.nn.BatchNorm()
            # batchnorm27_1_26_, output shape: {[8,60,60]}

            self.relu27_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_26_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_26_, output shape: {[8,60,60]}

            self.batchnorm28_1_26_ = gluon.nn.BatchNorm()
            # batchnorm28_1_26_, output shape: {[8,60,60]}

            self.relu28_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_26_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_26_, output shape: {[512,60,60]}

            self.batchnorm29_1_26_ = gluon.nn.BatchNorm()
            # batchnorm29_1_26_, output shape: {[512,60,60]}

            self.conv27_1_27_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_27_, output shape: {[8,60,60]}

            self.batchnorm27_1_27_ = gluon.nn.BatchNorm()
            # batchnorm27_1_27_, output shape: {[8,60,60]}

            self.relu27_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_27_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_27_, output shape: {[8,60,60]}

            self.batchnorm28_1_27_ = gluon.nn.BatchNorm()
            # batchnorm28_1_27_, output shape: {[8,60,60]}

            self.relu28_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_27_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_27_, output shape: {[512,60,60]}

            self.batchnorm29_1_27_ = gluon.nn.BatchNorm()
            # batchnorm29_1_27_, output shape: {[512,60,60]}

            self.conv27_1_28_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_28_, output shape: {[8,60,60]}

            self.batchnorm27_1_28_ = gluon.nn.BatchNorm()
            # batchnorm27_1_28_, output shape: {[8,60,60]}

            self.relu27_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_28_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_28_, output shape: {[8,60,60]}

            self.batchnorm28_1_28_ = gluon.nn.BatchNorm()
            # batchnorm28_1_28_, output shape: {[8,60,60]}

            self.relu28_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_28_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_28_, output shape: {[512,60,60]}

            self.batchnorm29_1_28_ = gluon.nn.BatchNorm()
            # batchnorm29_1_28_, output shape: {[512,60,60]}

            self.conv27_1_29_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_29_, output shape: {[8,60,60]}

            self.batchnorm27_1_29_ = gluon.nn.BatchNorm()
            # batchnorm27_1_29_, output shape: {[8,60,60]}

            self.relu27_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_29_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_29_, output shape: {[8,60,60]}

            self.batchnorm28_1_29_ = gluon.nn.BatchNorm()
            # batchnorm28_1_29_, output shape: {[8,60,60]}

            self.relu28_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_29_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_29_, output shape: {[512,60,60]}

            self.batchnorm29_1_29_ = gluon.nn.BatchNorm()
            # batchnorm29_1_29_, output shape: {[512,60,60]}

            self.conv27_1_30_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_30_, output shape: {[8,60,60]}

            self.batchnorm27_1_30_ = gluon.nn.BatchNorm()
            # batchnorm27_1_30_, output shape: {[8,60,60]}

            self.relu27_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_30_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_30_, output shape: {[8,60,60]}

            self.batchnorm28_1_30_ = gluon.nn.BatchNorm()
            # batchnorm28_1_30_, output shape: {[8,60,60]}

            self.relu28_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_30_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_30_, output shape: {[512,60,60]}

            self.batchnorm29_1_30_ = gluon.nn.BatchNorm()
            # batchnorm29_1_30_, output shape: {[512,60,60]}

            self.conv27_1_31_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_31_, output shape: {[8,60,60]}

            self.batchnorm27_1_31_ = gluon.nn.BatchNorm()
            # batchnorm27_1_31_, output shape: {[8,60,60]}

            self.relu27_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_31_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_31_, output shape: {[8,60,60]}

            self.batchnorm28_1_31_ = gluon.nn.BatchNorm()
            # batchnorm28_1_31_, output shape: {[8,60,60]}

            self.relu28_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_31_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_31_, output shape: {[512,60,60]}

            self.batchnorm29_1_31_ = gluon.nn.BatchNorm()
            # batchnorm29_1_31_, output shape: {[512,60,60]}

            self.conv27_1_32_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv27_1_32_, output shape: {[8,60,60]}

            self.batchnorm27_1_32_ = gluon.nn.BatchNorm()
            # batchnorm27_1_32_, output shape: {[8,60,60]}

            self.relu27_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv28_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv28_1_32_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv28_1_32_, output shape: {[8,60,60]}

            self.batchnorm28_1_32_ = gluon.nn.BatchNorm()
            # batchnorm28_1_32_, output shape: {[8,60,60]}

            self.relu28_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv29_1_32_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv29_1_32_, output shape: {[512,60,60]}

            self.batchnorm29_1_32_ = gluon.nn.BatchNorm()
            # batchnorm29_1_32_, output shape: {[512,60,60]}

            self.relu31_ = gluon.nn.Activation(activation='relu')
            self.conv33_1_1_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_1_, output shape: {[8,60,60]}

            self.batchnorm33_1_1_ = gluon.nn.BatchNorm()
            # batchnorm33_1_1_, output shape: {[8,60,60]}

            self.relu33_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_1_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_1_, output shape: {[8,60,60]}

            self.batchnorm34_1_1_ = gluon.nn.BatchNorm()
            # batchnorm34_1_1_, output shape: {[8,60,60]}

            self.relu34_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_1_, output shape: {[512,60,60]}

            self.batchnorm35_1_1_ = gluon.nn.BatchNorm()
            # batchnorm35_1_1_, output shape: {[512,60,60]}

            self.conv33_1_2_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_2_, output shape: {[8,60,60]}

            self.batchnorm33_1_2_ = gluon.nn.BatchNorm()
            # batchnorm33_1_2_, output shape: {[8,60,60]}

            self.relu33_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_2_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_2_, output shape: {[8,60,60]}

            self.batchnorm34_1_2_ = gluon.nn.BatchNorm()
            # batchnorm34_1_2_, output shape: {[8,60,60]}

            self.relu34_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_2_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_2_, output shape: {[512,60,60]}

            self.batchnorm35_1_2_ = gluon.nn.BatchNorm()
            # batchnorm35_1_2_, output shape: {[512,60,60]}

            self.conv33_1_3_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_3_, output shape: {[8,60,60]}

            self.batchnorm33_1_3_ = gluon.nn.BatchNorm()
            # batchnorm33_1_3_, output shape: {[8,60,60]}

            self.relu33_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_3_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_3_, output shape: {[8,60,60]}

            self.batchnorm34_1_3_ = gluon.nn.BatchNorm()
            # batchnorm34_1_3_, output shape: {[8,60,60]}

            self.relu34_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_3_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_3_, output shape: {[512,60,60]}

            self.batchnorm35_1_3_ = gluon.nn.BatchNorm()
            # batchnorm35_1_3_, output shape: {[512,60,60]}

            self.conv33_1_4_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_4_, output shape: {[8,60,60]}

            self.batchnorm33_1_4_ = gluon.nn.BatchNorm()
            # batchnorm33_1_4_, output shape: {[8,60,60]}

            self.relu33_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_4_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_4_, output shape: {[8,60,60]}

            self.batchnorm34_1_4_ = gluon.nn.BatchNorm()
            # batchnorm34_1_4_, output shape: {[8,60,60]}

            self.relu34_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_4_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_4_, output shape: {[512,60,60]}

            self.batchnorm35_1_4_ = gluon.nn.BatchNorm()
            # batchnorm35_1_4_, output shape: {[512,60,60]}

            self.conv33_1_5_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_5_, output shape: {[8,60,60]}

            self.batchnorm33_1_5_ = gluon.nn.BatchNorm()
            # batchnorm33_1_5_, output shape: {[8,60,60]}

            self.relu33_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_5_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_5_, output shape: {[8,60,60]}

            self.batchnorm34_1_5_ = gluon.nn.BatchNorm()
            # batchnorm34_1_5_, output shape: {[8,60,60]}

            self.relu34_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_5_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_5_, output shape: {[512,60,60]}

            self.batchnorm35_1_5_ = gluon.nn.BatchNorm()
            # batchnorm35_1_5_, output shape: {[512,60,60]}

            self.conv33_1_6_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_6_, output shape: {[8,60,60]}

            self.batchnorm33_1_6_ = gluon.nn.BatchNorm()
            # batchnorm33_1_6_, output shape: {[8,60,60]}

            self.relu33_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_6_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_6_, output shape: {[8,60,60]}

            self.batchnorm34_1_6_ = gluon.nn.BatchNorm()
            # batchnorm34_1_6_, output shape: {[8,60,60]}

            self.relu34_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_6_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_6_, output shape: {[512,60,60]}

            self.batchnorm35_1_6_ = gluon.nn.BatchNorm()
            # batchnorm35_1_6_, output shape: {[512,60,60]}

            self.conv33_1_7_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_7_, output shape: {[8,60,60]}

            self.batchnorm33_1_7_ = gluon.nn.BatchNorm()
            # batchnorm33_1_7_, output shape: {[8,60,60]}

            self.relu33_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_7_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_7_, output shape: {[8,60,60]}

            self.batchnorm34_1_7_ = gluon.nn.BatchNorm()
            # batchnorm34_1_7_, output shape: {[8,60,60]}

            self.relu34_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_7_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_7_, output shape: {[512,60,60]}

            self.batchnorm35_1_7_ = gluon.nn.BatchNorm()
            # batchnorm35_1_7_, output shape: {[512,60,60]}

            self.conv33_1_8_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_8_, output shape: {[8,60,60]}

            self.batchnorm33_1_8_ = gluon.nn.BatchNorm()
            # batchnorm33_1_8_, output shape: {[8,60,60]}

            self.relu33_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_8_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_8_, output shape: {[8,60,60]}

            self.batchnorm34_1_8_ = gluon.nn.BatchNorm()
            # batchnorm34_1_8_, output shape: {[8,60,60]}

            self.relu34_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_8_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_8_, output shape: {[512,60,60]}

            self.batchnorm35_1_8_ = gluon.nn.BatchNorm()
            # batchnorm35_1_8_, output shape: {[512,60,60]}

            self.conv33_1_9_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_9_, output shape: {[8,60,60]}

            self.batchnorm33_1_9_ = gluon.nn.BatchNorm()
            # batchnorm33_1_9_, output shape: {[8,60,60]}

            self.relu33_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_9_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_9_, output shape: {[8,60,60]}

            self.batchnorm34_1_9_ = gluon.nn.BatchNorm()
            # batchnorm34_1_9_, output shape: {[8,60,60]}

            self.relu34_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_9_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_9_, output shape: {[512,60,60]}

            self.batchnorm35_1_9_ = gluon.nn.BatchNorm()
            # batchnorm35_1_9_, output shape: {[512,60,60]}

            self.conv33_1_10_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_10_, output shape: {[8,60,60]}

            self.batchnorm33_1_10_ = gluon.nn.BatchNorm()
            # batchnorm33_1_10_, output shape: {[8,60,60]}

            self.relu33_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_10_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_10_, output shape: {[8,60,60]}

            self.batchnorm34_1_10_ = gluon.nn.BatchNorm()
            # batchnorm34_1_10_, output shape: {[8,60,60]}

            self.relu34_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_10_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_10_, output shape: {[512,60,60]}

            self.batchnorm35_1_10_ = gluon.nn.BatchNorm()
            # batchnorm35_1_10_, output shape: {[512,60,60]}

            self.conv33_1_11_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_11_, output shape: {[8,60,60]}

            self.batchnorm33_1_11_ = gluon.nn.BatchNorm()
            # batchnorm33_1_11_, output shape: {[8,60,60]}

            self.relu33_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_11_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_11_, output shape: {[8,60,60]}

            self.batchnorm34_1_11_ = gluon.nn.BatchNorm()
            # batchnorm34_1_11_, output shape: {[8,60,60]}

            self.relu34_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_11_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_11_, output shape: {[512,60,60]}

            self.batchnorm35_1_11_ = gluon.nn.BatchNorm()
            # batchnorm35_1_11_, output shape: {[512,60,60]}

            self.conv33_1_12_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_12_, output shape: {[8,60,60]}

            self.batchnorm33_1_12_ = gluon.nn.BatchNorm()
            # batchnorm33_1_12_, output shape: {[8,60,60]}

            self.relu33_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_12_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_12_, output shape: {[8,60,60]}

            self.batchnorm34_1_12_ = gluon.nn.BatchNorm()
            # batchnorm34_1_12_, output shape: {[8,60,60]}

            self.relu34_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_12_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_12_, output shape: {[512,60,60]}

            self.batchnorm35_1_12_ = gluon.nn.BatchNorm()
            # batchnorm35_1_12_, output shape: {[512,60,60]}

            self.conv33_1_13_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_13_, output shape: {[8,60,60]}

            self.batchnorm33_1_13_ = gluon.nn.BatchNorm()
            # batchnorm33_1_13_, output shape: {[8,60,60]}

            self.relu33_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_13_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_13_, output shape: {[8,60,60]}

            self.batchnorm34_1_13_ = gluon.nn.BatchNorm()
            # batchnorm34_1_13_, output shape: {[8,60,60]}

            self.relu34_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_13_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_13_, output shape: {[512,60,60]}

            self.batchnorm35_1_13_ = gluon.nn.BatchNorm()
            # batchnorm35_1_13_, output shape: {[512,60,60]}

            self.conv33_1_14_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_14_, output shape: {[8,60,60]}

            self.batchnorm33_1_14_ = gluon.nn.BatchNorm()
            # batchnorm33_1_14_, output shape: {[8,60,60]}

            self.relu33_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_14_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_14_, output shape: {[8,60,60]}

            self.batchnorm34_1_14_ = gluon.nn.BatchNorm()
            # batchnorm34_1_14_, output shape: {[8,60,60]}

            self.relu34_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_14_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_14_, output shape: {[512,60,60]}

            self.batchnorm35_1_14_ = gluon.nn.BatchNorm()
            # batchnorm35_1_14_, output shape: {[512,60,60]}

            self.conv33_1_15_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_15_, output shape: {[8,60,60]}

            self.batchnorm33_1_15_ = gluon.nn.BatchNorm()
            # batchnorm33_1_15_, output shape: {[8,60,60]}

            self.relu33_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_15_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_15_, output shape: {[8,60,60]}

            self.batchnorm34_1_15_ = gluon.nn.BatchNorm()
            # batchnorm34_1_15_, output shape: {[8,60,60]}

            self.relu34_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_15_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_15_, output shape: {[512,60,60]}

            self.batchnorm35_1_15_ = gluon.nn.BatchNorm()
            # batchnorm35_1_15_, output shape: {[512,60,60]}

            self.conv33_1_16_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_16_, output shape: {[8,60,60]}

            self.batchnorm33_1_16_ = gluon.nn.BatchNorm()
            # batchnorm33_1_16_, output shape: {[8,60,60]}

            self.relu33_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_16_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_16_, output shape: {[8,60,60]}

            self.batchnorm34_1_16_ = gluon.nn.BatchNorm()
            # batchnorm34_1_16_, output shape: {[8,60,60]}

            self.relu34_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_16_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_16_, output shape: {[512,60,60]}

            self.batchnorm35_1_16_ = gluon.nn.BatchNorm()
            # batchnorm35_1_16_, output shape: {[512,60,60]}

            self.conv33_1_17_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_17_, output shape: {[8,60,60]}

            self.batchnorm33_1_17_ = gluon.nn.BatchNorm()
            # batchnorm33_1_17_, output shape: {[8,60,60]}

            self.relu33_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_17_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_17_, output shape: {[8,60,60]}

            self.batchnorm34_1_17_ = gluon.nn.BatchNorm()
            # batchnorm34_1_17_, output shape: {[8,60,60]}

            self.relu34_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_17_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_17_, output shape: {[512,60,60]}

            self.batchnorm35_1_17_ = gluon.nn.BatchNorm()
            # batchnorm35_1_17_, output shape: {[512,60,60]}

            self.conv33_1_18_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_18_, output shape: {[8,60,60]}

            self.batchnorm33_1_18_ = gluon.nn.BatchNorm()
            # batchnorm33_1_18_, output shape: {[8,60,60]}

            self.relu33_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_18_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_18_, output shape: {[8,60,60]}

            self.batchnorm34_1_18_ = gluon.nn.BatchNorm()
            # batchnorm34_1_18_, output shape: {[8,60,60]}

            self.relu34_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_18_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_18_, output shape: {[512,60,60]}

            self.batchnorm35_1_18_ = gluon.nn.BatchNorm()
            # batchnorm35_1_18_, output shape: {[512,60,60]}

            self.conv33_1_19_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_19_, output shape: {[8,60,60]}

            self.batchnorm33_1_19_ = gluon.nn.BatchNorm()
            # batchnorm33_1_19_, output shape: {[8,60,60]}

            self.relu33_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_19_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_19_, output shape: {[8,60,60]}

            self.batchnorm34_1_19_ = gluon.nn.BatchNorm()
            # batchnorm34_1_19_, output shape: {[8,60,60]}

            self.relu34_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_19_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_19_, output shape: {[512,60,60]}

            self.batchnorm35_1_19_ = gluon.nn.BatchNorm()
            # batchnorm35_1_19_, output shape: {[512,60,60]}

            self.conv33_1_20_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_20_, output shape: {[8,60,60]}

            self.batchnorm33_1_20_ = gluon.nn.BatchNorm()
            # batchnorm33_1_20_, output shape: {[8,60,60]}

            self.relu33_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_20_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_20_, output shape: {[8,60,60]}

            self.batchnorm34_1_20_ = gluon.nn.BatchNorm()
            # batchnorm34_1_20_, output shape: {[8,60,60]}

            self.relu34_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_20_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_20_, output shape: {[512,60,60]}

            self.batchnorm35_1_20_ = gluon.nn.BatchNorm()
            # batchnorm35_1_20_, output shape: {[512,60,60]}

            self.conv33_1_21_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_21_, output shape: {[8,60,60]}

            self.batchnorm33_1_21_ = gluon.nn.BatchNorm()
            # batchnorm33_1_21_, output shape: {[8,60,60]}

            self.relu33_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_21_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_21_, output shape: {[8,60,60]}

            self.batchnorm34_1_21_ = gluon.nn.BatchNorm()
            # batchnorm34_1_21_, output shape: {[8,60,60]}

            self.relu34_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_21_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_21_, output shape: {[512,60,60]}

            self.batchnorm35_1_21_ = gluon.nn.BatchNorm()
            # batchnorm35_1_21_, output shape: {[512,60,60]}

            self.conv33_1_22_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_22_, output shape: {[8,60,60]}

            self.batchnorm33_1_22_ = gluon.nn.BatchNorm()
            # batchnorm33_1_22_, output shape: {[8,60,60]}

            self.relu33_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_22_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_22_, output shape: {[8,60,60]}

            self.batchnorm34_1_22_ = gluon.nn.BatchNorm()
            # batchnorm34_1_22_, output shape: {[8,60,60]}

            self.relu34_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_22_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_22_, output shape: {[512,60,60]}

            self.batchnorm35_1_22_ = gluon.nn.BatchNorm()
            # batchnorm35_1_22_, output shape: {[512,60,60]}

            self.conv33_1_23_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_23_, output shape: {[8,60,60]}

            self.batchnorm33_1_23_ = gluon.nn.BatchNorm()
            # batchnorm33_1_23_, output shape: {[8,60,60]}

            self.relu33_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_23_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_23_, output shape: {[8,60,60]}

            self.batchnorm34_1_23_ = gluon.nn.BatchNorm()
            # batchnorm34_1_23_, output shape: {[8,60,60]}

            self.relu34_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_23_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_23_, output shape: {[512,60,60]}

            self.batchnorm35_1_23_ = gluon.nn.BatchNorm()
            # batchnorm35_1_23_, output shape: {[512,60,60]}

            self.conv33_1_24_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_24_, output shape: {[8,60,60]}

            self.batchnorm33_1_24_ = gluon.nn.BatchNorm()
            # batchnorm33_1_24_, output shape: {[8,60,60]}

            self.relu33_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_24_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_24_, output shape: {[8,60,60]}

            self.batchnorm34_1_24_ = gluon.nn.BatchNorm()
            # batchnorm34_1_24_, output shape: {[8,60,60]}

            self.relu34_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_24_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_24_, output shape: {[512,60,60]}

            self.batchnorm35_1_24_ = gluon.nn.BatchNorm()
            # batchnorm35_1_24_, output shape: {[512,60,60]}

            self.conv33_1_25_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_25_, output shape: {[8,60,60]}

            self.batchnorm33_1_25_ = gluon.nn.BatchNorm()
            # batchnorm33_1_25_, output shape: {[8,60,60]}

            self.relu33_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_25_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_25_, output shape: {[8,60,60]}

            self.batchnorm34_1_25_ = gluon.nn.BatchNorm()
            # batchnorm34_1_25_, output shape: {[8,60,60]}

            self.relu34_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_25_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_25_, output shape: {[512,60,60]}

            self.batchnorm35_1_25_ = gluon.nn.BatchNorm()
            # batchnorm35_1_25_, output shape: {[512,60,60]}

            self.conv33_1_26_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_26_, output shape: {[8,60,60]}

            self.batchnorm33_1_26_ = gluon.nn.BatchNorm()
            # batchnorm33_1_26_, output shape: {[8,60,60]}

            self.relu33_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_26_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_26_, output shape: {[8,60,60]}

            self.batchnorm34_1_26_ = gluon.nn.BatchNorm()
            # batchnorm34_1_26_, output shape: {[8,60,60]}

            self.relu34_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_26_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_26_, output shape: {[512,60,60]}

            self.batchnorm35_1_26_ = gluon.nn.BatchNorm()
            # batchnorm35_1_26_, output shape: {[512,60,60]}

            self.conv33_1_27_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_27_, output shape: {[8,60,60]}

            self.batchnorm33_1_27_ = gluon.nn.BatchNorm()
            # batchnorm33_1_27_, output shape: {[8,60,60]}

            self.relu33_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_27_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_27_, output shape: {[8,60,60]}

            self.batchnorm34_1_27_ = gluon.nn.BatchNorm()
            # batchnorm34_1_27_, output shape: {[8,60,60]}

            self.relu34_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_27_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_27_, output shape: {[512,60,60]}

            self.batchnorm35_1_27_ = gluon.nn.BatchNorm()
            # batchnorm35_1_27_, output shape: {[512,60,60]}

            self.conv33_1_28_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_28_, output shape: {[8,60,60]}

            self.batchnorm33_1_28_ = gluon.nn.BatchNorm()
            # batchnorm33_1_28_, output shape: {[8,60,60]}

            self.relu33_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_28_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_28_, output shape: {[8,60,60]}

            self.batchnorm34_1_28_ = gluon.nn.BatchNorm()
            # batchnorm34_1_28_, output shape: {[8,60,60]}

            self.relu34_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_28_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_28_, output shape: {[512,60,60]}

            self.batchnorm35_1_28_ = gluon.nn.BatchNorm()
            # batchnorm35_1_28_, output shape: {[512,60,60]}

            self.conv33_1_29_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_29_, output shape: {[8,60,60]}

            self.batchnorm33_1_29_ = gluon.nn.BatchNorm()
            # batchnorm33_1_29_, output shape: {[8,60,60]}

            self.relu33_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_29_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_29_, output shape: {[8,60,60]}

            self.batchnorm34_1_29_ = gluon.nn.BatchNorm()
            # batchnorm34_1_29_, output shape: {[8,60,60]}

            self.relu34_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_29_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_29_, output shape: {[512,60,60]}

            self.batchnorm35_1_29_ = gluon.nn.BatchNorm()
            # batchnorm35_1_29_, output shape: {[512,60,60]}

            self.conv33_1_30_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_30_, output shape: {[8,60,60]}

            self.batchnorm33_1_30_ = gluon.nn.BatchNorm()
            # batchnorm33_1_30_, output shape: {[8,60,60]}

            self.relu33_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_30_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_30_, output shape: {[8,60,60]}

            self.batchnorm34_1_30_ = gluon.nn.BatchNorm()
            # batchnorm34_1_30_, output shape: {[8,60,60]}

            self.relu34_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_30_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_30_, output shape: {[512,60,60]}

            self.batchnorm35_1_30_ = gluon.nn.BatchNorm()
            # batchnorm35_1_30_, output shape: {[512,60,60]}

            self.conv33_1_31_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_31_, output shape: {[8,60,60]}

            self.batchnorm33_1_31_ = gluon.nn.BatchNorm()
            # batchnorm33_1_31_, output shape: {[8,60,60]}

            self.relu33_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_31_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_31_, output shape: {[8,60,60]}

            self.batchnorm34_1_31_ = gluon.nn.BatchNorm()
            # batchnorm34_1_31_, output shape: {[8,60,60]}

            self.relu34_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_31_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_31_, output shape: {[512,60,60]}

            self.batchnorm35_1_31_ = gluon.nn.BatchNorm()
            # batchnorm35_1_31_, output shape: {[512,60,60]}

            self.conv33_1_32_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv33_1_32_, output shape: {[8,60,60]}

            self.batchnorm33_1_32_ = gluon.nn.BatchNorm()
            # batchnorm33_1_32_, output shape: {[8,60,60]}

            self.relu33_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv34_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv34_1_32_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv34_1_32_, output shape: {[8,60,60]}

            self.batchnorm34_1_32_ = gluon.nn.BatchNorm()
            # batchnorm34_1_32_, output shape: {[8,60,60]}

            self.relu34_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv35_1_32_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv35_1_32_, output shape: {[512,60,60]}

            self.batchnorm35_1_32_ = gluon.nn.BatchNorm()
            # batchnorm35_1_32_, output shape: {[512,60,60]}

            self.relu37_ = gluon.nn.Activation(activation='relu')
            self.conv39_1_1_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_1_, output shape: {[8,60,60]}

            self.batchnorm39_1_1_ = gluon.nn.BatchNorm()
            # batchnorm39_1_1_, output shape: {[8,60,60]}

            self.relu39_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_1_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_1_, output shape: {[8,60,60]}

            self.batchnorm40_1_1_ = gluon.nn.BatchNorm()
            # batchnorm40_1_1_, output shape: {[8,60,60]}

            self.relu40_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_1_, output shape: {[512,60,60]}

            self.batchnorm41_1_1_ = gluon.nn.BatchNorm()
            # batchnorm41_1_1_, output shape: {[512,60,60]}

            self.conv39_1_2_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_2_, output shape: {[8,60,60]}

            self.batchnorm39_1_2_ = gluon.nn.BatchNorm()
            # batchnorm39_1_2_, output shape: {[8,60,60]}

            self.relu39_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_2_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_2_, output shape: {[8,60,60]}

            self.batchnorm40_1_2_ = gluon.nn.BatchNorm()
            # batchnorm40_1_2_, output shape: {[8,60,60]}

            self.relu40_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_2_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_2_, output shape: {[512,60,60]}

            self.batchnorm41_1_2_ = gluon.nn.BatchNorm()
            # batchnorm41_1_2_, output shape: {[512,60,60]}

            self.conv39_1_3_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_3_, output shape: {[8,60,60]}

            self.batchnorm39_1_3_ = gluon.nn.BatchNorm()
            # batchnorm39_1_3_, output shape: {[8,60,60]}

            self.relu39_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_3_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_3_, output shape: {[8,60,60]}

            self.batchnorm40_1_3_ = gluon.nn.BatchNorm()
            # batchnorm40_1_3_, output shape: {[8,60,60]}

            self.relu40_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_3_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_3_, output shape: {[512,60,60]}

            self.batchnorm41_1_3_ = gluon.nn.BatchNorm()
            # batchnorm41_1_3_, output shape: {[512,60,60]}

            self.conv39_1_4_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_4_, output shape: {[8,60,60]}

            self.batchnorm39_1_4_ = gluon.nn.BatchNorm()
            # batchnorm39_1_4_, output shape: {[8,60,60]}

            self.relu39_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_4_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_4_, output shape: {[8,60,60]}

            self.batchnorm40_1_4_ = gluon.nn.BatchNorm()
            # batchnorm40_1_4_, output shape: {[8,60,60]}

            self.relu40_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_4_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_4_, output shape: {[512,60,60]}

            self.batchnorm41_1_4_ = gluon.nn.BatchNorm()
            # batchnorm41_1_4_, output shape: {[512,60,60]}

            self.conv39_1_5_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_5_, output shape: {[8,60,60]}

            self.batchnorm39_1_5_ = gluon.nn.BatchNorm()
            # batchnorm39_1_5_, output shape: {[8,60,60]}

            self.relu39_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_5_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_5_, output shape: {[8,60,60]}

            self.batchnorm40_1_5_ = gluon.nn.BatchNorm()
            # batchnorm40_1_5_, output shape: {[8,60,60]}

            self.relu40_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_5_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_5_, output shape: {[512,60,60]}

            self.batchnorm41_1_5_ = gluon.nn.BatchNorm()
            # batchnorm41_1_5_, output shape: {[512,60,60]}

            self.conv39_1_6_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_6_, output shape: {[8,60,60]}

            self.batchnorm39_1_6_ = gluon.nn.BatchNorm()
            # batchnorm39_1_6_, output shape: {[8,60,60]}

            self.relu39_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_6_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_6_, output shape: {[8,60,60]}

            self.batchnorm40_1_6_ = gluon.nn.BatchNorm()
            # batchnorm40_1_6_, output shape: {[8,60,60]}

            self.relu40_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_6_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_6_, output shape: {[512,60,60]}

            self.batchnorm41_1_6_ = gluon.nn.BatchNorm()
            # batchnorm41_1_6_, output shape: {[512,60,60]}

            self.conv39_1_7_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_7_, output shape: {[8,60,60]}

            self.batchnorm39_1_7_ = gluon.nn.BatchNorm()
            # batchnorm39_1_7_, output shape: {[8,60,60]}

            self.relu39_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_7_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_7_, output shape: {[8,60,60]}

            self.batchnorm40_1_7_ = gluon.nn.BatchNorm()
            # batchnorm40_1_7_, output shape: {[8,60,60]}

            self.relu40_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_7_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_7_, output shape: {[512,60,60]}

            self.batchnorm41_1_7_ = gluon.nn.BatchNorm()
            # batchnorm41_1_7_, output shape: {[512,60,60]}

            self.conv39_1_8_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_8_, output shape: {[8,60,60]}

            self.batchnorm39_1_8_ = gluon.nn.BatchNorm()
            # batchnorm39_1_8_, output shape: {[8,60,60]}

            self.relu39_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_8_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_8_, output shape: {[8,60,60]}

            self.batchnorm40_1_8_ = gluon.nn.BatchNorm()
            # batchnorm40_1_8_, output shape: {[8,60,60]}

            self.relu40_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_8_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_8_, output shape: {[512,60,60]}

            self.batchnorm41_1_8_ = gluon.nn.BatchNorm()
            # batchnorm41_1_8_, output shape: {[512,60,60]}

            self.conv39_1_9_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_9_, output shape: {[8,60,60]}

            self.batchnorm39_1_9_ = gluon.nn.BatchNorm()
            # batchnorm39_1_9_, output shape: {[8,60,60]}

            self.relu39_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_9_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_9_, output shape: {[8,60,60]}

            self.batchnorm40_1_9_ = gluon.nn.BatchNorm()
            # batchnorm40_1_9_, output shape: {[8,60,60]}

            self.relu40_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_9_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_9_, output shape: {[512,60,60]}

            self.batchnorm41_1_9_ = gluon.nn.BatchNorm()
            # batchnorm41_1_9_, output shape: {[512,60,60]}

            self.conv39_1_10_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_10_, output shape: {[8,60,60]}

            self.batchnorm39_1_10_ = gluon.nn.BatchNorm()
            # batchnorm39_1_10_, output shape: {[8,60,60]}

            self.relu39_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_10_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_10_, output shape: {[8,60,60]}

            self.batchnorm40_1_10_ = gluon.nn.BatchNorm()
            # batchnorm40_1_10_, output shape: {[8,60,60]}

            self.relu40_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_10_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_10_, output shape: {[512,60,60]}

            self.batchnorm41_1_10_ = gluon.nn.BatchNorm()
            # batchnorm41_1_10_, output shape: {[512,60,60]}

            self.conv39_1_11_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_11_, output shape: {[8,60,60]}

            self.batchnorm39_1_11_ = gluon.nn.BatchNorm()
            # batchnorm39_1_11_, output shape: {[8,60,60]}

            self.relu39_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_11_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_11_, output shape: {[8,60,60]}

            self.batchnorm40_1_11_ = gluon.nn.BatchNorm()
            # batchnorm40_1_11_, output shape: {[8,60,60]}

            self.relu40_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_11_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_11_, output shape: {[512,60,60]}

            self.batchnorm41_1_11_ = gluon.nn.BatchNorm()
            # batchnorm41_1_11_, output shape: {[512,60,60]}

            self.conv39_1_12_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_12_, output shape: {[8,60,60]}

            self.batchnorm39_1_12_ = gluon.nn.BatchNorm()
            # batchnorm39_1_12_, output shape: {[8,60,60]}

            self.relu39_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_12_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_12_, output shape: {[8,60,60]}

            self.batchnorm40_1_12_ = gluon.nn.BatchNorm()
            # batchnorm40_1_12_, output shape: {[8,60,60]}

            self.relu40_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_12_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_12_, output shape: {[512,60,60]}

            self.batchnorm41_1_12_ = gluon.nn.BatchNorm()
            # batchnorm41_1_12_, output shape: {[512,60,60]}

            self.conv39_1_13_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_13_, output shape: {[8,60,60]}

            self.batchnorm39_1_13_ = gluon.nn.BatchNorm()
            # batchnorm39_1_13_, output shape: {[8,60,60]}

            self.relu39_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_13_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_13_, output shape: {[8,60,60]}

            self.batchnorm40_1_13_ = gluon.nn.BatchNorm()
            # batchnorm40_1_13_, output shape: {[8,60,60]}

            self.relu40_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_13_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_13_, output shape: {[512,60,60]}

            self.batchnorm41_1_13_ = gluon.nn.BatchNorm()
            # batchnorm41_1_13_, output shape: {[512,60,60]}

            self.conv39_1_14_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_14_, output shape: {[8,60,60]}

            self.batchnorm39_1_14_ = gluon.nn.BatchNorm()
            # batchnorm39_1_14_, output shape: {[8,60,60]}

            self.relu39_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_14_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_14_, output shape: {[8,60,60]}

            self.batchnorm40_1_14_ = gluon.nn.BatchNorm()
            # batchnorm40_1_14_, output shape: {[8,60,60]}

            self.relu40_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_14_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_14_, output shape: {[512,60,60]}

            self.batchnorm41_1_14_ = gluon.nn.BatchNorm()
            # batchnorm41_1_14_, output shape: {[512,60,60]}

            self.conv39_1_15_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_15_, output shape: {[8,60,60]}

            self.batchnorm39_1_15_ = gluon.nn.BatchNorm()
            # batchnorm39_1_15_, output shape: {[8,60,60]}

            self.relu39_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_15_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_15_, output shape: {[8,60,60]}

            self.batchnorm40_1_15_ = gluon.nn.BatchNorm()
            # batchnorm40_1_15_, output shape: {[8,60,60]}

            self.relu40_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_15_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_15_, output shape: {[512,60,60]}

            self.batchnorm41_1_15_ = gluon.nn.BatchNorm()
            # batchnorm41_1_15_, output shape: {[512,60,60]}

            self.conv39_1_16_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_16_, output shape: {[8,60,60]}

            self.batchnorm39_1_16_ = gluon.nn.BatchNorm()
            # batchnorm39_1_16_, output shape: {[8,60,60]}

            self.relu39_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_16_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_16_, output shape: {[8,60,60]}

            self.batchnorm40_1_16_ = gluon.nn.BatchNorm()
            # batchnorm40_1_16_, output shape: {[8,60,60]}

            self.relu40_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_16_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_16_, output shape: {[512,60,60]}

            self.batchnorm41_1_16_ = gluon.nn.BatchNorm()
            # batchnorm41_1_16_, output shape: {[512,60,60]}

            self.conv39_1_17_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_17_, output shape: {[8,60,60]}

            self.batchnorm39_1_17_ = gluon.nn.BatchNorm()
            # batchnorm39_1_17_, output shape: {[8,60,60]}

            self.relu39_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_17_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_17_, output shape: {[8,60,60]}

            self.batchnorm40_1_17_ = gluon.nn.BatchNorm()
            # batchnorm40_1_17_, output shape: {[8,60,60]}

            self.relu40_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_17_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_17_, output shape: {[512,60,60]}

            self.batchnorm41_1_17_ = gluon.nn.BatchNorm()
            # batchnorm41_1_17_, output shape: {[512,60,60]}

            self.conv39_1_18_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_18_, output shape: {[8,60,60]}

            self.batchnorm39_1_18_ = gluon.nn.BatchNorm()
            # batchnorm39_1_18_, output shape: {[8,60,60]}

            self.relu39_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_18_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_18_, output shape: {[8,60,60]}

            self.batchnorm40_1_18_ = gluon.nn.BatchNorm()
            # batchnorm40_1_18_, output shape: {[8,60,60]}

            self.relu40_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_18_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_18_, output shape: {[512,60,60]}

            self.batchnorm41_1_18_ = gluon.nn.BatchNorm()
            # batchnorm41_1_18_, output shape: {[512,60,60]}

            self.conv39_1_19_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_19_, output shape: {[8,60,60]}

            self.batchnorm39_1_19_ = gluon.nn.BatchNorm()
            # batchnorm39_1_19_, output shape: {[8,60,60]}

            self.relu39_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_19_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_19_, output shape: {[8,60,60]}

            self.batchnorm40_1_19_ = gluon.nn.BatchNorm()
            # batchnorm40_1_19_, output shape: {[8,60,60]}

            self.relu40_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_19_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_19_, output shape: {[512,60,60]}

            self.batchnorm41_1_19_ = gluon.nn.BatchNorm()
            # batchnorm41_1_19_, output shape: {[512,60,60]}

            self.conv39_1_20_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_20_, output shape: {[8,60,60]}

            self.batchnorm39_1_20_ = gluon.nn.BatchNorm()
            # batchnorm39_1_20_, output shape: {[8,60,60]}

            self.relu39_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_20_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_20_, output shape: {[8,60,60]}

            self.batchnorm40_1_20_ = gluon.nn.BatchNorm()
            # batchnorm40_1_20_, output shape: {[8,60,60]}

            self.relu40_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_20_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_20_, output shape: {[512,60,60]}

            self.batchnorm41_1_20_ = gluon.nn.BatchNorm()
            # batchnorm41_1_20_, output shape: {[512,60,60]}

            self.conv39_1_21_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_21_, output shape: {[8,60,60]}

            self.batchnorm39_1_21_ = gluon.nn.BatchNorm()
            # batchnorm39_1_21_, output shape: {[8,60,60]}

            self.relu39_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_21_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_21_, output shape: {[8,60,60]}

            self.batchnorm40_1_21_ = gluon.nn.BatchNorm()
            # batchnorm40_1_21_, output shape: {[8,60,60]}

            self.relu40_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_21_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_21_, output shape: {[512,60,60]}

            self.batchnorm41_1_21_ = gluon.nn.BatchNorm()
            # batchnorm41_1_21_, output shape: {[512,60,60]}

            self.conv39_1_22_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_22_, output shape: {[8,60,60]}

            self.batchnorm39_1_22_ = gluon.nn.BatchNorm()
            # batchnorm39_1_22_, output shape: {[8,60,60]}

            self.relu39_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_22_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_22_, output shape: {[8,60,60]}

            self.batchnorm40_1_22_ = gluon.nn.BatchNorm()
            # batchnorm40_1_22_, output shape: {[8,60,60]}

            self.relu40_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_22_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_22_, output shape: {[512,60,60]}

            self.batchnorm41_1_22_ = gluon.nn.BatchNorm()
            # batchnorm41_1_22_, output shape: {[512,60,60]}

            self.conv39_1_23_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_23_, output shape: {[8,60,60]}

            self.batchnorm39_1_23_ = gluon.nn.BatchNorm()
            # batchnorm39_1_23_, output shape: {[8,60,60]}

            self.relu39_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_23_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_23_, output shape: {[8,60,60]}

            self.batchnorm40_1_23_ = gluon.nn.BatchNorm()
            # batchnorm40_1_23_, output shape: {[8,60,60]}

            self.relu40_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_23_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_23_, output shape: {[512,60,60]}

            self.batchnorm41_1_23_ = gluon.nn.BatchNorm()
            # batchnorm41_1_23_, output shape: {[512,60,60]}

            self.conv39_1_24_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_24_, output shape: {[8,60,60]}

            self.batchnorm39_1_24_ = gluon.nn.BatchNorm()
            # batchnorm39_1_24_, output shape: {[8,60,60]}

            self.relu39_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_24_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_24_, output shape: {[8,60,60]}

            self.batchnorm40_1_24_ = gluon.nn.BatchNorm()
            # batchnorm40_1_24_, output shape: {[8,60,60]}

            self.relu40_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_24_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_24_, output shape: {[512,60,60]}

            self.batchnorm41_1_24_ = gluon.nn.BatchNorm()
            # batchnorm41_1_24_, output shape: {[512,60,60]}

            self.conv39_1_25_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_25_, output shape: {[8,60,60]}

            self.batchnorm39_1_25_ = gluon.nn.BatchNorm()
            # batchnorm39_1_25_, output shape: {[8,60,60]}

            self.relu39_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_25_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_25_, output shape: {[8,60,60]}

            self.batchnorm40_1_25_ = gluon.nn.BatchNorm()
            # batchnorm40_1_25_, output shape: {[8,60,60]}

            self.relu40_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_25_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_25_, output shape: {[512,60,60]}

            self.batchnorm41_1_25_ = gluon.nn.BatchNorm()
            # batchnorm41_1_25_, output shape: {[512,60,60]}

            self.conv39_1_26_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_26_, output shape: {[8,60,60]}

            self.batchnorm39_1_26_ = gluon.nn.BatchNorm()
            # batchnorm39_1_26_, output shape: {[8,60,60]}

            self.relu39_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_26_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_26_, output shape: {[8,60,60]}

            self.batchnorm40_1_26_ = gluon.nn.BatchNorm()
            # batchnorm40_1_26_, output shape: {[8,60,60]}

            self.relu40_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_26_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_26_, output shape: {[512,60,60]}

            self.batchnorm41_1_26_ = gluon.nn.BatchNorm()
            # batchnorm41_1_26_, output shape: {[512,60,60]}

            self.conv39_1_27_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_27_, output shape: {[8,60,60]}

            self.batchnorm39_1_27_ = gluon.nn.BatchNorm()
            # batchnorm39_1_27_, output shape: {[8,60,60]}

            self.relu39_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_27_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_27_, output shape: {[8,60,60]}

            self.batchnorm40_1_27_ = gluon.nn.BatchNorm()
            # batchnorm40_1_27_, output shape: {[8,60,60]}

            self.relu40_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_27_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_27_, output shape: {[512,60,60]}

            self.batchnorm41_1_27_ = gluon.nn.BatchNorm()
            # batchnorm41_1_27_, output shape: {[512,60,60]}

            self.conv39_1_28_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_28_, output shape: {[8,60,60]}

            self.batchnorm39_1_28_ = gluon.nn.BatchNorm()
            # batchnorm39_1_28_, output shape: {[8,60,60]}

            self.relu39_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_28_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_28_, output shape: {[8,60,60]}

            self.batchnorm40_1_28_ = gluon.nn.BatchNorm()
            # batchnorm40_1_28_, output shape: {[8,60,60]}

            self.relu40_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_28_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_28_, output shape: {[512,60,60]}

            self.batchnorm41_1_28_ = gluon.nn.BatchNorm()
            # batchnorm41_1_28_, output shape: {[512,60,60]}

            self.conv39_1_29_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_29_, output shape: {[8,60,60]}

            self.batchnorm39_1_29_ = gluon.nn.BatchNorm()
            # batchnorm39_1_29_, output shape: {[8,60,60]}

            self.relu39_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_29_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_29_, output shape: {[8,60,60]}

            self.batchnorm40_1_29_ = gluon.nn.BatchNorm()
            # batchnorm40_1_29_, output shape: {[8,60,60]}

            self.relu40_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_29_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_29_, output shape: {[512,60,60]}

            self.batchnorm41_1_29_ = gluon.nn.BatchNorm()
            # batchnorm41_1_29_, output shape: {[512,60,60]}

            self.conv39_1_30_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_30_, output shape: {[8,60,60]}

            self.batchnorm39_1_30_ = gluon.nn.BatchNorm()
            # batchnorm39_1_30_, output shape: {[8,60,60]}

            self.relu39_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_30_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_30_, output shape: {[8,60,60]}

            self.batchnorm40_1_30_ = gluon.nn.BatchNorm()
            # batchnorm40_1_30_, output shape: {[8,60,60]}

            self.relu40_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_30_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_30_, output shape: {[512,60,60]}

            self.batchnorm41_1_30_ = gluon.nn.BatchNorm()
            # batchnorm41_1_30_, output shape: {[512,60,60]}

            self.conv39_1_31_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_31_, output shape: {[8,60,60]}

            self.batchnorm39_1_31_ = gluon.nn.BatchNorm()
            # batchnorm39_1_31_, output shape: {[8,60,60]}

            self.relu39_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_31_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_31_, output shape: {[8,60,60]}

            self.batchnorm40_1_31_ = gluon.nn.BatchNorm()
            # batchnorm40_1_31_, output shape: {[8,60,60]}

            self.relu40_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_31_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_31_, output shape: {[512,60,60]}

            self.batchnorm41_1_31_ = gluon.nn.BatchNorm()
            # batchnorm41_1_31_, output shape: {[512,60,60]}

            self.conv39_1_32_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv39_1_32_, output shape: {[8,60,60]}

            self.batchnorm39_1_32_ = gluon.nn.BatchNorm()
            # batchnorm39_1_32_, output shape: {[8,60,60]}

            self.relu39_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv40_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv40_1_32_ = gluon.nn.Conv2D(channels=8,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv40_1_32_, output shape: {[8,60,60]}

            self.batchnorm40_1_32_ = gluon.nn.BatchNorm()
            # batchnorm40_1_32_, output shape: {[8,60,60]}

            self.relu40_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv41_1_32_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv41_1_32_, output shape: {[512,60,60]}

            self.batchnorm41_1_32_ = gluon.nn.BatchNorm()
            # batchnorm41_1_32_, output shape: {[512,60,60]}

            self.relu43_ = gluon.nn.Activation(activation='relu')
            self.conv45_1_1_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_1_, output shape: {[16,60,60]}

            self.batchnorm45_1_1_ = gluon.nn.BatchNorm()
            # batchnorm45_1_1_, output shape: {[16,60,60]}

            self.relu45_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_1_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_1_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_1_, output shape: {[16,30,30]}

            self.batchnorm46_1_1_ = gluon.nn.BatchNorm()
            # batchnorm46_1_1_, output shape: {[16,30,30]}

            self.relu46_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_1_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_1_, output shape: {[1024,30,30]}

            self.batchnorm47_1_1_ = gluon.nn.BatchNorm()
            # batchnorm47_1_1_, output shape: {[1024,30,30]}

            self.conv45_1_2_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_2_, output shape: {[16,60,60]}

            self.batchnorm45_1_2_ = gluon.nn.BatchNorm()
            # batchnorm45_1_2_, output shape: {[16,60,60]}

            self.relu45_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_2_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_2_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_2_, output shape: {[16,30,30]}

            self.batchnorm46_1_2_ = gluon.nn.BatchNorm()
            # batchnorm46_1_2_, output shape: {[16,30,30]}

            self.relu46_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_2_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_2_, output shape: {[1024,30,30]}

            self.batchnorm47_1_2_ = gluon.nn.BatchNorm()
            # batchnorm47_1_2_, output shape: {[1024,30,30]}

            self.conv45_1_3_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_3_, output shape: {[16,60,60]}

            self.batchnorm45_1_3_ = gluon.nn.BatchNorm()
            # batchnorm45_1_3_, output shape: {[16,60,60]}

            self.relu45_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_3_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_3_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_3_, output shape: {[16,30,30]}

            self.batchnorm46_1_3_ = gluon.nn.BatchNorm()
            # batchnorm46_1_3_, output shape: {[16,30,30]}

            self.relu46_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_3_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_3_, output shape: {[1024,30,30]}

            self.batchnorm47_1_3_ = gluon.nn.BatchNorm()
            # batchnorm47_1_3_, output shape: {[1024,30,30]}

            self.conv45_1_4_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_4_, output shape: {[16,60,60]}

            self.batchnorm45_1_4_ = gluon.nn.BatchNorm()
            # batchnorm45_1_4_, output shape: {[16,60,60]}

            self.relu45_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_4_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_4_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_4_, output shape: {[16,30,30]}

            self.batchnorm46_1_4_ = gluon.nn.BatchNorm()
            # batchnorm46_1_4_, output shape: {[16,30,30]}

            self.relu46_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_4_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_4_, output shape: {[1024,30,30]}

            self.batchnorm47_1_4_ = gluon.nn.BatchNorm()
            # batchnorm47_1_4_, output shape: {[1024,30,30]}

            self.conv45_1_5_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_5_, output shape: {[16,60,60]}

            self.batchnorm45_1_5_ = gluon.nn.BatchNorm()
            # batchnorm45_1_5_, output shape: {[16,60,60]}

            self.relu45_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_5_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_5_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_5_, output shape: {[16,30,30]}

            self.batchnorm46_1_5_ = gluon.nn.BatchNorm()
            # batchnorm46_1_5_, output shape: {[16,30,30]}

            self.relu46_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_5_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_5_, output shape: {[1024,30,30]}

            self.batchnorm47_1_5_ = gluon.nn.BatchNorm()
            # batchnorm47_1_5_, output shape: {[1024,30,30]}

            self.conv45_1_6_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_6_, output shape: {[16,60,60]}

            self.batchnorm45_1_6_ = gluon.nn.BatchNorm()
            # batchnorm45_1_6_, output shape: {[16,60,60]}

            self.relu45_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_6_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_6_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_6_, output shape: {[16,30,30]}

            self.batchnorm46_1_6_ = gluon.nn.BatchNorm()
            # batchnorm46_1_6_, output shape: {[16,30,30]}

            self.relu46_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_6_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_6_, output shape: {[1024,30,30]}

            self.batchnorm47_1_6_ = gluon.nn.BatchNorm()
            # batchnorm47_1_6_, output shape: {[1024,30,30]}

            self.conv45_1_7_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_7_, output shape: {[16,60,60]}

            self.batchnorm45_1_7_ = gluon.nn.BatchNorm()
            # batchnorm45_1_7_, output shape: {[16,60,60]}

            self.relu45_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_7_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_7_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_7_, output shape: {[16,30,30]}

            self.batchnorm46_1_7_ = gluon.nn.BatchNorm()
            # batchnorm46_1_7_, output shape: {[16,30,30]}

            self.relu46_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_7_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_7_, output shape: {[1024,30,30]}

            self.batchnorm47_1_7_ = gluon.nn.BatchNorm()
            # batchnorm47_1_7_, output shape: {[1024,30,30]}

            self.conv45_1_8_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_8_, output shape: {[16,60,60]}

            self.batchnorm45_1_8_ = gluon.nn.BatchNorm()
            # batchnorm45_1_8_, output shape: {[16,60,60]}

            self.relu45_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_8_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_8_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_8_, output shape: {[16,30,30]}

            self.batchnorm46_1_8_ = gluon.nn.BatchNorm()
            # batchnorm46_1_8_, output shape: {[16,30,30]}

            self.relu46_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_8_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_8_, output shape: {[1024,30,30]}

            self.batchnorm47_1_8_ = gluon.nn.BatchNorm()
            # batchnorm47_1_8_, output shape: {[1024,30,30]}

            self.conv45_1_9_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_9_, output shape: {[16,60,60]}

            self.batchnorm45_1_9_ = gluon.nn.BatchNorm()
            # batchnorm45_1_9_, output shape: {[16,60,60]}

            self.relu45_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_9_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_9_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_9_, output shape: {[16,30,30]}

            self.batchnorm46_1_9_ = gluon.nn.BatchNorm()
            # batchnorm46_1_9_, output shape: {[16,30,30]}

            self.relu46_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_9_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_9_, output shape: {[1024,30,30]}

            self.batchnorm47_1_9_ = gluon.nn.BatchNorm()
            # batchnorm47_1_9_, output shape: {[1024,30,30]}

            self.conv45_1_10_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_10_, output shape: {[16,60,60]}

            self.batchnorm45_1_10_ = gluon.nn.BatchNorm()
            # batchnorm45_1_10_, output shape: {[16,60,60]}

            self.relu45_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_10_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_10_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_10_, output shape: {[16,30,30]}

            self.batchnorm46_1_10_ = gluon.nn.BatchNorm()
            # batchnorm46_1_10_, output shape: {[16,30,30]}

            self.relu46_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_10_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_10_, output shape: {[1024,30,30]}

            self.batchnorm47_1_10_ = gluon.nn.BatchNorm()
            # batchnorm47_1_10_, output shape: {[1024,30,30]}

            self.conv45_1_11_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_11_, output shape: {[16,60,60]}

            self.batchnorm45_1_11_ = gluon.nn.BatchNorm()
            # batchnorm45_1_11_, output shape: {[16,60,60]}

            self.relu45_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_11_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_11_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_11_, output shape: {[16,30,30]}

            self.batchnorm46_1_11_ = gluon.nn.BatchNorm()
            # batchnorm46_1_11_, output shape: {[16,30,30]}

            self.relu46_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_11_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_11_, output shape: {[1024,30,30]}

            self.batchnorm47_1_11_ = gluon.nn.BatchNorm()
            # batchnorm47_1_11_, output shape: {[1024,30,30]}

            self.conv45_1_12_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_12_, output shape: {[16,60,60]}

            self.batchnorm45_1_12_ = gluon.nn.BatchNorm()
            # batchnorm45_1_12_, output shape: {[16,60,60]}

            self.relu45_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_12_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_12_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_12_, output shape: {[16,30,30]}

            self.batchnorm46_1_12_ = gluon.nn.BatchNorm()
            # batchnorm46_1_12_, output shape: {[16,30,30]}

            self.relu46_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_12_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_12_, output shape: {[1024,30,30]}

            self.batchnorm47_1_12_ = gluon.nn.BatchNorm()
            # batchnorm47_1_12_, output shape: {[1024,30,30]}

            self.conv45_1_13_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_13_, output shape: {[16,60,60]}

            self.batchnorm45_1_13_ = gluon.nn.BatchNorm()
            # batchnorm45_1_13_, output shape: {[16,60,60]}

            self.relu45_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_13_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_13_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_13_, output shape: {[16,30,30]}

            self.batchnorm46_1_13_ = gluon.nn.BatchNorm()
            # batchnorm46_1_13_, output shape: {[16,30,30]}

            self.relu46_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_13_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_13_, output shape: {[1024,30,30]}

            self.batchnorm47_1_13_ = gluon.nn.BatchNorm()
            # batchnorm47_1_13_, output shape: {[1024,30,30]}

            self.conv45_1_14_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_14_, output shape: {[16,60,60]}

            self.batchnorm45_1_14_ = gluon.nn.BatchNorm()
            # batchnorm45_1_14_, output shape: {[16,60,60]}

            self.relu45_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_14_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_14_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_14_, output shape: {[16,30,30]}

            self.batchnorm46_1_14_ = gluon.nn.BatchNorm()
            # batchnorm46_1_14_, output shape: {[16,30,30]}

            self.relu46_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_14_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_14_, output shape: {[1024,30,30]}

            self.batchnorm47_1_14_ = gluon.nn.BatchNorm()
            # batchnorm47_1_14_, output shape: {[1024,30,30]}

            self.conv45_1_15_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_15_, output shape: {[16,60,60]}

            self.batchnorm45_1_15_ = gluon.nn.BatchNorm()
            # batchnorm45_1_15_, output shape: {[16,60,60]}

            self.relu45_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_15_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_15_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_15_, output shape: {[16,30,30]}

            self.batchnorm46_1_15_ = gluon.nn.BatchNorm()
            # batchnorm46_1_15_, output shape: {[16,30,30]}

            self.relu46_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_15_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_15_, output shape: {[1024,30,30]}

            self.batchnorm47_1_15_ = gluon.nn.BatchNorm()
            # batchnorm47_1_15_, output shape: {[1024,30,30]}

            self.conv45_1_16_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_16_, output shape: {[16,60,60]}

            self.batchnorm45_1_16_ = gluon.nn.BatchNorm()
            # batchnorm45_1_16_, output shape: {[16,60,60]}

            self.relu45_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_16_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_16_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_16_, output shape: {[16,30,30]}

            self.batchnorm46_1_16_ = gluon.nn.BatchNorm()
            # batchnorm46_1_16_, output shape: {[16,30,30]}

            self.relu46_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_16_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_16_, output shape: {[1024,30,30]}

            self.batchnorm47_1_16_ = gluon.nn.BatchNorm()
            # batchnorm47_1_16_, output shape: {[1024,30,30]}

            self.conv45_1_17_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_17_, output shape: {[16,60,60]}

            self.batchnorm45_1_17_ = gluon.nn.BatchNorm()
            # batchnorm45_1_17_, output shape: {[16,60,60]}

            self.relu45_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_17_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_17_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_17_, output shape: {[16,30,30]}

            self.batchnorm46_1_17_ = gluon.nn.BatchNorm()
            # batchnorm46_1_17_, output shape: {[16,30,30]}

            self.relu46_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_17_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_17_, output shape: {[1024,30,30]}

            self.batchnorm47_1_17_ = gluon.nn.BatchNorm()
            # batchnorm47_1_17_, output shape: {[1024,30,30]}

            self.conv45_1_18_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_18_, output shape: {[16,60,60]}

            self.batchnorm45_1_18_ = gluon.nn.BatchNorm()
            # batchnorm45_1_18_, output shape: {[16,60,60]}

            self.relu45_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_18_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_18_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_18_, output shape: {[16,30,30]}

            self.batchnorm46_1_18_ = gluon.nn.BatchNorm()
            # batchnorm46_1_18_, output shape: {[16,30,30]}

            self.relu46_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_18_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_18_, output shape: {[1024,30,30]}

            self.batchnorm47_1_18_ = gluon.nn.BatchNorm()
            # batchnorm47_1_18_, output shape: {[1024,30,30]}

            self.conv45_1_19_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_19_, output shape: {[16,60,60]}

            self.batchnorm45_1_19_ = gluon.nn.BatchNorm()
            # batchnorm45_1_19_, output shape: {[16,60,60]}

            self.relu45_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_19_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_19_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_19_, output shape: {[16,30,30]}

            self.batchnorm46_1_19_ = gluon.nn.BatchNorm()
            # batchnorm46_1_19_, output shape: {[16,30,30]}

            self.relu46_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_19_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_19_, output shape: {[1024,30,30]}

            self.batchnorm47_1_19_ = gluon.nn.BatchNorm()
            # batchnorm47_1_19_, output shape: {[1024,30,30]}

            self.conv45_1_20_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_20_, output shape: {[16,60,60]}

            self.batchnorm45_1_20_ = gluon.nn.BatchNorm()
            # batchnorm45_1_20_, output shape: {[16,60,60]}

            self.relu45_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_20_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_20_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_20_, output shape: {[16,30,30]}

            self.batchnorm46_1_20_ = gluon.nn.BatchNorm()
            # batchnorm46_1_20_, output shape: {[16,30,30]}

            self.relu46_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_20_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_20_, output shape: {[1024,30,30]}

            self.batchnorm47_1_20_ = gluon.nn.BatchNorm()
            # batchnorm47_1_20_, output shape: {[1024,30,30]}

            self.conv45_1_21_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_21_, output shape: {[16,60,60]}

            self.batchnorm45_1_21_ = gluon.nn.BatchNorm()
            # batchnorm45_1_21_, output shape: {[16,60,60]}

            self.relu45_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_21_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_21_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_21_, output shape: {[16,30,30]}

            self.batchnorm46_1_21_ = gluon.nn.BatchNorm()
            # batchnorm46_1_21_, output shape: {[16,30,30]}

            self.relu46_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_21_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_21_, output shape: {[1024,30,30]}

            self.batchnorm47_1_21_ = gluon.nn.BatchNorm()
            # batchnorm47_1_21_, output shape: {[1024,30,30]}

            self.conv45_1_22_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_22_, output shape: {[16,60,60]}

            self.batchnorm45_1_22_ = gluon.nn.BatchNorm()
            # batchnorm45_1_22_, output shape: {[16,60,60]}

            self.relu45_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_22_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_22_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_22_, output shape: {[16,30,30]}

            self.batchnorm46_1_22_ = gluon.nn.BatchNorm()
            # batchnorm46_1_22_, output shape: {[16,30,30]}

            self.relu46_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_22_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_22_, output shape: {[1024,30,30]}

            self.batchnorm47_1_22_ = gluon.nn.BatchNorm()
            # batchnorm47_1_22_, output shape: {[1024,30,30]}

            self.conv45_1_23_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_23_, output shape: {[16,60,60]}

            self.batchnorm45_1_23_ = gluon.nn.BatchNorm()
            # batchnorm45_1_23_, output shape: {[16,60,60]}

            self.relu45_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_23_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_23_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_23_, output shape: {[16,30,30]}

            self.batchnorm46_1_23_ = gluon.nn.BatchNorm()
            # batchnorm46_1_23_, output shape: {[16,30,30]}

            self.relu46_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_23_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_23_, output shape: {[1024,30,30]}

            self.batchnorm47_1_23_ = gluon.nn.BatchNorm()
            # batchnorm47_1_23_, output shape: {[1024,30,30]}

            self.conv45_1_24_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_24_, output shape: {[16,60,60]}

            self.batchnorm45_1_24_ = gluon.nn.BatchNorm()
            # batchnorm45_1_24_, output shape: {[16,60,60]}

            self.relu45_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_24_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_24_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_24_, output shape: {[16,30,30]}

            self.batchnorm46_1_24_ = gluon.nn.BatchNorm()
            # batchnorm46_1_24_, output shape: {[16,30,30]}

            self.relu46_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_24_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_24_, output shape: {[1024,30,30]}

            self.batchnorm47_1_24_ = gluon.nn.BatchNorm()
            # batchnorm47_1_24_, output shape: {[1024,30,30]}

            self.conv45_1_25_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_25_, output shape: {[16,60,60]}

            self.batchnorm45_1_25_ = gluon.nn.BatchNorm()
            # batchnorm45_1_25_, output shape: {[16,60,60]}

            self.relu45_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_25_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_25_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_25_, output shape: {[16,30,30]}

            self.batchnorm46_1_25_ = gluon.nn.BatchNorm()
            # batchnorm46_1_25_, output shape: {[16,30,30]}

            self.relu46_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_25_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_25_, output shape: {[1024,30,30]}

            self.batchnorm47_1_25_ = gluon.nn.BatchNorm()
            # batchnorm47_1_25_, output shape: {[1024,30,30]}

            self.conv45_1_26_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_26_, output shape: {[16,60,60]}

            self.batchnorm45_1_26_ = gluon.nn.BatchNorm()
            # batchnorm45_1_26_, output shape: {[16,60,60]}

            self.relu45_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_26_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_26_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_26_, output shape: {[16,30,30]}

            self.batchnorm46_1_26_ = gluon.nn.BatchNorm()
            # batchnorm46_1_26_, output shape: {[16,30,30]}

            self.relu46_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_26_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_26_, output shape: {[1024,30,30]}

            self.batchnorm47_1_26_ = gluon.nn.BatchNorm()
            # batchnorm47_1_26_, output shape: {[1024,30,30]}

            self.conv45_1_27_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_27_, output shape: {[16,60,60]}

            self.batchnorm45_1_27_ = gluon.nn.BatchNorm()
            # batchnorm45_1_27_, output shape: {[16,60,60]}

            self.relu45_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_27_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_27_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_27_, output shape: {[16,30,30]}

            self.batchnorm46_1_27_ = gluon.nn.BatchNorm()
            # batchnorm46_1_27_, output shape: {[16,30,30]}

            self.relu46_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_27_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_27_, output shape: {[1024,30,30]}

            self.batchnorm47_1_27_ = gluon.nn.BatchNorm()
            # batchnorm47_1_27_, output shape: {[1024,30,30]}

            self.conv45_1_28_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_28_, output shape: {[16,60,60]}

            self.batchnorm45_1_28_ = gluon.nn.BatchNorm()
            # batchnorm45_1_28_, output shape: {[16,60,60]}

            self.relu45_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_28_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_28_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_28_, output shape: {[16,30,30]}

            self.batchnorm46_1_28_ = gluon.nn.BatchNorm()
            # batchnorm46_1_28_, output shape: {[16,30,30]}

            self.relu46_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_28_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_28_, output shape: {[1024,30,30]}

            self.batchnorm47_1_28_ = gluon.nn.BatchNorm()
            # batchnorm47_1_28_, output shape: {[1024,30,30]}

            self.conv45_1_29_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_29_, output shape: {[16,60,60]}

            self.batchnorm45_1_29_ = gluon.nn.BatchNorm()
            # batchnorm45_1_29_, output shape: {[16,60,60]}

            self.relu45_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_29_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_29_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_29_, output shape: {[16,30,30]}

            self.batchnorm46_1_29_ = gluon.nn.BatchNorm()
            # batchnorm46_1_29_, output shape: {[16,30,30]}

            self.relu46_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_29_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_29_, output shape: {[1024,30,30]}

            self.batchnorm47_1_29_ = gluon.nn.BatchNorm()
            # batchnorm47_1_29_, output shape: {[1024,30,30]}

            self.conv45_1_30_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_30_, output shape: {[16,60,60]}

            self.batchnorm45_1_30_ = gluon.nn.BatchNorm()
            # batchnorm45_1_30_, output shape: {[16,60,60]}

            self.relu45_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_30_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_30_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_30_, output shape: {[16,30,30]}

            self.batchnorm46_1_30_ = gluon.nn.BatchNorm()
            # batchnorm46_1_30_, output shape: {[16,30,30]}

            self.relu46_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_30_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_30_, output shape: {[1024,30,30]}

            self.batchnorm47_1_30_ = gluon.nn.BatchNorm()
            # batchnorm47_1_30_, output shape: {[1024,30,30]}

            self.conv45_1_31_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_31_, output shape: {[16,60,60]}

            self.batchnorm45_1_31_ = gluon.nn.BatchNorm()
            # batchnorm45_1_31_, output shape: {[16,60,60]}

            self.relu45_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_31_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_31_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_31_, output shape: {[16,30,30]}

            self.batchnorm46_1_31_ = gluon.nn.BatchNorm()
            # batchnorm46_1_31_, output shape: {[16,30,30]}

            self.relu46_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_31_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_31_, output shape: {[1024,30,30]}

            self.batchnorm47_1_31_ = gluon.nn.BatchNorm()
            # batchnorm47_1_31_, output shape: {[1024,30,30]}

            self.conv45_1_32_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv45_1_32_, output shape: {[16,60,60]}

            self.batchnorm45_1_32_ = gluon.nn.BatchNorm()
            # batchnorm45_1_32_, output shape: {[16,60,60]}

            self.relu45_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv46_1_32_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv46_1_32_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv46_1_32_, output shape: {[16,30,30]}

            self.batchnorm46_1_32_ = gluon.nn.BatchNorm()
            # batchnorm46_1_32_, output shape: {[16,30,30]}

            self.relu46_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv47_1_32_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv47_1_32_, output shape: {[1024,30,30]}

            self.batchnorm47_1_32_ = gluon.nn.BatchNorm()
            # batchnorm47_1_32_, output shape: {[1024,30,30]}

            self.conv44_2_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv44_2_, output shape: {[1024,30,30]}

            self.batchnorm44_2_ = gluon.nn.BatchNorm()
            # batchnorm44_2_, output shape: {[1024,30,30]}

            self.relu49_ = gluon.nn.Activation(activation='relu')
            self.conv52_1_1_1_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_1_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_1_, output shape: {[16,30,30]}

            self.relu52_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_1_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_1_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_1_, output shape: {[16,30,30]}

            self.relu53_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_1_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_1_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_1_, output shape: {[1024,30,30]}

            self.conv52_1_1_2_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_2_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_2_, output shape: {[16,30,30]}

            self.relu52_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_2_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_2_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_2_, output shape: {[16,30,30]}

            self.relu53_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_2_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_2_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_2_, output shape: {[1024,30,30]}

            self.conv52_1_1_3_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_3_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_3_, output shape: {[16,30,30]}

            self.relu52_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_3_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_3_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_3_, output shape: {[16,30,30]}

            self.relu53_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_3_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_3_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_3_, output shape: {[1024,30,30]}

            self.conv52_1_1_4_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_4_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_4_, output shape: {[16,30,30]}

            self.relu52_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_4_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_4_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_4_, output shape: {[16,30,30]}

            self.relu53_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_4_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_4_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_4_, output shape: {[1024,30,30]}

            self.conv52_1_1_5_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_5_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_5_, output shape: {[16,30,30]}

            self.relu52_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_5_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_5_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_5_, output shape: {[16,30,30]}

            self.relu53_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_5_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_5_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_5_, output shape: {[1024,30,30]}

            self.conv52_1_1_6_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_6_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_6_, output shape: {[16,30,30]}

            self.relu52_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_6_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_6_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_6_, output shape: {[16,30,30]}

            self.relu53_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_6_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_6_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_6_, output shape: {[1024,30,30]}

            self.conv52_1_1_7_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_7_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_7_, output shape: {[16,30,30]}

            self.relu52_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_7_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_7_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_7_, output shape: {[16,30,30]}

            self.relu53_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_7_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_7_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_7_, output shape: {[1024,30,30]}

            self.conv52_1_1_8_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_8_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_8_, output shape: {[16,30,30]}

            self.relu52_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_8_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_8_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_8_, output shape: {[16,30,30]}

            self.relu53_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_8_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_8_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_8_, output shape: {[1024,30,30]}

            self.conv52_1_1_9_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_9_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_9_, output shape: {[16,30,30]}

            self.relu52_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_9_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_9_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_9_, output shape: {[16,30,30]}

            self.relu53_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_9_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_9_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_9_, output shape: {[1024,30,30]}

            self.conv52_1_1_10_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_10_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_10_, output shape: {[16,30,30]}

            self.relu52_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_10_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_10_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_10_, output shape: {[16,30,30]}

            self.relu53_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_10_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_10_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_10_, output shape: {[1024,30,30]}

            self.conv52_1_1_11_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_11_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_11_, output shape: {[16,30,30]}

            self.relu52_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_11_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_11_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_11_, output shape: {[16,30,30]}

            self.relu53_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_11_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_11_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_11_, output shape: {[1024,30,30]}

            self.conv52_1_1_12_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_12_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_12_, output shape: {[16,30,30]}

            self.relu52_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_12_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_12_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_12_, output shape: {[16,30,30]}

            self.relu53_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_12_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_12_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_12_, output shape: {[1024,30,30]}

            self.conv52_1_1_13_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_13_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_13_, output shape: {[16,30,30]}

            self.relu52_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_13_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_13_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_13_, output shape: {[16,30,30]}

            self.relu53_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_13_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_13_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_13_, output shape: {[1024,30,30]}

            self.conv52_1_1_14_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_14_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_14_, output shape: {[16,30,30]}

            self.relu52_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_14_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_14_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_14_, output shape: {[16,30,30]}

            self.relu53_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_14_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_14_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_14_, output shape: {[1024,30,30]}

            self.conv52_1_1_15_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_15_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_15_, output shape: {[16,30,30]}

            self.relu52_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_15_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_15_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_15_, output shape: {[16,30,30]}

            self.relu53_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_15_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_15_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_15_, output shape: {[1024,30,30]}

            self.conv52_1_1_16_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_16_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_16_, output shape: {[16,30,30]}

            self.relu52_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_16_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_16_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_16_, output shape: {[16,30,30]}

            self.relu53_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_16_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_16_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_16_, output shape: {[1024,30,30]}

            self.conv52_1_1_17_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_17_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_17_, output shape: {[16,30,30]}

            self.relu52_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_17_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_17_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_17_, output shape: {[16,30,30]}

            self.relu53_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_17_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_17_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_17_, output shape: {[1024,30,30]}

            self.conv52_1_1_18_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_18_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_18_, output shape: {[16,30,30]}

            self.relu52_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_18_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_18_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_18_, output shape: {[16,30,30]}

            self.relu53_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_18_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_18_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_18_, output shape: {[1024,30,30]}

            self.conv52_1_1_19_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_19_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_19_, output shape: {[16,30,30]}

            self.relu52_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_19_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_19_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_19_, output shape: {[16,30,30]}

            self.relu53_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_19_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_19_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_19_, output shape: {[1024,30,30]}

            self.conv52_1_1_20_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_20_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_20_, output shape: {[16,30,30]}

            self.relu52_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_20_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_20_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_20_, output shape: {[16,30,30]}

            self.relu53_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_20_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_20_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_20_, output shape: {[1024,30,30]}

            self.conv52_1_1_21_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_21_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_21_, output shape: {[16,30,30]}

            self.relu52_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_21_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_21_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_21_, output shape: {[16,30,30]}

            self.relu53_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_21_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_21_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_21_, output shape: {[1024,30,30]}

            self.conv52_1_1_22_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_22_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_22_, output shape: {[16,30,30]}

            self.relu52_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_22_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_22_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_22_, output shape: {[16,30,30]}

            self.relu53_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_22_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_22_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_22_, output shape: {[1024,30,30]}

            self.conv52_1_1_23_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_23_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_23_, output shape: {[16,30,30]}

            self.relu52_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_23_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_23_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_23_, output shape: {[16,30,30]}

            self.relu53_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_23_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_23_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_23_, output shape: {[1024,30,30]}

            self.conv52_1_1_24_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_24_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_24_, output shape: {[16,30,30]}

            self.relu52_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_24_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_24_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_24_, output shape: {[16,30,30]}

            self.relu53_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_24_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_24_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_24_, output shape: {[1024,30,30]}

            self.conv52_1_1_25_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_25_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_25_, output shape: {[16,30,30]}

            self.relu52_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_25_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_25_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_25_, output shape: {[16,30,30]}

            self.relu53_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_25_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_25_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_25_, output shape: {[1024,30,30]}

            self.conv52_1_1_26_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_26_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_26_, output shape: {[16,30,30]}

            self.relu52_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_26_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_26_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_26_, output shape: {[16,30,30]}

            self.relu53_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_26_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_26_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_26_, output shape: {[1024,30,30]}

            self.conv52_1_1_27_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_27_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_27_, output shape: {[16,30,30]}

            self.relu52_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_27_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_27_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_27_, output shape: {[16,30,30]}

            self.relu53_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_27_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_27_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_27_, output shape: {[1024,30,30]}

            self.conv52_1_1_28_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_28_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_28_, output shape: {[16,30,30]}

            self.relu52_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_28_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_28_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_28_, output shape: {[16,30,30]}

            self.relu53_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_28_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_28_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_28_, output shape: {[1024,30,30]}

            self.conv52_1_1_29_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_29_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_29_, output shape: {[16,30,30]}

            self.relu52_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_29_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_29_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_29_, output shape: {[16,30,30]}

            self.relu53_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_29_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_29_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_29_, output shape: {[1024,30,30]}

            self.conv52_1_1_30_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_30_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_30_, output shape: {[16,30,30]}

            self.relu52_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_30_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_30_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_30_, output shape: {[16,30,30]}

            self.relu53_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_30_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_30_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_30_, output shape: {[1024,30,30]}

            self.conv52_1_1_31_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_31_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_31_, output shape: {[16,30,30]}

            self.relu52_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_31_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_31_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_31_, output shape: {[16,30,30]}

            self.relu53_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_31_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_31_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_31_, output shape: {[1024,30,30]}

            self.conv52_1_1_32_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv52_1_1_32_, output shape: {[16,30,30]}

            self.batchnorm52_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm52_1_1_32_, output shape: {[16,30,30]}

            self.relu52_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv53_1_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv53_1_1_32_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv53_1_1_32_, output shape: {[16,30,30]}

            self.batchnorm53_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm53_1_1_32_, output shape: {[16,30,30]}

            self.relu53_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv54_1_1_32_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv54_1_1_32_, output shape: {[1024,30,30]}

            self.batchnorm54_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm54_1_1_32_, output shape: {[1024,30,30]}

            self.relu56_1_ = gluon.nn.Activation(activation='relu')
            self.conv58_1_1_1_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_1_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_1_, output shape: {[16,30,30]}

            self.relu58_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_1_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_1_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_1_, output shape: {[16,30,30]}

            self.relu59_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_1_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_1_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_1_, output shape: {[1024,30,30]}

            self.conv58_1_1_2_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_2_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_2_, output shape: {[16,30,30]}

            self.relu58_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_2_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_2_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_2_, output shape: {[16,30,30]}

            self.relu59_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_2_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_2_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_2_, output shape: {[1024,30,30]}

            self.conv58_1_1_3_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_3_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_3_, output shape: {[16,30,30]}

            self.relu58_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_3_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_3_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_3_, output shape: {[16,30,30]}

            self.relu59_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_3_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_3_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_3_, output shape: {[1024,30,30]}

            self.conv58_1_1_4_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_4_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_4_, output shape: {[16,30,30]}

            self.relu58_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_4_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_4_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_4_, output shape: {[16,30,30]}

            self.relu59_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_4_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_4_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_4_, output shape: {[1024,30,30]}

            self.conv58_1_1_5_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_5_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_5_, output shape: {[16,30,30]}

            self.relu58_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_5_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_5_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_5_, output shape: {[16,30,30]}

            self.relu59_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_5_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_5_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_5_, output shape: {[1024,30,30]}

            self.conv58_1_1_6_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_6_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_6_, output shape: {[16,30,30]}

            self.relu58_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_6_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_6_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_6_, output shape: {[16,30,30]}

            self.relu59_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_6_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_6_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_6_, output shape: {[1024,30,30]}

            self.conv58_1_1_7_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_7_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_7_, output shape: {[16,30,30]}

            self.relu58_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_7_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_7_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_7_, output shape: {[16,30,30]}

            self.relu59_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_7_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_7_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_7_, output shape: {[1024,30,30]}

            self.conv58_1_1_8_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_8_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_8_, output shape: {[16,30,30]}

            self.relu58_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_8_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_8_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_8_, output shape: {[16,30,30]}

            self.relu59_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_8_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_8_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_8_, output shape: {[1024,30,30]}

            self.conv58_1_1_9_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_9_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_9_, output shape: {[16,30,30]}

            self.relu58_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_9_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_9_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_9_, output shape: {[16,30,30]}

            self.relu59_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_9_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_9_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_9_, output shape: {[1024,30,30]}

            self.conv58_1_1_10_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_10_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_10_, output shape: {[16,30,30]}

            self.relu58_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_10_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_10_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_10_, output shape: {[16,30,30]}

            self.relu59_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_10_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_10_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_10_, output shape: {[1024,30,30]}

            self.conv58_1_1_11_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_11_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_11_, output shape: {[16,30,30]}

            self.relu58_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_11_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_11_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_11_, output shape: {[16,30,30]}

            self.relu59_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_11_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_11_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_11_, output shape: {[1024,30,30]}

            self.conv58_1_1_12_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_12_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_12_, output shape: {[16,30,30]}

            self.relu58_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_12_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_12_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_12_, output shape: {[16,30,30]}

            self.relu59_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_12_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_12_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_12_, output shape: {[1024,30,30]}

            self.conv58_1_1_13_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_13_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_13_, output shape: {[16,30,30]}

            self.relu58_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_13_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_13_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_13_, output shape: {[16,30,30]}

            self.relu59_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_13_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_13_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_13_, output shape: {[1024,30,30]}

            self.conv58_1_1_14_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_14_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_14_, output shape: {[16,30,30]}

            self.relu58_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_14_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_14_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_14_, output shape: {[16,30,30]}

            self.relu59_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_14_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_14_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_14_, output shape: {[1024,30,30]}

            self.conv58_1_1_15_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_15_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_15_, output shape: {[16,30,30]}

            self.relu58_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_15_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_15_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_15_, output shape: {[16,30,30]}

            self.relu59_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_15_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_15_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_15_, output shape: {[1024,30,30]}

            self.conv58_1_1_16_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_16_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_16_, output shape: {[16,30,30]}

            self.relu58_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_16_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_16_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_16_, output shape: {[16,30,30]}

            self.relu59_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_16_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_16_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_16_, output shape: {[1024,30,30]}

            self.conv58_1_1_17_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_17_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_17_, output shape: {[16,30,30]}

            self.relu58_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_17_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_17_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_17_, output shape: {[16,30,30]}

            self.relu59_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_17_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_17_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_17_, output shape: {[1024,30,30]}

            self.conv58_1_1_18_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_18_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_18_, output shape: {[16,30,30]}

            self.relu58_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_18_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_18_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_18_, output shape: {[16,30,30]}

            self.relu59_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_18_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_18_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_18_, output shape: {[1024,30,30]}

            self.conv58_1_1_19_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_19_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_19_, output shape: {[16,30,30]}

            self.relu58_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_19_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_19_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_19_, output shape: {[16,30,30]}

            self.relu59_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_19_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_19_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_19_, output shape: {[1024,30,30]}

            self.conv58_1_1_20_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_20_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_20_, output shape: {[16,30,30]}

            self.relu58_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_20_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_20_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_20_, output shape: {[16,30,30]}

            self.relu59_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_20_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_20_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_20_, output shape: {[1024,30,30]}

            self.conv58_1_1_21_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_21_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_21_, output shape: {[16,30,30]}

            self.relu58_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_21_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_21_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_21_, output shape: {[16,30,30]}

            self.relu59_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_21_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_21_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_21_, output shape: {[1024,30,30]}

            self.conv58_1_1_22_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_22_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_22_, output shape: {[16,30,30]}

            self.relu58_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_22_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_22_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_22_, output shape: {[16,30,30]}

            self.relu59_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_22_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_22_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_22_, output shape: {[1024,30,30]}

            self.conv58_1_1_23_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_23_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_23_, output shape: {[16,30,30]}

            self.relu58_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_23_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_23_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_23_, output shape: {[16,30,30]}

            self.relu59_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_23_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_23_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_23_, output shape: {[1024,30,30]}

            self.conv58_1_1_24_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_24_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_24_, output shape: {[16,30,30]}

            self.relu58_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_24_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_24_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_24_, output shape: {[16,30,30]}

            self.relu59_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_24_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_24_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_24_, output shape: {[1024,30,30]}

            self.conv58_1_1_25_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_25_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_25_, output shape: {[16,30,30]}

            self.relu58_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_25_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_25_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_25_, output shape: {[16,30,30]}

            self.relu59_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_25_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_25_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_25_, output shape: {[1024,30,30]}

            self.conv58_1_1_26_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_26_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_26_, output shape: {[16,30,30]}

            self.relu58_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_26_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_26_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_26_, output shape: {[16,30,30]}

            self.relu59_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_26_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_26_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_26_, output shape: {[1024,30,30]}

            self.conv58_1_1_27_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_27_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_27_, output shape: {[16,30,30]}

            self.relu58_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_27_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_27_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_27_, output shape: {[16,30,30]}

            self.relu59_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_27_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_27_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_27_, output shape: {[1024,30,30]}

            self.conv58_1_1_28_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_28_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_28_, output shape: {[16,30,30]}

            self.relu58_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_28_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_28_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_28_, output shape: {[16,30,30]}

            self.relu59_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_28_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_28_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_28_, output shape: {[1024,30,30]}

            self.conv58_1_1_29_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_29_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_29_, output shape: {[16,30,30]}

            self.relu58_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_29_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_29_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_29_, output shape: {[16,30,30]}

            self.relu59_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_29_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_29_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_29_, output shape: {[1024,30,30]}

            self.conv58_1_1_30_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_30_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_30_, output shape: {[16,30,30]}

            self.relu58_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_30_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_30_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_30_, output shape: {[16,30,30]}

            self.relu59_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_30_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_30_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_30_, output shape: {[1024,30,30]}

            self.conv58_1_1_31_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_31_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_31_, output shape: {[16,30,30]}

            self.relu58_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_31_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_31_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_31_, output shape: {[16,30,30]}

            self.relu59_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_31_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_31_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_31_, output shape: {[1024,30,30]}

            self.conv58_1_1_32_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv58_1_1_32_, output shape: {[16,30,30]}

            self.batchnorm58_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm58_1_1_32_, output shape: {[16,30,30]}

            self.relu58_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv59_1_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv59_1_1_32_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv59_1_1_32_, output shape: {[16,30,30]}

            self.batchnorm59_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm59_1_1_32_, output shape: {[16,30,30]}

            self.relu59_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv60_1_1_32_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv60_1_1_32_, output shape: {[1024,30,30]}

            self.batchnorm60_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm60_1_1_32_, output shape: {[1024,30,30]}

            self.relu62_1_ = gluon.nn.Activation(activation='relu')
            self.conv64_1_1_1_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_1_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_1_, output shape: {[16,30,30]}

            self.relu64_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_1_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_1_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_1_, output shape: {[16,30,30]}

            self.relu65_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_1_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_1_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_1_, output shape: {[1024,30,30]}

            self.conv64_1_1_2_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_2_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_2_, output shape: {[16,30,30]}

            self.relu64_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_2_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_2_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_2_, output shape: {[16,30,30]}

            self.relu65_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_2_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_2_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_2_, output shape: {[1024,30,30]}

            self.conv64_1_1_3_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_3_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_3_, output shape: {[16,30,30]}

            self.relu64_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_3_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_3_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_3_, output shape: {[16,30,30]}

            self.relu65_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_3_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_3_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_3_, output shape: {[1024,30,30]}

            self.conv64_1_1_4_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_4_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_4_, output shape: {[16,30,30]}

            self.relu64_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_4_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_4_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_4_, output shape: {[16,30,30]}

            self.relu65_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_4_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_4_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_4_, output shape: {[1024,30,30]}

            self.conv64_1_1_5_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_5_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_5_, output shape: {[16,30,30]}

            self.relu64_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_5_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_5_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_5_, output shape: {[16,30,30]}

            self.relu65_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_5_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_5_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_5_, output shape: {[1024,30,30]}

            self.conv64_1_1_6_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_6_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_6_, output shape: {[16,30,30]}

            self.relu64_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_6_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_6_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_6_, output shape: {[16,30,30]}

            self.relu65_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_6_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_6_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_6_, output shape: {[1024,30,30]}

            self.conv64_1_1_7_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_7_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_7_, output shape: {[16,30,30]}

            self.relu64_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_7_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_7_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_7_, output shape: {[16,30,30]}

            self.relu65_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_7_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_7_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_7_, output shape: {[1024,30,30]}

            self.conv64_1_1_8_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_8_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_8_, output shape: {[16,30,30]}

            self.relu64_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_8_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_8_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_8_, output shape: {[16,30,30]}

            self.relu65_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_8_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_8_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_8_, output shape: {[1024,30,30]}

            self.conv64_1_1_9_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_9_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_9_, output shape: {[16,30,30]}

            self.relu64_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_9_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_9_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_9_, output shape: {[16,30,30]}

            self.relu65_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_9_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_9_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_9_, output shape: {[1024,30,30]}

            self.conv64_1_1_10_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_10_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_10_, output shape: {[16,30,30]}

            self.relu64_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_10_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_10_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_10_, output shape: {[16,30,30]}

            self.relu65_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_10_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_10_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_10_, output shape: {[1024,30,30]}

            self.conv64_1_1_11_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_11_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_11_, output shape: {[16,30,30]}

            self.relu64_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_11_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_11_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_11_, output shape: {[16,30,30]}

            self.relu65_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_11_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_11_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_11_, output shape: {[1024,30,30]}

            self.conv64_1_1_12_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_12_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_12_, output shape: {[16,30,30]}

            self.relu64_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_12_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_12_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_12_, output shape: {[16,30,30]}

            self.relu65_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_12_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_12_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_12_, output shape: {[1024,30,30]}

            self.conv64_1_1_13_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_13_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_13_, output shape: {[16,30,30]}

            self.relu64_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_13_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_13_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_13_, output shape: {[16,30,30]}

            self.relu65_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_13_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_13_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_13_, output shape: {[1024,30,30]}

            self.conv64_1_1_14_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_14_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_14_, output shape: {[16,30,30]}

            self.relu64_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_14_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_14_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_14_, output shape: {[16,30,30]}

            self.relu65_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_14_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_14_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_14_, output shape: {[1024,30,30]}

            self.conv64_1_1_15_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_15_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_15_, output shape: {[16,30,30]}

            self.relu64_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_15_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_15_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_15_, output shape: {[16,30,30]}

            self.relu65_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_15_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_15_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_15_, output shape: {[1024,30,30]}

            self.conv64_1_1_16_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_16_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_16_, output shape: {[16,30,30]}

            self.relu64_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_16_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_16_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_16_, output shape: {[16,30,30]}

            self.relu65_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_16_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_16_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_16_, output shape: {[1024,30,30]}

            self.conv64_1_1_17_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_17_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_17_, output shape: {[16,30,30]}

            self.relu64_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_17_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_17_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_17_, output shape: {[16,30,30]}

            self.relu65_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_17_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_17_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_17_, output shape: {[1024,30,30]}

            self.conv64_1_1_18_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_18_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_18_, output shape: {[16,30,30]}

            self.relu64_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_18_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_18_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_18_, output shape: {[16,30,30]}

            self.relu65_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_18_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_18_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_18_, output shape: {[1024,30,30]}

            self.conv64_1_1_19_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_19_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_19_, output shape: {[16,30,30]}

            self.relu64_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_19_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_19_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_19_, output shape: {[16,30,30]}

            self.relu65_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_19_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_19_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_19_, output shape: {[1024,30,30]}

            self.conv64_1_1_20_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_20_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_20_, output shape: {[16,30,30]}

            self.relu64_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_20_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_20_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_20_, output shape: {[16,30,30]}

            self.relu65_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_20_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_20_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_20_, output shape: {[1024,30,30]}

            self.conv64_1_1_21_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_21_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_21_, output shape: {[16,30,30]}

            self.relu64_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_21_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_21_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_21_, output shape: {[16,30,30]}

            self.relu65_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_21_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_21_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_21_, output shape: {[1024,30,30]}

            self.conv64_1_1_22_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_22_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_22_, output shape: {[16,30,30]}

            self.relu64_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_22_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_22_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_22_, output shape: {[16,30,30]}

            self.relu65_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_22_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_22_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_22_, output shape: {[1024,30,30]}

            self.conv64_1_1_23_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_23_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_23_, output shape: {[16,30,30]}

            self.relu64_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_23_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_23_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_23_, output shape: {[16,30,30]}

            self.relu65_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_23_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_23_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_23_, output shape: {[1024,30,30]}

            self.conv64_1_1_24_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_24_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_24_, output shape: {[16,30,30]}

            self.relu64_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_24_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_24_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_24_, output shape: {[16,30,30]}

            self.relu65_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_24_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_24_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_24_, output shape: {[1024,30,30]}

            self.conv64_1_1_25_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_25_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_25_, output shape: {[16,30,30]}

            self.relu64_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_25_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_25_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_25_, output shape: {[16,30,30]}

            self.relu65_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_25_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_25_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_25_, output shape: {[1024,30,30]}

            self.conv64_1_1_26_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_26_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_26_, output shape: {[16,30,30]}

            self.relu64_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_26_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_26_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_26_, output shape: {[16,30,30]}

            self.relu65_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_26_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_26_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_26_, output shape: {[1024,30,30]}

            self.conv64_1_1_27_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_27_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_27_, output shape: {[16,30,30]}

            self.relu64_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_27_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_27_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_27_, output shape: {[16,30,30]}

            self.relu65_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_27_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_27_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_27_, output shape: {[1024,30,30]}

            self.conv64_1_1_28_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_28_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_28_, output shape: {[16,30,30]}

            self.relu64_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_28_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_28_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_28_, output shape: {[16,30,30]}

            self.relu65_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_28_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_28_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_28_, output shape: {[1024,30,30]}

            self.conv64_1_1_29_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_29_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_29_, output shape: {[16,30,30]}

            self.relu64_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_29_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_29_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_29_, output shape: {[16,30,30]}

            self.relu65_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_29_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_29_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_29_, output shape: {[1024,30,30]}

            self.conv64_1_1_30_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_30_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_30_, output shape: {[16,30,30]}

            self.relu64_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_30_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_30_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_30_, output shape: {[16,30,30]}

            self.relu65_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_30_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_30_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_30_, output shape: {[1024,30,30]}

            self.conv64_1_1_31_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_31_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_31_, output shape: {[16,30,30]}

            self.relu64_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_31_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_31_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_31_, output shape: {[16,30,30]}

            self.relu65_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_31_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_31_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_31_, output shape: {[1024,30,30]}

            self.conv64_1_1_32_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv64_1_1_32_, output shape: {[16,30,30]}

            self.batchnorm64_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm64_1_1_32_, output shape: {[16,30,30]}

            self.relu64_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv65_1_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv65_1_1_32_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv65_1_1_32_, output shape: {[16,30,30]}

            self.batchnorm65_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm65_1_1_32_, output shape: {[16,30,30]}

            self.relu65_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv66_1_1_32_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv66_1_1_32_, output shape: {[1024,30,30]}

            self.batchnorm66_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm66_1_1_32_, output shape: {[1024,30,30]}

            self.relu68_1_ = gluon.nn.Activation(activation='relu')
            self.conv70_1_1_1_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_1_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_1_, output shape: {[16,30,30]}

            self.relu70_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_1_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_1_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_1_, output shape: {[16,30,30]}

            self.relu71_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_1_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_1_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_1_, output shape: {[1024,30,30]}

            self.conv70_1_1_2_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_2_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_2_, output shape: {[16,30,30]}

            self.relu70_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_2_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_2_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_2_, output shape: {[16,30,30]}

            self.relu71_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_2_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_2_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_2_, output shape: {[1024,30,30]}

            self.conv70_1_1_3_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_3_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_3_, output shape: {[16,30,30]}

            self.relu70_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_3_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_3_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_3_, output shape: {[16,30,30]}

            self.relu71_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_3_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_3_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_3_, output shape: {[1024,30,30]}

            self.conv70_1_1_4_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_4_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_4_, output shape: {[16,30,30]}

            self.relu70_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_4_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_4_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_4_, output shape: {[16,30,30]}

            self.relu71_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_4_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_4_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_4_, output shape: {[1024,30,30]}

            self.conv70_1_1_5_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_5_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_5_, output shape: {[16,30,30]}

            self.relu70_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_5_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_5_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_5_, output shape: {[16,30,30]}

            self.relu71_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_5_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_5_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_5_, output shape: {[1024,30,30]}

            self.conv70_1_1_6_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_6_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_6_, output shape: {[16,30,30]}

            self.relu70_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_6_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_6_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_6_, output shape: {[16,30,30]}

            self.relu71_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_6_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_6_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_6_, output shape: {[1024,30,30]}

            self.conv70_1_1_7_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_7_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_7_, output shape: {[16,30,30]}

            self.relu70_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_7_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_7_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_7_, output shape: {[16,30,30]}

            self.relu71_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_7_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_7_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_7_, output shape: {[1024,30,30]}

            self.conv70_1_1_8_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_8_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_8_, output shape: {[16,30,30]}

            self.relu70_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_8_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_8_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_8_, output shape: {[16,30,30]}

            self.relu71_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_8_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_8_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_8_, output shape: {[1024,30,30]}

            self.conv70_1_1_9_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_9_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_9_, output shape: {[16,30,30]}

            self.relu70_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_9_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_9_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_9_, output shape: {[16,30,30]}

            self.relu71_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_9_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_9_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_9_, output shape: {[1024,30,30]}

            self.conv70_1_1_10_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_10_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_10_, output shape: {[16,30,30]}

            self.relu70_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_10_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_10_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_10_, output shape: {[16,30,30]}

            self.relu71_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_10_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_10_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_10_, output shape: {[1024,30,30]}

            self.conv70_1_1_11_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_11_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_11_, output shape: {[16,30,30]}

            self.relu70_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_11_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_11_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_11_, output shape: {[16,30,30]}

            self.relu71_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_11_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_11_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_11_, output shape: {[1024,30,30]}

            self.conv70_1_1_12_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_12_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_12_, output shape: {[16,30,30]}

            self.relu70_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_12_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_12_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_12_, output shape: {[16,30,30]}

            self.relu71_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_12_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_12_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_12_, output shape: {[1024,30,30]}

            self.conv70_1_1_13_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_13_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_13_, output shape: {[16,30,30]}

            self.relu70_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_13_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_13_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_13_, output shape: {[16,30,30]}

            self.relu71_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_13_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_13_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_13_, output shape: {[1024,30,30]}

            self.conv70_1_1_14_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_14_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_14_, output shape: {[16,30,30]}

            self.relu70_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_14_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_14_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_14_, output shape: {[16,30,30]}

            self.relu71_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_14_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_14_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_14_, output shape: {[1024,30,30]}

            self.conv70_1_1_15_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_15_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_15_, output shape: {[16,30,30]}

            self.relu70_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_15_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_15_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_15_, output shape: {[16,30,30]}

            self.relu71_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_15_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_15_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_15_, output shape: {[1024,30,30]}

            self.conv70_1_1_16_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_16_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_16_, output shape: {[16,30,30]}

            self.relu70_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_16_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_16_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_16_, output shape: {[16,30,30]}

            self.relu71_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_16_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_16_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_16_, output shape: {[1024,30,30]}

            self.conv70_1_1_17_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_17_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_17_, output shape: {[16,30,30]}

            self.relu70_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_17_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_17_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_17_, output shape: {[16,30,30]}

            self.relu71_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_17_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_17_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_17_, output shape: {[1024,30,30]}

            self.conv70_1_1_18_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_18_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_18_, output shape: {[16,30,30]}

            self.relu70_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_18_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_18_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_18_, output shape: {[16,30,30]}

            self.relu71_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_18_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_18_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_18_, output shape: {[1024,30,30]}

            self.conv70_1_1_19_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_19_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_19_, output shape: {[16,30,30]}

            self.relu70_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_19_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_19_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_19_, output shape: {[16,30,30]}

            self.relu71_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_19_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_19_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_19_, output shape: {[1024,30,30]}

            self.conv70_1_1_20_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_20_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_20_, output shape: {[16,30,30]}

            self.relu70_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_20_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_20_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_20_, output shape: {[16,30,30]}

            self.relu71_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_20_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_20_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_20_, output shape: {[1024,30,30]}

            self.conv70_1_1_21_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_21_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_21_, output shape: {[16,30,30]}

            self.relu70_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_21_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_21_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_21_, output shape: {[16,30,30]}

            self.relu71_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_21_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_21_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_21_, output shape: {[1024,30,30]}

            self.conv70_1_1_22_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_22_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_22_, output shape: {[16,30,30]}

            self.relu70_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_22_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_22_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_22_, output shape: {[16,30,30]}

            self.relu71_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_22_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_22_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_22_, output shape: {[1024,30,30]}

            self.conv70_1_1_23_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_23_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_23_, output shape: {[16,30,30]}

            self.relu70_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_23_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_23_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_23_, output shape: {[16,30,30]}

            self.relu71_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_23_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_23_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_23_, output shape: {[1024,30,30]}

            self.conv70_1_1_24_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_24_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_24_, output shape: {[16,30,30]}

            self.relu70_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_24_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_24_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_24_, output shape: {[16,30,30]}

            self.relu71_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_24_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_24_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_24_, output shape: {[1024,30,30]}

            self.conv70_1_1_25_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_25_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_25_, output shape: {[16,30,30]}

            self.relu70_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_25_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_25_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_25_, output shape: {[16,30,30]}

            self.relu71_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_25_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_25_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_25_, output shape: {[1024,30,30]}

            self.conv70_1_1_26_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_26_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_26_, output shape: {[16,30,30]}

            self.relu70_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_26_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_26_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_26_, output shape: {[16,30,30]}

            self.relu71_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_26_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_26_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_26_, output shape: {[1024,30,30]}

            self.conv70_1_1_27_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_27_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_27_, output shape: {[16,30,30]}

            self.relu70_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_27_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_27_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_27_, output shape: {[16,30,30]}

            self.relu71_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_27_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_27_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_27_, output shape: {[1024,30,30]}

            self.conv70_1_1_28_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_28_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_28_, output shape: {[16,30,30]}

            self.relu70_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_28_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_28_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_28_, output shape: {[16,30,30]}

            self.relu71_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_28_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_28_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_28_, output shape: {[1024,30,30]}

            self.conv70_1_1_29_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_29_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_29_, output shape: {[16,30,30]}

            self.relu70_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_29_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_29_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_29_, output shape: {[16,30,30]}

            self.relu71_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_29_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_29_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_29_, output shape: {[1024,30,30]}

            self.conv70_1_1_30_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_30_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_30_, output shape: {[16,30,30]}

            self.relu70_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_30_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_30_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_30_, output shape: {[16,30,30]}

            self.relu71_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_30_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_30_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_30_, output shape: {[1024,30,30]}

            self.conv70_1_1_31_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_31_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_31_, output shape: {[16,30,30]}

            self.relu70_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_31_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_31_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_31_, output shape: {[16,30,30]}

            self.relu71_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_31_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_31_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_31_, output shape: {[1024,30,30]}

            self.conv70_1_1_32_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv70_1_1_32_, output shape: {[16,30,30]}

            self.batchnorm70_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm70_1_1_32_, output shape: {[16,30,30]}

            self.relu70_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv71_1_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv71_1_1_32_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv71_1_1_32_, output shape: {[16,30,30]}

            self.batchnorm71_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm71_1_1_32_, output shape: {[16,30,30]}

            self.relu71_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv72_1_1_32_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv72_1_1_32_, output shape: {[1024,30,30]}

            self.batchnorm72_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm72_1_1_32_, output shape: {[1024,30,30]}

            self.relu74_1_ = gluon.nn.Activation(activation='relu')
            self.conv76_1_1_1_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_1_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_1_, output shape: {[16,30,30]}

            self.relu76_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_1_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_1_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_1_, output shape: {[16,30,30]}

            self.relu77_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_1_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_1_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_1_, output shape: {[1024,30,30]}

            self.conv76_1_1_2_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_2_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_2_, output shape: {[16,30,30]}

            self.relu76_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_2_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_2_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_2_, output shape: {[16,30,30]}

            self.relu77_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_2_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_2_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_2_, output shape: {[1024,30,30]}

            self.conv76_1_1_3_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_3_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_3_, output shape: {[16,30,30]}

            self.relu76_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_3_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_3_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_3_, output shape: {[16,30,30]}

            self.relu77_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_3_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_3_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_3_, output shape: {[1024,30,30]}

            self.conv76_1_1_4_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_4_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_4_, output shape: {[16,30,30]}

            self.relu76_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_4_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_4_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_4_, output shape: {[16,30,30]}

            self.relu77_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_4_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_4_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_4_, output shape: {[1024,30,30]}

            self.conv76_1_1_5_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_5_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_5_, output shape: {[16,30,30]}

            self.relu76_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_5_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_5_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_5_, output shape: {[16,30,30]}

            self.relu77_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_5_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_5_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_5_, output shape: {[1024,30,30]}

            self.conv76_1_1_6_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_6_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_6_, output shape: {[16,30,30]}

            self.relu76_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_6_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_6_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_6_, output shape: {[16,30,30]}

            self.relu77_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_6_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_6_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_6_, output shape: {[1024,30,30]}

            self.conv76_1_1_7_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_7_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_7_, output shape: {[16,30,30]}

            self.relu76_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_7_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_7_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_7_, output shape: {[16,30,30]}

            self.relu77_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_7_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_7_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_7_, output shape: {[1024,30,30]}

            self.conv76_1_1_8_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_8_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_8_, output shape: {[16,30,30]}

            self.relu76_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_8_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_8_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_8_, output shape: {[16,30,30]}

            self.relu77_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_8_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_8_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_8_, output shape: {[1024,30,30]}

            self.conv76_1_1_9_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_9_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_9_, output shape: {[16,30,30]}

            self.relu76_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_9_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_9_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_9_, output shape: {[16,30,30]}

            self.relu77_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_9_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_9_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_9_, output shape: {[1024,30,30]}

            self.conv76_1_1_10_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_10_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_10_, output shape: {[16,30,30]}

            self.relu76_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_10_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_10_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_10_, output shape: {[16,30,30]}

            self.relu77_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_10_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_10_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_10_, output shape: {[1024,30,30]}

            self.conv76_1_1_11_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_11_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_11_, output shape: {[16,30,30]}

            self.relu76_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_11_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_11_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_11_, output shape: {[16,30,30]}

            self.relu77_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_11_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_11_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_11_, output shape: {[1024,30,30]}

            self.conv76_1_1_12_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_12_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_12_, output shape: {[16,30,30]}

            self.relu76_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_12_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_12_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_12_, output shape: {[16,30,30]}

            self.relu77_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_12_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_12_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_12_, output shape: {[1024,30,30]}

            self.conv76_1_1_13_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_13_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_13_, output shape: {[16,30,30]}

            self.relu76_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_13_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_13_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_13_, output shape: {[16,30,30]}

            self.relu77_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_13_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_13_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_13_, output shape: {[1024,30,30]}

            self.conv76_1_1_14_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_14_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_14_, output shape: {[16,30,30]}

            self.relu76_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_14_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_14_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_14_, output shape: {[16,30,30]}

            self.relu77_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_14_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_14_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_14_, output shape: {[1024,30,30]}

            self.conv76_1_1_15_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_15_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_15_, output shape: {[16,30,30]}

            self.relu76_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_15_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_15_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_15_, output shape: {[16,30,30]}

            self.relu77_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_15_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_15_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_15_, output shape: {[1024,30,30]}

            self.conv76_1_1_16_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_16_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_16_, output shape: {[16,30,30]}

            self.relu76_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_16_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_16_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_16_, output shape: {[16,30,30]}

            self.relu77_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_16_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_16_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_16_, output shape: {[1024,30,30]}

            self.conv76_1_1_17_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_17_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_17_, output shape: {[16,30,30]}

            self.relu76_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_17_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_17_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_17_, output shape: {[16,30,30]}

            self.relu77_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_17_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_17_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_17_, output shape: {[1024,30,30]}

            self.conv76_1_1_18_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_18_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_18_, output shape: {[16,30,30]}

            self.relu76_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_18_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_18_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_18_, output shape: {[16,30,30]}

            self.relu77_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_18_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_18_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_18_, output shape: {[1024,30,30]}

            self.conv76_1_1_19_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_19_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_19_, output shape: {[16,30,30]}

            self.relu76_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_19_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_19_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_19_, output shape: {[16,30,30]}

            self.relu77_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_19_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_19_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_19_, output shape: {[1024,30,30]}

            self.conv76_1_1_20_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_20_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_20_, output shape: {[16,30,30]}

            self.relu76_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_20_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_20_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_20_, output shape: {[16,30,30]}

            self.relu77_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_20_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_20_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_20_, output shape: {[1024,30,30]}

            self.conv76_1_1_21_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_21_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_21_, output shape: {[16,30,30]}

            self.relu76_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_21_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_21_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_21_, output shape: {[16,30,30]}

            self.relu77_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_21_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_21_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_21_, output shape: {[1024,30,30]}

            self.conv76_1_1_22_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_22_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_22_, output shape: {[16,30,30]}

            self.relu76_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_22_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_22_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_22_, output shape: {[16,30,30]}

            self.relu77_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_22_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_22_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_22_, output shape: {[1024,30,30]}

            self.conv76_1_1_23_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_23_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_23_, output shape: {[16,30,30]}

            self.relu76_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_23_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_23_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_23_, output shape: {[16,30,30]}

            self.relu77_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_23_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_23_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_23_, output shape: {[1024,30,30]}

            self.conv76_1_1_24_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_24_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_24_, output shape: {[16,30,30]}

            self.relu76_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_24_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_24_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_24_, output shape: {[16,30,30]}

            self.relu77_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_24_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_24_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_24_, output shape: {[1024,30,30]}

            self.conv76_1_1_25_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_25_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_25_, output shape: {[16,30,30]}

            self.relu76_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_25_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_25_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_25_, output shape: {[16,30,30]}

            self.relu77_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_25_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_25_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_25_, output shape: {[1024,30,30]}

            self.conv76_1_1_26_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_26_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_26_, output shape: {[16,30,30]}

            self.relu76_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_26_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_26_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_26_, output shape: {[16,30,30]}

            self.relu77_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_26_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_26_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_26_, output shape: {[1024,30,30]}

            self.conv76_1_1_27_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_27_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_27_, output shape: {[16,30,30]}

            self.relu76_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_27_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_27_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_27_, output shape: {[16,30,30]}

            self.relu77_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_27_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_27_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_27_, output shape: {[1024,30,30]}

            self.conv76_1_1_28_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_28_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_28_, output shape: {[16,30,30]}

            self.relu76_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_28_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_28_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_28_, output shape: {[16,30,30]}

            self.relu77_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_28_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_28_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_28_, output shape: {[1024,30,30]}

            self.conv76_1_1_29_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_29_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_29_, output shape: {[16,30,30]}

            self.relu76_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_29_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_29_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_29_, output shape: {[16,30,30]}

            self.relu77_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_29_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_29_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_29_, output shape: {[1024,30,30]}

            self.conv76_1_1_30_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_30_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_30_, output shape: {[16,30,30]}

            self.relu76_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_30_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_30_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_30_, output shape: {[16,30,30]}

            self.relu77_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_30_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_30_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_30_, output shape: {[1024,30,30]}

            self.conv76_1_1_31_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_31_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_31_, output shape: {[16,30,30]}

            self.relu76_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_31_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_31_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_31_, output shape: {[16,30,30]}

            self.relu77_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_31_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_31_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_31_, output shape: {[1024,30,30]}

            self.conv76_1_1_32_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv76_1_1_32_, output shape: {[16,30,30]}

            self.batchnorm76_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm76_1_1_32_, output shape: {[16,30,30]}

            self.relu76_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv77_1_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv77_1_1_32_ = gluon.nn.Conv2D(channels=16,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv77_1_1_32_, output shape: {[16,30,30]}

            self.batchnorm77_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm77_1_1_32_, output shape: {[16,30,30]}

            self.relu77_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv78_1_1_32_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv78_1_1_32_, output shape: {[1024,30,30]}

            self.batchnorm78_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm78_1_1_32_, output shape: {[1024,30,30]}

            self.relu80_1_ = gluon.nn.Activation(activation='relu')
            self.conv82_1_1_1_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_1_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_1_, output shape: {[32,30,30]}

            self.relu82_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_1_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_1_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_1_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_1_, output shape: {[32,15,15]}

            self.relu83_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_1_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_1_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_1_, output shape: {[2048,15,15]}

            self.conv82_1_1_2_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_2_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_2_, output shape: {[32,30,30]}

            self.relu82_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_2_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_2_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_2_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_2_, output shape: {[32,15,15]}

            self.relu83_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_2_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_2_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_2_, output shape: {[2048,15,15]}

            self.conv82_1_1_3_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_3_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_3_, output shape: {[32,30,30]}

            self.relu82_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_3_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_3_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_3_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_3_, output shape: {[32,15,15]}

            self.relu83_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_3_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_3_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_3_, output shape: {[2048,15,15]}

            self.conv82_1_1_4_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_4_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_4_, output shape: {[32,30,30]}

            self.relu82_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_4_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_4_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_4_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_4_, output shape: {[32,15,15]}

            self.relu83_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_4_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_4_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_4_, output shape: {[2048,15,15]}

            self.conv82_1_1_5_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_5_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_5_, output shape: {[32,30,30]}

            self.relu82_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_5_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_5_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_5_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_5_, output shape: {[32,15,15]}

            self.relu83_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_5_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_5_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_5_, output shape: {[2048,15,15]}

            self.conv82_1_1_6_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_6_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_6_, output shape: {[32,30,30]}

            self.relu82_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_6_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_6_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_6_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_6_, output shape: {[32,15,15]}

            self.relu83_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_6_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_6_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_6_, output shape: {[2048,15,15]}

            self.conv82_1_1_7_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_7_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_7_, output shape: {[32,30,30]}

            self.relu82_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_7_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_7_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_7_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_7_, output shape: {[32,15,15]}

            self.relu83_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_7_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_7_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_7_, output shape: {[2048,15,15]}

            self.conv82_1_1_8_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_8_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_8_, output shape: {[32,30,30]}

            self.relu82_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_8_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_8_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_8_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_8_, output shape: {[32,15,15]}

            self.relu83_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_8_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_8_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_8_, output shape: {[2048,15,15]}

            self.conv82_1_1_9_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_9_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_9_, output shape: {[32,30,30]}

            self.relu82_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_9_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_9_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_9_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_9_, output shape: {[32,15,15]}

            self.relu83_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_9_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_9_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_9_, output shape: {[2048,15,15]}

            self.conv82_1_1_10_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_10_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_10_, output shape: {[32,30,30]}

            self.relu82_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_10_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_10_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_10_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_10_, output shape: {[32,15,15]}

            self.relu83_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_10_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_10_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_10_, output shape: {[2048,15,15]}

            self.conv82_1_1_11_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_11_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_11_, output shape: {[32,30,30]}

            self.relu82_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_11_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_11_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_11_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_11_, output shape: {[32,15,15]}

            self.relu83_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_11_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_11_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_11_, output shape: {[2048,15,15]}

            self.conv82_1_1_12_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_12_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_12_, output shape: {[32,30,30]}

            self.relu82_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_12_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_12_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_12_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_12_, output shape: {[32,15,15]}

            self.relu83_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_12_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_12_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_12_, output shape: {[2048,15,15]}

            self.conv82_1_1_13_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_13_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_13_, output shape: {[32,30,30]}

            self.relu82_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_13_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_13_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_13_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_13_, output shape: {[32,15,15]}

            self.relu83_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_13_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_13_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_13_, output shape: {[2048,15,15]}

            self.conv82_1_1_14_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_14_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_14_, output shape: {[32,30,30]}

            self.relu82_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_14_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_14_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_14_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_14_, output shape: {[32,15,15]}

            self.relu83_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_14_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_14_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_14_, output shape: {[2048,15,15]}

            self.conv82_1_1_15_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_15_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_15_, output shape: {[32,30,30]}

            self.relu82_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_15_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_15_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_15_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_15_, output shape: {[32,15,15]}

            self.relu83_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_15_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_15_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_15_, output shape: {[2048,15,15]}

            self.conv82_1_1_16_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_16_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_16_, output shape: {[32,30,30]}

            self.relu82_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_16_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_16_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_16_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_16_, output shape: {[32,15,15]}

            self.relu83_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_16_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_16_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_16_, output shape: {[2048,15,15]}

            self.conv82_1_1_17_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_17_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_17_, output shape: {[32,30,30]}

            self.relu82_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_17_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_17_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_17_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_17_, output shape: {[32,15,15]}

            self.relu83_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_17_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_17_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_17_, output shape: {[2048,15,15]}

            self.conv82_1_1_18_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_18_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_18_, output shape: {[32,30,30]}

            self.relu82_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_18_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_18_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_18_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_18_, output shape: {[32,15,15]}

            self.relu83_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_18_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_18_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_18_, output shape: {[2048,15,15]}

            self.conv82_1_1_19_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_19_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_19_, output shape: {[32,30,30]}

            self.relu82_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_19_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_19_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_19_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_19_, output shape: {[32,15,15]}

            self.relu83_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_19_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_19_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_19_, output shape: {[2048,15,15]}

            self.conv82_1_1_20_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_20_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_20_, output shape: {[32,30,30]}

            self.relu82_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_20_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_20_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_20_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_20_, output shape: {[32,15,15]}

            self.relu83_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_20_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_20_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_20_, output shape: {[2048,15,15]}

            self.conv82_1_1_21_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_21_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_21_, output shape: {[32,30,30]}

            self.relu82_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_21_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_21_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_21_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_21_, output shape: {[32,15,15]}

            self.relu83_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_21_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_21_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_21_, output shape: {[2048,15,15]}

            self.conv82_1_1_22_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_22_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_22_, output shape: {[32,30,30]}

            self.relu82_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_22_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_22_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_22_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_22_, output shape: {[32,15,15]}

            self.relu83_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_22_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_22_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_22_, output shape: {[2048,15,15]}

            self.conv82_1_1_23_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_23_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_23_, output shape: {[32,30,30]}

            self.relu82_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_23_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_23_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_23_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_23_, output shape: {[32,15,15]}

            self.relu83_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_23_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_23_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_23_, output shape: {[2048,15,15]}

            self.conv82_1_1_24_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_24_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_24_, output shape: {[32,30,30]}

            self.relu82_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_24_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_24_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_24_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_24_, output shape: {[32,15,15]}

            self.relu83_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_24_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_24_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_24_, output shape: {[2048,15,15]}

            self.conv82_1_1_25_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_25_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_25_, output shape: {[32,30,30]}

            self.relu82_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_25_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_25_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_25_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_25_, output shape: {[32,15,15]}

            self.relu83_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_25_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_25_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_25_, output shape: {[2048,15,15]}

            self.conv82_1_1_26_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_26_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_26_, output shape: {[32,30,30]}

            self.relu82_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_26_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_26_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_26_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_26_, output shape: {[32,15,15]}

            self.relu83_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_26_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_26_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_26_, output shape: {[2048,15,15]}

            self.conv82_1_1_27_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_27_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_27_, output shape: {[32,30,30]}

            self.relu82_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_27_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_27_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_27_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_27_, output shape: {[32,15,15]}

            self.relu83_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_27_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_27_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_27_, output shape: {[2048,15,15]}

            self.conv82_1_1_28_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_28_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_28_, output shape: {[32,30,30]}

            self.relu82_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_28_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_28_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_28_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_28_, output shape: {[32,15,15]}

            self.relu83_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_28_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_28_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_28_, output shape: {[2048,15,15]}

            self.conv82_1_1_29_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_29_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_29_, output shape: {[32,30,30]}

            self.relu82_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_29_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_29_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_29_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_29_, output shape: {[32,15,15]}

            self.relu83_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_29_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_29_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_29_, output shape: {[2048,15,15]}

            self.conv82_1_1_30_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_30_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_30_, output shape: {[32,30,30]}

            self.relu82_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_30_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_30_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_30_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_30_, output shape: {[32,15,15]}

            self.relu83_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_30_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_30_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_30_, output shape: {[2048,15,15]}

            self.conv82_1_1_31_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_31_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_31_, output shape: {[32,30,30]}

            self.relu82_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_31_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_31_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_31_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_31_, output shape: {[32,15,15]}

            self.relu83_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_31_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_31_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_31_, output shape: {[2048,15,15]}

            self.conv82_1_1_32_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv82_1_1_32_, output shape: {[32,30,30]}

            self.batchnorm82_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm82_1_1_32_, output shape: {[32,30,30]}

            self.relu82_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv83_1_1_32_padding = Padding(padding=(0,0,0,0,1,0,1,0))
            self.conv83_1_1_32_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv83_1_1_32_, output shape: {[32,15,15]}

            self.batchnorm83_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm83_1_1_32_, output shape: {[32,15,15]}

            self.relu83_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv84_1_1_32_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv84_1_1_32_, output shape: {[2048,15,15]}

            self.batchnorm84_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm84_1_1_32_, output shape: {[2048,15,15]}

            self.conv81_1_2_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(2,2),
                groups=1,
                use_bias=True)
            # conv81_1_2_, output shape: {[2048,15,15]}

            self.batchnorm81_1_2_ = gluon.nn.BatchNorm()
            # batchnorm81_1_2_, output shape: {[2048,15,15]}

            self.relu86_1_ = gluon.nn.Activation(activation='relu')
            self.conv89_1_1_1_1_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_1_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_1_, output shape: {[32,15,15]}

            self.relu89_1_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_1_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_1_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_1_, output shape: {[32,15,15]}

            self.relu90_1_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_1_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_1_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_1_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_2_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_2_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_2_, output shape: {[32,15,15]}

            self.relu89_1_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_2_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_2_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_2_, output shape: {[32,15,15]}

            self.relu90_1_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_2_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_2_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_2_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_3_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_3_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_3_, output shape: {[32,15,15]}

            self.relu89_1_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_3_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_3_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_3_, output shape: {[32,15,15]}

            self.relu90_1_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_3_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_3_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_3_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_4_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_4_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_4_, output shape: {[32,15,15]}

            self.relu89_1_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_4_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_4_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_4_, output shape: {[32,15,15]}

            self.relu90_1_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_4_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_4_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_4_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_5_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_5_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_5_, output shape: {[32,15,15]}

            self.relu89_1_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_5_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_5_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_5_, output shape: {[32,15,15]}

            self.relu90_1_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_5_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_5_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_5_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_6_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_6_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_6_, output shape: {[32,15,15]}

            self.relu89_1_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_6_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_6_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_6_, output shape: {[32,15,15]}

            self.relu90_1_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_6_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_6_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_6_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_7_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_7_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_7_, output shape: {[32,15,15]}

            self.relu89_1_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_7_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_7_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_7_, output shape: {[32,15,15]}

            self.relu90_1_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_7_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_7_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_7_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_8_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_8_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_8_, output shape: {[32,15,15]}

            self.relu89_1_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_8_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_8_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_8_, output shape: {[32,15,15]}

            self.relu90_1_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_8_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_8_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_8_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_9_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_9_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_9_, output shape: {[32,15,15]}

            self.relu89_1_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_9_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_9_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_9_, output shape: {[32,15,15]}

            self.relu90_1_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_9_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_9_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_9_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_10_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_10_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_10_, output shape: {[32,15,15]}

            self.relu89_1_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_10_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_10_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_10_, output shape: {[32,15,15]}

            self.relu90_1_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_10_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_10_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_10_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_11_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_11_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_11_, output shape: {[32,15,15]}

            self.relu89_1_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_11_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_11_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_11_, output shape: {[32,15,15]}

            self.relu90_1_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_11_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_11_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_11_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_12_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_12_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_12_, output shape: {[32,15,15]}

            self.relu89_1_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_12_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_12_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_12_, output shape: {[32,15,15]}

            self.relu90_1_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_12_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_12_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_12_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_13_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_13_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_13_, output shape: {[32,15,15]}

            self.relu89_1_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_13_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_13_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_13_, output shape: {[32,15,15]}

            self.relu90_1_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_13_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_13_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_13_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_14_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_14_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_14_, output shape: {[32,15,15]}

            self.relu89_1_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_14_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_14_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_14_, output shape: {[32,15,15]}

            self.relu90_1_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_14_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_14_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_14_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_15_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_15_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_15_, output shape: {[32,15,15]}

            self.relu89_1_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_15_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_15_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_15_, output shape: {[32,15,15]}

            self.relu90_1_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_15_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_15_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_15_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_16_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_16_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_16_, output shape: {[32,15,15]}

            self.relu89_1_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_16_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_16_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_16_, output shape: {[32,15,15]}

            self.relu90_1_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_16_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_16_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_16_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_17_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_17_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_17_, output shape: {[32,15,15]}

            self.relu89_1_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_17_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_17_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_17_, output shape: {[32,15,15]}

            self.relu90_1_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_17_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_17_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_17_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_18_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_18_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_18_, output shape: {[32,15,15]}

            self.relu89_1_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_18_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_18_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_18_, output shape: {[32,15,15]}

            self.relu90_1_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_18_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_18_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_18_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_19_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_19_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_19_, output shape: {[32,15,15]}

            self.relu89_1_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_19_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_19_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_19_, output shape: {[32,15,15]}

            self.relu90_1_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_19_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_19_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_19_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_20_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_20_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_20_, output shape: {[32,15,15]}

            self.relu89_1_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_20_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_20_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_20_, output shape: {[32,15,15]}

            self.relu90_1_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_20_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_20_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_20_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_21_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_21_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_21_, output shape: {[32,15,15]}

            self.relu89_1_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_21_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_21_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_21_, output shape: {[32,15,15]}

            self.relu90_1_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_21_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_21_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_21_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_22_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_22_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_22_, output shape: {[32,15,15]}

            self.relu89_1_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_22_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_22_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_22_, output shape: {[32,15,15]}

            self.relu90_1_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_22_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_22_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_22_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_23_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_23_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_23_, output shape: {[32,15,15]}

            self.relu89_1_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_23_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_23_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_23_, output shape: {[32,15,15]}

            self.relu90_1_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_23_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_23_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_23_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_24_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_24_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_24_, output shape: {[32,15,15]}

            self.relu89_1_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_24_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_24_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_24_, output shape: {[32,15,15]}

            self.relu90_1_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_24_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_24_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_24_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_25_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_25_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_25_, output shape: {[32,15,15]}

            self.relu89_1_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_25_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_25_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_25_, output shape: {[32,15,15]}

            self.relu90_1_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_25_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_25_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_25_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_26_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_26_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_26_, output shape: {[32,15,15]}

            self.relu89_1_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_26_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_26_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_26_, output shape: {[32,15,15]}

            self.relu90_1_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_26_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_26_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_26_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_27_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_27_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_27_, output shape: {[32,15,15]}

            self.relu89_1_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_27_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_27_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_27_, output shape: {[32,15,15]}

            self.relu90_1_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_27_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_27_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_27_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_28_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_28_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_28_, output shape: {[32,15,15]}

            self.relu89_1_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_28_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_28_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_28_, output shape: {[32,15,15]}

            self.relu90_1_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_28_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_28_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_28_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_29_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_29_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_29_, output shape: {[32,15,15]}

            self.relu89_1_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_29_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_29_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_29_, output shape: {[32,15,15]}

            self.relu90_1_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_29_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_29_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_29_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_30_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_30_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_30_, output shape: {[32,15,15]}

            self.relu89_1_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_30_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_30_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_30_, output shape: {[32,15,15]}

            self.relu90_1_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_30_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_30_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_30_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_31_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_31_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_31_, output shape: {[32,15,15]}

            self.relu89_1_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_31_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_31_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_31_, output shape: {[32,15,15]}

            self.relu90_1_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_31_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_31_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_31_, output shape: {[2048,15,15]}

            self.conv89_1_1_1_32_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv89_1_1_1_32_, output shape: {[32,15,15]}

            self.batchnorm89_1_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm89_1_1_1_32_, output shape: {[32,15,15]}

            self.relu89_1_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv90_1_1_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv90_1_1_1_32_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv90_1_1_1_32_, output shape: {[32,15,15]}

            self.batchnorm90_1_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm90_1_1_1_32_, output shape: {[32,15,15]}

            self.relu90_1_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv91_1_1_1_32_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv91_1_1_1_32_, output shape: {[2048,15,15]}

            self.batchnorm91_1_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm91_1_1_1_32_, output shape: {[2048,15,15]}

            self.relu93_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv95_1_1_1_1_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_1_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_1_, output shape: {[32,15,15]}

            self.relu95_1_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_1_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_1_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_1_, output shape: {[32,15,15]}

            self.relu96_1_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_1_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_1_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_1_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_1_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_2_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_2_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_2_, output shape: {[32,15,15]}

            self.relu95_1_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_2_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_2_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_2_, output shape: {[32,15,15]}

            self.relu96_1_1_1_2_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_2_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_2_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_2_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_2_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_3_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_3_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_3_, output shape: {[32,15,15]}

            self.relu95_1_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_3_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_3_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_3_, output shape: {[32,15,15]}

            self.relu96_1_1_1_3_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_3_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_3_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_3_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_3_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_4_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_4_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_4_, output shape: {[32,15,15]}

            self.relu95_1_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_4_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_4_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_4_, output shape: {[32,15,15]}

            self.relu96_1_1_1_4_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_4_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_4_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_4_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_4_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_5_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_5_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_5_, output shape: {[32,15,15]}

            self.relu95_1_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_5_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_5_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_5_, output shape: {[32,15,15]}

            self.relu96_1_1_1_5_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_5_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_5_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_5_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_5_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_6_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_6_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_6_, output shape: {[32,15,15]}

            self.relu95_1_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_6_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_6_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_6_, output shape: {[32,15,15]}

            self.relu96_1_1_1_6_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_6_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_6_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_6_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_6_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_7_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_7_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_7_, output shape: {[32,15,15]}

            self.relu95_1_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_7_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_7_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_7_, output shape: {[32,15,15]}

            self.relu96_1_1_1_7_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_7_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_7_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_7_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_7_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_8_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_8_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_8_, output shape: {[32,15,15]}

            self.relu95_1_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_8_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_8_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_8_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_8_, output shape: {[32,15,15]}

            self.relu96_1_1_1_8_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_8_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_8_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_8_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_8_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_9_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_9_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_9_, output shape: {[32,15,15]}

            self.relu95_1_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_9_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_9_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_9_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_9_, output shape: {[32,15,15]}

            self.relu96_1_1_1_9_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_9_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_9_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_9_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_9_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_10_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_10_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_10_, output shape: {[32,15,15]}

            self.relu95_1_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_10_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_10_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_10_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_10_, output shape: {[32,15,15]}

            self.relu96_1_1_1_10_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_10_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_10_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_10_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_10_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_11_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_11_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_11_, output shape: {[32,15,15]}

            self.relu95_1_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_11_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_11_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_11_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_11_, output shape: {[32,15,15]}

            self.relu96_1_1_1_11_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_11_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_11_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_11_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_11_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_12_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_12_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_12_, output shape: {[32,15,15]}

            self.relu95_1_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_12_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_12_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_12_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_12_, output shape: {[32,15,15]}

            self.relu96_1_1_1_12_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_12_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_12_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_12_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_12_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_13_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_13_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_13_, output shape: {[32,15,15]}

            self.relu95_1_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_13_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_13_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_13_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_13_, output shape: {[32,15,15]}

            self.relu96_1_1_1_13_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_13_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_13_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_13_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_13_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_14_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_14_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_14_, output shape: {[32,15,15]}

            self.relu95_1_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_14_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_14_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_14_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_14_, output shape: {[32,15,15]}

            self.relu96_1_1_1_14_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_14_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_14_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_14_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_14_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_15_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_15_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_15_, output shape: {[32,15,15]}

            self.relu95_1_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_15_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_15_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_15_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_15_, output shape: {[32,15,15]}

            self.relu96_1_1_1_15_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_15_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_15_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_15_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_15_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_16_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_16_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_16_, output shape: {[32,15,15]}

            self.relu95_1_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_16_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_16_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_16_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_16_, output shape: {[32,15,15]}

            self.relu96_1_1_1_16_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_16_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_16_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_16_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_16_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_17_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_17_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_17_, output shape: {[32,15,15]}

            self.relu95_1_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_17_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_17_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_17_, output shape: {[32,15,15]}

            self.relu96_1_1_1_17_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_17_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_17_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_17_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_17_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_18_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_18_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_18_, output shape: {[32,15,15]}

            self.relu95_1_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_18_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_18_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_18_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_18_, output shape: {[32,15,15]}

            self.relu96_1_1_1_18_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_18_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_18_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_18_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_18_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_19_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_19_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_19_, output shape: {[32,15,15]}

            self.relu95_1_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_19_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_19_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_19_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_19_, output shape: {[32,15,15]}

            self.relu96_1_1_1_19_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_19_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_19_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_19_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_19_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_20_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_20_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_20_, output shape: {[32,15,15]}

            self.relu95_1_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_20_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_20_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_20_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_20_, output shape: {[32,15,15]}

            self.relu96_1_1_1_20_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_20_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_20_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_20_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_20_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_21_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_21_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_21_, output shape: {[32,15,15]}

            self.relu95_1_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_21_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_21_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_21_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_21_, output shape: {[32,15,15]}

            self.relu96_1_1_1_21_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_21_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_21_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_21_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_21_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_22_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_22_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_22_, output shape: {[32,15,15]}

            self.relu95_1_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_22_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_22_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_22_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_22_, output shape: {[32,15,15]}

            self.relu96_1_1_1_22_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_22_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_22_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_22_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_22_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_23_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_23_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_23_, output shape: {[32,15,15]}

            self.relu95_1_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_23_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_23_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_23_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_23_, output shape: {[32,15,15]}

            self.relu96_1_1_1_23_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_23_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_23_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_23_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_23_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_24_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_24_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_24_, output shape: {[32,15,15]}

            self.relu95_1_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_24_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_24_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_24_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_24_, output shape: {[32,15,15]}

            self.relu96_1_1_1_24_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_24_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_24_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_24_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_24_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_25_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_25_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_25_, output shape: {[32,15,15]}

            self.relu95_1_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_25_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_25_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_25_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_25_, output shape: {[32,15,15]}

            self.relu96_1_1_1_25_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_25_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_25_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_25_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_25_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_26_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_26_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_26_, output shape: {[32,15,15]}

            self.relu95_1_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_26_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_26_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_26_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_26_, output shape: {[32,15,15]}

            self.relu96_1_1_1_26_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_26_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_26_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_26_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_26_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_27_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_27_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_27_, output shape: {[32,15,15]}

            self.relu95_1_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_27_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_27_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_27_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_27_, output shape: {[32,15,15]}

            self.relu96_1_1_1_27_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_27_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_27_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_27_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_27_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_28_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_28_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_28_, output shape: {[32,15,15]}

            self.relu95_1_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_28_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_28_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_28_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_28_, output shape: {[32,15,15]}

            self.relu96_1_1_1_28_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_28_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_28_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_28_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_28_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_29_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_29_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_29_, output shape: {[32,15,15]}

            self.relu95_1_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_29_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_29_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_29_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_29_, output shape: {[32,15,15]}

            self.relu96_1_1_1_29_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_29_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_29_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_29_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_29_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_30_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_30_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_30_, output shape: {[32,15,15]}

            self.relu95_1_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_30_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_30_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_30_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_30_, output shape: {[32,15,15]}

            self.relu96_1_1_1_30_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_30_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_30_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_30_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_30_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_31_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_31_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_31_, output shape: {[32,15,15]}

            self.relu95_1_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_31_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_31_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_31_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_31_, output shape: {[32,15,15]}

            self.relu96_1_1_1_31_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_31_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_31_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_31_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_31_, output shape: {[2048,15,15]}

            self.conv95_1_1_1_32_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv95_1_1_1_32_, output shape: {[32,15,15]}

            self.batchnorm95_1_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm95_1_1_1_32_, output shape: {[32,15,15]}

            self.relu95_1_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv96_1_1_1_32_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv96_1_1_1_32_ = gluon.nn.Conv2D(channels=32,
                kernel_size=(3,3),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv96_1_1_1_32_, output shape: {[32,15,15]}

            self.batchnorm96_1_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm96_1_1_1_32_, output shape: {[32,15,15]}

            self.relu96_1_1_1_32_ = gluon.nn.Activation(activation='relu')
            self.conv97_1_1_1_32_ = gluon.nn.Conv2D(channels=2048,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv97_1_1_1_32_, output shape: {[2048,15,15]}

            self.batchnorm97_1_1_1_32_ = gluon.nn.BatchNorm()
            # batchnorm97_1_1_1_32_, output shape: {[2048,15,15]}

            self.relu99_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv99_1_1_padding = Padding(padding=(0,0,0,0,3,3,3,3))
            self.conv99_1_1_ = gluon.nn.Conv2D(channels=4096,
                kernel_size=(7,7),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv99_1_1_, output shape: {[4096,15,15]}

            self.batchnorm99_1_1_ = gluon.nn.BatchNorm()
            # batchnorm99_1_1_, output shape: {[4096,15,15]}

            self.relu100_1_1_ = gluon.nn.Activation(activation='relu')
            self.relu101_1_1_ = gluon.nn.Activation(activation='relu')
            self.dropout101_1_1_ = gluon.nn.Dropout(rate=0.5)
            self.conv101_1_1_padding = Padding(padding=(0,0,0,0,3,3,3,3))
            self.conv101_1_1_ = gluon.nn.Conv2D(channels=4096,
                kernel_size=(7,7),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv101_1_1_, output shape: {[4096,15,15]}

            self.batchnorm101_1_1_ = gluon.nn.BatchNorm()
            # batchnorm101_1_1_, output shape: {[4096,15,15]}

            self.relu102_1_1_ = gluon.nn.Activation(activation='relu')
            self.relu103_1_1_ = gluon.nn.Activation(activation='relu')
            self.dropout103_1_1_ = gluon.nn.Dropout(rate=0.5)
            self.conv103_1_1_ = gluon.nn.Conv2D(channels=21,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv103_1_1_, output shape: {[21,15,15]}

            self.transconv103_1_1_padding = (1,1)
            self.transconv103_1_1_ = gluon.nn.Conv2DTranspose(channels=21,
                kernel_size=(4,4),
                strides=(2,2),
                padding=self.transconv103_1_1_padding,
                groups=1,
                use_bias=True)
            # transconv103_1_1_, output shape: {[21,30,30]}

            self.conv87_1_2_padding = Padding(padding=(0,0,0,0,2,1,2,1))
            self.conv87_1_2_ = gluon.nn.Conv2D(channels=21,
                kernel_size=(4,4),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv87_1_2_, output shape: {[21,15,15]}

            self.transconv104_1_padding = (1,1)
            self.transconv104_1_ = gluon.nn.Conv2DTranspose(channels=21,
                kernel_size=(4,4),
                strides=(2,2),
                padding=self.transconv104_1_padding,
                groups=1,
                use_bias=True)
            # transconv104_1_, output shape: {[21,60,60]}

            self.conv50_2_ = gluon.nn.Conv2D(channels=21,
                kernel_size=(1,1),
                strides=(1,1),
                groups=1,
                use_bias=True)
            # conv50_2_, output shape: {[21,30,30]}

            self.transconv105_padding = (4,4)
            self.transconv105_ = gluon.nn.Conv2DTranspose(channels=21,
                kernel_size=(16,16),
                strides=(8,8),
                padding=self.transconv105_padding,
                groups=1,
                use_bias=True)
            # transconv105_, output shape: {[21,480,480]}


            pass

    def hybrid_forward(self, F, data_):
        data_ = self.input_normalization_data_(data_)
        conv1_padding = self.conv1_padding(data_)
        conv1_ = self.conv1_(conv1_padding)
        batchnorm1_ = self.batchnorm1_(conv1_)
        relu1_ = self.relu1_(batchnorm1_)
        pool1_padding = self.pool1_padding(relu1_)
        pool1_ = self.pool1_(pool1_padding)
        conv3_1_1_ = self.conv3_1_1_(pool1_)
        batchnorm3_1_1_ = self.batchnorm3_1_1_(conv3_1_1_)
        relu3_1_1_ = self.relu3_1_1_(batchnorm3_1_1_)
        conv4_1_1_padding = self.conv4_1_1_padding(relu3_1_1_)
        conv4_1_1_ = self.conv4_1_1_(conv4_1_1_padding)
        batchnorm4_1_1_ = self.batchnorm4_1_1_(conv4_1_1_)
        relu4_1_1_ = self.relu4_1_1_(batchnorm4_1_1_)
        conv5_1_1_ = self.conv5_1_1_(relu4_1_1_)
        batchnorm5_1_1_ = self.batchnorm5_1_1_(conv5_1_1_)
        conv3_1_2_ = self.conv3_1_2_(pool1_)
        batchnorm3_1_2_ = self.batchnorm3_1_2_(conv3_1_2_)
        relu3_1_2_ = self.relu3_1_2_(batchnorm3_1_2_)
        conv4_1_2_padding = self.conv4_1_2_padding(relu3_1_2_)
        conv4_1_2_ = self.conv4_1_2_(conv4_1_2_padding)
        batchnorm4_1_2_ = self.batchnorm4_1_2_(conv4_1_2_)
        relu4_1_2_ = self.relu4_1_2_(batchnorm4_1_2_)
        conv5_1_2_ = self.conv5_1_2_(relu4_1_2_)
        batchnorm5_1_2_ = self.batchnorm5_1_2_(conv5_1_2_)
        conv3_1_3_ = self.conv3_1_3_(pool1_)
        batchnorm3_1_3_ = self.batchnorm3_1_3_(conv3_1_3_)
        relu3_1_3_ = self.relu3_1_3_(batchnorm3_1_3_)
        conv4_1_3_padding = self.conv4_1_3_padding(relu3_1_3_)
        conv4_1_3_ = self.conv4_1_3_(conv4_1_3_padding)
        batchnorm4_1_3_ = self.batchnorm4_1_3_(conv4_1_3_)
        relu4_1_3_ = self.relu4_1_3_(batchnorm4_1_3_)
        conv5_1_3_ = self.conv5_1_3_(relu4_1_3_)
        batchnorm5_1_3_ = self.batchnorm5_1_3_(conv5_1_3_)
        conv3_1_4_ = self.conv3_1_4_(pool1_)
        batchnorm3_1_4_ = self.batchnorm3_1_4_(conv3_1_4_)
        relu3_1_4_ = self.relu3_1_4_(batchnorm3_1_4_)
        conv4_1_4_padding = self.conv4_1_4_padding(relu3_1_4_)
        conv4_1_4_ = self.conv4_1_4_(conv4_1_4_padding)
        batchnorm4_1_4_ = self.batchnorm4_1_4_(conv4_1_4_)
        relu4_1_4_ = self.relu4_1_4_(batchnorm4_1_4_)
        conv5_1_4_ = self.conv5_1_4_(relu4_1_4_)
        batchnorm5_1_4_ = self.batchnorm5_1_4_(conv5_1_4_)
        conv3_1_5_ = self.conv3_1_5_(pool1_)
        batchnorm3_1_5_ = self.batchnorm3_1_5_(conv3_1_5_)
        relu3_1_5_ = self.relu3_1_5_(batchnorm3_1_5_)
        conv4_1_5_padding = self.conv4_1_5_padding(relu3_1_5_)
        conv4_1_5_ = self.conv4_1_5_(conv4_1_5_padding)
        batchnorm4_1_5_ = self.batchnorm4_1_5_(conv4_1_5_)
        relu4_1_5_ = self.relu4_1_5_(batchnorm4_1_5_)
        conv5_1_5_ = self.conv5_1_5_(relu4_1_5_)
        batchnorm5_1_5_ = self.batchnorm5_1_5_(conv5_1_5_)
        conv3_1_6_ = self.conv3_1_6_(pool1_)
        batchnorm3_1_6_ = self.batchnorm3_1_6_(conv3_1_6_)
        relu3_1_6_ = self.relu3_1_6_(batchnorm3_1_6_)
        conv4_1_6_padding = self.conv4_1_6_padding(relu3_1_6_)
        conv4_1_6_ = self.conv4_1_6_(conv4_1_6_padding)
        batchnorm4_1_6_ = self.batchnorm4_1_6_(conv4_1_6_)
        relu4_1_6_ = self.relu4_1_6_(batchnorm4_1_6_)
        conv5_1_6_ = self.conv5_1_6_(relu4_1_6_)
        batchnorm5_1_6_ = self.batchnorm5_1_6_(conv5_1_6_)
        conv3_1_7_ = self.conv3_1_7_(pool1_)
        batchnorm3_1_7_ = self.batchnorm3_1_7_(conv3_1_7_)
        relu3_1_7_ = self.relu3_1_7_(batchnorm3_1_7_)
        conv4_1_7_padding = self.conv4_1_7_padding(relu3_1_7_)
        conv4_1_7_ = self.conv4_1_7_(conv4_1_7_padding)
        batchnorm4_1_7_ = self.batchnorm4_1_7_(conv4_1_7_)
        relu4_1_7_ = self.relu4_1_7_(batchnorm4_1_7_)
        conv5_1_7_ = self.conv5_1_7_(relu4_1_7_)
        batchnorm5_1_7_ = self.batchnorm5_1_7_(conv5_1_7_)
        conv3_1_8_ = self.conv3_1_8_(pool1_)
        batchnorm3_1_8_ = self.batchnorm3_1_8_(conv3_1_8_)
        relu3_1_8_ = self.relu3_1_8_(batchnorm3_1_8_)
        conv4_1_8_padding = self.conv4_1_8_padding(relu3_1_8_)
        conv4_1_8_ = self.conv4_1_8_(conv4_1_8_padding)
        batchnorm4_1_8_ = self.batchnorm4_1_8_(conv4_1_8_)
        relu4_1_8_ = self.relu4_1_8_(batchnorm4_1_8_)
        conv5_1_8_ = self.conv5_1_8_(relu4_1_8_)
        batchnorm5_1_8_ = self.batchnorm5_1_8_(conv5_1_8_)
        conv3_1_9_ = self.conv3_1_9_(pool1_)
        batchnorm3_1_9_ = self.batchnorm3_1_9_(conv3_1_9_)
        relu3_1_9_ = self.relu3_1_9_(batchnorm3_1_9_)
        conv4_1_9_padding = self.conv4_1_9_padding(relu3_1_9_)
        conv4_1_9_ = self.conv4_1_9_(conv4_1_9_padding)
        batchnorm4_1_9_ = self.batchnorm4_1_9_(conv4_1_9_)
        relu4_1_9_ = self.relu4_1_9_(batchnorm4_1_9_)
        conv5_1_9_ = self.conv5_1_9_(relu4_1_9_)
        batchnorm5_1_9_ = self.batchnorm5_1_9_(conv5_1_9_)
        conv3_1_10_ = self.conv3_1_10_(pool1_)
        batchnorm3_1_10_ = self.batchnorm3_1_10_(conv3_1_10_)
        relu3_1_10_ = self.relu3_1_10_(batchnorm3_1_10_)
        conv4_1_10_padding = self.conv4_1_10_padding(relu3_1_10_)
        conv4_1_10_ = self.conv4_1_10_(conv4_1_10_padding)
        batchnorm4_1_10_ = self.batchnorm4_1_10_(conv4_1_10_)
        relu4_1_10_ = self.relu4_1_10_(batchnorm4_1_10_)
        conv5_1_10_ = self.conv5_1_10_(relu4_1_10_)
        batchnorm5_1_10_ = self.batchnorm5_1_10_(conv5_1_10_)
        conv3_1_11_ = self.conv3_1_11_(pool1_)
        batchnorm3_1_11_ = self.batchnorm3_1_11_(conv3_1_11_)
        relu3_1_11_ = self.relu3_1_11_(batchnorm3_1_11_)
        conv4_1_11_padding = self.conv4_1_11_padding(relu3_1_11_)
        conv4_1_11_ = self.conv4_1_11_(conv4_1_11_padding)
        batchnorm4_1_11_ = self.batchnorm4_1_11_(conv4_1_11_)
        relu4_1_11_ = self.relu4_1_11_(batchnorm4_1_11_)
        conv5_1_11_ = self.conv5_1_11_(relu4_1_11_)
        batchnorm5_1_11_ = self.batchnorm5_1_11_(conv5_1_11_)
        conv3_1_12_ = self.conv3_1_12_(pool1_)
        batchnorm3_1_12_ = self.batchnorm3_1_12_(conv3_1_12_)
        relu3_1_12_ = self.relu3_1_12_(batchnorm3_1_12_)
        conv4_1_12_padding = self.conv4_1_12_padding(relu3_1_12_)
        conv4_1_12_ = self.conv4_1_12_(conv4_1_12_padding)
        batchnorm4_1_12_ = self.batchnorm4_1_12_(conv4_1_12_)
        relu4_1_12_ = self.relu4_1_12_(batchnorm4_1_12_)
        conv5_1_12_ = self.conv5_1_12_(relu4_1_12_)
        batchnorm5_1_12_ = self.batchnorm5_1_12_(conv5_1_12_)
        conv3_1_13_ = self.conv3_1_13_(pool1_)
        batchnorm3_1_13_ = self.batchnorm3_1_13_(conv3_1_13_)
        relu3_1_13_ = self.relu3_1_13_(batchnorm3_1_13_)
        conv4_1_13_padding = self.conv4_1_13_padding(relu3_1_13_)
        conv4_1_13_ = self.conv4_1_13_(conv4_1_13_padding)
        batchnorm4_1_13_ = self.batchnorm4_1_13_(conv4_1_13_)
        relu4_1_13_ = self.relu4_1_13_(batchnorm4_1_13_)
        conv5_1_13_ = self.conv5_1_13_(relu4_1_13_)
        batchnorm5_1_13_ = self.batchnorm5_1_13_(conv5_1_13_)
        conv3_1_14_ = self.conv3_1_14_(pool1_)
        batchnorm3_1_14_ = self.batchnorm3_1_14_(conv3_1_14_)
        relu3_1_14_ = self.relu3_1_14_(batchnorm3_1_14_)
        conv4_1_14_padding = self.conv4_1_14_padding(relu3_1_14_)
        conv4_1_14_ = self.conv4_1_14_(conv4_1_14_padding)
        batchnorm4_1_14_ = self.batchnorm4_1_14_(conv4_1_14_)
        relu4_1_14_ = self.relu4_1_14_(batchnorm4_1_14_)
        conv5_1_14_ = self.conv5_1_14_(relu4_1_14_)
        batchnorm5_1_14_ = self.batchnorm5_1_14_(conv5_1_14_)
        conv3_1_15_ = self.conv3_1_15_(pool1_)
        batchnorm3_1_15_ = self.batchnorm3_1_15_(conv3_1_15_)
        relu3_1_15_ = self.relu3_1_15_(batchnorm3_1_15_)
        conv4_1_15_padding = self.conv4_1_15_padding(relu3_1_15_)
        conv4_1_15_ = self.conv4_1_15_(conv4_1_15_padding)
        batchnorm4_1_15_ = self.batchnorm4_1_15_(conv4_1_15_)
        relu4_1_15_ = self.relu4_1_15_(batchnorm4_1_15_)
        conv5_1_15_ = self.conv5_1_15_(relu4_1_15_)
        batchnorm5_1_15_ = self.batchnorm5_1_15_(conv5_1_15_)
        conv3_1_16_ = self.conv3_1_16_(pool1_)
        batchnorm3_1_16_ = self.batchnorm3_1_16_(conv3_1_16_)
        relu3_1_16_ = self.relu3_1_16_(batchnorm3_1_16_)
        conv4_1_16_padding = self.conv4_1_16_padding(relu3_1_16_)
        conv4_1_16_ = self.conv4_1_16_(conv4_1_16_padding)
        batchnorm4_1_16_ = self.batchnorm4_1_16_(conv4_1_16_)
        relu4_1_16_ = self.relu4_1_16_(batchnorm4_1_16_)
        conv5_1_16_ = self.conv5_1_16_(relu4_1_16_)
        batchnorm5_1_16_ = self.batchnorm5_1_16_(conv5_1_16_)
        conv3_1_17_ = self.conv3_1_17_(pool1_)
        batchnorm3_1_17_ = self.batchnorm3_1_17_(conv3_1_17_)
        relu3_1_17_ = self.relu3_1_17_(batchnorm3_1_17_)
        conv4_1_17_padding = self.conv4_1_17_padding(relu3_1_17_)
        conv4_1_17_ = self.conv4_1_17_(conv4_1_17_padding)
        batchnorm4_1_17_ = self.batchnorm4_1_17_(conv4_1_17_)
        relu4_1_17_ = self.relu4_1_17_(batchnorm4_1_17_)
        conv5_1_17_ = self.conv5_1_17_(relu4_1_17_)
        batchnorm5_1_17_ = self.batchnorm5_1_17_(conv5_1_17_)
        conv3_1_18_ = self.conv3_1_18_(pool1_)
        batchnorm3_1_18_ = self.batchnorm3_1_18_(conv3_1_18_)
        relu3_1_18_ = self.relu3_1_18_(batchnorm3_1_18_)
        conv4_1_18_padding = self.conv4_1_18_padding(relu3_1_18_)
        conv4_1_18_ = self.conv4_1_18_(conv4_1_18_padding)
        batchnorm4_1_18_ = self.batchnorm4_1_18_(conv4_1_18_)
        relu4_1_18_ = self.relu4_1_18_(batchnorm4_1_18_)
        conv5_1_18_ = self.conv5_1_18_(relu4_1_18_)
        batchnorm5_1_18_ = self.batchnorm5_1_18_(conv5_1_18_)
        conv3_1_19_ = self.conv3_1_19_(pool1_)
        batchnorm3_1_19_ = self.batchnorm3_1_19_(conv3_1_19_)
        relu3_1_19_ = self.relu3_1_19_(batchnorm3_1_19_)
        conv4_1_19_padding = self.conv4_1_19_padding(relu3_1_19_)
        conv4_1_19_ = self.conv4_1_19_(conv4_1_19_padding)
        batchnorm4_1_19_ = self.batchnorm4_1_19_(conv4_1_19_)
        relu4_1_19_ = self.relu4_1_19_(batchnorm4_1_19_)
        conv5_1_19_ = self.conv5_1_19_(relu4_1_19_)
        batchnorm5_1_19_ = self.batchnorm5_1_19_(conv5_1_19_)
        conv3_1_20_ = self.conv3_1_20_(pool1_)
        batchnorm3_1_20_ = self.batchnorm3_1_20_(conv3_1_20_)
        relu3_1_20_ = self.relu3_1_20_(batchnorm3_1_20_)
        conv4_1_20_padding = self.conv4_1_20_padding(relu3_1_20_)
        conv4_1_20_ = self.conv4_1_20_(conv4_1_20_padding)
        batchnorm4_1_20_ = self.batchnorm4_1_20_(conv4_1_20_)
        relu4_1_20_ = self.relu4_1_20_(batchnorm4_1_20_)
        conv5_1_20_ = self.conv5_1_20_(relu4_1_20_)
        batchnorm5_1_20_ = self.batchnorm5_1_20_(conv5_1_20_)
        conv3_1_21_ = self.conv3_1_21_(pool1_)
        batchnorm3_1_21_ = self.batchnorm3_1_21_(conv3_1_21_)
        relu3_1_21_ = self.relu3_1_21_(batchnorm3_1_21_)
        conv4_1_21_padding = self.conv4_1_21_padding(relu3_1_21_)
        conv4_1_21_ = self.conv4_1_21_(conv4_1_21_padding)
        batchnorm4_1_21_ = self.batchnorm4_1_21_(conv4_1_21_)
        relu4_1_21_ = self.relu4_1_21_(batchnorm4_1_21_)
        conv5_1_21_ = self.conv5_1_21_(relu4_1_21_)
        batchnorm5_1_21_ = self.batchnorm5_1_21_(conv5_1_21_)
        conv3_1_22_ = self.conv3_1_22_(pool1_)
        batchnorm3_1_22_ = self.batchnorm3_1_22_(conv3_1_22_)
        relu3_1_22_ = self.relu3_1_22_(batchnorm3_1_22_)
        conv4_1_22_padding = self.conv4_1_22_padding(relu3_1_22_)
        conv4_1_22_ = self.conv4_1_22_(conv4_1_22_padding)
        batchnorm4_1_22_ = self.batchnorm4_1_22_(conv4_1_22_)
        relu4_1_22_ = self.relu4_1_22_(batchnorm4_1_22_)
        conv5_1_22_ = self.conv5_1_22_(relu4_1_22_)
        batchnorm5_1_22_ = self.batchnorm5_1_22_(conv5_1_22_)
        conv3_1_23_ = self.conv3_1_23_(pool1_)
        batchnorm3_1_23_ = self.batchnorm3_1_23_(conv3_1_23_)
        relu3_1_23_ = self.relu3_1_23_(batchnorm3_1_23_)
        conv4_1_23_padding = self.conv4_1_23_padding(relu3_1_23_)
        conv4_1_23_ = self.conv4_1_23_(conv4_1_23_padding)
        batchnorm4_1_23_ = self.batchnorm4_1_23_(conv4_1_23_)
        relu4_1_23_ = self.relu4_1_23_(batchnorm4_1_23_)
        conv5_1_23_ = self.conv5_1_23_(relu4_1_23_)
        batchnorm5_1_23_ = self.batchnorm5_1_23_(conv5_1_23_)
        conv3_1_24_ = self.conv3_1_24_(pool1_)
        batchnorm3_1_24_ = self.batchnorm3_1_24_(conv3_1_24_)
        relu3_1_24_ = self.relu3_1_24_(batchnorm3_1_24_)
        conv4_1_24_padding = self.conv4_1_24_padding(relu3_1_24_)
        conv4_1_24_ = self.conv4_1_24_(conv4_1_24_padding)
        batchnorm4_1_24_ = self.batchnorm4_1_24_(conv4_1_24_)
        relu4_1_24_ = self.relu4_1_24_(batchnorm4_1_24_)
        conv5_1_24_ = self.conv5_1_24_(relu4_1_24_)
        batchnorm5_1_24_ = self.batchnorm5_1_24_(conv5_1_24_)
        conv3_1_25_ = self.conv3_1_25_(pool1_)
        batchnorm3_1_25_ = self.batchnorm3_1_25_(conv3_1_25_)
        relu3_1_25_ = self.relu3_1_25_(batchnorm3_1_25_)
        conv4_1_25_padding = self.conv4_1_25_padding(relu3_1_25_)
        conv4_1_25_ = self.conv4_1_25_(conv4_1_25_padding)
        batchnorm4_1_25_ = self.batchnorm4_1_25_(conv4_1_25_)
        relu4_1_25_ = self.relu4_1_25_(batchnorm4_1_25_)
        conv5_1_25_ = self.conv5_1_25_(relu4_1_25_)
        batchnorm5_1_25_ = self.batchnorm5_1_25_(conv5_1_25_)
        conv3_1_26_ = self.conv3_1_26_(pool1_)
        batchnorm3_1_26_ = self.batchnorm3_1_26_(conv3_1_26_)
        relu3_1_26_ = self.relu3_1_26_(batchnorm3_1_26_)
        conv4_1_26_padding = self.conv4_1_26_padding(relu3_1_26_)
        conv4_1_26_ = self.conv4_1_26_(conv4_1_26_padding)
        batchnorm4_1_26_ = self.batchnorm4_1_26_(conv4_1_26_)
        relu4_1_26_ = self.relu4_1_26_(batchnorm4_1_26_)
        conv5_1_26_ = self.conv5_1_26_(relu4_1_26_)
        batchnorm5_1_26_ = self.batchnorm5_1_26_(conv5_1_26_)
        conv3_1_27_ = self.conv3_1_27_(pool1_)
        batchnorm3_1_27_ = self.batchnorm3_1_27_(conv3_1_27_)
        relu3_1_27_ = self.relu3_1_27_(batchnorm3_1_27_)
        conv4_1_27_padding = self.conv4_1_27_padding(relu3_1_27_)
        conv4_1_27_ = self.conv4_1_27_(conv4_1_27_padding)
        batchnorm4_1_27_ = self.batchnorm4_1_27_(conv4_1_27_)
        relu4_1_27_ = self.relu4_1_27_(batchnorm4_1_27_)
        conv5_1_27_ = self.conv5_1_27_(relu4_1_27_)
        batchnorm5_1_27_ = self.batchnorm5_1_27_(conv5_1_27_)
        conv3_1_28_ = self.conv3_1_28_(pool1_)
        batchnorm3_1_28_ = self.batchnorm3_1_28_(conv3_1_28_)
        relu3_1_28_ = self.relu3_1_28_(batchnorm3_1_28_)
        conv4_1_28_padding = self.conv4_1_28_padding(relu3_1_28_)
        conv4_1_28_ = self.conv4_1_28_(conv4_1_28_padding)
        batchnorm4_1_28_ = self.batchnorm4_1_28_(conv4_1_28_)
        relu4_1_28_ = self.relu4_1_28_(batchnorm4_1_28_)
        conv5_1_28_ = self.conv5_1_28_(relu4_1_28_)
        batchnorm5_1_28_ = self.batchnorm5_1_28_(conv5_1_28_)
        conv3_1_29_ = self.conv3_1_29_(pool1_)
        batchnorm3_1_29_ = self.batchnorm3_1_29_(conv3_1_29_)
        relu3_1_29_ = self.relu3_1_29_(batchnorm3_1_29_)
        conv4_1_29_padding = self.conv4_1_29_padding(relu3_1_29_)
        conv4_1_29_ = self.conv4_1_29_(conv4_1_29_padding)
        batchnorm4_1_29_ = self.batchnorm4_1_29_(conv4_1_29_)
        relu4_1_29_ = self.relu4_1_29_(batchnorm4_1_29_)
        conv5_1_29_ = self.conv5_1_29_(relu4_1_29_)
        batchnorm5_1_29_ = self.batchnorm5_1_29_(conv5_1_29_)
        conv3_1_30_ = self.conv3_1_30_(pool1_)
        batchnorm3_1_30_ = self.batchnorm3_1_30_(conv3_1_30_)
        relu3_1_30_ = self.relu3_1_30_(batchnorm3_1_30_)
        conv4_1_30_padding = self.conv4_1_30_padding(relu3_1_30_)
        conv4_1_30_ = self.conv4_1_30_(conv4_1_30_padding)
        batchnorm4_1_30_ = self.batchnorm4_1_30_(conv4_1_30_)
        relu4_1_30_ = self.relu4_1_30_(batchnorm4_1_30_)
        conv5_1_30_ = self.conv5_1_30_(relu4_1_30_)
        batchnorm5_1_30_ = self.batchnorm5_1_30_(conv5_1_30_)
        conv3_1_31_ = self.conv3_1_31_(pool1_)
        batchnorm3_1_31_ = self.batchnorm3_1_31_(conv3_1_31_)
        relu3_1_31_ = self.relu3_1_31_(batchnorm3_1_31_)
        conv4_1_31_padding = self.conv4_1_31_padding(relu3_1_31_)
        conv4_1_31_ = self.conv4_1_31_(conv4_1_31_padding)
        batchnorm4_1_31_ = self.batchnorm4_1_31_(conv4_1_31_)
        relu4_1_31_ = self.relu4_1_31_(batchnorm4_1_31_)
        conv5_1_31_ = self.conv5_1_31_(relu4_1_31_)
        batchnorm5_1_31_ = self.batchnorm5_1_31_(conv5_1_31_)
        conv3_1_32_ = self.conv3_1_32_(pool1_)
        batchnorm3_1_32_ = self.batchnorm3_1_32_(conv3_1_32_)
        relu3_1_32_ = self.relu3_1_32_(batchnorm3_1_32_)
        conv4_1_32_padding = self.conv4_1_32_padding(relu3_1_32_)
        conv4_1_32_ = self.conv4_1_32_(conv4_1_32_padding)
        batchnorm4_1_32_ = self.batchnorm4_1_32_(conv4_1_32_)
        relu4_1_32_ = self.relu4_1_32_(batchnorm4_1_32_)
        conv5_1_32_ = self.conv5_1_32_(relu4_1_32_)
        batchnorm5_1_32_ = self.batchnorm5_1_32_(conv5_1_32_)
        add6_1_ = batchnorm5_1_1_ + batchnorm5_1_2_ + batchnorm5_1_3_ + batchnorm5_1_4_ + batchnorm5_1_5_ + batchnorm5_1_6_ + batchnorm5_1_7_ + batchnorm5_1_8_ + batchnorm5_1_9_ + batchnorm5_1_10_ + batchnorm5_1_11_ + batchnorm5_1_12_ + batchnorm5_1_13_ + batchnorm5_1_14_ + batchnorm5_1_15_ + batchnorm5_1_16_ + batchnorm5_1_17_ + batchnorm5_1_18_ + batchnorm5_1_19_ + batchnorm5_1_20_ + batchnorm5_1_21_ + batchnorm5_1_22_ + batchnorm5_1_23_ + batchnorm5_1_24_ + batchnorm5_1_25_ + batchnorm5_1_26_ + batchnorm5_1_27_ + batchnorm5_1_28_ + batchnorm5_1_29_ + batchnorm5_1_30_ + batchnorm5_1_31_ + batchnorm5_1_32_
        conv2_2_ = self.conv2_2_(pool1_)
        batchnorm2_2_ = self.batchnorm2_2_(conv2_2_)
        add7_ = add6_1_ + batchnorm2_2_
        relu7_ = self.relu7_(add7_)
        conv9_1_1_ = self.conv9_1_1_(relu7_)
        batchnorm9_1_1_ = self.batchnorm9_1_1_(conv9_1_1_)
        relu9_1_1_ = self.relu9_1_1_(batchnorm9_1_1_)
        conv10_1_1_padding = self.conv10_1_1_padding(relu9_1_1_)
        conv10_1_1_ = self.conv10_1_1_(conv10_1_1_padding)
        batchnorm10_1_1_ = self.batchnorm10_1_1_(conv10_1_1_)
        relu10_1_1_ = self.relu10_1_1_(batchnorm10_1_1_)
        conv11_1_1_ = self.conv11_1_1_(relu10_1_1_)
        batchnorm11_1_1_ = self.batchnorm11_1_1_(conv11_1_1_)
        conv9_1_2_ = self.conv9_1_2_(relu7_)
        batchnorm9_1_2_ = self.batchnorm9_1_2_(conv9_1_2_)
        relu9_1_2_ = self.relu9_1_2_(batchnorm9_1_2_)
        conv10_1_2_padding = self.conv10_1_2_padding(relu9_1_2_)
        conv10_1_2_ = self.conv10_1_2_(conv10_1_2_padding)
        batchnorm10_1_2_ = self.batchnorm10_1_2_(conv10_1_2_)
        relu10_1_2_ = self.relu10_1_2_(batchnorm10_1_2_)
        conv11_1_2_ = self.conv11_1_2_(relu10_1_2_)
        batchnorm11_1_2_ = self.batchnorm11_1_2_(conv11_1_2_)
        conv9_1_3_ = self.conv9_1_3_(relu7_)
        batchnorm9_1_3_ = self.batchnorm9_1_3_(conv9_1_3_)
        relu9_1_3_ = self.relu9_1_3_(batchnorm9_1_3_)
        conv10_1_3_padding = self.conv10_1_3_padding(relu9_1_3_)
        conv10_1_3_ = self.conv10_1_3_(conv10_1_3_padding)
        batchnorm10_1_3_ = self.batchnorm10_1_3_(conv10_1_3_)
        relu10_1_3_ = self.relu10_1_3_(batchnorm10_1_3_)
        conv11_1_3_ = self.conv11_1_3_(relu10_1_3_)
        batchnorm11_1_3_ = self.batchnorm11_1_3_(conv11_1_3_)
        conv9_1_4_ = self.conv9_1_4_(relu7_)
        batchnorm9_1_4_ = self.batchnorm9_1_4_(conv9_1_4_)
        relu9_1_4_ = self.relu9_1_4_(batchnorm9_1_4_)
        conv10_1_4_padding = self.conv10_1_4_padding(relu9_1_4_)
        conv10_1_4_ = self.conv10_1_4_(conv10_1_4_padding)
        batchnorm10_1_4_ = self.batchnorm10_1_4_(conv10_1_4_)
        relu10_1_4_ = self.relu10_1_4_(batchnorm10_1_4_)
        conv11_1_4_ = self.conv11_1_4_(relu10_1_4_)
        batchnorm11_1_4_ = self.batchnorm11_1_4_(conv11_1_4_)
        conv9_1_5_ = self.conv9_1_5_(relu7_)
        batchnorm9_1_5_ = self.batchnorm9_1_5_(conv9_1_5_)
        relu9_1_5_ = self.relu9_1_5_(batchnorm9_1_5_)
        conv10_1_5_padding = self.conv10_1_5_padding(relu9_1_5_)
        conv10_1_5_ = self.conv10_1_5_(conv10_1_5_padding)
        batchnorm10_1_5_ = self.batchnorm10_1_5_(conv10_1_5_)
        relu10_1_5_ = self.relu10_1_5_(batchnorm10_1_5_)
        conv11_1_5_ = self.conv11_1_5_(relu10_1_5_)
        batchnorm11_1_5_ = self.batchnorm11_1_5_(conv11_1_5_)
        conv9_1_6_ = self.conv9_1_6_(relu7_)
        batchnorm9_1_6_ = self.batchnorm9_1_6_(conv9_1_6_)
        relu9_1_6_ = self.relu9_1_6_(batchnorm9_1_6_)
        conv10_1_6_padding = self.conv10_1_6_padding(relu9_1_6_)
        conv10_1_6_ = self.conv10_1_6_(conv10_1_6_padding)
        batchnorm10_1_6_ = self.batchnorm10_1_6_(conv10_1_6_)
        relu10_1_6_ = self.relu10_1_6_(batchnorm10_1_6_)
        conv11_1_6_ = self.conv11_1_6_(relu10_1_6_)
        batchnorm11_1_6_ = self.batchnorm11_1_6_(conv11_1_6_)
        conv9_1_7_ = self.conv9_1_7_(relu7_)
        batchnorm9_1_7_ = self.batchnorm9_1_7_(conv9_1_7_)
        relu9_1_7_ = self.relu9_1_7_(batchnorm9_1_7_)
        conv10_1_7_padding = self.conv10_1_7_padding(relu9_1_7_)
        conv10_1_7_ = self.conv10_1_7_(conv10_1_7_padding)
        batchnorm10_1_7_ = self.batchnorm10_1_7_(conv10_1_7_)
        relu10_1_7_ = self.relu10_1_7_(batchnorm10_1_7_)
        conv11_1_7_ = self.conv11_1_7_(relu10_1_7_)
        batchnorm11_1_7_ = self.batchnorm11_1_7_(conv11_1_7_)
        conv9_1_8_ = self.conv9_1_8_(relu7_)
        batchnorm9_1_8_ = self.batchnorm9_1_8_(conv9_1_8_)
        relu9_1_8_ = self.relu9_1_8_(batchnorm9_1_8_)
        conv10_1_8_padding = self.conv10_1_8_padding(relu9_1_8_)
        conv10_1_8_ = self.conv10_1_8_(conv10_1_8_padding)
        batchnorm10_1_8_ = self.batchnorm10_1_8_(conv10_1_8_)
        relu10_1_8_ = self.relu10_1_8_(batchnorm10_1_8_)
        conv11_1_8_ = self.conv11_1_8_(relu10_1_8_)
        batchnorm11_1_8_ = self.batchnorm11_1_8_(conv11_1_8_)
        conv9_1_9_ = self.conv9_1_9_(relu7_)
        batchnorm9_1_9_ = self.batchnorm9_1_9_(conv9_1_9_)
        relu9_1_9_ = self.relu9_1_9_(batchnorm9_1_9_)
        conv10_1_9_padding = self.conv10_1_9_padding(relu9_1_9_)
        conv10_1_9_ = self.conv10_1_9_(conv10_1_9_padding)
        batchnorm10_1_9_ = self.batchnorm10_1_9_(conv10_1_9_)
        relu10_1_9_ = self.relu10_1_9_(batchnorm10_1_9_)
        conv11_1_9_ = self.conv11_1_9_(relu10_1_9_)
        batchnorm11_1_9_ = self.batchnorm11_1_9_(conv11_1_9_)
        conv9_1_10_ = self.conv9_1_10_(relu7_)
        batchnorm9_1_10_ = self.batchnorm9_1_10_(conv9_1_10_)
        relu9_1_10_ = self.relu9_1_10_(batchnorm9_1_10_)
        conv10_1_10_padding = self.conv10_1_10_padding(relu9_1_10_)
        conv10_1_10_ = self.conv10_1_10_(conv10_1_10_padding)
        batchnorm10_1_10_ = self.batchnorm10_1_10_(conv10_1_10_)
        relu10_1_10_ = self.relu10_1_10_(batchnorm10_1_10_)
        conv11_1_10_ = self.conv11_1_10_(relu10_1_10_)
        batchnorm11_1_10_ = self.batchnorm11_1_10_(conv11_1_10_)
        conv9_1_11_ = self.conv9_1_11_(relu7_)
        batchnorm9_1_11_ = self.batchnorm9_1_11_(conv9_1_11_)
        relu9_1_11_ = self.relu9_1_11_(batchnorm9_1_11_)
        conv10_1_11_padding = self.conv10_1_11_padding(relu9_1_11_)
        conv10_1_11_ = self.conv10_1_11_(conv10_1_11_padding)
        batchnorm10_1_11_ = self.batchnorm10_1_11_(conv10_1_11_)
        relu10_1_11_ = self.relu10_1_11_(batchnorm10_1_11_)
        conv11_1_11_ = self.conv11_1_11_(relu10_1_11_)
        batchnorm11_1_11_ = self.batchnorm11_1_11_(conv11_1_11_)
        conv9_1_12_ = self.conv9_1_12_(relu7_)
        batchnorm9_1_12_ = self.batchnorm9_1_12_(conv9_1_12_)
        relu9_1_12_ = self.relu9_1_12_(batchnorm9_1_12_)
        conv10_1_12_padding = self.conv10_1_12_padding(relu9_1_12_)
        conv10_1_12_ = self.conv10_1_12_(conv10_1_12_padding)
        batchnorm10_1_12_ = self.batchnorm10_1_12_(conv10_1_12_)
        relu10_1_12_ = self.relu10_1_12_(batchnorm10_1_12_)
        conv11_1_12_ = self.conv11_1_12_(relu10_1_12_)
        batchnorm11_1_12_ = self.batchnorm11_1_12_(conv11_1_12_)
        conv9_1_13_ = self.conv9_1_13_(relu7_)
        batchnorm9_1_13_ = self.batchnorm9_1_13_(conv9_1_13_)
        relu9_1_13_ = self.relu9_1_13_(batchnorm9_1_13_)
        conv10_1_13_padding = self.conv10_1_13_padding(relu9_1_13_)
        conv10_1_13_ = self.conv10_1_13_(conv10_1_13_padding)
        batchnorm10_1_13_ = self.batchnorm10_1_13_(conv10_1_13_)
        relu10_1_13_ = self.relu10_1_13_(batchnorm10_1_13_)
        conv11_1_13_ = self.conv11_1_13_(relu10_1_13_)
        batchnorm11_1_13_ = self.batchnorm11_1_13_(conv11_1_13_)
        conv9_1_14_ = self.conv9_1_14_(relu7_)
        batchnorm9_1_14_ = self.batchnorm9_1_14_(conv9_1_14_)
        relu9_1_14_ = self.relu9_1_14_(batchnorm9_1_14_)
        conv10_1_14_padding = self.conv10_1_14_padding(relu9_1_14_)
        conv10_1_14_ = self.conv10_1_14_(conv10_1_14_padding)
        batchnorm10_1_14_ = self.batchnorm10_1_14_(conv10_1_14_)
        relu10_1_14_ = self.relu10_1_14_(batchnorm10_1_14_)
        conv11_1_14_ = self.conv11_1_14_(relu10_1_14_)
        batchnorm11_1_14_ = self.batchnorm11_1_14_(conv11_1_14_)
        conv9_1_15_ = self.conv9_1_15_(relu7_)
        batchnorm9_1_15_ = self.batchnorm9_1_15_(conv9_1_15_)
        relu9_1_15_ = self.relu9_1_15_(batchnorm9_1_15_)
        conv10_1_15_padding = self.conv10_1_15_padding(relu9_1_15_)
        conv10_1_15_ = self.conv10_1_15_(conv10_1_15_padding)
        batchnorm10_1_15_ = self.batchnorm10_1_15_(conv10_1_15_)
        relu10_1_15_ = self.relu10_1_15_(batchnorm10_1_15_)
        conv11_1_15_ = self.conv11_1_15_(relu10_1_15_)
        batchnorm11_1_15_ = self.batchnorm11_1_15_(conv11_1_15_)
        conv9_1_16_ = self.conv9_1_16_(relu7_)
        batchnorm9_1_16_ = self.batchnorm9_1_16_(conv9_1_16_)
        relu9_1_16_ = self.relu9_1_16_(batchnorm9_1_16_)
        conv10_1_16_padding = self.conv10_1_16_padding(relu9_1_16_)
        conv10_1_16_ = self.conv10_1_16_(conv10_1_16_padding)
        batchnorm10_1_16_ = self.batchnorm10_1_16_(conv10_1_16_)
        relu10_1_16_ = self.relu10_1_16_(batchnorm10_1_16_)
        conv11_1_16_ = self.conv11_1_16_(relu10_1_16_)
        batchnorm11_1_16_ = self.batchnorm11_1_16_(conv11_1_16_)
        conv9_1_17_ = self.conv9_1_17_(relu7_)
        batchnorm9_1_17_ = self.batchnorm9_1_17_(conv9_1_17_)
        relu9_1_17_ = self.relu9_1_17_(batchnorm9_1_17_)
        conv10_1_17_padding = self.conv10_1_17_padding(relu9_1_17_)
        conv10_1_17_ = self.conv10_1_17_(conv10_1_17_padding)
        batchnorm10_1_17_ = self.batchnorm10_1_17_(conv10_1_17_)
        relu10_1_17_ = self.relu10_1_17_(batchnorm10_1_17_)
        conv11_1_17_ = self.conv11_1_17_(relu10_1_17_)
        batchnorm11_1_17_ = self.batchnorm11_1_17_(conv11_1_17_)
        conv9_1_18_ = self.conv9_1_18_(relu7_)
        batchnorm9_1_18_ = self.batchnorm9_1_18_(conv9_1_18_)
        relu9_1_18_ = self.relu9_1_18_(batchnorm9_1_18_)
        conv10_1_18_padding = self.conv10_1_18_padding(relu9_1_18_)
        conv10_1_18_ = self.conv10_1_18_(conv10_1_18_padding)
        batchnorm10_1_18_ = self.batchnorm10_1_18_(conv10_1_18_)
        relu10_1_18_ = self.relu10_1_18_(batchnorm10_1_18_)
        conv11_1_18_ = self.conv11_1_18_(relu10_1_18_)
        batchnorm11_1_18_ = self.batchnorm11_1_18_(conv11_1_18_)
        conv9_1_19_ = self.conv9_1_19_(relu7_)
        batchnorm9_1_19_ = self.batchnorm9_1_19_(conv9_1_19_)
        relu9_1_19_ = self.relu9_1_19_(batchnorm9_1_19_)
        conv10_1_19_padding = self.conv10_1_19_padding(relu9_1_19_)
        conv10_1_19_ = self.conv10_1_19_(conv10_1_19_padding)
        batchnorm10_1_19_ = self.batchnorm10_1_19_(conv10_1_19_)
        relu10_1_19_ = self.relu10_1_19_(batchnorm10_1_19_)
        conv11_1_19_ = self.conv11_1_19_(relu10_1_19_)
        batchnorm11_1_19_ = self.batchnorm11_1_19_(conv11_1_19_)
        conv9_1_20_ = self.conv9_1_20_(relu7_)
        batchnorm9_1_20_ = self.batchnorm9_1_20_(conv9_1_20_)
        relu9_1_20_ = self.relu9_1_20_(batchnorm9_1_20_)
        conv10_1_20_padding = self.conv10_1_20_padding(relu9_1_20_)
        conv10_1_20_ = self.conv10_1_20_(conv10_1_20_padding)
        batchnorm10_1_20_ = self.batchnorm10_1_20_(conv10_1_20_)
        relu10_1_20_ = self.relu10_1_20_(batchnorm10_1_20_)
        conv11_1_20_ = self.conv11_1_20_(relu10_1_20_)
        batchnorm11_1_20_ = self.batchnorm11_1_20_(conv11_1_20_)
        conv9_1_21_ = self.conv9_1_21_(relu7_)
        batchnorm9_1_21_ = self.batchnorm9_1_21_(conv9_1_21_)
        relu9_1_21_ = self.relu9_1_21_(batchnorm9_1_21_)
        conv10_1_21_padding = self.conv10_1_21_padding(relu9_1_21_)
        conv10_1_21_ = self.conv10_1_21_(conv10_1_21_padding)
        batchnorm10_1_21_ = self.batchnorm10_1_21_(conv10_1_21_)
        relu10_1_21_ = self.relu10_1_21_(batchnorm10_1_21_)
        conv11_1_21_ = self.conv11_1_21_(relu10_1_21_)
        batchnorm11_1_21_ = self.batchnorm11_1_21_(conv11_1_21_)
        conv9_1_22_ = self.conv9_1_22_(relu7_)
        batchnorm9_1_22_ = self.batchnorm9_1_22_(conv9_1_22_)
        relu9_1_22_ = self.relu9_1_22_(batchnorm9_1_22_)
        conv10_1_22_padding = self.conv10_1_22_padding(relu9_1_22_)
        conv10_1_22_ = self.conv10_1_22_(conv10_1_22_padding)
        batchnorm10_1_22_ = self.batchnorm10_1_22_(conv10_1_22_)
        relu10_1_22_ = self.relu10_1_22_(batchnorm10_1_22_)
        conv11_1_22_ = self.conv11_1_22_(relu10_1_22_)
        batchnorm11_1_22_ = self.batchnorm11_1_22_(conv11_1_22_)
        conv9_1_23_ = self.conv9_1_23_(relu7_)
        batchnorm9_1_23_ = self.batchnorm9_1_23_(conv9_1_23_)
        relu9_1_23_ = self.relu9_1_23_(batchnorm9_1_23_)
        conv10_1_23_padding = self.conv10_1_23_padding(relu9_1_23_)
        conv10_1_23_ = self.conv10_1_23_(conv10_1_23_padding)
        batchnorm10_1_23_ = self.batchnorm10_1_23_(conv10_1_23_)
        relu10_1_23_ = self.relu10_1_23_(batchnorm10_1_23_)
        conv11_1_23_ = self.conv11_1_23_(relu10_1_23_)
        batchnorm11_1_23_ = self.batchnorm11_1_23_(conv11_1_23_)
        conv9_1_24_ = self.conv9_1_24_(relu7_)
        batchnorm9_1_24_ = self.batchnorm9_1_24_(conv9_1_24_)
        relu9_1_24_ = self.relu9_1_24_(batchnorm9_1_24_)
        conv10_1_24_padding = self.conv10_1_24_padding(relu9_1_24_)
        conv10_1_24_ = self.conv10_1_24_(conv10_1_24_padding)
        batchnorm10_1_24_ = self.batchnorm10_1_24_(conv10_1_24_)
        relu10_1_24_ = self.relu10_1_24_(batchnorm10_1_24_)
        conv11_1_24_ = self.conv11_1_24_(relu10_1_24_)
        batchnorm11_1_24_ = self.batchnorm11_1_24_(conv11_1_24_)
        conv9_1_25_ = self.conv9_1_25_(relu7_)
        batchnorm9_1_25_ = self.batchnorm9_1_25_(conv9_1_25_)
        relu9_1_25_ = self.relu9_1_25_(batchnorm9_1_25_)
        conv10_1_25_padding = self.conv10_1_25_padding(relu9_1_25_)
        conv10_1_25_ = self.conv10_1_25_(conv10_1_25_padding)
        batchnorm10_1_25_ = self.batchnorm10_1_25_(conv10_1_25_)
        relu10_1_25_ = self.relu10_1_25_(batchnorm10_1_25_)
        conv11_1_25_ = self.conv11_1_25_(relu10_1_25_)
        batchnorm11_1_25_ = self.batchnorm11_1_25_(conv11_1_25_)
        conv9_1_26_ = self.conv9_1_26_(relu7_)
        batchnorm9_1_26_ = self.batchnorm9_1_26_(conv9_1_26_)
        relu9_1_26_ = self.relu9_1_26_(batchnorm9_1_26_)
        conv10_1_26_padding = self.conv10_1_26_padding(relu9_1_26_)
        conv10_1_26_ = self.conv10_1_26_(conv10_1_26_padding)
        batchnorm10_1_26_ = self.batchnorm10_1_26_(conv10_1_26_)
        relu10_1_26_ = self.relu10_1_26_(batchnorm10_1_26_)
        conv11_1_26_ = self.conv11_1_26_(relu10_1_26_)
        batchnorm11_1_26_ = self.batchnorm11_1_26_(conv11_1_26_)
        conv9_1_27_ = self.conv9_1_27_(relu7_)
        batchnorm9_1_27_ = self.batchnorm9_1_27_(conv9_1_27_)
        relu9_1_27_ = self.relu9_1_27_(batchnorm9_1_27_)
        conv10_1_27_padding = self.conv10_1_27_padding(relu9_1_27_)
        conv10_1_27_ = self.conv10_1_27_(conv10_1_27_padding)
        batchnorm10_1_27_ = self.batchnorm10_1_27_(conv10_1_27_)
        relu10_1_27_ = self.relu10_1_27_(batchnorm10_1_27_)
        conv11_1_27_ = self.conv11_1_27_(relu10_1_27_)
        batchnorm11_1_27_ = self.batchnorm11_1_27_(conv11_1_27_)
        conv9_1_28_ = self.conv9_1_28_(relu7_)
        batchnorm9_1_28_ = self.batchnorm9_1_28_(conv9_1_28_)
        relu9_1_28_ = self.relu9_1_28_(batchnorm9_1_28_)
        conv10_1_28_padding = self.conv10_1_28_padding(relu9_1_28_)
        conv10_1_28_ = self.conv10_1_28_(conv10_1_28_padding)
        batchnorm10_1_28_ = self.batchnorm10_1_28_(conv10_1_28_)
        relu10_1_28_ = self.relu10_1_28_(batchnorm10_1_28_)
        conv11_1_28_ = self.conv11_1_28_(relu10_1_28_)
        batchnorm11_1_28_ = self.batchnorm11_1_28_(conv11_1_28_)
        conv9_1_29_ = self.conv9_1_29_(relu7_)
        batchnorm9_1_29_ = self.batchnorm9_1_29_(conv9_1_29_)
        relu9_1_29_ = self.relu9_1_29_(batchnorm9_1_29_)
        conv10_1_29_padding = self.conv10_1_29_padding(relu9_1_29_)
        conv10_1_29_ = self.conv10_1_29_(conv10_1_29_padding)
        batchnorm10_1_29_ = self.batchnorm10_1_29_(conv10_1_29_)
        relu10_1_29_ = self.relu10_1_29_(batchnorm10_1_29_)
        conv11_1_29_ = self.conv11_1_29_(relu10_1_29_)
        batchnorm11_1_29_ = self.batchnorm11_1_29_(conv11_1_29_)
        conv9_1_30_ = self.conv9_1_30_(relu7_)
        batchnorm9_1_30_ = self.batchnorm9_1_30_(conv9_1_30_)
        relu9_1_30_ = self.relu9_1_30_(batchnorm9_1_30_)
        conv10_1_30_padding = self.conv10_1_30_padding(relu9_1_30_)
        conv10_1_30_ = self.conv10_1_30_(conv10_1_30_padding)
        batchnorm10_1_30_ = self.batchnorm10_1_30_(conv10_1_30_)
        relu10_1_30_ = self.relu10_1_30_(batchnorm10_1_30_)
        conv11_1_30_ = self.conv11_1_30_(relu10_1_30_)
        batchnorm11_1_30_ = self.batchnorm11_1_30_(conv11_1_30_)
        conv9_1_31_ = self.conv9_1_31_(relu7_)
        batchnorm9_1_31_ = self.batchnorm9_1_31_(conv9_1_31_)
        relu9_1_31_ = self.relu9_1_31_(batchnorm9_1_31_)
        conv10_1_31_padding = self.conv10_1_31_padding(relu9_1_31_)
        conv10_1_31_ = self.conv10_1_31_(conv10_1_31_padding)
        batchnorm10_1_31_ = self.batchnorm10_1_31_(conv10_1_31_)
        relu10_1_31_ = self.relu10_1_31_(batchnorm10_1_31_)
        conv11_1_31_ = self.conv11_1_31_(relu10_1_31_)
        batchnorm11_1_31_ = self.batchnorm11_1_31_(conv11_1_31_)
        conv9_1_32_ = self.conv9_1_32_(relu7_)
        batchnorm9_1_32_ = self.batchnorm9_1_32_(conv9_1_32_)
        relu9_1_32_ = self.relu9_1_32_(batchnorm9_1_32_)
        conv10_1_32_padding = self.conv10_1_32_padding(relu9_1_32_)
        conv10_1_32_ = self.conv10_1_32_(conv10_1_32_padding)
        batchnorm10_1_32_ = self.batchnorm10_1_32_(conv10_1_32_)
        relu10_1_32_ = self.relu10_1_32_(batchnorm10_1_32_)
        conv11_1_32_ = self.conv11_1_32_(relu10_1_32_)
        batchnorm11_1_32_ = self.batchnorm11_1_32_(conv11_1_32_)
        add12_1_ = batchnorm11_1_1_ + batchnorm11_1_2_ + batchnorm11_1_3_ + batchnorm11_1_4_ + batchnorm11_1_5_ + batchnorm11_1_6_ + batchnorm11_1_7_ + batchnorm11_1_8_ + batchnorm11_1_9_ + batchnorm11_1_10_ + batchnorm11_1_11_ + batchnorm11_1_12_ + batchnorm11_1_13_ + batchnorm11_1_14_ + batchnorm11_1_15_ + batchnorm11_1_16_ + batchnorm11_1_17_ + batchnorm11_1_18_ + batchnorm11_1_19_ + batchnorm11_1_20_ + batchnorm11_1_21_ + batchnorm11_1_22_ + batchnorm11_1_23_ + batchnorm11_1_24_ + batchnorm11_1_25_ + batchnorm11_1_26_ + batchnorm11_1_27_ + batchnorm11_1_28_ + batchnorm11_1_29_ + batchnorm11_1_30_ + batchnorm11_1_31_ + batchnorm11_1_32_
        conv8_2_ = self.conv8_2_(relu7_)
        batchnorm8_2_ = self.batchnorm8_2_(conv8_2_)
        add13_ = add12_1_ + batchnorm8_2_
        relu13_ = self.relu13_(add13_)
        conv15_1_1_ = self.conv15_1_1_(relu13_)
        batchnorm15_1_1_ = self.batchnorm15_1_1_(conv15_1_1_)
        relu15_1_1_ = self.relu15_1_1_(batchnorm15_1_1_)
        conv16_1_1_padding = self.conv16_1_1_padding(relu15_1_1_)
        conv16_1_1_ = self.conv16_1_1_(conv16_1_1_padding)
        batchnorm16_1_1_ = self.batchnorm16_1_1_(conv16_1_1_)
        relu16_1_1_ = self.relu16_1_1_(batchnorm16_1_1_)
        conv17_1_1_ = self.conv17_1_1_(relu16_1_1_)
        batchnorm17_1_1_ = self.batchnorm17_1_1_(conv17_1_1_)
        conv15_1_2_ = self.conv15_1_2_(relu13_)
        batchnorm15_1_2_ = self.batchnorm15_1_2_(conv15_1_2_)
        relu15_1_2_ = self.relu15_1_2_(batchnorm15_1_2_)
        conv16_1_2_padding = self.conv16_1_2_padding(relu15_1_2_)
        conv16_1_2_ = self.conv16_1_2_(conv16_1_2_padding)
        batchnorm16_1_2_ = self.batchnorm16_1_2_(conv16_1_2_)
        relu16_1_2_ = self.relu16_1_2_(batchnorm16_1_2_)
        conv17_1_2_ = self.conv17_1_2_(relu16_1_2_)
        batchnorm17_1_2_ = self.batchnorm17_1_2_(conv17_1_2_)
        conv15_1_3_ = self.conv15_1_3_(relu13_)
        batchnorm15_1_3_ = self.batchnorm15_1_3_(conv15_1_3_)
        relu15_1_3_ = self.relu15_1_3_(batchnorm15_1_3_)
        conv16_1_3_padding = self.conv16_1_3_padding(relu15_1_3_)
        conv16_1_3_ = self.conv16_1_3_(conv16_1_3_padding)
        batchnorm16_1_3_ = self.batchnorm16_1_3_(conv16_1_3_)
        relu16_1_3_ = self.relu16_1_3_(batchnorm16_1_3_)
        conv17_1_3_ = self.conv17_1_3_(relu16_1_3_)
        batchnorm17_1_3_ = self.batchnorm17_1_3_(conv17_1_3_)
        conv15_1_4_ = self.conv15_1_4_(relu13_)
        batchnorm15_1_4_ = self.batchnorm15_1_4_(conv15_1_4_)
        relu15_1_4_ = self.relu15_1_4_(batchnorm15_1_4_)
        conv16_1_4_padding = self.conv16_1_4_padding(relu15_1_4_)
        conv16_1_4_ = self.conv16_1_4_(conv16_1_4_padding)
        batchnorm16_1_4_ = self.batchnorm16_1_4_(conv16_1_4_)
        relu16_1_4_ = self.relu16_1_4_(batchnorm16_1_4_)
        conv17_1_4_ = self.conv17_1_4_(relu16_1_4_)
        batchnorm17_1_4_ = self.batchnorm17_1_4_(conv17_1_4_)
        conv15_1_5_ = self.conv15_1_5_(relu13_)
        batchnorm15_1_5_ = self.batchnorm15_1_5_(conv15_1_5_)
        relu15_1_5_ = self.relu15_1_5_(batchnorm15_1_5_)
        conv16_1_5_padding = self.conv16_1_5_padding(relu15_1_5_)
        conv16_1_5_ = self.conv16_1_5_(conv16_1_5_padding)
        batchnorm16_1_5_ = self.batchnorm16_1_5_(conv16_1_5_)
        relu16_1_5_ = self.relu16_1_5_(batchnorm16_1_5_)
        conv17_1_5_ = self.conv17_1_5_(relu16_1_5_)
        batchnorm17_1_5_ = self.batchnorm17_1_5_(conv17_1_5_)
        conv15_1_6_ = self.conv15_1_6_(relu13_)
        batchnorm15_1_6_ = self.batchnorm15_1_6_(conv15_1_6_)
        relu15_1_6_ = self.relu15_1_6_(batchnorm15_1_6_)
        conv16_1_6_padding = self.conv16_1_6_padding(relu15_1_6_)
        conv16_1_6_ = self.conv16_1_6_(conv16_1_6_padding)
        batchnorm16_1_6_ = self.batchnorm16_1_6_(conv16_1_6_)
        relu16_1_6_ = self.relu16_1_6_(batchnorm16_1_6_)
        conv17_1_6_ = self.conv17_1_6_(relu16_1_6_)
        batchnorm17_1_6_ = self.batchnorm17_1_6_(conv17_1_6_)
        conv15_1_7_ = self.conv15_1_7_(relu13_)
        batchnorm15_1_7_ = self.batchnorm15_1_7_(conv15_1_7_)
        relu15_1_7_ = self.relu15_1_7_(batchnorm15_1_7_)
        conv16_1_7_padding = self.conv16_1_7_padding(relu15_1_7_)
        conv16_1_7_ = self.conv16_1_7_(conv16_1_7_padding)
        batchnorm16_1_7_ = self.batchnorm16_1_7_(conv16_1_7_)
        relu16_1_7_ = self.relu16_1_7_(batchnorm16_1_7_)
        conv17_1_7_ = self.conv17_1_7_(relu16_1_7_)
        batchnorm17_1_7_ = self.batchnorm17_1_7_(conv17_1_7_)
        conv15_1_8_ = self.conv15_1_8_(relu13_)
        batchnorm15_1_8_ = self.batchnorm15_1_8_(conv15_1_8_)
        relu15_1_8_ = self.relu15_1_8_(batchnorm15_1_8_)
        conv16_1_8_padding = self.conv16_1_8_padding(relu15_1_8_)
        conv16_1_8_ = self.conv16_1_8_(conv16_1_8_padding)
        batchnorm16_1_8_ = self.batchnorm16_1_8_(conv16_1_8_)
        relu16_1_8_ = self.relu16_1_8_(batchnorm16_1_8_)
        conv17_1_8_ = self.conv17_1_8_(relu16_1_8_)
        batchnorm17_1_8_ = self.batchnorm17_1_8_(conv17_1_8_)
        conv15_1_9_ = self.conv15_1_9_(relu13_)
        batchnorm15_1_9_ = self.batchnorm15_1_9_(conv15_1_9_)
        relu15_1_9_ = self.relu15_1_9_(batchnorm15_1_9_)
        conv16_1_9_padding = self.conv16_1_9_padding(relu15_1_9_)
        conv16_1_9_ = self.conv16_1_9_(conv16_1_9_padding)
        batchnorm16_1_9_ = self.batchnorm16_1_9_(conv16_1_9_)
        relu16_1_9_ = self.relu16_1_9_(batchnorm16_1_9_)
        conv17_1_9_ = self.conv17_1_9_(relu16_1_9_)
        batchnorm17_1_9_ = self.batchnorm17_1_9_(conv17_1_9_)
        conv15_1_10_ = self.conv15_1_10_(relu13_)
        batchnorm15_1_10_ = self.batchnorm15_1_10_(conv15_1_10_)
        relu15_1_10_ = self.relu15_1_10_(batchnorm15_1_10_)
        conv16_1_10_padding = self.conv16_1_10_padding(relu15_1_10_)
        conv16_1_10_ = self.conv16_1_10_(conv16_1_10_padding)
        batchnorm16_1_10_ = self.batchnorm16_1_10_(conv16_1_10_)
        relu16_1_10_ = self.relu16_1_10_(batchnorm16_1_10_)
        conv17_1_10_ = self.conv17_1_10_(relu16_1_10_)
        batchnorm17_1_10_ = self.batchnorm17_1_10_(conv17_1_10_)
        conv15_1_11_ = self.conv15_1_11_(relu13_)
        batchnorm15_1_11_ = self.batchnorm15_1_11_(conv15_1_11_)
        relu15_1_11_ = self.relu15_1_11_(batchnorm15_1_11_)
        conv16_1_11_padding = self.conv16_1_11_padding(relu15_1_11_)
        conv16_1_11_ = self.conv16_1_11_(conv16_1_11_padding)
        batchnorm16_1_11_ = self.batchnorm16_1_11_(conv16_1_11_)
        relu16_1_11_ = self.relu16_1_11_(batchnorm16_1_11_)
        conv17_1_11_ = self.conv17_1_11_(relu16_1_11_)
        batchnorm17_1_11_ = self.batchnorm17_1_11_(conv17_1_11_)
        conv15_1_12_ = self.conv15_1_12_(relu13_)
        batchnorm15_1_12_ = self.batchnorm15_1_12_(conv15_1_12_)
        relu15_1_12_ = self.relu15_1_12_(batchnorm15_1_12_)
        conv16_1_12_padding = self.conv16_1_12_padding(relu15_1_12_)
        conv16_1_12_ = self.conv16_1_12_(conv16_1_12_padding)
        batchnorm16_1_12_ = self.batchnorm16_1_12_(conv16_1_12_)
        relu16_1_12_ = self.relu16_1_12_(batchnorm16_1_12_)
        conv17_1_12_ = self.conv17_1_12_(relu16_1_12_)
        batchnorm17_1_12_ = self.batchnorm17_1_12_(conv17_1_12_)
        conv15_1_13_ = self.conv15_1_13_(relu13_)
        batchnorm15_1_13_ = self.batchnorm15_1_13_(conv15_1_13_)
        relu15_1_13_ = self.relu15_1_13_(batchnorm15_1_13_)
        conv16_1_13_padding = self.conv16_1_13_padding(relu15_1_13_)
        conv16_1_13_ = self.conv16_1_13_(conv16_1_13_padding)
        batchnorm16_1_13_ = self.batchnorm16_1_13_(conv16_1_13_)
        relu16_1_13_ = self.relu16_1_13_(batchnorm16_1_13_)
        conv17_1_13_ = self.conv17_1_13_(relu16_1_13_)
        batchnorm17_1_13_ = self.batchnorm17_1_13_(conv17_1_13_)
        conv15_1_14_ = self.conv15_1_14_(relu13_)
        batchnorm15_1_14_ = self.batchnorm15_1_14_(conv15_1_14_)
        relu15_1_14_ = self.relu15_1_14_(batchnorm15_1_14_)
        conv16_1_14_padding = self.conv16_1_14_padding(relu15_1_14_)
        conv16_1_14_ = self.conv16_1_14_(conv16_1_14_padding)
        batchnorm16_1_14_ = self.batchnorm16_1_14_(conv16_1_14_)
        relu16_1_14_ = self.relu16_1_14_(batchnorm16_1_14_)
        conv17_1_14_ = self.conv17_1_14_(relu16_1_14_)
        batchnorm17_1_14_ = self.batchnorm17_1_14_(conv17_1_14_)
        conv15_1_15_ = self.conv15_1_15_(relu13_)
        batchnorm15_1_15_ = self.batchnorm15_1_15_(conv15_1_15_)
        relu15_1_15_ = self.relu15_1_15_(batchnorm15_1_15_)
        conv16_1_15_padding = self.conv16_1_15_padding(relu15_1_15_)
        conv16_1_15_ = self.conv16_1_15_(conv16_1_15_padding)
        batchnorm16_1_15_ = self.batchnorm16_1_15_(conv16_1_15_)
        relu16_1_15_ = self.relu16_1_15_(batchnorm16_1_15_)
        conv17_1_15_ = self.conv17_1_15_(relu16_1_15_)
        batchnorm17_1_15_ = self.batchnorm17_1_15_(conv17_1_15_)
        conv15_1_16_ = self.conv15_1_16_(relu13_)
        batchnorm15_1_16_ = self.batchnorm15_1_16_(conv15_1_16_)
        relu15_1_16_ = self.relu15_1_16_(batchnorm15_1_16_)
        conv16_1_16_padding = self.conv16_1_16_padding(relu15_1_16_)
        conv16_1_16_ = self.conv16_1_16_(conv16_1_16_padding)
        batchnorm16_1_16_ = self.batchnorm16_1_16_(conv16_1_16_)
        relu16_1_16_ = self.relu16_1_16_(batchnorm16_1_16_)
        conv17_1_16_ = self.conv17_1_16_(relu16_1_16_)
        batchnorm17_1_16_ = self.batchnorm17_1_16_(conv17_1_16_)
        conv15_1_17_ = self.conv15_1_17_(relu13_)
        batchnorm15_1_17_ = self.batchnorm15_1_17_(conv15_1_17_)
        relu15_1_17_ = self.relu15_1_17_(batchnorm15_1_17_)
        conv16_1_17_padding = self.conv16_1_17_padding(relu15_1_17_)
        conv16_1_17_ = self.conv16_1_17_(conv16_1_17_padding)
        batchnorm16_1_17_ = self.batchnorm16_1_17_(conv16_1_17_)
        relu16_1_17_ = self.relu16_1_17_(batchnorm16_1_17_)
        conv17_1_17_ = self.conv17_1_17_(relu16_1_17_)
        batchnorm17_1_17_ = self.batchnorm17_1_17_(conv17_1_17_)
        conv15_1_18_ = self.conv15_1_18_(relu13_)
        batchnorm15_1_18_ = self.batchnorm15_1_18_(conv15_1_18_)
        relu15_1_18_ = self.relu15_1_18_(batchnorm15_1_18_)
        conv16_1_18_padding = self.conv16_1_18_padding(relu15_1_18_)
        conv16_1_18_ = self.conv16_1_18_(conv16_1_18_padding)
        batchnorm16_1_18_ = self.batchnorm16_1_18_(conv16_1_18_)
        relu16_1_18_ = self.relu16_1_18_(batchnorm16_1_18_)
        conv17_1_18_ = self.conv17_1_18_(relu16_1_18_)
        batchnorm17_1_18_ = self.batchnorm17_1_18_(conv17_1_18_)
        conv15_1_19_ = self.conv15_1_19_(relu13_)
        batchnorm15_1_19_ = self.batchnorm15_1_19_(conv15_1_19_)
        relu15_1_19_ = self.relu15_1_19_(batchnorm15_1_19_)
        conv16_1_19_padding = self.conv16_1_19_padding(relu15_1_19_)
        conv16_1_19_ = self.conv16_1_19_(conv16_1_19_padding)
        batchnorm16_1_19_ = self.batchnorm16_1_19_(conv16_1_19_)
        relu16_1_19_ = self.relu16_1_19_(batchnorm16_1_19_)
        conv17_1_19_ = self.conv17_1_19_(relu16_1_19_)
        batchnorm17_1_19_ = self.batchnorm17_1_19_(conv17_1_19_)
        conv15_1_20_ = self.conv15_1_20_(relu13_)
        batchnorm15_1_20_ = self.batchnorm15_1_20_(conv15_1_20_)
        relu15_1_20_ = self.relu15_1_20_(batchnorm15_1_20_)
        conv16_1_20_padding = self.conv16_1_20_padding(relu15_1_20_)
        conv16_1_20_ = self.conv16_1_20_(conv16_1_20_padding)
        batchnorm16_1_20_ = self.batchnorm16_1_20_(conv16_1_20_)
        relu16_1_20_ = self.relu16_1_20_(batchnorm16_1_20_)
        conv17_1_20_ = self.conv17_1_20_(relu16_1_20_)
        batchnorm17_1_20_ = self.batchnorm17_1_20_(conv17_1_20_)
        conv15_1_21_ = self.conv15_1_21_(relu13_)
        batchnorm15_1_21_ = self.batchnorm15_1_21_(conv15_1_21_)
        relu15_1_21_ = self.relu15_1_21_(batchnorm15_1_21_)
        conv16_1_21_padding = self.conv16_1_21_padding(relu15_1_21_)
        conv16_1_21_ = self.conv16_1_21_(conv16_1_21_padding)
        batchnorm16_1_21_ = self.batchnorm16_1_21_(conv16_1_21_)
        relu16_1_21_ = self.relu16_1_21_(batchnorm16_1_21_)
        conv17_1_21_ = self.conv17_1_21_(relu16_1_21_)
        batchnorm17_1_21_ = self.batchnorm17_1_21_(conv17_1_21_)
        conv15_1_22_ = self.conv15_1_22_(relu13_)
        batchnorm15_1_22_ = self.batchnorm15_1_22_(conv15_1_22_)
        relu15_1_22_ = self.relu15_1_22_(batchnorm15_1_22_)
        conv16_1_22_padding = self.conv16_1_22_padding(relu15_1_22_)
        conv16_1_22_ = self.conv16_1_22_(conv16_1_22_padding)
        batchnorm16_1_22_ = self.batchnorm16_1_22_(conv16_1_22_)
        relu16_1_22_ = self.relu16_1_22_(batchnorm16_1_22_)
        conv17_1_22_ = self.conv17_1_22_(relu16_1_22_)
        batchnorm17_1_22_ = self.batchnorm17_1_22_(conv17_1_22_)
        conv15_1_23_ = self.conv15_1_23_(relu13_)
        batchnorm15_1_23_ = self.batchnorm15_1_23_(conv15_1_23_)
        relu15_1_23_ = self.relu15_1_23_(batchnorm15_1_23_)
        conv16_1_23_padding = self.conv16_1_23_padding(relu15_1_23_)
        conv16_1_23_ = self.conv16_1_23_(conv16_1_23_padding)
        batchnorm16_1_23_ = self.batchnorm16_1_23_(conv16_1_23_)
        relu16_1_23_ = self.relu16_1_23_(batchnorm16_1_23_)
        conv17_1_23_ = self.conv17_1_23_(relu16_1_23_)
        batchnorm17_1_23_ = self.batchnorm17_1_23_(conv17_1_23_)
        conv15_1_24_ = self.conv15_1_24_(relu13_)
        batchnorm15_1_24_ = self.batchnorm15_1_24_(conv15_1_24_)
        relu15_1_24_ = self.relu15_1_24_(batchnorm15_1_24_)
        conv16_1_24_padding = self.conv16_1_24_padding(relu15_1_24_)
        conv16_1_24_ = self.conv16_1_24_(conv16_1_24_padding)
        batchnorm16_1_24_ = self.batchnorm16_1_24_(conv16_1_24_)
        relu16_1_24_ = self.relu16_1_24_(batchnorm16_1_24_)
        conv17_1_24_ = self.conv17_1_24_(relu16_1_24_)
        batchnorm17_1_24_ = self.batchnorm17_1_24_(conv17_1_24_)
        conv15_1_25_ = self.conv15_1_25_(relu13_)
        batchnorm15_1_25_ = self.batchnorm15_1_25_(conv15_1_25_)
        relu15_1_25_ = self.relu15_1_25_(batchnorm15_1_25_)
        conv16_1_25_padding = self.conv16_1_25_padding(relu15_1_25_)
        conv16_1_25_ = self.conv16_1_25_(conv16_1_25_padding)
        batchnorm16_1_25_ = self.batchnorm16_1_25_(conv16_1_25_)
        relu16_1_25_ = self.relu16_1_25_(batchnorm16_1_25_)
        conv17_1_25_ = self.conv17_1_25_(relu16_1_25_)
        batchnorm17_1_25_ = self.batchnorm17_1_25_(conv17_1_25_)
        conv15_1_26_ = self.conv15_1_26_(relu13_)
        batchnorm15_1_26_ = self.batchnorm15_1_26_(conv15_1_26_)
        relu15_1_26_ = self.relu15_1_26_(batchnorm15_1_26_)
        conv16_1_26_padding = self.conv16_1_26_padding(relu15_1_26_)
        conv16_1_26_ = self.conv16_1_26_(conv16_1_26_padding)
        batchnorm16_1_26_ = self.batchnorm16_1_26_(conv16_1_26_)
        relu16_1_26_ = self.relu16_1_26_(batchnorm16_1_26_)
        conv17_1_26_ = self.conv17_1_26_(relu16_1_26_)
        batchnorm17_1_26_ = self.batchnorm17_1_26_(conv17_1_26_)
        conv15_1_27_ = self.conv15_1_27_(relu13_)
        batchnorm15_1_27_ = self.batchnorm15_1_27_(conv15_1_27_)
        relu15_1_27_ = self.relu15_1_27_(batchnorm15_1_27_)
        conv16_1_27_padding = self.conv16_1_27_padding(relu15_1_27_)
        conv16_1_27_ = self.conv16_1_27_(conv16_1_27_padding)
        batchnorm16_1_27_ = self.batchnorm16_1_27_(conv16_1_27_)
        relu16_1_27_ = self.relu16_1_27_(batchnorm16_1_27_)
        conv17_1_27_ = self.conv17_1_27_(relu16_1_27_)
        batchnorm17_1_27_ = self.batchnorm17_1_27_(conv17_1_27_)
        conv15_1_28_ = self.conv15_1_28_(relu13_)
        batchnorm15_1_28_ = self.batchnorm15_1_28_(conv15_1_28_)
        relu15_1_28_ = self.relu15_1_28_(batchnorm15_1_28_)
        conv16_1_28_padding = self.conv16_1_28_padding(relu15_1_28_)
        conv16_1_28_ = self.conv16_1_28_(conv16_1_28_padding)
        batchnorm16_1_28_ = self.batchnorm16_1_28_(conv16_1_28_)
        relu16_1_28_ = self.relu16_1_28_(batchnorm16_1_28_)
        conv17_1_28_ = self.conv17_1_28_(relu16_1_28_)
        batchnorm17_1_28_ = self.batchnorm17_1_28_(conv17_1_28_)
        conv15_1_29_ = self.conv15_1_29_(relu13_)
        batchnorm15_1_29_ = self.batchnorm15_1_29_(conv15_1_29_)
        relu15_1_29_ = self.relu15_1_29_(batchnorm15_1_29_)
        conv16_1_29_padding = self.conv16_1_29_padding(relu15_1_29_)
        conv16_1_29_ = self.conv16_1_29_(conv16_1_29_padding)
        batchnorm16_1_29_ = self.batchnorm16_1_29_(conv16_1_29_)
        relu16_1_29_ = self.relu16_1_29_(batchnorm16_1_29_)
        conv17_1_29_ = self.conv17_1_29_(relu16_1_29_)
        batchnorm17_1_29_ = self.batchnorm17_1_29_(conv17_1_29_)
        conv15_1_30_ = self.conv15_1_30_(relu13_)
        batchnorm15_1_30_ = self.batchnorm15_1_30_(conv15_1_30_)
        relu15_1_30_ = self.relu15_1_30_(batchnorm15_1_30_)
        conv16_1_30_padding = self.conv16_1_30_padding(relu15_1_30_)
        conv16_1_30_ = self.conv16_1_30_(conv16_1_30_padding)
        batchnorm16_1_30_ = self.batchnorm16_1_30_(conv16_1_30_)
        relu16_1_30_ = self.relu16_1_30_(batchnorm16_1_30_)
        conv17_1_30_ = self.conv17_1_30_(relu16_1_30_)
        batchnorm17_1_30_ = self.batchnorm17_1_30_(conv17_1_30_)
        conv15_1_31_ = self.conv15_1_31_(relu13_)
        batchnorm15_1_31_ = self.batchnorm15_1_31_(conv15_1_31_)
        relu15_1_31_ = self.relu15_1_31_(batchnorm15_1_31_)
        conv16_1_31_padding = self.conv16_1_31_padding(relu15_1_31_)
        conv16_1_31_ = self.conv16_1_31_(conv16_1_31_padding)
        batchnorm16_1_31_ = self.batchnorm16_1_31_(conv16_1_31_)
        relu16_1_31_ = self.relu16_1_31_(batchnorm16_1_31_)
        conv17_1_31_ = self.conv17_1_31_(relu16_1_31_)
        batchnorm17_1_31_ = self.batchnorm17_1_31_(conv17_1_31_)
        conv15_1_32_ = self.conv15_1_32_(relu13_)
        batchnorm15_1_32_ = self.batchnorm15_1_32_(conv15_1_32_)
        relu15_1_32_ = self.relu15_1_32_(batchnorm15_1_32_)
        conv16_1_32_padding = self.conv16_1_32_padding(relu15_1_32_)
        conv16_1_32_ = self.conv16_1_32_(conv16_1_32_padding)
        batchnorm16_1_32_ = self.batchnorm16_1_32_(conv16_1_32_)
        relu16_1_32_ = self.relu16_1_32_(batchnorm16_1_32_)
        conv17_1_32_ = self.conv17_1_32_(relu16_1_32_)
        batchnorm17_1_32_ = self.batchnorm17_1_32_(conv17_1_32_)
        add18_1_ = batchnorm17_1_1_ + batchnorm17_1_2_ + batchnorm17_1_3_ + batchnorm17_1_4_ + batchnorm17_1_5_ + batchnorm17_1_6_ + batchnorm17_1_7_ + batchnorm17_1_8_ + batchnorm17_1_9_ + batchnorm17_1_10_ + batchnorm17_1_11_ + batchnorm17_1_12_ + batchnorm17_1_13_ + batchnorm17_1_14_ + batchnorm17_1_15_ + batchnorm17_1_16_ + batchnorm17_1_17_ + batchnorm17_1_18_ + batchnorm17_1_19_ + batchnorm17_1_20_ + batchnorm17_1_21_ + batchnorm17_1_22_ + batchnorm17_1_23_ + batchnorm17_1_24_ + batchnorm17_1_25_ + batchnorm17_1_26_ + batchnorm17_1_27_ + batchnorm17_1_28_ + batchnorm17_1_29_ + batchnorm17_1_30_ + batchnorm17_1_31_ + batchnorm17_1_32_
        conv14_2_ = self.conv14_2_(relu13_)
        batchnorm14_2_ = self.batchnorm14_2_(conv14_2_)
        add19_ = add18_1_ + batchnorm14_2_
        relu19_ = self.relu19_(add19_)
        conv21_1_1_ = self.conv21_1_1_(relu19_)
        batchnorm21_1_1_ = self.batchnorm21_1_1_(conv21_1_1_)
        relu21_1_1_ = self.relu21_1_1_(batchnorm21_1_1_)
        conv22_1_1_padding = self.conv22_1_1_padding(relu21_1_1_)
        conv22_1_1_ = self.conv22_1_1_(conv22_1_1_padding)
        batchnorm22_1_1_ = self.batchnorm22_1_1_(conv22_1_1_)
        relu22_1_1_ = self.relu22_1_1_(batchnorm22_1_1_)
        conv23_1_1_ = self.conv23_1_1_(relu22_1_1_)
        batchnorm23_1_1_ = self.batchnorm23_1_1_(conv23_1_1_)
        conv21_1_2_ = self.conv21_1_2_(relu19_)
        batchnorm21_1_2_ = self.batchnorm21_1_2_(conv21_1_2_)
        relu21_1_2_ = self.relu21_1_2_(batchnorm21_1_2_)
        conv22_1_2_padding = self.conv22_1_2_padding(relu21_1_2_)
        conv22_1_2_ = self.conv22_1_2_(conv22_1_2_padding)
        batchnorm22_1_2_ = self.batchnorm22_1_2_(conv22_1_2_)
        relu22_1_2_ = self.relu22_1_2_(batchnorm22_1_2_)
        conv23_1_2_ = self.conv23_1_2_(relu22_1_2_)
        batchnorm23_1_2_ = self.batchnorm23_1_2_(conv23_1_2_)
        conv21_1_3_ = self.conv21_1_3_(relu19_)
        batchnorm21_1_3_ = self.batchnorm21_1_3_(conv21_1_3_)
        relu21_1_3_ = self.relu21_1_3_(batchnorm21_1_3_)
        conv22_1_3_padding = self.conv22_1_3_padding(relu21_1_3_)
        conv22_1_3_ = self.conv22_1_3_(conv22_1_3_padding)
        batchnorm22_1_3_ = self.batchnorm22_1_3_(conv22_1_3_)
        relu22_1_3_ = self.relu22_1_3_(batchnorm22_1_3_)
        conv23_1_3_ = self.conv23_1_3_(relu22_1_3_)
        batchnorm23_1_3_ = self.batchnorm23_1_3_(conv23_1_3_)
        conv21_1_4_ = self.conv21_1_4_(relu19_)
        batchnorm21_1_4_ = self.batchnorm21_1_4_(conv21_1_4_)
        relu21_1_4_ = self.relu21_1_4_(batchnorm21_1_4_)
        conv22_1_4_padding = self.conv22_1_4_padding(relu21_1_4_)
        conv22_1_4_ = self.conv22_1_4_(conv22_1_4_padding)
        batchnorm22_1_4_ = self.batchnorm22_1_4_(conv22_1_4_)
        relu22_1_4_ = self.relu22_1_4_(batchnorm22_1_4_)
        conv23_1_4_ = self.conv23_1_4_(relu22_1_4_)
        batchnorm23_1_4_ = self.batchnorm23_1_4_(conv23_1_4_)
        conv21_1_5_ = self.conv21_1_5_(relu19_)
        batchnorm21_1_5_ = self.batchnorm21_1_5_(conv21_1_5_)
        relu21_1_5_ = self.relu21_1_5_(batchnorm21_1_5_)
        conv22_1_5_padding = self.conv22_1_5_padding(relu21_1_5_)
        conv22_1_5_ = self.conv22_1_5_(conv22_1_5_padding)
        batchnorm22_1_5_ = self.batchnorm22_1_5_(conv22_1_5_)
        relu22_1_5_ = self.relu22_1_5_(batchnorm22_1_5_)
        conv23_1_5_ = self.conv23_1_5_(relu22_1_5_)
        batchnorm23_1_5_ = self.batchnorm23_1_5_(conv23_1_5_)
        conv21_1_6_ = self.conv21_1_6_(relu19_)
        batchnorm21_1_6_ = self.batchnorm21_1_6_(conv21_1_6_)
        relu21_1_6_ = self.relu21_1_6_(batchnorm21_1_6_)
        conv22_1_6_padding = self.conv22_1_6_padding(relu21_1_6_)
        conv22_1_6_ = self.conv22_1_6_(conv22_1_6_padding)
        batchnorm22_1_6_ = self.batchnorm22_1_6_(conv22_1_6_)
        relu22_1_6_ = self.relu22_1_6_(batchnorm22_1_6_)
        conv23_1_6_ = self.conv23_1_6_(relu22_1_6_)
        batchnorm23_1_6_ = self.batchnorm23_1_6_(conv23_1_6_)
        conv21_1_7_ = self.conv21_1_7_(relu19_)
        batchnorm21_1_7_ = self.batchnorm21_1_7_(conv21_1_7_)
        relu21_1_7_ = self.relu21_1_7_(batchnorm21_1_7_)
        conv22_1_7_padding = self.conv22_1_7_padding(relu21_1_7_)
        conv22_1_7_ = self.conv22_1_7_(conv22_1_7_padding)
        batchnorm22_1_7_ = self.batchnorm22_1_7_(conv22_1_7_)
        relu22_1_7_ = self.relu22_1_7_(batchnorm22_1_7_)
        conv23_1_7_ = self.conv23_1_7_(relu22_1_7_)
        batchnorm23_1_7_ = self.batchnorm23_1_7_(conv23_1_7_)
        conv21_1_8_ = self.conv21_1_8_(relu19_)
        batchnorm21_1_8_ = self.batchnorm21_1_8_(conv21_1_8_)
        relu21_1_8_ = self.relu21_1_8_(batchnorm21_1_8_)
        conv22_1_8_padding = self.conv22_1_8_padding(relu21_1_8_)
        conv22_1_8_ = self.conv22_1_8_(conv22_1_8_padding)
        batchnorm22_1_8_ = self.batchnorm22_1_8_(conv22_1_8_)
        relu22_1_8_ = self.relu22_1_8_(batchnorm22_1_8_)
        conv23_1_8_ = self.conv23_1_8_(relu22_1_8_)
        batchnorm23_1_8_ = self.batchnorm23_1_8_(conv23_1_8_)
        conv21_1_9_ = self.conv21_1_9_(relu19_)
        batchnorm21_1_9_ = self.batchnorm21_1_9_(conv21_1_9_)
        relu21_1_9_ = self.relu21_1_9_(batchnorm21_1_9_)
        conv22_1_9_padding = self.conv22_1_9_padding(relu21_1_9_)
        conv22_1_9_ = self.conv22_1_9_(conv22_1_9_padding)
        batchnorm22_1_9_ = self.batchnorm22_1_9_(conv22_1_9_)
        relu22_1_9_ = self.relu22_1_9_(batchnorm22_1_9_)
        conv23_1_9_ = self.conv23_1_9_(relu22_1_9_)
        batchnorm23_1_9_ = self.batchnorm23_1_9_(conv23_1_9_)
        conv21_1_10_ = self.conv21_1_10_(relu19_)
        batchnorm21_1_10_ = self.batchnorm21_1_10_(conv21_1_10_)
        relu21_1_10_ = self.relu21_1_10_(batchnorm21_1_10_)
        conv22_1_10_padding = self.conv22_1_10_padding(relu21_1_10_)
        conv22_1_10_ = self.conv22_1_10_(conv22_1_10_padding)
        batchnorm22_1_10_ = self.batchnorm22_1_10_(conv22_1_10_)
        relu22_1_10_ = self.relu22_1_10_(batchnorm22_1_10_)
        conv23_1_10_ = self.conv23_1_10_(relu22_1_10_)
        batchnorm23_1_10_ = self.batchnorm23_1_10_(conv23_1_10_)
        conv21_1_11_ = self.conv21_1_11_(relu19_)
        batchnorm21_1_11_ = self.batchnorm21_1_11_(conv21_1_11_)
        relu21_1_11_ = self.relu21_1_11_(batchnorm21_1_11_)
        conv22_1_11_padding = self.conv22_1_11_padding(relu21_1_11_)
        conv22_1_11_ = self.conv22_1_11_(conv22_1_11_padding)
        batchnorm22_1_11_ = self.batchnorm22_1_11_(conv22_1_11_)
        relu22_1_11_ = self.relu22_1_11_(batchnorm22_1_11_)
        conv23_1_11_ = self.conv23_1_11_(relu22_1_11_)
        batchnorm23_1_11_ = self.batchnorm23_1_11_(conv23_1_11_)
        conv21_1_12_ = self.conv21_1_12_(relu19_)
        batchnorm21_1_12_ = self.batchnorm21_1_12_(conv21_1_12_)
        relu21_1_12_ = self.relu21_1_12_(batchnorm21_1_12_)
        conv22_1_12_padding = self.conv22_1_12_padding(relu21_1_12_)
        conv22_1_12_ = self.conv22_1_12_(conv22_1_12_padding)
        batchnorm22_1_12_ = self.batchnorm22_1_12_(conv22_1_12_)
        relu22_1_12_ = self.relu22_1_12_(batchnorm22_1_12_)
        conv23_1_12_ = self.conv23_1_12_(relu22_1_12_)
        batchnorm23_1_12_ = self.batchnorm23_1_12_(conv23_1_12_)
        conv21_1_13_ = self.conv21_1_13_(relu19_)
        batchnorm21_1_13_ = self.batchnorm21_1_13_(conv21_1_13_)
        relu21_1_13_ = self.relu21_1_13_(batchnorm21_1_13_)
        conv22_1_13_padding = self.conv22_1_13_padding(relu21_1_13_)
        conv22_1_13_ = self.conv22_1_13_(conv22_1_13_padding)
        batchnorm22_1_13_ = self.batchnorm22_1_13_(conv22_1_13_)
        relu22_1_13_ = self.relu22_1_13_(batchnorm22_1_13_)
        conv23_1_13_ = self.conv23_1_13_(relu22_1_13_)
        batchnorm23_1_13_ = self.batchnorm23_1_13_(conv23_1_13_)
        conv21_1_14_ = self.conv21_1_14_(relu19_)
        batchnorm21_1_14_ = self.batchnorm21_1_14_(conv21_1_14_)
        relu21_1_14_ = self.relu21_1_14_(batchnorm21_1_14_)
        conv22_1_14_padding = self.conv22_1_14_padding(relu21_1_14_)
        conv22_1_14_ = self.conv22_1_14_(conv22_1_14_padding)
        batchnorm22_1_14_ = self.batchnorm22_1_14_(conv22_1_14_)
        relu22_1_14_ = self.relu22_1_14_(batchnorm22_1_14_)
        conv23_1_14_ = self.conv23_1_14_(relu22_1_14_)
        batchnorm23_1_14_ = self.batchnorm23_1_14_(conv23_1_14_)
        conv21_1_15_ = self.conv21_1_15_(relu19_)
        batchnorm21_1_15_ = self.batchnorm21_1_15_(conv21_1_15_)
        relu21_1_15_ = self.relu21_1_15_(batchnorm21_1_15_)
        conv22_1_15_padding = self.conv22_1_15_padding(relu21_1_15_)
        conv22_1_15_ = self.conv22_1_15_(conv22_1_15_padding)
        batchnorm22_1_15_ = self.batchnorm22_1_15_(conv22_1_15_)
        relu22_1_15_ = self.relu22_1_15_(batchnorm22_1_15_)
        conv23_1_15_ = self.conv23_1_15_(relu22_1_15_)
        batchnorm23_1_15_ = self.batchnorm23_1_15_(conv23_1_15_)
        conv21_1_16_ = self.conv21_1_16_(relu19_)
        batchnorm21_1_16_ = self.batchnorm21_1_16_(conv21_1_16_)
        relu21_1_16_ = self.relu21_1_16_(batchnorm21_1_16_)
        conv22_1_16_padding = self.conv22_1_16_padding(relu21_1_16_)
        conv22_1_16_ = self.conv22_1_16_(conv22_1_16_padding)
        batchnorm22_1_16_ = self.batchnorm22_1_16_(conv22_1_16_)
        relu22_1_16_ = self.relu22_1_16_(batchnorm22_1_16_)
        conv23_1_16_ = self.conv23_1_16_(relu22_1_16_)
        batchnorm23_1_16_ = self.batchnorm23_1_16_(conv23_1_16_)
        conv21_1_17_ = self.conv21_1_17_(relu19_)
        batchnorm21_1_17_ = self.batchnorm21_1_17_(conv21_1_17_)
        relu21_1_17_ = self.relu21_1_17_(batchnorm21_1_17_)
        conv22_1_17_padding = self.conv22_1_17_padding(relu21_1_17_)
        conv22_1_17_ = self.conv22_1_17_(conv22_1_17_padding)
        batchnorm22_1_17_ = self.batchnorm22_1_17_(conv22_1_17_)
        relu22_1_17_ = self.relu22_1_17_(batchnorm22_1_17_)
        conv23_1_17_ = self.conv23_1_17_(relu22_1_17_)
        batchnorm23_1_17_ = self.batchnorm23_1_17_(conv23_1_17_)
        conv21_1_18_ = self.conv21_1_18_(relu19_)
        batchnorm21_1_18_ = self.batchnorm21_1_18_(conv21_1_18_)
        relu21_1_18_ = self.relu21_1_18_(batchnorm21_1_18_)
        conv22_1_18_padding = self.conv22_1_18_padding(relu21_1_18_)
        conv22_1_18_ = self.conv22_1_18_(conv22_1_18_padding)
        batchnorm22_1_18_ = self.batchnorm22_1_18_(conv22_1_18_)
        relu22_1_18_ = self.relu22_1_18_(batchnorm22_1_18_)
        conv23_1_18_ = self.conv23_1_18_(relu22_1_18_)
        batchnorm23_1_18_ = self.batchnorm23_1_18_(conv23_1_18_)
        conv21_1_19_ = self.conv21_1_19_(relu19_)
        batchnorm21_1_19_ = self.batchnorm21_1_19_(conv21_1_19_)
        relu21_1_19_ = self.relu21_1_19_(batchnorm21_1_19_)
        conv22_1_19_padding = self.conv22_1_19_padding(relu21_1_19_)
        conv22_1_19_ = self.conv22_1_19_(conv22_1_19_padding)
        batchnorm22_1_19_ = self.batchnorm22_1_19_(conv22_1_19_)
        relu22_1_19_ = self.relu22_1_19_(batchnorm22_1_19_)
        conv23_1_19_ = self.conv23_1_19_(relu22_1_19_)
        batchnorm23_1_19_ = self.batchnorm23_1_19_(conv23_1_19_)
        conv21_1_20_ = self.conv21_1_20_(relu19_)
        batchnorm21_1_20_ = self.batchnorm21_1_20_(conv21_1_20_)
        relu21_1_20_ = self.relu21_1_20_(batchnorm21_1_20_)
        conv22_1_20_padding = self.conv22_1_20_padding(relu21_1_20_)
        conv22_1_20_ = self.conv22_1_20_(conv22_1_20_padding)
        batchnorm22_1_20_ = self.batchnorm22_1_20_(conv22_1_20_)
        relu22_1_20_ = self.relu22_1_20_(batchnorm22_1_20_)
        conv23_1_20_ = self.conv23_1_20_(relu22_1_20_)
        batchnorm23_1_20_ = self.batchnorm23_1_20_(conv23_1_20_)
        conv21_1_21_ = self.conv21_1_21_(relu19_)
        batchnorm21_1_21_ = self.batchnorm21_1_21_(conv21_1_21_)
        relu21_1_21_ = self.relu21_1_21_(batchnorm21_1_21_)
        conv22_1_21_padding = self.conv22_1_21_padding(relu21_1_21_)
        conv22_1_21_ = self.conv22_1_21_(conv22_1_21_padding)
        batchnorm22_1_21_ = self.batchnorm22_1_21_(conv22_1_21_)
        relu22_1_21_ = self.relu22_1_21_(batchnorm22_1_21_)
        conv23_1_21_ = self.conv23_1_21_(relu22_1_21_)
        batchnorm23_1_21_ = self.batchnorm23_1_21_(conv23_1_21_)
        conv21_1_22_ = self.conv21_1_22_(relu19_)
        batchnorm21_1_22_ = self.batchnorm21_1_22_(conv21_1_22_)
        relu21_1_22_ = self.relu21_1_22_(batchnorm21_1_22_)
        conv22_1_22_padding = self.conv22_1_22_padding(relu21_1_22_)
        conv22_1_22_ = self.conv22_1_22_(conv22_1_22_padding)
        batchnorm22_1_22_ = self.batchnorm22_1_22_(conv22_1_22_)
        relu22_1_22_ = self.relu22_1_22_(batchnorm22_1_22_)
        conv23_1_22_ = self.conv23_1_22_(relu22_1_22_)
        batchnorm23_1_22_ = self.batchnorm23_1_22_(conv23_1_22_)
        conv21_1_23_ = self.conv21_1_23_(relu19_)
        batchnorm21_1_23_ = self.batchnorm21_1_23_(conv21_1_23_)
        relu21_1_23_ = self.relu21_1_23_(batchnorm21_1_23_)
        conv22_1_23_padding = self.conv22_1_23_padding(relu21_1_23_)
        conv22_1_23_ = self.conv22_1_23_(conv22_1_23_padding)
        batchnorm22_1_23_ = self.batchnorm22_1_23_(conv22_1_23_)
        relu22_1_23_ = self.relu22_1_23_(batchnorm22_1_23_)
        conv23_1_23_ = self.conv23_1_23_(relu22_1_23_)
        batchnorm23_1_23_ = self.batchnorm23_1_23_(conv23_1_23_)
        conv21_1_24_ = self.conv21_1_24_(relu19_)
        batchnorm21_1_24_ = self.batchnorm21_1_24_(conv21_1_24_)
        relu21_1_24_ = self.relu21_1_24_(batchnorm21_1_24_)
        conv22_1_24_padding = self.conv22_1_24_padding(relu21_1_24_)
        conv22_1_24_ = self.conv22_1_24_(conv22_1_24_padding)
        batchnorm22_1_24_ = self.batchnorm22_1_24_(conv22_1_24_)
        relu22_1_24_ = self.relu22_1_24_(batchnorm22_1_24_)
        conv23_1_24_ = self.conv23_1_24_(relu22_1_24_)
        batchnorm23_1_24_ = self.batchnorm23_1_24_(conv23_1_24_)
        conv21_1_25_ = self.conv21_1_25_(relu19_)
        batchnorm21_1_25_ = self.batchnorm21_1_25_(conv21_1_25_)
        relu21_1_25_ = self.relu21_1_25_(batchnorm21_1_25_)
        conv22_1_25_padding = self.conv22_1_25_padding(relu21_1_25_)
        conv22_1_25_ = self.conv22_1_25_(conv22_1_25_padding)
        batchnorm22_1_25_ = self.batchnorm22_1_25_(conv22_1_25_)
        relu22_1_25_ = self.relu22_1_25_(batchnorm22_1_25_)
        conv23_1_25_ = self.conv23_1_25_(relu22_1_25_)
        batchnorm23_1_25_ = self.batchnorm23_1_25_(conv23_1_25_)
        conv21_1_26_ = self.conv21_1_26_(relu19_)
        batchnorm21_1_26_ = self.batchnorm21_1_26_(conv21_1_26_)
        relu21_1_26_ = self.relu21_1_26_(batchnorm21_1_26_)
        conv22_1_26_padding = self.conv22_1_26_padding(relu21_1_26_)
        conv22_1_26_ = self.conv22_1_26_(conv22_1_26_padding)
        batchnorm22_1_26_ = self.batchnorm22_1_26_(conv22_1_26_)
        relu22_1_26_ = self.relu22_1_26_(batchnorm22_1_26_)
        conv23_1_26_ = self.conv23_1_26_(relu22_1_26_)
        batchnorm23_1_26_ = self.batchnorm23_1_26_(conv23_1_26_)
        conv21_1_27_ = self.conv21_1_27_(relu19_)
        batchnorm21_1_27_ = self.batchnorm21_1_27_(conv21_1_27_)
        relu21_1_27_ = self.relu21_1_27_(batchnorm21_1_27_)
        conv22_1_27_padding = self.conv22_1_27_padding(relu21_1_27_)
        conv22_1_27_ = self.conv22_1_27_(conv22_1_27_padding)
        batchnorm22_1_27_ = self.batchnorm22_1_27_(conv22_1_27_)
        relu22_1_27_ = self.relu22_1_27_(batchnorm22_1_27_)
        conv23_1_27_ = self.conv23_1_27_(relu22_1_27_)
        batchnorm23_1_27_ = self.batchnorm23_1_27_(conv23_1_27_)
        conv21_1_28_ = self.conv21_1_28_(relu19_)
        batchnorm21_1_28_ = self.batchnorm21_1_28_(conv21_1_28_)
        relu21_1_28_ = self.relu21_1_28_(batchnorm21_1_28_)
        conv22_1_28_padding = self.conv22_1_28_padding(relu21_1_28_)
        conv22_1_28_ = self.conv22_1_28_(conv22_1_28_padding)
        batchnorm22_1_28_ = self.batchnorm22_1_28_(conv22_1_28_)
        relu22_1_28_ = self.relu22_1_28_(batchnorm22_1_28_)
        conv23_1_28_ = self.conv23_1_28_(relu22_1_28_)
        batchnorm23_1_28_ = self.batchnorm23_1_28_(conv23_1_28_)
        conv21_1_29_ = self.conv21_1_29_(relu19_)
        batchnorm21_1_29_ = self.batchnorm21_1_29_(conv21_1_29_)
        relu21_1_29_ = self.relu21_1_29_(batchnorm21_1_29_)
        conv22_1_29_padding = self.conv22_1_29_padding(relu21_1_29_)
        conv22_1_29_ = self.conv22_1_29_(conv22_1_29_padding)
        batchnorm22_1_29_ = self.batchnorm22_1_29_(conv22_1_29_)
        relu22_1_29_ = self.relu22_1_29_(batchnorm22_1_29_)
        conv23_1_29_ = self.conv23_1_29_(relu22_1_29_)
        batchnorm23_1_29_ = self.batchnorm23_1_29_(conv23_1_29_)
        conv21_1_30_ = self.conv21_1_30_(relu19_)
        batchnorm21_1_30_ = self.batchnorm21_1_30_(conv21_1_30_)
        relu21_1_30_ = self.relu21_1_30_(batchnorm21_1_30_)
        conv22_1_30_padding = self.conv22_1_30_padding(relu21_1_30_)
        conv22_1_30_ = self.conv22_1_30_(conv22_1_30_padding)
        batchnorm22_1_30_ = self.batchnorm22_1_30_(conv22_1_30_)
        relu22_1_30_ = self.relu22_1_30_(batchnorm22_1_30_)
        conv23_1_30_ = self.conv23_1_30_(relu22_1_30_)
        batchnorm23_1_30_ = self.batchnorm23_1_30_(conv23_1_30_)
        conv21_1_31_ = self.conv21_1_31_(relu19_)
        batchnorm21_1_31_ = self.batchnorm21_1_31_(conv21_1_31_)
        relu21_1_31_ = self.relu21_1_31_(batchnorm21_1_31_)
        conv22_1_31_padding = self.conv22_1_31_padding(relu21_1_31_)
        conv22_1_31_ = self.conv22_1_31_(conv22_1_31_padding)
        batchnorm22_1_31_ = self.batchnorm22_1_31_(conv22_1_31_)
        relu22_1_31_ = self.relu22_1_31_(batchnorm22_1_31_)
        conv23_1_31_ = self.conv23_1_31_(relu22_1_31_)
        batchnorm23_1_31_ = self.batchnorm23_1_31_(conv23_1_31_)
        conv21_1_32_ = self.conv21_1_32_(relu19_)
        batchnorm21_1_32_ = self.batchnorm21_1_32_(conv21_1_32_)
        relu21_1_32_ = self.relu21_1_32_(batchnorm21_1_32_)
        conv22_1_32_padding = self.conv22_1_32_padding(relu21_1_32_)
        conv22_1_32_ = self.conv22_1_32_(conv22_1_32_padding)
        batchnorm22_1_32_ = self.batchnorm22_1_32_(conv22_1_32_)
        relu22_1_32_ = self.relu22_1_32_(batchnorm22_1_32_)
        conv23_1_32_ = self.conv23_1_32_(relu22_1_32_)
        batchnorm23_1_32_ = self.batchnorm23_1_32_(conv23_1_32_)
        add24_1_ = batchnorm23_1_1_ + batchnorm23_1_2_ + batchnorm23_1_3_ + batchnorm23_1_4_ + batchnorm23_1_5_ + batchnorm23_1_6_ + batchnorm23_1_7_ + batchnorm23_1_8_ + batchnorm23_1_9_ + batchnorm23_1_10_ + batchnorm23_1_11_ + batchnorm23_1_12_ + batchnorm23_1_13_ + batchnorm23_1_14_ + batchnorm23_1_15_ + batchnorm23_1_16_ + batchnorm23_1_17_ + batchnorm23_1_18_ + batchnorm23_1_19_ + batchnorm23_1_20_ + batchnorm23_1_21_ + batchnorm23_1_22_ + batchnorm23_1_23_ + batchnorm23_1_24_ + batchnorm23_1_25_ + batchnorm23_1_26_ + batchnorm23_1_27_ + batchnorm23_1_28_ + batchnorm23_1_29_ + batchnorm23_1_30_ + batchnorm23_1_31_ + batchnorm23_1_32_
        conv20_2_ = self.conv20_2_(relu19_)
        batchnorm20_2_ = self.batchnorm20_2_(conv20_2_)
        add25_ = add24_1_ + batchnorm20_2_
        relu25_ = self.relu25_(add25_)
        conv27_1_1_ = self.conv27_1_1_(relu25_)
        batchnorm27_1_1_ = self.batchnorm27_1_1_(conv27_1_1_)
        relu27_1_1_ = self.relu27_1_1_(batchnorm27_1_1_)
        conv28_1_1_padding = self.conv28_1_1_padding(relu27_1_1_)
        conv28_1_1_ = self.conv28_1_1_(conv28_1_1_padding)
        batchnorm28_1_1_ = self.batchnorm28_1_1_(conv28_1_1_)
        relu28_1_1_ = self.relu28_1_1_(batchnorm28_1_1_)
        conv29_1_1_ = self.conv29_1_1_(relu28_1_1_)
        batchnorm29_1_1_ = self.batchnorm29_1_1_(conv29_1_1_)
        conv27_1_2_ = self.conv27_1_2_(relu25_)
        batchnorm27_1_2_ = self.batchnorm27_1_2_(conv27_1_2_)
        relu27_1_2_ = self.relu27_1_2_(batchnorm27_1_2_)
        conv28_1_2_padding = self.conv28_1_2_padding(relu27_1_2_)
        conv28_1_2_ = self.conv28_1_2_(conv28_1_2_padding)
        batchnorm28_1_2_ = self.batchnorm28_1_2_(conv28_1_2_)
        relu28_1_2_ = self.relu28_1_2_(batchnorm28_1_2_)
        conv29_1_2_ = self.conv29_1_2_(relu28_1_2_)
        batchnorm29_1_2_ = self.batchnorm29_1_2_(conv29_1_2_)
        conv27_1_3_ = self.conv27_1_3_(relu25_)
        batchnorm27_1_3_ = self.batchnorm27_1_3_(conv27_1_3_)
        relu27_1_3_ = self.relu27_1_3_(batchnorm27_1_3_)
        conv28_1_3_padding = self.conv28_1_3_padding(relu27_1_3_)
        conv28_1_3_ = self.conv28_1_3_(conv28_1_3_padding)
        batchnorm28_1_3_ = self.batchnorm28_1_3_(conv28_1_3_)
        relu28_1_3_ = self.relu28_1_3_(batchnorm28_1_3_)
        conv29_1_3_ = self.conv29_1_3_(relu28_1_3_)
        batchnorm29_1_3_ = self.batchnorm29_1_3_(conv29_1_3_)
        conv27_1_4_ = self.conv27_1_4_(relu25_)
        batchnorm27_1_4_ = self.batchnorm27_1_4_(conv27_1_4_)
        relu27_1_4_ = self.relu27_1_4_(batchnorm27_1_4_)
        conv28_1_4_padding = self.conv28_1_4_padding(relu27_1_4_)
        conv28_1_4_ = self.conv28_1_4_(conv28_1_4_padding)
        batchnorm28_1_4_ = self.batchnorm28_1_4_(conv28_1_4_)
        relu28_1_4_ = self.relu28_1_4_(batchnorm28_1_4_)
        conv29_1_4_ = self.conv29_1_4_(relu28_1_4_)
        batchnorm29_1_4_ = self.batchnorm29_1_4_(conv29_1_4_)
        conv27_1_5_ = self.conv27_1_5_(relu25_)
        batchnorm27_1_5_ = self.batchnorm27_1_5_(conv27_1_5_)
        relu27_1_5_ = self.relu27_1_5_(batchnorm27_1_5_)
        conv28_1_5_padding = self.conv28_1_5_padding(relu27_1_5_)
        conv28_1_5_ = self.conv28_1_5_(conv28_1_5_padding)
        batchnorm28_1_5_ = self.batchnorm28_1_5_(conv28_1_5_)
        relu28_1_5_ = self.relu28_1_5_(batchnorm28_1_5_)
        conv29_1_5_ = self.conv29_1_5_(relu28_1_5_)
        batchnorm29_1_5_ = self.batchnorm29_1_5_(conv29_1_5_)
        conv27_1_6_ = self.conv27_1_6_(relu25_)
        batchnorm27_1_6_ = self.batchnorm27_1_6_(conv27_1_6_)
        relu27_1_6_ = self.relu27_1_6_(batchnorm27_1_6_)
        conv28_1_6_padding = self.conv28_1_6_padding(relu27_1_6_)
        conv28_1_6_ = self.conv28_1_6_(conv28_1_6_padding)
        batchnorm28_1_6_ = self.batchnorm28_1_6_(conv28_1_6_)
        relu28_1_6_ = self.relu28_1_6_(batchnorm28_1_6_)
        conv29_1_6_ = self.conv29_1_6_(relu28_1_6_)
        batchnorm29_1_6_ = self.batchnorm29_1_6_(conv29_1_6_)
        conv27_1_7_ = self.conv27_1_7_(relu25_)
        batchnorm27_1_7_ = self.batchnorm27_1_7_(conv27_1_7_)
        relu27_1_7_ = self.relu27_1_7_(batchnorm27_1_7_)
        conv28_1_7_padding = self.conv28_1_7_padding(relu27_1_7_)
        conv28_1_7_ = self.conv28_1_7_(conv28_1_7_padding)
        batchnorm28_1_7_ = self.batchnorm28_1_7_(conv28_1_7_)
        relu28_1_7_ = self.relu28_1_7_(batchnorm28_1_7_)
        conv29_1_7_ = self.conv29_1_7_(relu28_1_7_)
        batchnorm29_1_7_ = self.batchnorm29_1_7_(conv29_1_7_)
        conv27_1_8_ = self.conv27_1_8_(relu25_)
        batchnorm27_1_8_ = self.batchnorm27_1_8_(conv27_1_8_)
        relu27_1_8_ = self.relu27_1_8_(batchnorm27_1_8_)
        conv28_1_8_padding = self.conv28_1_8_padding(relu27_1_8_)
        conv28_1_8_ = self.conv28_1_8_(conv28_1_8_padding)
        batchnorm28_1_8_ = self.batchnorm28_1_8_(conv28_1_8_)
        relu28_1_8_ = self.relu28_1_8_(batchnorm28_1_8_)
        conv29_1_8_ = self.conv29_1_8_(relu28_1_8_)
        batchnorm29_1_8_ = self.batchnorm29_1_8_(conv29_1_8_)
        conv27_1_9_ = self.conv27_1_9_(relu25_)
        batchnorm27_1_9_ = self.batchnorm27_1_9_(conv27_1_9_)
        relu27_1_9_ = self.relu27_1_9_(batchnorm27_1_9_)
        conv28_1_9_padding = self.conv28_1_9_padding(relu27_1_9_)
        conv28_1_9_ = self.conv28_1_9_(conv28_1_9_padding)
        batchnorm28_1_9_ = self.batchnorm28_1_9_(conv28_1_9_)
        relu28_1_9_ = self.relu28_1_9_(batchnorm28_1_9_)
        conv29_1_9_ = self.conv29_1_9_(relu28_1_9_)
        batchnorm29_1_9_ = self.batchnorm29_1_9_(conv29_1_9_)
        conv27_1_10_ = self.conv27_1_10_(relu25_)
        batchnorm27_1_10_ = self.batchnorm27_1_10_(conv27_1_10_)
        relu27_1_10_ = self.relu27_1_10_(batchnorm27_1_10_)
        conv28_1_10_padding = self.conv28_1_10_padding(relu27_1_10_)
        conv28_1_10_ = self.conv28_1_10_(conv28_1_10_padding)
        batchnorm28_1_10_ = self.batchnorm28_1_10_(conv28_1_10_)
        relu28_1_10_ = self.relu28_1_10_(batchnorm28_1_10_)
        conv29_1_10_ = self.conv29_1_10_(relu28_1_10_)
        batchnorm29_1_10_ = self.batchnorm29_1_10_(conv29_1_10_)
        conv27_1_11_ = self.conv27_1_11_(relu25_)
        batchnorm27_1_11_ = self.batchnorm27_1_11_(conv27_1_11_)
        relu27_1_11_ = self.relu27_1_11_(batchnorm27_1_11_)
        conv28_1_11_padding = self.conv28_1_11_padding(relu27_1_11_)
        conv28_1_11_ = self.conv28_1_11_(conv28_1_11_padding)
        batchnorm28_1_11_ = self.batchnorm28_1_11_(conv28_1_11_)
        relu28_1_11_ = self.relu28_1_11_(batchnorm28_1_11_)
        conv29_1_11_ = self.conv29_1_11_(relu28_1_11_)
        batchnorm29_1_11_ = self.batchnorm29_1_11_(conv29_1_11_)
        conv27_1_12_ = self.conv27_1_12_(relu25_)
        batchnorm27_1_12_ = self.batchnorm27_1_12_(conv27_1_12_)
        relu27_1_12_ = self.relu27_1_12_(batchnorm27_1_12_)
        conv28_1_12_padding = self.conv28_1_12_padding(relu27_1_12_)
        conv28_1_12_ = self.conv28_1_12_(conv28_1_12_padding)
        batchnorm28_1_12_ = self.batchnorm28_1_12_(conv28_1_12_)
        relu28_1_12_ = self.relu28_1_12_(batchnorm28_1_12_)
        conv29_1_12_ = self.conv29_1_12_(relu28_1_12_)
        batchnorm29_1_12_ = self.batchnorm29_1_12_(conv29_1_12_)
        conv27_1_13_ = self.conv27_1_13_(relu25_)
        batchnorm27_1_13_ = self.batchnorm27_1_13_(conv27_1_13_)
        relu27_1_13_ = self.relu27_1_13_(batchnorm27_1_13_)
        conv28_1_13_padding = self.conv28_1_13_padding(relu27_1_13_)
        conv28_1_13_ = self.conv28_1_13_(conv28_1_13_padding)
        batchnorm28_1_13_ = self.batchnorm28_1_13_(conv28_1_13_)
        relu28_1_13_ = self.relu28_1_13_(batchnorm28_1_13_)
        conv29_1_13_ = self.conv29_1_13_(relu28_1_13_)
        batchnorm29_1_13_ = self.batchnorm29_1_13_(conv29_1_13_)
        conv27_1_14_ = self.conv27_1_14_(relu25_)
        batchnorm27_1_14_ = self.batchnorm27_1_14_(conv27_1_14_)
        relu27_1_14_ = self.relu27_1_14_(batchnorm27_1_14_)
        conv28_1_14_padding = self.conv28_1_14_padding(relu27_1_14_)
        conv28_1_14_ = self.conv28_1_14_(conv28_1_14_padding)
        batchnorm28_1_14_ = self.batchnorm28_1_14_(conv28_1_14_)
        relu28_1_14_ = self.relu28_1_14_(batchnorm28_1_14_)
        conv29_1_14_ = self.conv29_1_14_(relu28_1_14_)
        batchnorm29_1_14_ = self.batchnorm29_1_14_(conv29_1_14_)
        conv27_1_15_ = self.conv27_1_15_(relu25_)
        batchnorm27_1_15_ = self.batchnorm27_1_15_(conv27_1_15_)
        relu27_1_15_ = self.relu27_1_15_(batchnorm27_1_15_)
        conv28_1_15_padding = self.conv28_1_15_padding(relu27_1_15_)
        conv28_1_15_ = self.conv28_1_15_(conv28_1_15_padding)
        batchnorm28_1_15_ = self.batchnorm28_1_15_(conv28_1_15_)
        relu28_1_15_ = self.relu28_1_15_(batchnorm28_1_15_)
        conv29_1_15_ = self.conv29_1_15_(relu28_1_15_)
        batchnorm29_1_15_ = self.batchnorm29_1_15_(conv29_1_15_)
        conv27_1_16_ = self.conv27_1_16_(relu25_)
        batchnorm27_1_16_ = self.batchnorm27_1_16_(conv27_1_16_)
        relu27_1_16_ = self.relu27_1_16_(batchnorm27_1_16_)
        conv28_1_16_padding = self.conv28_1_16_padding(relu27_1_16_)
        conv28_1_16_ = self.conv28_1_16_(conv28_1_16_padding)
        batchnorm28_1_16_ = self.batchnorm28_1_16_(conv28_1_16_)
        relu28_1_16_ = self.relu28_1_16_(batchnorm28_1_16_)
        conv29_1_16_ = self.conv29_1_16_(relu28_1_16_)
        batchnorm29_1_16_ = self.batchnorm29_1_16_(conv29_1_16_)
        conv27_1_17_ = self.conv27_1_17_(relu25_)
        batchnorm27_1_17_ = self.batchnorm27_1_17_(conv27_1_17_)
        relu27_1_17_ = self.relu27_1_17_(batchnorm27_1_17_)
        conv28_1_17_padding = self.conv28_1_17_padding(relu27_1_17_)
        conv28_1_17_ = self.conv28_1_17_(conv28_1_17_padding)
        batchnorm28_1_17_ = self.batchnorm28_1_17_(conv28_1_17_)
        relu28_1_17_ = self.relu28_1_17_(batchnorm28_1_17_)
        conv29_1_17_ = self.conv29_1_17_(relu28_1_17_)
        batchnorm29_1_17_ = self.batchnorm29_1_17_(conv29_1_17_)
        conv27_1_18_ = self.conv27_1_18_(relu25_)
        batchnorm27_1_18_ = self.batchnorm27_1_18_(conv27_1_18_)
        relu27_1_18_ = self.relu27_1_18_(batchnorm27_1_18_)
        conv28_1_18_padding = self.conv28_1_18_padding(relu27_1_18_)
        conv28_1_18_ = self.conv28_1_18_(conv28_1_18_padding)
        batchnorm28_1_18_ = self.batchnorm28_1_18_(conv28_1_18_)
        relu28_1_18_ = self.relu28_1_18_(batchnorm28_1_18_)
        conv29_1_18_ = self.conv29_1_18_(relu28_1_18_)
        batchnorm29_1_18_ = self.batchnorm29_1_18_(conv29_1_18_)
        conv27_1_19_ = self.conv27_1_19_(relu25_)
        batchnorm27_1_19_ = self.batchnorm27_1_19_(conv27_1_19_)
        relu27_1_19_ = self.relu27_1_19_(batchnorm27_1_19_)
        conv28_1_19_padding = self.conv28_1_19_padding(relu27_1_19_)
        conv28_1_19_ = self.conv28_1_19_(conv28_1_19_padding)
        batchnorm28_1_19_ = self.batchnorm28_1_19_(conv28_1_19_)
        relu28_1_19_ = self.relu28_1_19_(batchnorm28_1_19_)
        conv29_1_19_ = self.conv29_1_19_(relu28_1_19_)
        batchnorm29_1_19_ = self.batchnorm29_1_19_(conv29_1_19_)
        conv27_1_20_ = self.conv27_1_20_(relu25_)
        batchnorm27_1_20_ = self.batchnorm27_1_20_(conv27_1_20_)
        relu27_1_20_ = self.relu27_1_20_(batchnorm27_1_20_)
        conv28_1_20_padding = self.conv28_1_20_padding(relu27_1_20_)
        conv28_1_20_ = self.conv28_1_20_(conv28_1_20_padding)
        batchnorm28_1_20_ = self.batchnorm28_1_20_(conv28_1_20_)
        relu28_1_20_ = self.relu28_1_20_(batchnorm28_1_20_)
        conv29_1_20_ = self.conv29_1_20_(relu28_1_20_)
        batchnorm29_1_20_ = self.batchnorm29_1_20_(conv29_1_20_)
        conv27_1_21_ = self.conv27_1_21_(relu25_)
        batchnorm27_1_21_ = self.batchnorm27_1_21_(conv27_1_21_)
        relu27_1_21_ = self.relu27_1_21_(batchnorm27_1_21_)
        conv28_1_21_padding = self.conv28_1_21_padding(relu27_1_21_)
        conv28_1_21_ = self.conv28_1_21_(conv28_1_21_padding)
        batchnorm28_1_21_ = self.batchnorm28_1_21_(conv28_1_21_)
        relu28_1_21_ = self.relu28_1_21_(batchnorm28_1_21_)
        conv29_1_21_ = self.conv29_1_21_(relu28_1_21_)
        batchnorm29_1_21_ = self.batchnorm29_1_21_(conv29_1_21_)
        conv27_1_22_ = self.conv27_1_22_(relu25_)
        batchnorm27_1_22_ = self.batchnorm27_1_22_(conv27_1_22_)
        relu27_1_22_ = self.relu27_1_22_(batchnorm27_1_22_)
        conv28_1_22_padding = self.conv28_1_22_padding(relu27_1_22_)
        conv28_1_22_ = self.conv28_1_22_(conv28_1_22_padding)
        batchnorm28_1_22_ = self.batchnorm28_1_22_(conv28_1_22_)
        relu28_1_22_ = self.relu28_1_22_(batchnorm28_1_22_)
        conv29_1_22_ = self.conv29_1_22_(relu28_1_22_)
        batchnorm29_1_22_ = self.batchnorm29_1_22_(conv29_1_22_)
        conv27_1_23_ = self.conv27_1_23_(relu25_)
        batchnorm27_1_23_ = self.batchnorm27_1_23_(conv27_1_23_)
        relu27_1_23_ = self.relu27_1_23_(batchnorm27_1_23_)
        conv28_1_23_padding = self.conv28_1_23_padding(relu27_1_23_)
        conv28_1_23_ = self.conv28_1_23_(conv28_1_23_padding)
        batchnorm28_1_23_ = self.batchnorm28_1_23_(conv28_1_23_)
        relu28_1_23_ = self.relu28_1_23_(batchnorm28_1_23_)
        conv29_1_23_ = self.conv29_1_23_(relu28_1_23_)
        batchnorm29_1_23_ = self.batchnorm29_1_23_(conv29_1_23_)
        conv27_1_24_ = self.conv27_1_24_(relu25_)
        batchnorm27_1_24_ = self.batchnorm27_1_24_(conv27_1_24_)
        relu27_1_24_ = self.relu27_1_24_(batchnorm27_1_24_)
        conv28_1_24_padding = self.conv28_1_24_padding(relu27_1_24_)
        conv28_1_24_ = self.conv28_1_24_(conv28_1_24_padding)
        batchnorm28_1_24_ = self.batchnorm28_1_24_(conv28_1_24_)
        relu28_1_24_ = self.relu28_1_24_(batchnorm28_1_24_)
        conv29_1_24_ = self.conv29_1_24_(relu28_1_24_)
        batchnorm29_1_24_ = self.batchnorm29_1_24_(conv29_1_24_)
        conv27_1_25_ = self.conv27_1_25_(relu25_)
        batchnorm27_1_25_ = self.batchnorm27_1_25_(conv27_1_25_)
        relu27_1_25_ = self.relu27_1_25_(batchnorm27_1_25_)
        conv28_1_25_padding = self.conv28_1_25_padding(relu27_1_25_)
        conv28_1_25_ = self.conv28_1_25_(conv28_1_25_padding)
        batchnorm28_1_25_ = self.batchnorm28_1_25_(conv28_1_25_)
        relu28_1_25_ = self.relu28_1_25_(batchnorm28_1_25_)
        conv29_1_25_ = self.conv29_1_25_(relu28_1_25_)
        batchnorm29_1_25_ = self.batchnorm29_1_25_(conv29_1_25_)
        conv27_1_26_ = self.conv27_1_26_(relu25_)
        batchnorm27_1_26_ = self.batchnorm27_1_26_(conv27_1_26_)
        relu27_1_26_ = self.relu27_1_26_(batchnorm27_1_26_)
        conv28_1_26_padding = self.conv28_1_26_padding(relu27_1_26_)
        conv28_1_26_ = self.conv28_1_26_(conv28_1_26_padding)
        batchnorm28_1_26_ = self.batchnorm28_1_26_(conv28_1_26_)
        relu28_1_26_ = self.relu28_1_26_(batchnorm28_1_26_)
        conv29_1_26_ = self.conv29_1_26_(relu28_1_26_)
        batchnorm29_1_26_ = self.batchnorm29_1_26_(conv29_1_26_)
        conv27_1_27_ = self.conv27_1_27_(relu25_)
        batchnorm27_1_27_ = self.batchnorm27_1_27_(conv27_1_27_)
        relu27_1_27_ = self.relu27_1_27_(batchnorm27_1_27_)
        conv28_1_27_padding = self.conv28_1_27_padding(relu27_1_27_)
        conv28_1_27_ = self.conv28_1_27_(conv28_1_27_padding)
        batchnorm28_1_27_ = self.batchnorm28_1_27_(conv28_1_27_)
        relu28_1_27_ = self.relu28_1_27_(batchnorm28_1_27_)
        conv29_1_27_ = self.conv29_1_27_(relu28_1_27_)
        batchnorm29_1_27_ = self.batchnorm29_1_27_(conv29_1_27_)
        conv27_1_28_ = self.conv27_1_28_(relu25_)
        batchnorm27_1_28_ = self.batchnorm27_1_28_(conv27_1_28_)
        relu27_1_28_ = self.relu27_1_28_(batchnorm27_1_28_)
        conv28_1_28_padding = self.conv28_1_28_padding(relu27_1_28_)
        conv28_1_28_ = self.conv28_1_28_(conv28_1_28_padding)
        batchnorm28_1_28_ = self.batchnorm28_1_28_(conv28_1_28_)
        relu28_1_28_ = self.relu28_1_28_(batchnorm28_1_28_)
        conv29_1_28_ = self.conv29_1_28_(relu28_1_28_)
        batchnorm29_1_28_ = self.batchnorm29_1_28_(conv29_1_28_)
        conv27_1_29_ = self.conv27_1_29_(relu25_)
        batchnorm27_1_29_ = self.batchnorm27_1_29_(conv27_1_29_)
        relu27_1_29_ = self.relu27_1_29_(batchnorm27_1_29_)
        conv28_1_29_padding = self.conv28_1_29_padding(relu27_1_29_)
        conv28_1_29_ = self.conv28_1_29_(conv28_1_29_padding)
        batchnorm28_1_29_ = self.batchnorm28_1_29_(conv28_1_29_)
        relu28_1_29_ = self.relu28_1_29_(batchnorm28_1_29_)
        conv29_1_29_ = self.conv29_1_29_(relu28_1_29_)
        batchnorm29_1_29_ = self.batchnorm29_1_29_(conv29_1_29_)
        conv27_1_30_ = self.conv27_1_30_(relu25_)
        batchnorm27_1_30_ = self.batchnorm27_1_30_(conv27_1_30_)
        relu27_1_30_ = self.relu27_1_30_(batchnorm27_1_30_)
        conv28_1_30_padding = self.conv28_1_30_padding(relu27_1_30_)
        conv28_1_30_ = self.conv28_1_30_(conv28_1_30_padding)
        batchnorm28_1_30_ = self.batchnorm28_1_30_(conv28_1_30_)
        relu28_1_30_ = self.relu28_1_30_(batchnorm28_1_30_)
        conv29_1_30_ = self.conv29_1_30_(relu28_1_30_)
        batchnorm29_1_30_ = self.batchnorm29_1_30_(conv29_1_30_)
        conv27_1_31_ = self.conv27_1_31_(relu25_)
        batchnorm27_1_31_ = self.batchnorm27_1_31_(conv27_1_31_)
        relu27_1_31_ = self.relu27_1_31_(batchnorm27_1_31_)
        conv28_1_31_padding = self.conv28_1_31_padding(relu27_1_31_)
        conv28_1_31_ = self.conv28_1_31_(conv28_1_31_padding)
        batchnorm28_1_31_ = self.batchnorm28_1_31_(conv28_1_31_)
        relu28_1_31_ = self.relu28_1_31_(batchnorm28_1_31_)
        conv29_1_31_ = self.conv29_1_31_(relu28_1_31_)
        batchnorm29_1_31_ = self.batchnorm29_1_31_(conv29_1_31_)
        conv27_1_32_ = self.conv27_1_32_(relu25_)
        batchnorm27_1_32_ = self.batchnorm27_1_32_(conv27_1_32_)
        relu27_1_32_ = self.relu27_1_32_(batchnorm27_1_32_)
        conv28_1_32_padding = self.conv28_1_32_padding(relu27_1_32_)
        conv28_1_32_ = self.conv28_1_32_(conv28_1_32_padding)
        batchnorm28_1_32_ = self.batchnorm28_1_32_(conv28_1_32_)
        relu28_1_32_ = self.relu28_1_32_(batchnorm28_1_32_)
        conv29_1_32_ = self.conv29_1_32_(relu28_1_32_)
        batchnorm29_1_32_ = self.batchnorm29_1_32_(conv29_1_32_)
        add30_1_ = batchnorm29_1_1_ + batchnorm29_1_2_ + batchnorm29_1_3_ + batchnorm29_1_4_ + batchnorm29_1_5_ + batchnorm29_1_6_ + batchnorm29_1_7_ + batchnorm29_1_8_ + batchnorm29_1_9_ + batchnorm29_1_10_ + batchnorm29_1_11_ + batchnorm29_1_12_ + batchnorm29_1_13_ + batchnorm29_1_14_ + batchnorm29_1_15_ + batchnorm29_1_16_ + batchnorm29_1_17_ + batchnorm29_1_18_ + batchnorm29_1_19_ + batchnorm29_1_20_ + batchnorm29_1_21_ + batchnorm29_1_22_ + batchnorm29_1_23_ + batchnorm29_1_24_ + batchnorm29_1_25_ + batchnorm29_1_26_ + batchnorm29_1_27_ + batchnorm29_1_28_ + batchnorm29_1_29_ + batchnorm29_1_30_ + batchnorm29_1_31_ + batchnorm29_1_32_
        add31_ = add30_1_ + relu25_
        relu31_ = self.relu31_(add31_)
        conv33_1_1_ = self.conv33_1_1_(relu31_)
        batchnorm33_1_1_ = self.batchnorm33_1_1_(conv33_1_1_)
        relu33_1_1_ = self.relu33_1_1_(batchnorm33_1_1_)
        conv34_1_1_padding = self.conv34_1_1_padding(relu33_1_1_)
        conv34_1_1_ = self.conv34_1_1_(conv34_1_1_padding)
        batchnorm34_1_1_ = self.batchnorm34_1_1_(conv34_1_1_)
        relu34_1_1_ = self.relu34_1_1_(batchnorm34_1_1_)
        conv35_1_1_ = self.conv35_1_1_(relu34_1_1_)
        batchnorm35_1_1_ = self.batchnorm35_1_1_(conv35_1_1_)
        conv33_1_2_ = self.conv33_1_2_(relu31_)
        batchnorm33_1_2_ = self.batchnorm33_1_2_(conv33_1_2_)
        relu33_1_2_ = self.relu33_1_2_(batchnorm33_1_2_)
        conv34_1_2_padding = self.conv34_1_2_padding(relu33_1_2_)
        conv34_1_2_ = self.conv34_1_2_(conv34_1_2_padding)
        batchnorm34_1_2_ = self.batchnorm34_1_2_(conv34_1_2_)
        relu34_1_2_ = self.relu34_1_2_(batchnorm34_1_2_)
        conv35_1_2_ = self.conv35_1_2_(relu34_1_2_)
        batchnorm35_1_2_ = self.batchnorm35_1_2_(conv35_1_2_)
        conv33_1_3_ = self.conv33_1_3_(relu31_)
        batchnorm33_1_3_ = self.batchnorm33_1_3_(conv33_1_3_)
        relu33_1_3_ = self.relu33_1_3_(batchnorm33_1_3_)
        conv34_1_3_padding = self.conv34_1_3_padding(relu33_1_3_)
        conv34_1_3_ = self.conv34_1_3_(conv34_1_3_padding)
        batchnorm34_1_3_ = self.batchnorm34_1_3_(conv34_1_3_)
        relu34_1_3_ = self.relu34_1_3_(batchnorm34_1_3_)
        conv35_1_3_ = self.conv35_1_3_(relu34_1_3_)
        batchnorm35_1_3_ = self.batchnorm35_1_3_(conv35_1_3_)
        conv33_1_4_ = self.conv33_1_4_(relu31_)
        batchnorm33_1_4_ = self.batchnorm33_1_4_(conv33_1_4_)
        relu33_1_4_ = self.relu33_1_4_(batchnorm33_1_4_)
        conv34_1_4_padding = self.conv34_1_4_padding(relu33_1_4_)
        conv34_1_4_ = self.conv34_1_4_(conv34_1_4_padding)
        batchnorm34_1_4_ = self.batchnorm34_1_4_(conv34_1_4_)
        relu34_1_4_ = self.relu34_1_4_(batchnorm34_1_4_)
        conv35_1_4_ = self.conv35_1_4_(relu34_1_4_)
        batchnorm35_1_4_ = self.batchnorm35_1_4_(conv35_1_4_)
        conv33_1_5_ = self.conv33_1_5_(relu31_)
        batchnorm33_1_5_ = self.batchnorm33_1_5_(conv33_1_5_)
        relu33_1_5_ = self.relu33_1_5_(batchnorm33_1_5_)
        conv34_1_5_padding = self.conv34_1_5_padding(relu33_1_5_)
        conv34_1_5_ = self.conv34_1_5_(conv34_1_5_padding)
        batchnorm34_1_5_ = self.batchnorm34_1_5_(conv34_1_5_)
        relu34_1_5_ = self.relu34_1_5_(batchnorm34_1_5_)
        conv35_1_5_ = self.conv35_1_5_(relu34_1_5_)
        batchnorm35_1_5_ = self.batchnorm35_1_5_(conv35_1_5_)
        conv33_1_6_ = self.conv33_1_6_(relu31_)
        batchnorm33_1_6_ = self.batchnorm33_1_6_(conv33_1_6_)
        relu33_1_6_ = self.relu33_1_6_(batchnorm33_1_6_)
        conv34_1_6_padding = self.conv34_1_6_padding(relu33_1_6_)
        conv34_1_6_ = self.conv34_1_6_(conv34_1_6_padding)
        batchnorm34_1_6_ = self.batchnorm34_1_6_(conv34_1_6_)
        relu34_1_6_ = self.relu34_1_6_(batchnorm34_1_6_)
        conv35_1_6_ = self.conv35_1_6_(relu34_1_6_)
        batchnorm35_1_6_ = self.batchnorm35_1_6_(conv35_1_6_)
        conv33_1_7_ = self.conv33_1_7_(relu31_)
        batchnorm33_1_7_ = self.batchnorm33_1_7_(conv33_1_7_)
        relu33_1_7_ = self.relu33_1_7_(batchnorm33_1_7_)
        conv34_1_7_padding = self.conv34_1_7_padding(relu33_1_7_)
        conv34_1_7_ = self.conv34_1_7_(conv34_1_7_padding)
        batchnorm34_1_7_ = self.batchnorm34_1_7_(conv34_1_7_)
        relu34_1_7_ = self.relu34_1_7_(batchnorm34_1_7_)
        conv35_1_7_ = self.conv35_1_7_(relu34_1_7_)
        batchnorm35_1_7_ = self.batchnorm35_1_7_(conv35_1_7_)
        conv33_1_8_ = self.conv33_1_8_(relu31_)
        batchnorm33_1_8_ = self.batchnorm33_1_8_(conv33_1_8_)
        relu33_1_8_ = self.relu33_1_8_(batchnorm33_1_8_)
        conv34_1_8_padding = self.conv34_1_8_padding(relu33_1_8_)
        conv34_1_8_ = self.conv34_1_8_(conv34_1_8_padding)
        batchnorm34_1_8_ = self.batchnorm34_1_8_(conv34_1_8_)
        relu34_1_8_ = self.relu34_1_8_(batchnorm34_1_8_)
        conv35_1_8_ = self.conv35_1_8_(relu34_1_8_)
        batchnorm35_1_8_ = self.batchnorm35_1_8_(conv35_1_8_)
        conv33_1_9_ = self.conv33_1_9_(relu31_)
        batchnorm33_1_9_ = self.batchnorm33_1_9_(conv33_1_9_)
        relu33_1_9_ = self.relu33_1_9_(batchnorm33_1_9_)
        conv34_1_9_padding = self.conv34_1_9_padding(relu33_1_9_)
        conv34_1_9_ = self.conv34_1_9_(conv34_1_9_padding)
        batchnorm34_1_9_ = self.batchnorm34_1_9_(conv34_1_9_)
        relu34_1_9_ = self.relu34_1_9_(batchnorm34_1_9_)
        conv35_1_9_ = self.conv35_1_9_(relu34_1_9_)
        batchnorm35_1_9_ = self.batchnorm35_1_9_(conv35_1_9_)
        conv33_1_10_ = self.conv33_1_10_(relu31_)
        batchnorm33_1_10_ = self.batchnorm33_1_10_(conv33_1_10_)
        relu33_1_10_ = self.relu33_1_10_(batchnorm33_1_10_)
        conv34_1_10_padding = self.conv34_1_10_padding(relu33_1_10_)
        conv34_1_10_ = self.conv34_1_10_(conv34_1_10_padding)
        batchnorm34_1_10_ = self.batchnorm34_1_10_(conv34_1_10_)
        relu34_1_10_ = self.relu34_1_10_(batchnorm34_1_10_)
        conv35_1_10_ = self.conv35_1_10_(relu34_1_10_)
        batchnorm35_1_10_ = self.batchnorm35_1_10_(conv35_1_10_)
        conv33_1_11_ = self.conv33_1_11_(relu31_)
        batchnorm33_1_11_ = self.batchnorm33_1_11_(conv33_1_11_)
        relu33_1_11_ = self.relu33_1_11_(batchnorm33_1_11_)
        conv34_1_11_padding = self.conv34_1_11_padding(relu33_1_11_)
        conv34_1_11_ = self.conv34_1_11_(conv34_1_11_padding)
        batchnorm34_1_11_ = self.batchnorm34_1_11_(conv34_1_11_)
        relu34_1_11_ = self.relu34_1_11_(batchnorm34_1_11_)
        conv35_1_11_ = self.conv35_1_11_(relu34_1_11_)
        batchnorm35_1_11_ = self.batchnorm35_1_11_(conv35_1_11_)
        conv33_1_12_ = self.conv33_1_12_(relu31_)
        batchnorm33_1_12_ = self.batchnorm33_1_12_(conv33_1_12_)
        relu33_1_12_ = self.relu33_1_12_(batchnorm33_1_12_)
        conv34_1_12_padding = self.conv34_1_12_padding(relu33_1_12_)
        conv34_1_12_ = self.conv34_1_12_(conv34_1_12_padding)
        batchnorm34_1_12_ = self.batchnorm34_1_12_(conv34_1_12_)
        relu34_1_12_ = self.relu34_1_12_(batchnorm34_1_12_)
        conv35_1_12_ = self.conv35_1_12_(relu34_1_12_)
        batchnorm35_1_12_ = self.batchnorm35_1_12_(conv35_1_12_)
        conv33_1_13_ = self.conv33_1_13_(relu31_)
        batchnorm33_1_13_ = self.batchnorm33_1_13_(conv33_1_13_)
        relu33_1_13_ = self.relu33_1_13_(batchnorm33_1_13_)
        conv34_1_13_padding = self.conv34_1_13_padding(relu33_1_13_)
        conv34_1_13_ = self.conv34_1_13_(conv34_1_13_padding)
        batchnorm34_1_13_ = self.batchnorm34_1_13_(conv34_1_13_)
        relu34_1_13_ = self.relu34_1_13_(batchnorm34_1_13_)
        conv35_1_13_ = self.conv35_1_13_(relu34_1_13_)
        batchnorm35_1_13_ = self.batchnorm35_1_13_(conv35_1_13_)
        conv33_1_14_ = self.conv33_1_14_(relu31_)
        batchnorm33_1_14_ = self.batchnorm33_1_14_(conv33_1_14_)
        relu33_1_14_ = self.relu33_1_14_(batchnorm33_1_14_)
        conv34_1_14_padding = self.conv34_1_14_padding(relu33_1_14_)
        conv34_1_14_ = self.conv34_1_14_(conv34_1_14_padding)
        batchnorm34_1_14_ = self.batchnorm34_1_14_(conv34_1_14_)
        relu34_1_14_ = self.relu34_1_14_(batchnorm34_1_14_)
        conv35_1_14_ = self.conv35_1_14_(relu34_1_14_)
        batchnorm35_1_14_ = self.batchnorm35_1_14_(conv35_1_14_)
        conv33_1_15_ = self.conv33_1_15_(relu31_)
        batchnorm33_1_15_ = self.batchnorm33_1_15_(conv33_1_15_)
        relu33_1_15_ = self.relu33_1_15_(batchnorm33_1_15_)
        conv34_1_15_padding = self.conv34_1_15_padding(relu33_1_15_)
        conv34_1_15_ = self.conv34_1_15_(conv34_1_15_padding)
        batchnorm34_1_15_ = self.batchnorm34_1_15_(conv34_1_15_)
        relu34_1_15_ = self.relu34_1_15_(batchnorm34_1_15_)
        conv35_1_15_ = self.conv35_1_15_(relu34_1_15_)
        batchnorm35_1_15_ = self.batchnorm35_1_15_(conv35_1_15_)
        conv33_1_16_ = self.conv33_1_16_(relu31_)
        batchnorm33_1_16_ = self.batchnorm33_1_16_(conv33_1_16_)
        relu33_1_16_ = self.relu33_1_16_(batchnorm33_1_16_)
        conv34_1_16_padding = self.conv34_1_16_padding(relu33_1_16_)
        conv34_1_16_ = self.conv34_1_16_(conv34_1_16_padding)
        batchnorm34_1_16_ = self.batchnorm34_1_16_(conv34_1_16_)
        relu34_1_16_ = self.relu34_1_16_(batchnorm34_1_16_)
        conv35_1_16_ = self.conv35_1_16_(relu34_1_16_)
        batchnorm35_1_16_ = self.batchnorm35_1_16_(conv35_1_16_)
        conv33_1_17_ = self.conv33_1_17_(relu31_)
        batchnorm33_1_17_ = self.batchnorm33_1_17_(conv33_1_17_)
        relu33_1_17_ = self.relu33_1_17_(batchnorm33_1_17_)
        conv34_1_17_padding = self.conv34_1_17_padding(relu33_1_17_)
        conv34_1_17_ = self.conv34_1_17_(conv34_1_17_padding)
        batchnorm34_1_17_ = self.batchnorm34_1_17_(conv34_1_17_)
        relu34_1_17_ = self.relu34_1_17_(batchnorm34_1_17_)
        conv35_1_17_ = self.conv35_1_17_(relu34_1_17_)
        batchnorm35_1_17_ = self.batchnorm35_1_17_(conv35_1_17_)
        conv33_1_18_ = self.conv33_1_18_(relu31_)
        batchnorm33_1_18_ = self.batchnorm33_1_18_(conv33_1_18_)
        relu33_1_18_ = self.relu33_1_18_(batchnorm33_1_18_)
        conv34_1_18_padding = self.conv34_1_18_padding(relu33_1_18_)
        conv34_1_18_ = self.conv34_1_18_(conv34_1_18_padding)
        batchnorm34_1_18_ = self.batchnorm34_1_18_(conv34_1_18_)
        relu34_1_18_ = self.relu34_1_18_(batchnorm34_1_18_)
        conv35_1_18_ = self.conv35_1_18_(relu34_1_18_)
        batchnorm35_1_18_ = self.batchnorm35_1_18_(conv35_1_18_)
        conv33_1_19_ = self.conv33_1_19_(relu31_)
        batchnorm33_1_19_ = self.batchnorm33_1_19_(conv33_1_19_)
        relu33_1_19_ = self.relu33_1_19_(batchnorm33_1_19_)
        conv34_1_19_padding = self.conv34_1_19_padding(relu33_1_19_)
        conv34_1_19_ = self.conv34_1_19_(conv34_1_19_padding)
        batchnorm34_1_19_ = self.batchnorm34_1_19_(conv34_1_19_)
        relu34_1_19_ = self.relu34_1_19_(batchnorm34_1_19_)
        conv35_1_19_ = self.conv35_1_19_(relu34_1_19_)
        batchnorm35_1_19_ = self.batchnorm35_1_19_(conv35_1_19_)
        conv33_1_20_ = self.conv33_1_20_(relu31_)
        batchnorm33_1_20_ = self.batchnorm33_1_20_(conv33_1_20_)
        relu33_1_20_ = self.relu33_1_20_(batchnorm33_1_20_)
        conv34_1_20_padding = self.conv34_1_20_padding(relu33_1_20_)
        conv34_1_20_ = self.conv34_1_20_(conv34_1_20_padding)
        batchnorm34_1_20_ = self.batchnorm34_1_20_(conv34_1_20_)
        relu34_1_20_ = self.relu34_1_20_(batchnorm34_1_20_)
        conv35_1_20_ = self.conv35_1_20_(relu34_1_20_)
        batchnorm35_1_20_ = self.batchnorm35_1_20_(conv35_1_20_)
        conv33_1_21_ = self.conv33_1_21_(relu31_)
        batchnorm33_1_21_ = self.batchnorm33_1_21_(conv33_1_21_)
        relu33_1_21_ = self.relu33_1_21_(batchnorm33_1_21_)
        conv34_1_21_padding = self.conv34_1_21_padding(relu33_1_21_)
        conv34_1_21_ = self.conv34_1_21_(conv34_1_21_padding)
        batchnorm34_1_21_ = self.batchnorm34_1_21_(conv34_1_21_)
        relu34_1_21_ = self.relu34_1_21_(batchnorm34_1_21_)
        conv35_1_21_ = self.conv35_1_21_(relu34_1_21_)
        batchnorm35_1_21_ = self.batchnorm35_1_21_(conv35_1_21_)
        conv33_1_22_ = self.conv33_1_22_(relu31_)
        batchnorm33_1_22_ = self.batchnorm33_1_22_(conv33_1_22_)
        relu33_1_22_ = self.relu33_1_22_(batchnorm33_1_22_)
        conv34_1_22_padding = self.conv34_1_22_padding(relu33_1_22_)
        conv34_1_22_ = self.conv34_1_22_(conv34_1_22_padding)
        batchnorm34_1_22_ = self.batchnorm34_1_22_(conv34_1_22_)
        relu34_1_22_ = self.relu34_1_22_(batchnorm34_1_22_)
        conv35_1_22_ = self.conv35_1_22_(relu34_1_22_)
        batchnorm35_1_22_ = self.batchnorm35_1_22_(conv35_1_22_)
        conv33_1_23_ = self.conv33_1_23_(relu31_)
        batchnorm33_1_23_ = self.batchnorm33_1_23_(conv33_1_23_)
        relu33_1_23_ = self.relu33_1_23_(batchnorm33_1_23_)
        conv34_1_23_padding = self.conv34_1_23_padding(relu33_1_23_)
        conv34_1_23_ = self.conv34_1_23_(conv34_1_23_padding)
        batchnorm34_1_23_ = self.batchnorm34_1_23_(conv34_1_23_)
        relu34_1_23_ = self.relu34_1_23_(batchnorm34_1_23_)
        conv35_1_23_ = self.conv35_1_23_(relu34_1_23_)
        batchnorm35_1_23_ = self.batchnorm35_1_23_(conv35_1_23_)
        conv33_1_24_ = self.conv33_1_24_(relu31_)
        batchnorm33_1_24_ = self.batchnorm33_1_24_(conv33_1_24_)
        relu33_1_24_ = self.relu33_1_24_(batchnorm33_1_24_)
        conv34_1_24_padding = self.conv34_1_24_padding(relu33_1_24_)
        conv34_1_24_ = self.conv34_1_24_(conv34_1_24_padding)
        batchnorm34_1_24_ = self.batchnorm34_1_24_(conv34_1_24_)
        relu34_1_24_ = self.relu34_1_24_(batchnorm34_1_24_)
        conv35_1_24_ = self.conv35_1_24_(relu34_1_24_)
        batchnorm35_1_24_ = self.batchnorm35_1_24_(conv35_1_24_)
        conv33_1_25_ = self.conv33_1_25_(relu31_)
        batchnorm33_1_25_ = self.batchnorm33_1_25_(conv33_1_25_)
        relu33_1_25_ = self.relu33_1_25_(batchnorm33_1_25_)
        conv34_1_25_padding = self.conv34_1_25_padding(relu33_1_25_)
        conv34_1_25_ = self.conv34_1_25_(conv34_1_25_padding)
        batchnorm34_1_25_ = self.batchnorm34_1_25_(conv34_1_25_)
        relu34_1_25_ = self.relu34_1_25_(batchnorm34_1_25_)
        conv35_1_25_ = self.conv35_1_25_(relu34_1_25_)
        batchnorm35_1_25_ = self.batchnorm35_1_25_(conv35_1_25_)
        conv33_1_26_ = self.conv33_1_26_(relu31_)
        batchnorm33_1_26_ = self.batchnorm33_1_26_(conv33_1_26_)
        relu33_1_26_ = self.relu33_1_26_(batchnorm33_1_26_)
        conv34_1_26_padding = self.conv34_1_26_padding(relu33_1_26_)
        conv34_1_26_ = self.conv34_1_26_(conv34_1_26_padding)
        batchnorm34_1_26_ = self.batchnorm34_1_26_(conv34_1_26_)
        relu34_1_26_ = self.relu34_1_26_(batchnorm34_1_26_)
        conv35_1_26_ = self.conv35_1_26_(relu34_1_26_)
        batchnorm35_1_26_ = self.batchnorm35_1_26_(conv35_1_26_)
        conv33_1_27_ = self.conv33_1_27_(relu31_)
        batchnorm33_1_27_ = self.batchnorm33_1_27_(conv33_1_27_)
        relu33_1_27_ = self.relu33_1_27_(batchnorm33_1_27_)
        conv34_1_27_padding = self.conv34_1_27_padding(relu33_1_27_)
        conv34_1_27_ = self.conv34_1_27_(conv34_1_27_padding)
        batchnorm34_1_27_ = self.batchnorm34_1_27_(conv34_1_27_)
        relu34_1_27_ = self.relu34_1_27_(batchnorm34_1_27_)
        conv35_1_27_ = self.conv35_1_27_(relu34_1_27_)
        batchnorm35_1_27_ = self.batchnorm35_1_27_(conv35_1_27_)
        conv33_1_28_ = self.conv33_1_28_(relu31_)
        batchnorm33_1_28_ = self.batchnorm33_1_28_(conv33_1_28_)
        relu33_1_28_ = self.relu33_1_28_(batchnorm33_1_28_)
        conv34_1_28_padding = self.conv34_1_28_padding(relu33_1_28_)
        conv34_1_28_ = self.conv34_1_28_(conv34_1_28_padding)
        batchnorm34_1_28_ = self.batchnorm34_1_28_(conv34_1_28_)
        relu34_1_28_ = self.relu34_1_28_(batchnorm34_1_28_)
        conv35_1_28_ = self.conv35_1_28_(relu34_1_28_)
        batchnorm35_1_28_ = self.batchnorm35_1_28_(conv35_1_28_)
        conv33_1_29_ = self.conv33_1_29_(relu31_)
        batchnorm33_1_29_ = self.batchnorm33_1_29_(conv33_1_29_)
        relu33_1_29_ = self.relu33_1_29_(batchnorm33_1_29_)
        conv34_1_29_padding = self.conv34_1_29_padding(relu33_1_29_)
        conv34_1_29_ = self.conv34_1_29_(conv34_1_29_padding)
        batchnorm34_1_29_ = self.batchnorm34_1_29_(conv34_1_29_)
        relu34_1_29_ = self.relu34_1_29_(batchnorm34_1_29_)
        conv35_1_29_ = self.conv35_1_29_(relu34_1_29_)
        batchnorm35_1_29_ = self.batchnorm35_1_29_(conv35_1_29_)
        conv33_1_30_ = self.conv33_1_30_(relu31_)
        batchnorm33_1_30_ = self.batchnorm33_1_30_(conv33_1_30_)
        relu33_1_30_ = self.relu33_1_30_(batchnorm33_1_30_)
        conv34_1_30_padding = self.conv34_1_30_padding(relu33_1_30_)
        conv34_1_30_ = self.conv34_1_30_(conv34_1_30_padding)
        batchnorm34_1_30_ = self.batchnorm34_1_30_(conv34_1_30_)
        relu34_1_30_ = self.relu34_1_30_(batchnorm34_1_30_)
        conv35_1_30_ = self.conv35_1_30_(relu34_1_30_)
        batchnorm35_1_30_ = self.batchnorm35_1_30_(conv35_1_30_)
        conv33_1_31_ = self.conv33_1_31_(relu31_)
        batchnorm33_1_31_ = self.batchnorm33_1_31_(conv33_1_31_)
        relu33_1_31_ = self.relu33_1_31_(batchnorm33_1_31_)
        conv34_1_31_padding = self.conv34_1_31_padding(relu33_1_31_)
        conv34_1_31_ = self.conv34_1_31_(conv34_1_31_padding)
        batchnorm34_1_31_ = self.batchnorm34_1_31_(conv34_1_31_)
        relu34_1_31_ = self.relu34_1_31_(batchnorm34_1_31_)
        conv35_1_31_ = self.conv35_1_31_(relu34_1_31_)
        batchnorm35_1_31_ = self.batchnorm35_1_31_(conv35_1_31_)
        conv33_1_32_ = self.conv33_1_32_(relu31_)
        batchnorm33_1_32_ = self.batchnorm33_1_32_(conv33_1_32_)
        relu33_1_32_ = self.relu33_1_32_(batchnorm33_1_32_)
        conv34_1_32_padding = self.conv34_1_32_padding(relu33_1_32_)
        conv34_1_32_ = self.conv34_1_32_(conv34_1_32_padding)
        batchnorm34_1_32_ = self.batchnorm34_1_32_(conv34_1_32_)
        relu34_1_32_ = self.relu34_1_32_(batchnorm34_1_32_)
        conv35_1_32_ = self.conv35_1_32_(relu34_1_32_)
        batchnorm35_1_32_ = self.batchnorm35_1_32_(conv35_1_32_)
        add36_1_ = batchnorm35_1_1_ + batchnorm35_1_2_ + batchnorm35_1_3_ + batchnorm35_1_4_ + batchnorm35_1_5_ + batchnorm35_1_6_ + batchnorm35_1_7_ + batchnorm35_1_8_ + batchnorm35_1_9_ + batchnorm35_1_10_ + batchnorm35_1_11_ + batchnorm35_1_12_ + batchnorm35_1_13_ + batchnorm35_1_14_ + batchnorm35_1_15_ + batchnorm35_1_16_ + batchnorm35_1_17_ + batchnorm35_1_18_ + batchnorm35_1_19_ + batchnorm35_1_20_ + batchnorm35_1_21_ + batchnorm35_1_22_ + batchnorm35_1_23_ + batchnorm35_1_24_ + batchnorm35_1_25_ + batchnorm35_1_26_ + batchnorm35_1_27_ + batchnorm35_1_28_ + batchnorm35_1_29_ + batchnorm35_1_30_ + batchnorm35_1_31_ + batchnorm35_1_32_
        add37_ = add36_1_ + relu31_
        relu37_ = self.relu37_(add37_)
        conv39_1_1_ = self.conv39_1_1_(relu37_)
        batchnorm39_1_1_ = self.batchnorm39_1_1_(conv39_1_1_)
        relu39_1_1_ = self.relu39_1_1_(batchnorm39_1_1_)
        conv40_1_1_padding = self.conv40_1_1_padding(relu39_1_1_)
        conv40_1_1_ = self.conv40_1_1_(conv40_1_1_padding)
        batchnorm40_1_1_ = self.batchnorm40_1_1_(conv40_1_1_)
        relu40_1_1_ = self.relu40_1_1_(batchnorm40_1_1_)
        conv41_1_1_ = self.conv41_1_1_(relu40_1_1_)
        batchnorm41_1_1_ = self.batchnorm41_1_1_(conv41_1_1_)
        conv39_1_2_ = self.conv39_1_2_(relu37_)
        batchnorm39_1_2_ = self.batchnorm39_1_2_(conv39_1_2_)
        relu39_1_2_ = self.relu39_1_2_(batchnorm39_1_2_)
        conv40_1_2_padding = self.conv40_1_2_padding(relu39_1_2_)
        conv40_1_2_ = self.conv40_1_2_(conv40_1_2_padding)
        batchnorm40_1_2_ = self.batchnorm40_1_2_(conv40_1_2_)
        relu40_1_2_ = self.relu40_1_2_(batchnorm40_1_2_)
        conv41_1_2_ = self.conv41_1_2_(relu40_1_2_)
        batchnorm41_1_2_ = self.batchnorm41_1_2_(conv41_1_2_)
        conv39_1_3_ = self.conv39_1_3_(relu37_)
        batchnorm39_1_3_ = self.batchnorm39_1_3_(conv39_1_3_)
        relu39_1_3_ = self.relu39_1_3_(batchnorm39_1_3_)
        conv40_1_3_padding = self.conv40_1_3_padding(relu39_1_3_)
        conv40_1_3_ = self.conv40_1_3_(conv40_1_3_padding)
        batchnorm40_1_3_ = self.batchnorm40_1_3_(conv40_1_3_)
        relu40_1_3_ = self.relu40_1_3_(batchnorm40_1_3_)
        conv41_1_3_ = self.conv41_1_3_(relu40_1_3_)
        batchnorm41_1_3_ = self.batchnorm41_1_3_(conv41_1_3_)
        conv39_1_4_ = self.conv39_1_4_(relu37_)
        batchnorm39_1_4_ = self.batchnorm39_1_4_(conv39_1_4_)
        relu39_1_4_ = self.relu39_1_4_(batchnorm39_1_4_)
        conv40_1_4_padding = self.conv40_1_4_padding(relu39_1_4_)
        conv40_1_4_ = self.conv40_1_4_(conv40_1_4_padding)
        batchnorm40_1_4_ = self.batchnorm40_1_4_(conv40_1_4_)
        relu40_1_4_ = self.relu40_1_4_(batchnorm40_1_4_)
        conv41_1_4_ = self.conv41_1_4_(relu40_1_4_)
        batchnorm41_1_4_ = self.batchnorm41_1_4_(conv41_1_4_)
        conv39_1_5_ = self.conv39_1_5_(relu37_)
        batchnorm39_1_5_ = self.batchnorm39_1_5_(conv39_1_5_)
        relu39_1_5_ = self.relu39_1_5_(batchnorm39_1_5_)
        conv40_1_5_padding = self.conv40_1_5_padding(relu39_1_5_)
        conv40_1_5_ = self.conv40_1_5_(conv40_1_5_padding)
        batchnorm40_1_5_ = self.batchnorm40_1_5_(conv40_1_5_)
        relu40_1_5_ = self.relu40_1_5_(batchnorm40_1_5_)
        conv41_1_5_ = self.conv41_1_5_(relu40_1_5_)
        batchnorm41_1_5_ = self.batchnorm41_1_5_(conv41_1_5_)
        conv39_1_6_ = self.conv39_1_6_(relu37_)
        batchnorm39_1_6_ = self.batchnorm39_1_6_(conv39_1_6_)
        relu39_1_6_ = self.relu39_1_6_(batchnorm39_1_6_)
        conv40_1_6_padding = self.conv40_1_6_padding(relu39_1_6_)
        conv40_1_6_ = self.conv40_1_6_(conv40_1_6_padding)
        batchnorm40_1_6_ = self.batchnorm40_1_6_(conv40_1_6_)
        relu40_1_6_ = self.relu40_1_6_(batchnorm40_1_6_)
        conv41_1_6_ = self.conv41_1_6_(relu40_1_6_)
        batchnorm41_1_6_ = self.batchnorm41_1_6_(conv41_1_6_)
        conv39_1_7_ = self.conv39_1_7_(relu37_)
        batchnorm39_1_7_ = self.batchnorm39_1_7_(conv39_1_7_)
        relu39_1_7_ = self.relu39_1_7_(batchnorm39_1_7_)
        conv40_1_7_padding = self.conv40_1_7_padding(relu39_1_7_)
        conv40_1_7_ = self.conv40_1_7_(conv40_1_7_padding)
        batchnorm40_1_7_ = self.batchnorm40_1_7_(conv40_1_7_)
        relu40_1_7_ = self.relu40_1_7_(batchnorm40_1_7_)
        conv41_1_7_ = self.conv41_1_7_(relu40_1_7_)
        batchnorm41_1_7_ = self.batchnorm41_1_7_(conv41_1_7_)
        conv39_1_8_ = self.conv39_1_8_(relu37_)
        batchnorm39_1_8_ = self.batchnorm39_1_8_(conv39_1_8_)
        relu39_1_8_ = self.relu39_1_8_(batchnorm39_1_8_)
        conv40_1_8_padding = self.conv40_1_8_padding(relu39_1_8_)
        conv40_1_8_ = self.conv40_1_8_(conv40_1_8_padding)
        batchnorm40_1_8_ = self.batchnorm40_1_8_(conv40_1_8_)
        relu40_1_8_ = self.relu40_1_8_(batchnorm40_1_8_)
        conv41_1_8_ = self.conv41_1_8_(relu40_1_8_)
        batchnorm41_1_8_ = self.batchnorm41_1_8_(conv41_1_8_)
        conv39_1_9_ = self.conv39_1_9_(relu37_)
        batchnorm39_1_9_ = self.batchnorm39_1_9_(conv39_1_9_)
        relu39_1_9_ = self.relu39_1_9_(batchnorm39_1_9_)
        conv40_1_9_padding = self.conv40_1_9_padding(relu39_1_9_)
        conv40_1_9_ = self.conv40_1_9_(conv40_1_9_padding)
        batchnorm40_1_9_ = self.batchnorm40_1_9_(conv40_1_9_)
        relu40_1_9_ = self.relu40_1_9_(batchnorm40_1_9_)
        conv41_1_9_ = self.conv41_1_9_(relu40_1_9_)
        batchnorm41_1_9_ = self.batchnorm41_1_9_(conv41_1_9_)
        conv39_1_10_ = self.conv39_1_10_(relu37_)
        batchnorm39_1_10_ = self.batchnorm39_1_10_(conv39_1_10_)
        relu39_1_10_ = self.relu39_1_10_(batchnorm39_1_10_)
        conv40_1_10_padding = self.conv40_1_10_padding(relu39_1_10_)
        conv40_1_10_ = self.conv40_1_10_(conv40_1_10_padding)
        batchnorm40_1_10_ = self.batchnorm40_1_10_(conv40_1_10_)
        relu40_1_10_ = self.relu40_1_10_(batchnorm40_1_10_)
        conv41_1_10_ = self.conv41_1_10_(relu40_1_10_)
        batchnorm41_1_10_ = self.batchnorm41_1_10_(conv41_1_10_)
        conv39_1_11_ = self.conv39_1_11_(relu37_)
        batchnorm39_1_11_ = self.batchnorm39_1_11_(conv39_1_11_)
        relu39_1_11_ = self.relu39_1_11_(batchnorm39_1_11_)
        conv40_1_11_padding = self.conv40_1_11_padding(relu39_1_11_)
        conv40_1_11_ = self.conv40_1_11_(conv40_1_11_padding)
        batchnorm40_1_11_ = self.batchnorm40_1_11_(conv40_1_11_)
        relu40_1_11_ = self.relu40_1_11_(batchnorm40_1_11_)
        conv41_1_11_ = self.conv41_1_11_(relu40_1_11_)
        batchnorm41_1_11_ = self.batchnorm41_1_11_(conv41_1_11_)
        conv39_1_12_ = self.conv39_1_12_(relu37_)
        batchnorm39_1_12_ = self.batchnorm39_1_12_(conv39_1_12_)
        relu39_1_12_ = self.relu39_1_12_(batchnorm39_1_12_)
        conv40_1_12_padding = self.conv40_1_12_padding(relu39_1_12_)
        conv40_1_12_ = self.conv40_1_12_(conv40_1_12_padding)
        batchnorm40_1_12_ = self.batchnorm40_1_12_(conv40_1_12_)
        relu40_1_12_ = self.relu40_1_12_(batchnorm40_1_12_)
        conv41_1_12_ = self.conv41_1_12_(relu40_1_12_)
        batchnorm41_1_12_ = self.batchnorm41_1_12_(conv41_1_12_)
        conv39_1_13_ = self.conv39_1_13_(relu37_)
        batchnorm39_1_13_ = self.batchnorm39_1_13_(conv39_1_13_)
        relu39_1_13_ = self.relu39_1_13_(batchnorm39_1_13_)
        conv40_1_13_padding = self.conv40_1_13_padding(relu39_1_13_)
        conv40_1_13_ = self.conv40_1_13_(conv40_1_13_padding)
        batchnorm40_1_13_ = self.batchnorm40_1_13_(conv40_1_13_)
        relu40_1_13_ = self.relu40_1_13_(batchnorm40_1_13_)
        conv41_1_13_ = self.conv41_1_13_(relu40_1_13_)
        batchnorm41_1_13_ = self.batchnorm41_1_13_(conv41_1_13_)
        conv39_1_14_ = self.conv39_1_14_(relu37_)
        batchnorm39_1_14_ = self.batchnorm39_1_14_(conv39_1_14_)
        relu39_1_14_ = self.relu39_1_14_(batchnorm39_1_14_)
        conv40_1_14_padding = self.conv40_1_14_padding(relu39_1_14_)
        conv40_1_14_ = self.conv40_1_14_(conv40_1_14_padding)
        batchnorm40_1_14_ = self.batchnorm40_1_14_(conv40_1_14_)
        relu40_1_14_ = self.relu40_1_14_(batchnorm40_1_14_)
        conv41_1_14_ = self.conv41_1_14_(relu40_1_14_)
        batchnorm41_1_14_ = self.batchnorm41_1_14_(conv41_1_14_)
        conv39_1_15_ = self.conv39_1_15_(relu37_)
        batchnorm39_1_15_ = self.batchnorm39_1_15_(conv39_1_15_)
        relu39_1_15_ = self.relu39_1_15_(batchnorm39_1_15_)
        conv40_1_15_padding = self.conv40_1_15_padding(relu39_1_15_)
        conv40_1_15_ = self.conv40_1_15_(conv40_1_15_padding)
        batchnorm40_1_15_ = self.batchnorm40_1_15_(conv40_1_15_)
        relu40_1_15_ = self.relu40_1_15_(batchnorm40_1_15_)
        conv41_1_15_ = self.conv41_1_15_(relu40_1_15_)
        batchnorm41_1_15_ = self.batchnorm41_1_15_(conv41_1_15_)
        conv39_1_16_ = self.conv39_1_16_(relu37_)
        batchnorm39_1_16_ = self.batchnorm39_1_16_(conv39_1_16_)
        relu39_1_16_ = self.relu39_1_16_(batchnorm39_1_16_)
        conv40_1_16_padding = self.conv40_1_16_padding(relu39_1_16_)
        conv40_1_16_ = self.conv40_1_16_(conv40_1_16_padding)
        batchnorm40_1_16_ = self.batchnorm40_1_16_(conv40_1_16_)
        relu40_1_16_ = self.relu40_1_16_(batchnorm40_1_16_)
        conv41_1_16_ = self.conv41_1_16_(relu40_1_16_)
        batchnorm41_1_16_ = self.batchnorm41_1_16_(conv41_1_16_)
        conv39_1_17_ = self.conv39_1_17_(relu37_)
        batchnorm39_1_17_ = self.batchnorm39_1_17_(conv39_1_17_)
        relu39_1_17_ = self.relu39_1_17_(batchnorm39_1_17_)
        conv40_1_17_padding = self.conv40_1_17_padding(relu39_1_17_)
        conv40_1_17_ = self.conv40_1_17_(conv40_1_17_padding)
        batchnorm40_1_17_ = self.batchnorm40_1_17_(conv40_1_17_)
        relu40_1_17_ = self.relu40_1_17_(batchnorm40_1_17_)
        conv41_1_17_ = self.conv41_1_17_(relu40_1_17_)
        batchnorm41_1_17_ = self.batchnorm41_1_17_(conv41_1_17_)
        conv39_1_18_ = self.conv39_1_18_(relu37_)
        batchnorm39_1_18_ = self.batchnorm39_1_18_(conv39_1_18_)
        relu39_1_18_ = self.relu39_1_18_(batchnorm39_1_18_)
        conv40_1_18_padding = self.conv40_1_18_padding(relu39_1_18_)
        conv40_1_18_ = self.conv40_1_18_(conv40_1_18_padding)
        batchnorm40_1_18_ = self.batchnorm40_1_18_(conv40_1_18_)
        relu40_1_18_ = self.relu40_1_18_(batchnorm40_1_18_)
        conv41_1_18_ = self.conv41_1_18_(relu40_1_18_)
        batchnorm41_1_18_ = self.batchnorm41_1_18_(conv41_1_18_)
        conv39_1_19_ = self.conv39_1_19_(relu37_)
        batchnorm39_1_19_ = self.batchnorm39_1_19_(conv39_1_19_)
        relu39_1_19_ = self.relu39_1_19_(batchnorm39_1_19_)
        conv40_1_19_padding = self.conv40_1_19_padding(relu39_1_19_)
        conv40_1_19_ = self.conv40_1_19_(conv40_1_19_padding)
        batchnorm40_1_19_ = self.batchnorm40_1_19_(conv40_1_19_)
        relu40_1_19_ = self.relu40_1_19_(batchnorm40_1_19_)
        conv41_1_19_ = self.conv41_1_19_(relu40_1_19_)
        batchnorm41_1_19_ = self.batchnorm41_1_19_(conv41_1_19_)
        conv39_1_20_ = self.conv39_1_20_(relu37_)
        batchnorm39_1_20_ = self.batchnorm39_1_20_(conv39_1_20_)
        relu39_1_20_ = self.relu39_1_20_(batchnorm39_1_20_)
        conv40_1_20_padding = self.conv40_1_20_padding(relu39_1_20_)
        conv40_1_20_ = self.conv40_1_20_(conv40_1_20_padding)
        batchnorm40_1_20_ = self.batchnorm40_1_20_(conv40_1_20_)
        relu40_1_20_ = self.relu40_1_20_(batchnorm40_1_20_)
        conv41_1_20_ = self.conv41_1_20_(relu40_1_20_)
        batchnorm41_1_20_ = self.batchnorm41_1_20_(conv41_1_20_)
        conv39_1_21_ = self.conv39_1_21_(relu37_)
        batchnorm39_1_21_ = self.batchnorm39_1_21_(conv39_1_21_)
        relu39_1_21_ = self.relu39_1_21_(batchnorm39_1_21_)
        conv40_1_21_padding = self.conv40_1_21_padding(relu39_1_21_)
        conv40_1_21_ = self.conv40_1_21_(conv40_1_21_padding)
        batchnorm40_1_21_ = self.batchnorm40_1_21_(conv40_1_21_)
        relu40_1_21_ = self.relu40_1_21_(batchnorm40_1_21_)
        conv41_1_21_ = self.conv41_1_21_(relu40_1_21_)
        batchnorm41_1_21_ = self.batchnorm41_1_21_(conv41_1_21_)
        conv39_1_22_ = self.conv39_1_22_(relu37_)
        batchnorm39_1_22_ = self.batchnorm39_1_22_(conv39_1_22_)
        relu39_1_22_ = self.relu39_1_22_(batchnorm39_1_22_)
        conv40_1_22_padding = self.conv40_1_22_padding(relu39_1_22_)
        conv40_1_22_ = self.conv40_1_22_(conv40_1_22_padding)
        batchnorm40_1_22_ = self.batchnorm40_1_22_(conv40_1_22_)
        relu40_1_22_ = self.relu40_1_22_(batchnorm40_1_22_)
        conv41_1_22_ = self.conv41_1_22_(relu40_1_22_)
        batchnorm41_1_22_ = self.batchnorm41_1_22_(conv41_1_22_)
        conv39_1_23_ = self.conv39_1_23_(relu37_)
        batchnorm39_1_23_ = self.batchnorm39_1_23_(conv39_1_23_)
        relu39_1_23_ = self.relu39_1_23_(batchnorm39_1_23_)
        conv40_1_23_padding = self.conv40_1_23_padding(relu39_1_23_)
        conv40_1_23_ = self.conv40_1_23_(conv40_1_23_padding)
        batchnorm40_1_23_ = self.batchnorm40_1_23_(conv40_1_23_)
        relu40_1_23_ = self.relu40_1_23_(batchnorm40_1_23_)
        conv41_1_23_ = self.conv41_1_23_(relu40_1_23_)
        batchnorm41_1_23_ = self.batchnorm41_1_23_(conv41_1_23_)
        conv39_1_24_ = self.conv39_1_24_(relu37_)
        batchnorm39_1_24_ = self.batchnorm39_1_24_(conv39_1_24_)
        relu39_1_24_ = self.relu39_1_24_(batchnorm39_1_24_)
        conv40_1_24_padding = self.conv40_1_24_padding(relu39_1_24_)
        conv40_1_24_ = self.conv40_1_24_(conv40_1_24_padding)
        batchnorm40_1_24_ = self.batchnorm40_1_24_(conv40_1_24_)
        relu40_1_24_ = self.relu40_1_24_(batchnorm40_1_24_)
        conv41_1_24_ = self.conv41_1_24_(relu40_1_24_)
        batchnorm41_1_24_ = self.batchnorm41_1_24_(conv41_1_24_)
        conv39_1_25_ = self.conv39_1_25_(relu37_)
        batchnorm39_1_25_ = self.batchnorm39_1_25_(conv39_1_25_)
        relu39_1_25_ = self.relu39_1_25_(batchnorm39_1_25_)
        conv40_1_25_padding = self.conv40_1_25_padding(relu39_1_25_)
        conv40_1_25_ = self.conv40_1_25_(conv40_1_25_padding)
        batchnorm40_1_25_ = self.batchnorm40_1_25_(conv40_1_25_)
        relu40_1_25_ = self.relu40_1_25_(batchnorm40_1_25_)
        conv41_1_25_ = self.conv41_1_25_(relu40_1_25_)
        batchnorm41_1_25_ = self.batchnorm41_1_25_(conv41_1_25_)
        conv39_1_26_ = self.conv39_1_26_(relu37_)
        batchnorm39_1_26_ = self.batchnorm39_1_26_(conv39_1_26_)
        relu39_1_26_ = self.relu39_1_26_(batchnorm39_1_26_)
        conv40_1_26_padding = self.conv40_1_26_padding(relu39_1_26_)
        conv40_1_26_ = self.conv40_1_26_(conv40_1_26_padding)
        batchnorm40_1_26_ = self.batchnorm40_1_26_(conv40_1_26_)
        relu40_1_26_ = self.relu40_1_26_(batchnorm40_1_26_)
        conv41_1_26_ = self.conv41_1_26_(relu40_1_26_)
        batchnorm41_1_26_ = self.batchnorm41_1_26_(conv41_1_26_)
        conv39_1_27_ = self.conv39_1_27_(relu37_)
        batchnorm39_1_27_ = self.batchnorm39_1_27_(conv39_1_27_)
        relu39_1_27_ = self.relu39_1_27_(batchnorm39_1_27_)
        conv40_1_27_padding = self.conv40_1_27_padding(relu39_1_27_)
        conv40_1_27_ = self.conv40_1_27_(conv40_1_27_padding)
        batchnorm40_1_27_ = self.batchnorm40_1_27_(conv40_1_27_)
        relu40_1_27_ = self.relu40_1_27_(batchnorm40_1_27_)
        conv41_1_27_ = self.conv41_1_27_(relu40_1_27_)
        batchnorm41_1_27_ = self.batchnorm41_1_27_(conv41_1_27_)
        conv39_1_28_ = self.conv39_1_28_(relu37_)
        batchnorm39_1_28_ = self.batchnorm39_1_28_(conv39_1_28_)
        relu39_1_28_ = self.relu39_1_28_(batchnorm39_1_28_)
        conv40_1_28_padding = self.conv40_1_28_padding(relu39_1_28_)
        conv40_1_28_ = self.conv40_1_28_(conv40_1_28_padding)
        batchnorm40_1_28_ = self.batchnorm40_1_28_(conv40_1_28_)
        relu40_1_28_ = self.relu40_1_28_(batchnorm40_1_28_)
        conv41_1_28_ = self.conv41_1_28_(relu40_1_28_)
        batchnorm41_1_28_ = self.batchnorm41_1_28_(conv41_1_28_)
        conv39_1_29_ = self.conv39_1_29_(relu37_)
        batchnorm39_1_29_ = self.batchnorm39_1_29_(conv39_1_29_)
        relu39_1_29_ = self.relu39_1_29_(batchnorm39_1_29_)
        conv40_1_29_padding = self.conv40_1_29_padding(relu39_1_29_)
        conv40_1_29_ = self.conv40_1_29_(conv40_1_29_padding)
        batchnorm40_1_29_ = self.batchnorm40_1_29_(conv40_1_29_)
        relu40_1_29_ = self.relu40_1_29_(batchnorm40_1_29_)
        conv41_1_29_ = self.conv41_1_29_(relu40_1_29_)
        batchnorm41_1_29_ = self.batchnorm41_1_29_(conv41_1_29_)
        conv39_1_30_ = self.conv39_1_30_(relu37_)
        batchnorm39_1_30_ = self.batchnorm39_1_30_(conv39_1_30_)
        relu39_1_30_ = self.relu39_1_30_(batchnorm39_1_30_)
        conv40_1_30_padding = self.conv40_1_30_padding(relu39_1_30_)
        conv40_1_30_ = self.conv40_1_30_(conv40_1_30_padding)
        batchnorm40_1_30_ = self.batchnorm40_1_30_(conv40_1_30_)
        relu40_1_30_ = self.relu40_1_30_(batchnorm40_1_30_)
        conv41_1_30_ = self.conv41_1_30_(relu40_1_30_)
        batchnorm41_1_30_ = self.batchnorm41_1_30_(conv41_1_30_)
        conv39_1_31_ = self.conv39_1_31_(relu37_)
        batchnorm39_1_31_ = self.batchnorm39_1_31_(conv39_1_31_)
        relu39_1_31_ = self.relu39_1_31_(batchnorm39_1_31_)
        conv40_1_31_padding = self.conv40_1_31_padding(relu39_1_31_)
        conv40_1_31_ = self.conv40_1_31_(conv40_1_31_padding)
        batchnorm40_1_31_ = self.batchnorm40_1_31_(conv40_1_31_)
        relu40_1_31_ = self.relu40_1_31_(batchnorm40_1_31_)
        conv41_1_31_ = self.conv41_1_31_(relu40_1_31_)
        batchnorm41_1_31_ = self.batchnorm41_1_31_(conv41_1_31_)
        conv39_1_32_ = self.conv39_1_32_(relu37_)
        batchnorm39_1_32_ = self.batchnorm39_1_32_(conv39_1_32_)
        relu39_1_32_ = self.relu39_1_32_(batchnorm39_1_32_)
        conv40_1_32_padding = self.conv40_1_32_padding(relu39_1_32_)
        conv40_1_32_ = self.conv40_1_32_(conv40_1_32_padding)
        batchnorm40_1_32_ = self.batchnorm40_1_32_(conv40_1_32_)
        relu40_1_32_ = self.relu40_1_32_(batchnorm40_1_32_)
        conv41_1_32_ = self.conv41_1_32_(relu40_1_32_)
        batchnorm41_1_32_ = self.batchnorm41_1_32_(conv41_1_32_)
        add42_1_ = batchnorm41_1_1_ + batchnorm41_1_2_ + batchnorm41_1_3_ + batchnorm41_1_4_ + batchnorm41_1_5_ + batchnorm41_1_6_ + batchnorm41_1_7_ + batchnorm41_1_8_ + batchnorm41_1_9_ + batchnorm41_1_10_ + batchnorm41_1_11_ + batchnorm41_1_12_ + batchnorm41_1_13_ + batchnorm41_1_14_ + batchnorm41_1_15_ + batchnorm41_1_16_ + batchnorm41_1_17_ + batchnorm41_1_18_ + batchnorm41_1_19_ + batchnorm41_1_20_ + batchnorm41_1_21_ + batchnorm41_1_22_ + batchnorm41_1_23_ + batchnorm41_1_24_ + batchnorm41_1_25_ + batchnorm41_1_26_ + batchnorm41_1_27_ + batchnorm41_1_28_ + batchnorm41_1_29_ + batchnorm41_1_30_ + batchnorm41_1_31_ + batchnorm41_1_32_
        add43_ = add42_1_ + relu37_
        relu43_ = self.relu43_(add43_)
        conv45_1_1_ = self.conv45_1_1_(relu43_)
        batchnorm45_1_1_ = self.batchnorm45_1_1_(conv45_1_1_)
        relu45_1_1_ = self.relu45_1_1_(batchnorm45_1_1_)
        conv46_1_1_padding = self.conv46_1_1_padding(relu45_1_1_)
        conv46_1_1_ = self.conv46_1_1_(conv46_1_1_padding)
        batchnorm46_1_1_ = self.batchnorm46_1_1_(conv46_1_1_)
        relu46_1_1_ = self.relu46_1_1_(batchnorm46_1_1_)
        conv47_1_1_ = self.conv47_1_1_(relu46_1_1_)
        batchnorm47_1_1_ = self.batchnorm47_1_1_(conv47_1_1_)
        conv45_1_2_ = self.conv45_1_2_(relu43_)
        batchnorm45_1_2_ = self.batchnorm45_1_2_(conv45_1_2_)
        relu45_1_2_ = self.relu45_1_2_(batchnorm45_1_2_)
        conv46_1_2_padding = self.conv46_1_2_padding(relu45_1_2_)
        conv46_1_2_ = self.conv46_1_2_(conv46_1_2_padding)
        batchnorm46_1_2_ = self.batchnorm46_1_2_(conv46_1_2_)
        relu46_1_2_ = self.relu46_1_2_(batchnorm46_1_2_)
        conv47_1_2_ = self.conv47_1_2_(relu46_1_2_)
        batchnorm47_1_2_ = self.batchnorm47_1_2_(conv47_1_2_)
        conv45_1_3_ = self.conv45_1_3_(relu43_)
        batchnorm45_1_3_ = self.batchnorm45_1_3_(conv45_1_3_)
        relu45_1_3_ = self.relu45_1_3_(batchnorm45_1_3_)
        conv46_1_3_padding = self.conv46_1_3_padding(relu45_1_3_)
        conv46_1_3_ = self.conv46_1_3_(conv46_1_3_padding)
        batchnorm46_1_3_ = self.batchnorm46_1_3_(conv46_1_3_)
        relu46_1_3_ = self.relu46_1_3_(batchnorm46_1_3_)
        conv47_1_3_ = self.conv47_1_3_(relu46_1_3_)
        batchnorm47_1_3_ = self.batchnorm47_1_3_(conv47_1_3_)
        conv45_1_4_ = self.conv45_1_4_(relu43_)
        batchnorm45_1_4_ = self.batchnorm45_1_4_(conv45_1_4_)
        relu45_1_4_ = self.relu45_1_4_(batchnorm45_1_4_)
        conv46_1_4_padding = self.conv46_1_4_padding(relu45_1_4_)
        conv46_1_4_ = self.conv46_1_4_(conv46_1_4_padding)
        batchnorm46_1_4_ = self.batchnorm46_1_4_(conv46_1_4_)
        relu46_1_4_ = self.relu46_1_4_(batchnorm46_1_4_)
        conv47_1_4_ = self.conv47_1_4_(relu46_1_4_)
        batchnorm47_1_4_ = self.batchnorm47_1_4_(conv47_1_4_)
        conv45_1_5_ = self.conv45_1_5_(relu43_)
        batchnorm45_1_5_ = self.batchnorm45_1_5_(conv45_1_5_)
        relu45_1_5_ = self.relu45_1_5_(batchnorm45_1_5_)
        conv46_1_5_padding = self.conv46_1_5_padding(relu45_1_5_)
        conv46_1_5_ = self.conv46_1_5_(conv46_1_5_padding)
        batchnorm46_1_5_ = self.batchnorm46_1_5_(conv46_1_5_)
        relu46_1_5_ = self.relu46_1_5_(batchnorm46_1_5_)
        conv47_1_5_ = self.conv47_1_5_(relu46_1_5_)
        batchnorm47_1_5_ = self.batchnorm47_1_5_(conv47_1_5_)
        conv45_1_6_ = self.conv45_1_6_(relu43_)
        batchnorm45_1_6_ = self.batchnorm45_1_6_(conv45_1_6_)
        relu45_1_6_ = self.relu45_1_6_(batchnorm45_1_6_)
        conv46_1_6_padding = self.conv46_1_6_padding(relu45_1_6_)
        conv46_1_6_ = self.conv46_1_6_(conv46_1_6_padding)
        batchnorm46_1_6_ = self.batchnorm46_1_6_(conv46_1_6_)
        relu46_1_6_ = self.relu46_1_6_(batchnorm46_1_6_)
        conv47_1_6_ = self.conv47_1_6_(relu46_1_6_)
        batchnorm47_1_6_ = self.batchnorm47_1_6_(conv47_1_6_)
        conv45_1_7_ = self.conv45_1_7_(relu43_)
        batchnorm45_1_7_ = self.batchnorm45_1_7_(conv45_1_7_)
        relu45_1_7_ = self.relu45_1_7_(batchnorm45_1_7_)
        conv46_1_7_padding = self.conv46_1_7_padding(relu45_1_7_)
        conv46_1_7_ = self.conv46_1_7_(conv46_1_7_padding)
        batchnorm46_1_7_ = self.batchnorm46_1_7_(conv46_1_7_)
        relu46_1_7_ = self.relu46_1_7_(batchnorm46_1_7_)
        conv47_1_7_ = self.conv47_1_7_(relu46_1_7_)
        batchnorm47_1_7_ = self.batchnorm47_1_7_(conv47_1_7_)
        conv45_1_8_ = self.conv45_1_8_(relu43_)
        batchnorm45_1_8_ = self.batchnorm45_1_8_(conv45_1_8_)
        relu45_1_8_ = self.relu45_1_8_(batchnorm45_1_8_)
        conv46_1_8_padding = self.conv46_1_8_padding(relu45_1_8_)
        conv46_1_8_ = self.conv46_1_8_(conv46_1_8_padding)
        batchnorm46_1_8_ = self.batchnorm46_1_8_(conv46_1_8_)
        relu46_1_8_ = self.relu46_1_8_(batchnorm46_1_8_)
        conv47_1_8_ = self.conv47_1_8_(relu46_1_8_)
        batchnorm47_1_8_ = self.batchnorm47_1_8_(conv47_1_8_)
        conv45_1_9_ = self.conv45_1_9_(relu43_)
        batchnorm45_1_9_ = self.batchnorm45_1_9_(conv45_1_9_)
        relu45_1_9_ = self.relu45_1_9_(batchnorm45_1_9_)
        conv46_1_9_padding = self.conv46_1_9_padding(relu45_1_9_)
        conv46_1_9_ = self.conv46_1_9_(conv46_1_9_padding)
        batchnorm46_1_9_ = self.batchnorm46_1_9_(conv46_1_9_)
        relu46_1_9_ = self.relu46_1_9_(batchnorm46_1_9_)
        conv47_1_9_ = self.conv47_1_9_(relu46_1_9_)
        batchnorm47_1_9_ = self.batchnorm47_1_9_(conv47_1_9_)
        conv45_1_10_ = self.conv45_1_10_(relu43_)
        batchnorm45_1_10_ = self.batchnorm45_1_10_(conv45_1_10_)
        relu45_1_10_ = self.relu45_1_10_(batchnorm45_1_10_)
        conv46_1_10_padding = self.conv46_1_10_padding(relu45_1_10_)
        conv46_1_10_ = self.conv46_1_10_(conv46_1_10_padding)
        batchnorm46_1_10_ = self.batchnorm46_1_10_(conv46_1_10_)
        relu46_1_10_ = self.relu46_1_10_(batchnorm46_1_10_)
        conv47_1_10_ = self.conv47_1_10_(relu46_1_10_)
        batchnorm47_1_10_ = self.batchnorm47_1_10_(conv47_1_10_)
        conv45_1_11_ = self.conv45_1_11_(relu43_)
        batchnorm45_1_11_ = self.batchnorm45_1_11_(conv45_1_11_)
        relu45_1_11_ = self.relu45_1_11_(batchnorm45_1_11_)
        conv46_1_11_padding = self.conv46_1_11_padding(relu45_1_11_)
        conv46_1_11_ = self.conv46_1_11_(conv46_1_11_padding)
        batchnorm46_1_11_ = self.batchnorm46_1_11_(conv46_1_11_)
        relu46_1_11_ = self.relu46_1_11_(batchnorm46_1_11_)
        conv47_1_11_ = self.conv47_1_11_(relu46_1_11_)
        batchnorm47_1_11_ = self.batchnorm47_1_11_(conv47_1_11_)
        conv45_1_12_ = self.conv45_1_12_(relu43_)
        batchnorm45_1_12_ = self.batchnorm45_1_12_(conv45_1_12_)
        relu45_1_12_ = self.relu45_1_12_(batchnorm45_1_12_)
        conv46_1_12_padding = self.conv46_1_12_padding(relu45_1_12_)
        conv46_1_12_ = self.conv46_1_12_(conv46_1_12_padding)
        batchnorm46_1_12_ = self.batchnorm46_1_12_(conv46_1_12_)
        relu46_1_12_ = self.relu46_1_12_(batchnorm46_1_12_)
        conv47_1_12_ = self.conv47_1_12_(relu46_1_12_)
        batchnorm47_1_12_ = self.batchnorm47_1_12_(conv47_1_12_)
        conv45_1_13_ = self.conv45_1_13_(relu43_)
        batchnorm45_1_13_ = self.batchnorm45_1_13_(conv45_1_13_)
        relu45_1_13_ = self.relu45_1_13_(batchnorm45_1_13_)
        conv46_1_13_padding = self.conv46_1_13_padding(relu45_1_13_)
        conv46_1_13_ = self.conv46_1_13_(conv46_1_13_padding)
        batchnorm46_1_13_ = self.batchnorm46_1_13_(conv46_1_13_)
        relu46_1_13_ = self.relu46_1_13_(batchnorm46_1_13_)
        conv47_1_13_ = self.conv47_1_13_(relu46_1_13_)
        batchnorm47_1_13_ = self.batchnorm47_1_13_(conv47_1_13_)
        conv45_1_14_ = self.conv45_1_14_(relu43_)
        batchnorm45_1_14_ = self.batchnorm45_1_14_(conv45_1_14_)
        relu45_1_14_ = self.relu45_1_14_(batchnorm45_1_14_)
        conv46_1_14_padding = self.conv46_1_14_padding(relu45_1_14_)
        conv46_1_14_ = self.conv46_1_14_(conv46_1_14_padding)
        batchnorm46_1_14_ = self.batchnorm46_1_14_(conv46_1_14_)
        relu46_1_14_ = self.relu46_1_14_(batchnorm46_1_14_)
        conv47_1_14_ = self.conv47_1_14_(relu46_1_14_)
        batchnorm47_1_14_ = self.batchnorm47_1_14_(conv47_1_14_)
        conv45_1_15_ = self.conv45_1_15_(relu43_)
        batchnorm45_1_15_ = self.batchnorm45_1_15_(conv45_1_15_)
        relu45_1_15_ = self.relu45_1_15_(batchnorm45_1_15_)
        conv46_1_15_padding = self.conv46_1_15_padding(relu45_1_15_)
        conv46_1_15_ = self.conv46_1_15_(conv46_1_15_padding)
        batchnorm46_1_15_ = self.batchnorm46_1_15_(conv46_1_15_)
        relu46_1_15_ = self.relu46_1_15_(batchnorm46_1_15_)
        conv47_1_15_ = self.conv47_1_15_(relu46_1_15_)
        batchnorm47_1_15_ = self.batchnorm47_1_15_(conv47_1_15_)
        conv45_1_16_ = self.conv45_1_16_(relu43_)
        batchnorm45_1_16_ = self.batchnorm45_1_16_(conv45_1_16_)
        relu45_1_16_ = self.relu45_1_16_(batchnorm45_1_16_)
        conv46_1_16_padding = self.conv46_1_16_padding(relu45_1_16_)
        conv46_1_16_ = self.conv46_1_16_(conv46_1_16_padding)
        batchnorm46_1_16_ = self.batchnorm46_1_16_(conv46_1_16_)
        relu46_1_16_ = self.relu46_1_16_(batchnorm46_1_16_)
        conv47_1_16_ = self.conv47_1_16_(relu46_1_16_)
        batchnorm47_1_16_ = self.batchnorm47_1_16_(conv47_1_16_)
        conv45_1_17_ = self.conv45_1_17_(relu43_)
        batchnorm45_1_17_ = self.batchnorm45_1_17_(conv45_1_17_)
        relu45_1_17_ = self.relu45_1_17_(batchnorm45_1_17_)
        conv46_1_17_padding = self.conv46_1_17_padding(relu45_1_17_)
        conv46_1_17_ = self.conv46_1_17_(conv46_1_17_padding)
        batchnorm46_1_17_ = self.batchnorm46_1_17_(conv46_1_17_)
        relu46_1_17_ = self.relu46_1_17_(batchnorm46_1_17_)
        conv47_1_17_ = self.conv47_1_17_(relu46_1_17_)
        batchnorm47_1_17_ = self.batchnorm47_1_17_(conv47_1_17_)
        conv45_1_18_ = self.conv45_1_18_(relu43_)
        batchnorm45_1_18_ = self.batchnorm45_1_18_(conv45_1_18_)
        relu45_1_18_ = self.relu45_1_18_(batchnorm45_1_18_)
        conv46_1_18_padding = self.conv46_1_18_padding(relu45_1_18_)
        conv46_1_18_ = self.conv46_1_18_(conv46_1_18_padding)
        batchnorm46_1_18_ = self.batchnorm46_1_18_(conv46_1_18_)
        relu46_1_18_ = self.relu46_1_18_(batchnorm46_1_18_)
        conv47_1_18_ = self.conv47_1_18_(relu46_1_18_)
        batchnorm47_1_18_ = self.batchnorm47_1_18_(conv47_1_18_)
        conv45_1_19_ = self.conv45_1_19_(relu43_)
        batchnorm45_1_19_ = self.batchnorm45_1_19_(conv45_1_19_)
        relu45_1_19_ = self.relu45_1_19_(batchnorm45_1_19_)
        conv46_1_19_padding = self.conv46_1_19_padding(relu45_1_19_)
        conv46_1_19_ = self.conv46_1_19_(conv46_1_19_padding)
        batchnorm46_1_19_ = self.batchnorm46_1_19_(conv46_1_19_)
        relu46_1_19_ = self.relu46_1_19_(batchnorm46_1_19_)
        conv47_1_19_ = self.conv47_1_19_(relu46_1_19_)
        batchnorm47_1_19_ = self.batchnorm47_1_19_(conv47_1_19_)
        conv45_1_20_ = self.conv45_1_20_(relu43_)
        batchnorm45_1_20_ = self.batchnorm45_1_20_(conv45_1_20_)
        relu45_1_20_ = self.relu45_1_20_(batchnorm45_1_20_)
        conv46_1_20_padding = self.conv46_1_20_padding(relu45_1_20_)
        conv46_1_20_ = self.conv46_1_20_(conv46_1_20_padding)
        batchnorm46_1_20_ = self.batchnorm46_1_20_(conv46_1_20_)
        relu46_1_20_ = self.relu46_1_20_(batchnorm46_1_20_)
        conv47_1_20_ = self.conv47_1_20_(relu46_1_20_)
        batchnorm47_1_20_ = self.batchnorm47_1_20_(conv47_1_20_)
        conv45_1_21_ = self.conv45_1_21_(relu43_)
        batchnorm45_1_21_ = self.batchnorm45_1_21_(conv45_1_21_)
        relu45_1_21_ = self.relu45_1_21_(batchnorm45_1_21_)
        conv46_1_21_padding = self.conv46_1_21_padding(relu45_1_21_)
        conv46_1_21_ = self.conv46_1_21_(conv46_1_21_padding)
        batchnorm46_1_21_ = self.batchnorm46_1_21_(conv46_1_21_)
        relu46_1_21_ = self.relu46_1_21_(batchnorm46_1_21_)
        conv47_1_21_ = self.conv47_1_21_(relu46_1_21_)
        batchnorm47_1_21_ = self.batchnorm47_1_21_(conv47_1_21_)
        conv45_1_22_ = self.conv45_1_22_(relu43_)
        batchnorm45_1_22_ = self.batchnorm45_1_22_(conv45_1_22_)
        relu45_1_22_ = self.relu45_1_22_(batchnorm45_1_22_)
        conv46_1_22_padding = self.conv46_1_22_padding(relu45_1_22_)
        conv46_1_22_ = self.conv46_1_22_(conv46_1_22_padding)
        batchnorm46_1_22_ = self.batchnorm46_1_22_(conv46_1_22_)
        relu46_1_22_ = self.relu46_1_22_(batchnorm46_1_22_)
        conv47_1_22_ = self.conv47_1_22_(relu46_1_22_)
        batchnorm47_1_22_ = self.batchnorm47_1_22_(conv47_1_22_)
        conv45_1_23_ = self.conv45_1_23_(relu43_)
        batchnorm45_1_23_ = self.batchnorm45_1_23_(conv45_1_23_)
        relu45_1_23_ = self.relu45_1_23_(batchnorm45_1_23_)
        conv46_1_23_padding = self.conv46_1_23_padding(relu45_1_23_)
        conv46_1_23_ = self.conv46_1_23_(conv46_1_23_padding)
        batchnorm46_1_23_ = self.batchnorm46_1_23_(conv46_1_23_)
        relu46_1_23_ = self.relu46_1_23_(batchnorm46_1_23_)
        conv47_1_23_ = self.conv47_1_23_(relu46_1_23_)
        batchnorm47_1_23_ = self.batchnorm47_1_23_(conv47_1_23_)
        conv45_1_24_ = self.conv45_1_24_(relu43_)
        batchnorm45_1_24_ = self.batchnorm45_1_24_(conv45_1_24_)
        relu45_1_24_ = self.relu45_1_24_(batchnorm45_1_24_)
        conv46_1_24_padding = self.conv46_1_24_padding(relu45_1_24_)
        conv46_1_24_ = self.conv46_1_24_(conv46_1_24_padding)
        batchnorm46_1_24_ = self.batchnorm46_1_24_(conv46_1_24_)
        relu46_1_24_ = self.relu46_1_24_(batchnorm46_1_24_)
        conv47_1_24_ = self.conv47_1_24_(relu46_1_24_)
        batchnorm47_1_24_ = self.batchnorm47_1_24_(conv47_1_24_)
        conv45_1_25_ = self.conv45_1_25_(relu43_)
        batchnorm45_1_25_ = self.batchnorm45_1_25_(conv45_1_25_)
        relu45_1_25_ = self.relu45_1_25_(batchnorm45_1_25_)
        conv46_1_25_padding = self.conv46_1_25_padding(relu45_1_25_)
        conv46_1_25_ = self.conv46_1_25_(conv46_1_25_padding)
        batchnorm46_1_25_ = self.batchnorm46_1_25_(conv46_1_25_)
        relu46_1_25_ = self.relu46_1_25_(batchnorm46_1_25_)
        conv47_1_25_ = self.conv47_1_25_(relu46_1_25_)
        batchnorm47_1_25_ = self.batchnorm47_1_25_(conv47_1_25_)
        conv45_1_26_ = self.conv45_1_26_(relu43_)
        batchnorm45_1_26_ = self.batchnorm45_1_26_(conv45_1_26_)
        relu45_1_26_ = self.relu45_1_26_(batchnorm45_1_26_)
        conv46_1_26_padding = self.conv46_1_26_padding(relu45_1_26_)
        conv46_1_26_ = self.conv46_1_26_(conv46_1_26_padding)
        batchnorm46_1_26_ = self.batchnorm46_1_26_(conv46_1_26_)
        relu46_1_26_ = self.relu46_1_26_(batchnorm46_1_26_)
        conv47_1_26_ = self.conv47_1_26_(relu46_1_26_)
        batchnorm47_1_26_ = self.batchnorm47_1_26_(conv47_1_26_)
        conv45_1_27_ = self.conv45_1_27_(relu43_)
        batchnorm45_1_27_ = self.batchnorm45_1_27_(conv45_1_27_)
        relu45_1_27_ = self.relu45_1_27_(batchnorm45_1_27_)
        conv46_1_27_padding = self.conv46_1_27_padding(relu45_1_27_)
        conv46_1_27_ = self.conv46_1_27_(conv46_1_27_padding)
        batchnorm46_1_27_ = self.batchnorm46_1_27_(conv46_1_27_)
        relu46_1_27_ = self.relu46_1_27_(batchnorm46_1_27_)
        conv47_1_27_ = self.conv47_1_27_(relu46_1_27_)
        batchnorm47_1_27_ = self.batchnorm47_1_27_(conv47_1_27_)
        conv45_1_28_ = self.conv45_1_28_(relu43_)
        batchnorm45_1_28_ = self.batchnorm45_1_28_(conv45_1_28_)
        relu45_1_28_ = self.relu45_1_28_(batchnorm45_1_28_)
        conv46_1_28_padding = self.conv46_1_28_padding(relu45_1_28_)
        conv46_1_28_ = self.conv46_1_28_(conv46_1_28_padding)
        batchnorm46_1_28_ = self.batchnorm46_1_28_(conv46_1_28_)
        relu46_1_28_ = self.relu46_1_28_(batchnorm46_1_28_)
        conv47_1_28_ = self.conv47_1_28_(relu46_1_28_)
        batchnorm47_1_28_ = self.batchnorm47_1_28_(conv47_1_28_)
        conv45_1_29_ = self.conv45_1_29_(relu43_)
        batchnorm45_1_29_ = self.batchnorm45_1_29_(conv45_1_29_)
        relu45_1_29_ = self.relu45_1_29_(batchnorm45_1_29_)
        conv46_1_29_padding = self.conv46_1_29_padding(relu45_1_29_)
        conv46_1_29_ = self.conv46_1_29_(conv46_1_29_padding)
        batchnorm46_1_29_ = self.batchnorm46_1_29_(conv46_1_29_)
        relu46_1_29_ = self.relu46_1_29_(batchnorm46_1_29_)
        conv47_1_29_ = self.conv47_1_29_(relu46_1_29_)
        batchnorm47_1_29_ = self.batchnorm47_1_29_(conv47_1_29_)
        conv45_1_30_ = self.conv45_1_30_(relu43_)
        batchnorm45_1_30_ = self.batchnorm45_1_30_(conv45_1_30_)
        relu45_1_30_ = self.relu45_1_30_(batchnorm45_1_30_)
        conv46_1_30_padding = self.conv46_1_30_padding(relu45_1_30_)
        conv46_1_30_ = self.conv46_1_30_(conv46_1_30_padding)
        batchnorm46_1_30_ = self.batchnorm46_1_30_(conv46_1_30_)
        relu46_1_30_ = self.relu46_1_30_(batchnorm46_1_30_)
        conv47_1_30_ = self.conv47_1_30_(relu46_1_30_)
        batchnorm47_1_30_ = self.batchnorm47_1_30_(conv47_1_30_)
        conv45_1_31_ = self.conv45_1_31_(relu43_)
        batchnorm45_1_31_ = self.batchnorm45_1_31_(conv45_1_31_)
        relu45_1_31_ = self.relu45_1_31_(batchnorm45_1_31_)
        conv46_1_31_padding = self.conv46_1_31_padding(relu45_1_31_)
        conv46_1_31_ = self.conv46_1_31_(conv46_1_31_padding)
        batchnorm46_1_31_ = self.batchnorm46_1_31_(conv46_1_31_)
        relu46_1_31_ = self.relu46_1_31_(batchnorm46_1_31_)
        conv47_1_31_ = self.conv47_1_31_(relu46_1_31_)
        batchnorm47_1_31_ = self.batchnorm47_1_31_(conv47_1_31_)
        conv45_1_32_ = self.conv45_1_32_(relu43_)
        batchnorm45_1_32_ = self.batchnorm45_1_32_(conv45_1_32_)
        relu45_1_32_ = self.relu45_1_32_(batchnorm45_1_32_)
        conv46_1_32_padding = self.conv46_1_32_padding(relu45_1_32_)
        conv46_1_32_ = self.conv46_1_32_(conv46_1_32_padding)
        batchnorm46_1_32_ = self.batchnorm46_1_32_(conv46_1_32_)
        relu46_1_32_ = self.relu46_1_32_(batchnorm46_1_32_)
        conv47_1_32_ = self.conv47_1_32_(relu46_1_32_)
        batchnorm47_1_32_ = self.batchnorm47_1_32_(conv47_1_32_)
        add48_1_ = batchnorm47_1_1_ + batchnorm47_1_2_ + batchnorm47_1_3_ + batchnorm47_1_4_ + batchnorm47_1_5_ + batchnorm47_1_6_ + batchnorm47_1_7_ + batchnorm47_1_8_ + batchnorm47_1_9_ + batchnorm47_1_10_ + batchnorm47_1_11_ + batchnorm47_1_12_ + batchnorm47_1_13_ + batchnorm47_1_14_ + batchnorm47_1_15_ + batchnorm47_1_16_ + batchnorm47_1_17_ + batchnorm47_1_18_ + batchnorm47_1_19_ + batchnorm47_1_20_ + batchnorm47_1_21_ + batchnorm47_1_22_ + batchnorm47_1_23_ + batchnorm47_1_24_ + batchnorm47_1_25_ + batchnorm47_1_26_ + batchnorm47_1_27_ + batchnorm47_1_28_ + batchnorm47_1_29_ + batchnorm47_1_30_ + batchnorm47_1_31_ + batchnorm47_1_32_
        conv44_2_ = self.conv44_2_(relu43_)
        batchnorm44_2_ = self.batchnorm44_2_(conv44_2_)
        add49_ = add48_1_ + batchnorm44_2_
        relu49_ = self.relu49_(add49_)
        conv52_1_1_1_ = self.conv52_1_1_1_(relu49_)
        batchnorm52_1_1_1_ = self.batchnorm52_1_1_1_(conv52_1_1_1_)
        relu52_1_1_1_ = self.relu52_1_1_1_(batchnorm52_1_1_1_)
        conv53_1_1_1_padding = self.conv53_1_1_1_padding(relu52_1_1_1_)
        conv53_1_1_1_ = self.conv53_1_1_1_(conv53_1_1_1_padding)
        batchnorm53_1_1_1_ = self.batchnorm53_1_1_1_(conv53_1_1_1_)
        relu53_1_1_1_ = self.relu53_1_1_1_(batchnorm53_1_1_1_)
        conv54_1_1_1_ = self.conv54_1_1_1_(relu53_1_1_1_)
        batchnorm54_1_1_1_ = self.batchnorm54_1_1_1_(conv54_1_1_1_)
        conv52_1_1_2_ = self.conv52_1_1_2_(relu49_)
        batchnorm52_1_1_2_ = self.batchnorm52_1_1_2_(conv52_1_1_2_)
        relu52_1_1_2_ = self.relu52_1_1_2_(batchnorm52_1_1_2_)
        conv53_1_1_2_padding = self.conv53_1_1_2_padding(relu52_1_1_2_)
        conv53_1_1_2_ = self.conv53_1_1_2_(conv53_1_1_2_padding)
        batchnorm53_1_1_2_ = self.batchnorm53_1_1_2_(conv53_1_1_2_)
        relu53_1_1_2_ = self.relu53_1_1_2_(batchnorm53_1_1_2_)
        conv54_1_1_2_ = self.conv54_1_1_2_(relu53_1_1_2_)
        batchnorm54_1_1_2_ = self.batchnorm54_1_1_2_(conv54_1_1_2_)
        conv52_1_1_3_ = self.conv52_1_1_3_(relu49_)
        batchnorm52_1_1_3_ = self.batchnorm52_1_1_3_(conv52_1_1_3_)
        relu52_1_1_3_ = self.relu52_1_1_3_(batchnorm52_1_1_3_)
        conv53_1_1_3_padding = self.conv53_1_1_3_padding(relu52_1_1_3_)
        conv53_1_1_3_ = self.conv53_1_1_3_(conv53_1_1_3_padding)
        batchnorm53_1_1_3_ = self.batchnorm53_1_1_3_(conv53_1_1_3_)
        relu53_1_1_3_ = self.relu53_1_1_3_(batchnorm53_1_1_3_)
        conv54_1_1_3_ = self.conv54_1_1_3_(relu53_1_1_3_)
        batchnorm54_1_1_3_ = self.batchnorm54_1_1_3_(conv54_1_1_3_)
        conv52_1_1_4_ = self.conv52_1_1_4_(relu49_)
        batchnorm52_1_1_4_ = self.batchnorm52_1_1_4_(conv52_1_1_4_)
        relu52_1_1_4_ = self.relu52_1_1_4_(batchnorm52_1_1_4_)
        conv53_1_1_4_padding = self.conv53_1_1_4_padding(relu52_1_1_4_)
        conv53_1_1_4_ = self.conv53_1_1_4_(conv53_1_1_4_padding)
        batchnorm53_1_1_4_ = self.batchnorm53_1_1_4_(conv53_1_1_4_)
        relu53_1_1_4_ = self.relu53_1_1_4_(batchnorm53_1_1_4_)
        conv54_1_1_4_ = self.conv54_1_1_4_(relu53_1_1_4_)
        batchnorm54_1_1_4_ = self.batchnorm54_1_1_4_(conv54_1_1_4_)
        conv52_1_1_5_ = self.conv52_1_1_5_(relu49_)
        batchnorm52_1_1_5_ = self.batchnorm52_1_1_5_(conv52_1_1_5_)
        relu52_1_1_5_ = self.relu52_1_1_5_(batchnorm52_1_1_5_)
        conv53_1_1_5_padding = self.conv53_1_1_5_padding(relu52_1_1_5_)
        conv53_1_1_5_ = self.conv53_1_1_5_(conv53_1_1_5_padding)
        batchnorm53_1_1_5_ = self.batchnorm53_1_1_5_(conv53_1_1_5_)
        relu53_1_1_5_ = self.relu53_1_1_5_(batchnorm53_1_1_5_)
        conv54_1_1_5_ = self.conv54_1_1_5_(relu53_1_1_5_)
        batchnorm54_1_1_5_ = self.batchnorm54_1_1_5_(conv54_1_1_5_)
        conv52_1_1_6_ = self.conv52_1_1_6_(relu49_)
        batchnorm52_1_1_6_ = self.batchnorm52_1_1_6_(conv52_1_1_6_)
        relu52_1_1_6_ = self.relu52_1_1_6_(batchnorm52_1_1_6_)
        conv53_1_1_6_padding = self.conv53_1_1_6_padding(relu52_1_1_6_)
        conv53_1_1_6_ = self.conv53_1_1_6_(conv53_1_1_6_padding)
        batchnorm53_1_1_6_ = self.batchnorm53_1_1_6_(conv53_1_1_6_)
        relu53_1_1_6_ = self.relu53_1_1_6_(batchnorm53_1_1_6_)
        conv54_1_1_6_ = self.conv54_1_1_6_(relu53_1_1_6_)
        batchnorm54_1_1_6_ = self.batchnorm54_1_1_6_(conv54_1_1_6_)
        conv52_1_1_7_ = self.conv52_1_1_7_(relu49_)
        batchnorm52_1_1_7_ = self.batchnorm52_1_1_7_(conv52_1_1_7_)
        relu52_1_1_7_ = self.relu52_1_1_7_(batchnorm52_1_1_7_)
        conv53_1_1_7_padding = self.conv53_1_1_7_padding(relu52_1_1_7_)
        conv53_1_1_7_ = self.conv53_1_1_7_(conv53_1_1_7_padding)
        batchnorm53_1_1_7_ = self.batchnorm53_1_1_7_(conv53_1_1_7_)
        relu53_1_1_7_ = self.relu53_1_1_7_(batchnorm53_1_1_7_)
        conv54_1_1_7_ = self.conv54_1_1_7_(relu53_1_1_7_)
        batchnorm54_1_1_7_ = self.batchnorm54_1_1_7_(conv54_1_1_7_)
        conv52_1_1_8_ = self.conv52_1_1_8_(relu49_)
        batchnorm52_1_1_8_ = self.batchnorm52_1_1_8_(conv52_1_1_8_)
        relu52_1_1_8_ = self.relu52_1_1_8_(batchnorm52_1_1_8_)
        conv53_1_1_8_padding = self.conv53_1_1_8_padding(relu52_1_1_8_)
        conv53_1_1_8_ = self.conv53_1_1_8_(conv53_1_1_8_padding)
        batchnorm53_1_1_8_ = self.batchnorm53_1_1_8_(conv53_1_1_8_)
        relu53_1_1_8_ = self.relu53_1_1_8_(batchnorm53_1_1_8_)
        conv54_1_1_8_ = self.conv54_1_1_8_(relu53_1_1_8_)
        batchnorm54_1_1_8_ = self.batchnorm54_1_1_8_(conv54_1_1_8_)
        conv52_1_1_9_ = self.conv52_1_1_9_(relu49_)
        batchnorm52_1_1_9_ = self.batchnorm52_1_1_9_(conv52_1_1_9_)
        relu52_1_1_9_ = self.relu52_1_1_9_(batchnorm52_1_1_9_)
        conv53_1_1_9_padding = self.conv53_1_1_9_padding(relu52_1_1_9_)
        conv53_1_1_9_ = self.conv53_1_1_9_(conv53_1_1_9_padding)
        batchnorm53_1_1_9_ = self.batchnorm53_1_1_9_(conv53_1_1_9_)
        relu53_1_1_9_ = self.relu53_1_1_9_(batchnorm53_1_1_9_)
        conv54_1_1_9_ = self.conv54_1_1_9_(relu53_1_1_9_)
        batchnorm54_1_1_9_ = self.batchnorm54_1_1_9_(conv54_1_1_9_)
        conv52_1_1_10_ = self.conv52_1_1_10_(relu49_)
        batchnorm52_1_1_10_ = self.batchnorm52_1_1_10_(conv52_1_1_10_)
        relu52_1_1_10_ = self.relu52_1_1_10_(batchnorm52_1_1_10_)
        conv53_1_1_10_padding = self.conv53_1_1_10_padding(relu52_1_1_10_)
        conv53_1_1_10_ = self.conv53_1_1_10_(conv53_1_1_10_padding)
        batchnorm53_1_1_10_ = self.batchnorm53_1_1_10_(conv53_1_1_10_)
        relu53_1_1_10_ = self.relu53_1_1_10_(batchnorm53_1_1_10_)
        conv54_1_1_10_ = self.conv54_1_1_10_(relu53_1_1_10_)
        batchnorm54_1_1_10_ = self.batchnorm54_1_1_10_(conv54_1_1_10_)
        conv52_1_1_11_ = self.conv52_1_1_11_(relu49_)
        batchnorm52_1_1_11_ = self.batchnorm52_1_1_11_(conv52_1_1_11_)
        relu52_1_1_11_ = self.relu52_1_1_11_(batchnorm52_1_1_11_)
        conv53_1_1_11_padding = self.conv53_1_1_11_padding(relu52_1_1_11_)
        conv53_1_1_11_ = self.conv53_1_1_11_(conv53_1_1_11_padding)
        batchnorm53_1_1_11_ = self.batchnorm53_1_1_11_(conv53_1_1_11_)
        relu53_1_1_11_ = self.relu53_1_1_11_(batchnorm53_1_1_11_)
        conv54_1_1_11_ = self.conv54_1_1_11_(relu53_1_1_11_)
        batchnorm54_1_1_11_ = self.batchnorm54_1_1_11_(conv54_1_1_11_)
        conv52_1_1_12_ = self.conv52_1_1_12_(relu49_)
        batchnorm52_1_1_12_ = self.batchnorm52_1_1_12_(conv52_1_1_12_)
        relu52_1_1_12_ = self.relu52_1_1_12_(batchnorm52_1_1_12_)
        conv53_1_1_12_padding = self.conv53_1_1_12_padding(relu52_1_1_12_)
        conv53_1_1_12_ = self.conv53_1_1_12_(conv53_1_1_12_padding)
        batchnorm53_1_1_12_ = self.batchnorm53_1_1_12_(conv53_1_1_12_)
        relu53_1_1_12_ = self.relu53_1_1_12_(batchnorm53_1_1_12_)
        conv54_1_1_12_ = self.conv54_1_1_12_(relu53_1_1_12_)
        batchnorm54_1_1_12_ = self.batchnorm54_1_1_12_(conv54_1_1_12_)
        conv52_1_1_13_ = self.conv52_1_1_13_(relu49_)
        batchnorm52_1_1_13_ = self.batchnorm52_1_1_13_(conv52_1_1_13_)
        relu52_1_1_13_ = self.relu52_1_1_13_(batchnorm52_1_1_13_)
        conv53_1_1_13_padding = self.conv53_1_1_13_padding(relu52_1_1_13_)
        conv53_1_1_13_ = self.conv53_1_1_13_(conv53_1_1_13_padding)
        batchnorm53_1_1_13_ = self.batchnorm53_1_1_13_(conv53_1_1_13_)
        relu53_1_1_13_ = self.relu53_1_1_13_(batchnorm53_1_1_13_)
        conv54_1_1_13_ = self.conv54_1_1_13_(relu53_1_1_13_)
        batchnorm54_1_1_13_ = self.batchnorm54_1_1_13_(conv54_1_1_13_)
        conv52_1_1_14_ = self.conv52_1_1_14_(relu49_)
        batchnorm52_1_1_14_ = self.batchnorm52_1_1_14_(conv52_1_1_14_)
        relu52_1_1_14_ = self.relu52_1_1_14_(batchnorm52_1_1_14_)
        conv53_1_1_14_padding = self.conv53_1_1_14_padding(relu52_1_1_14_)
        conv53_1_1_14_ = self.conv53_1_1_14_(conv53_1_1_14_padding)
        batchnorm53_1_1_14_ = self.batchnorm53_1_1_14_(conv53_1_1_14_)
        relu53_1_1_14_ = self.relu53_1_1_14_(batchnorm53_1_1_14_)
        conv54_1_1_14_ = self.conv54_1_1_14_(relu53_1_1_14_)
        batchnorm54_1_1_14_ = self.batchnorm54_1_1_14_(conv54_1_1_14_)
        conv52_1_1_15_ = self.conv52_1_1_15_(relu49_)
        batchnorm52_1_1_15_ = self.batchnorm52_1_1_15_(conv52_1_1_15_)
        relu52_1_1_15_ = self.relu52_1_1_15_(batchnorm52_1_1_15_)
        conv53_1_1_15_padding = self.conv53_1_1_15_padding(relu52_1_1_15_)
        conv53_1_1_15_ = self.conv53_1_1_15_(conv53_1_1_15_padding)
        batchnorm53_1_1_15_ = self.batchnorm53_1_1_15_(conv53_1_1_15_)
        relu53_1_1_15_ = self.relu53_1_1_15_(batchnorm53_1_1_15_)
        conv54_1_1_15_ = self.conv54_1_1_15_(relu53_1_1_15_)
        batchnorm54_1_1_15_ = self.batchnorm54_1_1_15_(conv54_1_1_15_)
        conv52_1_1_16_ = self.conv52_1_1_16_(relu49_)
        batchnorm52_1_1_16_ = self.batchnorm52_1_1_16_(conv52_1_1_16_)
        relu52_1_1_16_ = self.relu52_1_1_16_(batchnorm52_1_1_16_)
        conv53_1_1_16_padding = self.conv53_1_1_16_padding(relu52_1_1_16_)
        conv53_1_1_16_ = self.conv53_1_1_16_(conv53_1_1_16_padding)
        batchnorm53_1_1_16_ = self.batchnorm53_1_1_16_(conv53_1_1_16_)
        relu53_1_1_16_ = self.relu53_1_1_16_(batchnorm53_1_1_16_)
        conv54_1_1_16_ = self.conv54_1_1_16_(relu53_1_1_16_)
        batchnorm54_1_1_16_ = self.batchnorm54_1_1_16_(conv54_1_1_16_)
        conv52_1_1_17_ = self.conv52_1_1_17_(relu49_)
        batchnorm52_1_1_17_ = self.batchnorm52_1_1_17_(conv52_1_1_17_)
        relu52_1_1_17_ = self.relu52_1_1_17_(batchnorm52_1_1_17_)
        conv53_1_1_17_padding = self.conv53_1_1_17_padding(relu52_1_1_17_)
        conv53_1_1_17_ = self.conv53_1_1_17_(conv53_1_1_17_padding)
        batchnorm53_1_1_17_ = self.batchnorm53_1_1_17_(conv53_1_1_17_)
        relu53_1_1_17_ = self.relu53_1_1_17_(batchnorm53_1_1_17_)
        conv54_1_1_17_ = self.conv54_1_1_17_(relu53_1_1_17_)
        batchnorm54_1_1_17_ = self.batchnorm54_1_1_17_(conv54_1_1_17_)
        conv52_1_1_18_ = self.conv52_1_1_18_(relu49_)
        batchnorm52_1_1_18_ = self.batchnorm52_1_1_18_(conv52_1_1_18_)
        relu52_1_1_18_ = self.relu52_1_1_18_(batchnorm52_1_1_18_)
        conv53_1_1_18_padding = self.conv53_1_1_18_padding(relu52_1_1_18_)
        conv53_1_1_18_ = self.conv53_1_1_18_(conv53_1_1_18_padding)
        batchnorm53_1_1_18_ = self.batchnorm53_1_1_18_(conv53_1_1_18_)
        relu53_1_1_18_ = self.relu53_1_1_18_(batchnorm53_1_1_18_)
        conv54_1_1_18_ = self.conv54_1_1_18_(relu53_1_1_18_)
        batchnorm54_1_1_18_ = self.batchnorm54_1_1_18_(conv54_1_1_18_)
        conv52_1_1_19_ = self.conv52_1_1_19_(relu49_)
        batchnorm52_1_1_19_ = self.batchnorm52_1_1_19_(conv52_1_1_19_)
        relu52_1_1_19_ = self.relu52_1_1_19_(batchnorm52_1_1_19_)
        conv53_1_1_19_padding = self.conv53_1_1_19_padding(relu52_1_1_19_)
        conv53_1_1_19_ = self.conv53_1_1_19_(conv53_1_1_19_padding)
        batchnorm53_1_1_19_ = self.batchnorm53_1_1_19_(conv53_1_1_19_)
        relu53_1_1_19_ = self.relu53_1_1_19_(batchnorm53_1_1_19_)
        conv54_1_1_19_ = self.conv54_1_1_19_(relu53_1_1_19_)
        batchnorm54_1_1_19_ = self.batchnorm54_1_1_19_(conv54_1_1_19_)
        conv52_1_1_20_ = self.conv52_1_1_20_(relu49_)
        batchnorm52_1_1_20_ = self.batchnorm52_1_1_20_(conv52_1_1_20_)
        relu52_1_1_20_ = self.relu52_1_1_20_(batchnorm52_1_1_20_)
        conv53_1_1_20_padding = self.conv53_1_1_20_padding(relu52_1_1_20_)
        conv53_1_1_20_ = self.conv53_1_1_20_(conv53_1_1_20_padding)
        batchnorm53_1_1_20_ = self.batchnorm53_1_1_20_(conv53_1_1_20_)
        relu53_1_1_20_ = self.relu53_1_1_20_(batchnorm53_1_1_20_)
        conv54_1_1_20_ = self.conv54_1_1_20_(relu53_1_1_20_)
        batchnorm54_1_1_20_ = self.batchnorm54_1_1_20_(conv54_1_1_20_)
        conv52_1_1_21_ = self.conv52_1_1_21_(relu49_)
        batchnorm52_1_1_21_ = self.batchnorm52_1_1_21_(conv52_1_1_21_)
        relu52_1_1_21_ = self.relu52_1_1_21_(batchnorm52_1_1_21_)
        conv53_1_1_21_padding = self.conv53_1_1_21_padding(relu52_1_1_21_)
        conv53_1_1_21_ = self.conv53_1_1_21_(conv53_1_1_21_padding)
        batchnorm53_1_1_21_ = self.batchnorm53_1_1_21_(conv53_1_1_21_)
        relu53_1_1_21_ = self.relu53_1_1_21_(batchnorm53_1_1_21_)
        conv54_1_1_21_ = self.conv54_1_1_21_(relu53_1_1_21_)
        batchnorm54_1_1_21_ = self.batchnorm54_1_1_21_(conv54_1_1_21_)
        conv52_1_1_22_ = self.conv52_1_1_22_(relu49_)
        batchnorm52_1_1_22_ = self.batchnorm52_1_1_22_(conv52_1_1_22_)
        relu52_1_1_22_ = self.relu52_1_1_22_(batchnorm52_1_1_22_)
        conv53_1_1_22_padding = self.conv53_1_1_22_padding(relu52_1_1_22_)
        conv53_1_1_22_ = self.conv53_1_1_22_(conv53_1_1_22_padding)
        batchnorm53_1_1_22_ = self.batchnorm53_1_1_22_(conv53_1_1_22_)
        relu53_1_1_22_ = self.relu53_1_1_22_(batchnorm53_1_1_22_)
        conv54_1_1_22_ = self.conv54_1_1_22_(relu53_1_1_22_)
        batchnorm54_1_1_22_ = self.batchnorm54_1_1_22_(conv54_1_1_22_)
        conv52_1_1_23_ = self.conv52_1_1_23_(relu49_)
        batchnorm52_1_1_23_ = self.batchnorm52_1_1_23_(conv52_1_1_23_)
        relu52_1_1_23_ = self.relu52_1_1_23_(batchnorm52_1_1_23_)
        conv53_1_1_23_padding = self.conv53_1_1_23_padding(relu52_1_1_23_)
        conv53_1_1_23_ = self.conv53_1_1_23_(conv53_1_1_23_padding)
        batchnorm53_1_1_23_ = self.batchnorm53_1_1_23_(conv53_1_1_23_)
        relu53_1_1_23_ = self.relu53_1_1_23_(batchnorm53_1_1_23_)
        conv54_1_1_23_ = self.conv54_1_1_23_(relu53_1_1_23_)
        batchnorm54_1_1_23_ = self.batchnorm54_1_1_23_(conv54_1_1_23_)
        conv52_1_1_24_ = self.conv52_1_1_24_(relu49_)
        batchnorm52_1_1_24_ = self.batchnorm52_1_1_24_(conv52_1_1_24_)
        relu52_1_1_24_ = self.relu52_1_1_24_(batchnorm52_1_1_24_)
        conv53_1_1_24_padding = self.conv53_1_1_24_padding(relu52_1_1_24_)
        conv53_1_1_24_ = self.conv53_1_1_24_(conv53_1_1_24_padding)
        batchnorm53_1_1_24_ = self.batchnorm53_1_1_24_(conv53_1_1_24_)
        relu53_1_1_24_ = self.relu53_1_1_24_(batchnorm53_1_1_24_)
        conv54_1_1_24_ = self.conv54_1_1_24_(relu53_1_1_24_)
        batchnorm54_1_1_24_ = self.batchnorm54_1_1_24_(conv54_1_1_24_)
        conv52_1_1_25_ = self.conv52_1_1_25_(relu49_)
        batchnorm52_1_1_25_ = self.batchnorm52_1_1_25_(conv52_1_1_25_)
        relu52_1_1_25_ = self.relu52_1_1_25_(batchnorm52_1_1_25_)
        conv53_1_1_25_padding = self.conv53_1_1_25_padding(relu52_1_1_25_)
        conv53_1_1_25_ = self.conv53_1_1_25_(conv53_1_1_25_padding)
        batchnorm53_1_1_25_ = self.batchnorm53_1_1_25_(conv53_1_1_25_)
        relu53_1_1_25_ = self.relu53_1_1_25_(batchnorm53_1_1_25_)
        conv54_1_1_25_ = self.conv54_1_1_25_(relu53_1_1_25_)
        batchnorm54_1_1_25_ = self.batchnorm54_1_1_25_(conv54_1_1_25_)
        conv52_1_1_26_ = self.conv52_1_1_26_(relu49_)
        batchnorm52_1_1_26_ = self.batchnorm52_1_1_26_(conv52_1_1_26_)
        relu52_1_1_26_ = self.relu52_1_1_26_(batchnorm52_1_1_26_)
        conv53_1_1_26_padding = self.conv53_1_1_26_padding(relu52_1_1_26_)
        conv53_1_1_26_ = self.conv53_1_1_26_(conv53_1_1_26_padding)
        batchnorm53_1_1_26_ = self.batchnorm53_1_1_26_(conv53_1_1_26_)
        relu53_1_1_26_ = self.relu53_1_1_26_(batchnorm53_1_1_26_)
        conv54_1_1_26_ = self.conv54_1_1_26_(relu53_1_1_26_)
        batchnorm54_1_1_26_ = self.batchnorm54_1_1_26_(conv54_1_1_26_)
        conv52_1_1_27_ = self.conv52_1_1_27_(relu49_)
        batchnorm52_1_1_27_ = self.batchnorm52_1_1_27_(conv52_1_1_27_)
        relu52_1_1_27_ = self.relu52_1_1_27_(batchnorm52_1_1_27_)
        conv53_1_1_27_padding = self.conv53_1_1_27_padding(relu52_1_1_27_)
        conv53_1_1_27_ = self.conv53_1_1_27_(conv53_1_1_27_padding)
        batchnorm53_1_1_27_ = self.batchnorm53_1_1_27_(conv53_1_1_27_)
        relu53_1_1_27_ = self.relu53_1_1_27_(batchnorm53_1_1_27_)
        conv54_1_1_27_ = self.conv54_1_1_27_(relu53_1_1_27_)
        batchnorm54_1_1_27_ = self.batchnorm54_1_1_27_(conv54_1_1_27_)
        conv52_1_1_28_ = self.conv52_1_1_28_(relu49_)
        batchnorm52_1_1_28_ = self.batchnorm52_1_1_28_(conv52_1_1_28_)
        relu52_1_1_28_ = self.relu52_1_1_28_(batchnorm52_1_1_28_)
        conv53_1_1_28_padding = self.conv53_1_1_28_padding(relu52_1_1_28_)
        conv53_1_1_28_ = self.conv53_1_1_28_(conv53_1_1_28_padding)
        batchnorm53_1_1_28_ = self.batchnorm53_1_1_28_(conv53_1_1_28_)
        relu53_1_1_28_ = self.relu53_1_1_28_(batchnorm53_1_1_28_)
        conv54_1_1_28_ = self.conv54_1_1_28_(relu53_1_1_28_)
        batchnorm54_1_1_28_ = self.batchnorm54_1_1_28_(conv54_1_1_28_)
        conv52_1_1_29_ = self.conv52_1_1_29_(relu49_)
        batchnorm52_1_1_29_ = self.batchnorm52_1_1_29_(conv52_1_1_29_)
        relu52_1_1_29_ = self.relu52_1_1_29_(batchnorm52_1_1_29_)
        conv53_1_1_29_padding = self.conv53_1_1_29_padding(relu52_1_1_29_)
        conv53_1_1_29_ = self.conv53_1_1_29_(conv53_1_1_29_padding)
        batchnorm53_1_1_29_ = self.batchnorm53_1_1_29_(conv53_1_1_29_)
        relu53_1_1_29_ = self.relu53_1_1_29_(batchnorm53_1_1_29_)
        conv54_1_1_29_ = self.conv54_1_1_29_(relu53_1_1_29_)
        batchnorm54_1_1_29_ = self.batchnorm54_1_1_29_(conv54_1_1_29_)
        conv52_1_1_30_ = self.conv52_1_1_30_(relu49_)
        batchnorm52_1_1_30_ = self.batchnorm52_1_1_30_(conv52_1_1_30_)
        relu52_1_1_30_ = self.relu52_1_1_30_(batchnorm52_1_1_30_)
        conv53_1_1_30_padding = self.conv53_1_1_30_padding(relu52_1_1_30_)
        conv53_1_1_30_ = self.conv53_1_1_30_(conv53_1_1_30_padding)
        batchnorm53_1_1_30_ = self.batchnorm53_1_1_30_(conv53_1_1_30_)
        relu53_1_1_30_ = self.relu53_1_1_30_(batchnorm53_1_1_30_)
        conv54_1_1_30_ = self.conv54_1_1_30_(relu53_1_1_30_)
        batchnorm54_1_1_30_ = self.batchnorm54_1_1_30_(conv54_1_1_30_)
        conv52_1_1_31_ = self.conv52_1_1_31_(relu49_)
        batchnorm52_1_1_31_ = self.batchnorm52_1_1_31_(conv52_1_1_31_)
        relu52_1_1_31_ = self.relu52_1_1_31_(batchnorm52_1_1_31_)
        conv53_1_1_31_padding = self.conv53_1_1_31_padding(relu52_1_1_31_)
        conv53_1_1_31_ = self.conv53_1_1_31_(conv53_1_1_31_padding)
        batchnorm53_1_1_31_ = self.batchnorm53_1_1_31_(conv53_1_1_31_)
        relu53_1_1_31_ = self.relu53_1_1_31_(batchnorm53_1_1_31_)
        conv54_1_1_31_ = self.conv54_1_1_31_(relu53_1_1_31_)
        batchnorm54_1_1_31_ = self.batchnorm54_1_1_31_(conv54_1_1_31_)
        conv52_1_1_32_ = self.conv52_1_1_32_(relu49_)
        batchnorm52_1_1_32_ = self.batchnorm52_1_1_32_(conv52_1_1_32_)
        relu52_1_1_32_ = self.relu52_1_1_32_(batchnorm52_1_1_32_)
        conv53_1_1_32_padding = self.conv53_1_1_32_padding(relu52_1_1_32_)
        conv53_1_1_32_ = self.conv53_1_1_32_(conv53_1_1_32_padding)
        batchnorm53_1_1_32_ = self.batchnorm53_1_1_32_(conv53_1_1_32_)
        relu53_1_1_32_ = self.relu53_1_1_32_(batchnorm53_1_1_32_)
        conv54_1_1_32_ = self.conv54_1_1_32_(relu53_1_1_32_)
        batchnorm54_1_1_32_ = self.batchnorm54_1_1_32_(conv54_1_1_32_)
        add55_1_1_ = batchnorm54_1_1_1_ + batchnorm54_1_1_2_ + batchnorm54_1_1_3_ + batchnorm54_1_1_4_ + batchnorm54_1_1_5_ + batchnorm54_1_1_6_ + batchnorm54_1_1_7_ + batchnorm54_1_1_8_ + batchnorm54_1_1_9_ + batchnorm54_1_1_10_ + batchnorm54_1_1_11_ + batchnorm54_1_1_12_ + batchnorm54_1_1_13_ + batchnorm54_1_1_14_ + batchnorm54_1_1_15_ + batchnorm54_1_1_16_ + batchnorm54_1_1_17_ + batchnorm54_1_1_18_ + batchnorm54_1_1_19_ + batchnorm54_1_1_20_ + batchnorm54_1_1_21_ + batchnorm54_1_1_22_ + batchnorm54_1_1_23_ + batchnorm54_1_1_24_ + batchnorm54_1_1_25_ + batchnorm54_1_1_26_ + batchnorm54_1_1_27_ + batchnorm54_1_1_28_ + batchnorm54_1_1_29_ + batchnorm54_1_1_30_ + batchnorm54_1_1_31_ + batchnorm54_1_1_32_
        add56_1_ = add55_1_1_ + relu49_
        relu56_1_ = self.relu56_1_(add56_1_)
        conv58_1_1_1_ = self.conv58_1_1_1_(relu56_1_)
        batchnorm58_1_1_1_ = self.batchnorm58_1_1_1_(conv58_1_1_1_)
        relu58_1_1_1_ = self.relu58_1_1_1_(batchnorm58_1_1_1_)
        conv59_1_1_1_padding = self.conv59_1_1_1_padding(relu58_1_1_1_)
        conv59_1_1_1_ = self.conv59_1_1_1_(conv59_1_1_1_padding)
        batchnorm59_1_1_1_ = self.batchnorm59_1_1_1_(conv59_1_1_1_)
        relu59_1_1_1_ = self.relu59_1_1_1_(batchnorm59_1_1_1_)
        conv60_1_1_1_ = self.conv60_1_1_1_(relu59_1_1_1_)
        batchnorm60_1_1_1_ = self.batchnorm60_1_1_1_(conv60_1_1_1_)
        conv58_1_1_2_ = self.conv58_1_1_2_(relu56_1_)
        batchnorm58_1_1_2_ = self.batchnorm58_1_1_2_(conv58_1_1_2_)
        relu58_1_1_2_ = self.relu58_1_1_2_(batchnorm58_1_1_2_)
        conv59_1_1_2_padding = self.conv59_1_1_2_padding(relu58_1_1_2_)
        conv59_1_1_2_ = self.conv59_1_1_2_(conv59_1_1_2_padding)
        batchnorm59_1_1_2_ = self.batchnorm59_1_1_2_(conv59_1_1_2_)
        relu59_1_1_2_ = self.relu59_1_1_2_(batchnorm59_1_1_2_)
        conv60_1_1_2_ = self.conv60_1_1_2_(relu59_1_1_2_)
        batchnorm60_1_1_2_ = self.batchnorm60_1_1_2_(conv60_1_1_2_)
        conv58_1_1_3_ = self.conv58_1_1_3_(relu56_1_)
        batchnorm58_1_1_3_ = self.batchnorm58_1_1_3_(conv58_1_1_3_)
        relu58_1_1_3_ = self.relu58_1_1_3_(batchnorm58_1_1_3_)
        conv59_1_1_3_padding = self.conv59_1_1_3_padding(relu58_1_1_3_)
        conv59_1_1_3_ = self.conv59_1_1_3_(conv59_1_1_3_padding)
        batchnorm59_1_1_3_ = self.batchnorm59_1_1_3_(conv59_1_1_3_)
        relu59_1_1_3_ = self.relu59_1_1_3_(batchnorm59_1_1_3_)
        conv60_1_1_3_ = self.conv60_1_1_3_(relu59_1_1_3_)
        batchnorm60_1_1_3_ = self.batchnorm60_1_1_3_(conv60_1_1_3_)
        conv58_1_1_4_ = self.conv58_1_1_4_(relu56_1_)
        batchnorm58_1_1_4_ = self.batchnorm58_1_1_4_(conv58_1_1_4_)
        relu58_1_1_4_ = self.relu58_1_1_4_(batchnorm58_1_1_4_)
        conv59_1_1_4_padding = self.conv59_1_1_4_padding(relu58_1_1_4_)
        conv59_1_1_4_ = self.conv59_1_1_4_(conv59_1_1_4_padding)
        batchnorm59_1_1_4_ = self.batchnorm59_1_1_4_(conv59_1_1_4_)
        relu59_1_1_4_ = self.relu59_1_1_4_(batchnorm59_1_1_4_)
        conv60_1_1_4_ = self.conv60_1_1_4_(relu59_1_1_4_)
        batchnorm60_1_1_4_ = self.batchnorm60_1_1_4_(conv60_1_1_4_)
        conv58_1_1_5_ = self.conv58_1_1_5_(relu56_1_)
        batchnorm58_1_1_5_ = self.batchnorm58_1_1_5_(conv58_1_1_5_)
        relu58_1_1_5_ = self.relu58_1_1_5_(batchnorm58_1_1_5_)
        conv59_1_1_5_padding = self.conv59_1_1_5_padding(relu58_1_1_5_)
        conv59_1_1_5_ = self.conv59_1_1_5_(conv59_1_1_5_padding)
        batchnorm59_1_1_5_ = self.batchnorm59_1_1_5_(conv59_1_1_5_)
        relu59_1_1_5_ = self.relu59_1_1_5_(batchnorm59_1_1_5_)
        conv60_1_1_5_ = self.conv60_1_1_5_(relu59_1_1_5_)
        batchnorm60_1_1_5_ = self.batchnorm60_1_1_5_(conv60_1_1_5_)
        conv58_1_1_6_ = self.conv58_1_1_6_(relu56_1_)
        batchnorm58_1_1_6_ = self.batchnorm58_1_1_6_(conv58_1_1_6_)
        relu58_1_1_6_ = self.relu58_1_1_6_(batchnorm58_1_1_6_)
        conv59_1_1_6_padding = self.conv59_1_1_6_padding(relu58_1_1_6_)
        conv59_1_1_6_ = self.conv59_1_1_6_(conv59_1_1_6_padding)
        batchnorm59_1_1_6_ = self.batchnorm59_1_1_6_(conv59_1_1_6_)
        relu59_1_1_6_ = self.relu59_1_1_6_(batchnorm59_1_1_6_)
        conv60_1_1_6_ = self.conv60_1_1_6_(relu59_1_1_6_)
        batchnorm60_1_1_6_ = self.batchnorm60_1_1_6_(conv60_1_1_6_)
        conv58_1_1_7_ = self.conv58_1_1_7_(relu56_1_)
        batchnorm58_1_1_7_ = self.batchnorm58_1_1_7_(conv58_1_1_7_)
        relu58_1_1_7_ = self.relu58_1_1_7_(batchnorm58_1_1_7_)
        conv59_1_1_7_padding = self.conv59_1_1_7_padding(relu58_1_1_7_)
        conv59_1_1_7_ = self.conv59_1_1_7_(conv59_1_1_7_padding)
        batchnorm59_1_1_7_ = self.batchnorm59_1_1_7_(conv59_1_1_7_)
        relu59_1_1_7_ = self.relu59_1_1_7_(batchnorm59_1_1_7_)
        conv60_1_1_7_ = self.conv60_1_1_7_(relu59_1_1_7_)
        batchnorm60_1_1_7_ = self.batchnorm60_1_1_7_(conv60_1_1_7_)
        conv58_1_1_8_ = self.conv58_1_1_8_(relu56_1_)
        batchnorm58_1_1_8_ = self.batchnorm58_1_1_8_(conv58_1_1_8_)
        relu58_1_1_8_ = self.relu58_1_1_8_(batchnorm58_1_1_8_)
        conv59_1_1_8_padding = self.conv59_1_1_8_padding(relu58_1_1_8_)
        conv59_1_1_8_ = self.conv59_1_1_8_(conv59_1_1_8_padding)
        batchnorm59_1_1_8_ = self.batchnorm59_1_1_8_(conv59_1_1_8_)
        relu59_1_1_8_ = self.relu59_1_1_8_(batchnorm59_1_1_8_)
        conv60_1_1_8_ = self.conv60_1_1_8_(relu59_1_1_8_)
        batchnorm60_1_1_8_ = self.batchnorm60_1_1_8_(conv60_1_1_8_)
        conv58_1_1_9_ = self.conv58_1_1_9_(relu56_1_)
        batchnorm58_1_1_9_ = self.batchnorm58_1_1_9_(conv58_1_1_9_)
        relu58_1_1_9_ = self.relu58_1_1_9_(batchnorm58_1_1_9_)
        conv59_1_1_9_padding = self.conv59_1_1_9_padding(relu58_1_1_9_)
        conv59_1_1_9_ = self.conv59_1_1_9_(conv59_1_1_9_padding)
        batchnorm59_1_1_9_ = self.batchnorm59_1_1_9_(conv59_1_1_9_)
        relu59_1_1_9_ = self.relu59_1_1_9_(batchnorm59_1_1_9_)
        conv60_1_1_9_ = self.conv60_1_1_9_(relu59_1_1_9_)
        batchnorm60_1_1_9_ = self.batchnorm60_1_1_9_(conv60_1_1_9_)
        conv58_1_1_10_ = self.conv58_1_1_10_(relu56_1_)
        batchnorm58_1_1_10_ = self.batchnorm58_1_1_10_(conv58_1_1_10_)
        relu58_1_1_10_ = self.relu58_1_1_10_(batchnorm58_1_1_10_)
        conv59_1_1_10_padding = self.conv59_1_1_10_padding(relu58_1_1_10_)
        conv59_1_1_10_ = self.conv59_1_1_10_(conv59_1_1_10_padding)
        batchnorm59_1_1_10_ = self.batchnorm59_1_1_10_(conv59_1_1_10_)
        relu59_1_1_10_ = self.relu59_1_1_10_(batchnorm59_1_1_10_)
        conv60_1_1_10_ = self.conv60_1_1_10_(relu59_1_1_10_)
        batchnorm60_1_1_10_ = self.batchnorm60_1_1_10_(conv60_1_1_10_)
        conv58_1_1_11_ = self.conv58_1_1_11_(relu56_1_)
        batchnorm58_1_1_11_ = self.batchnorm58_1_1_11_(conv58_1_1_11_)
        relu58_1_1_11_ = self.relu58_1_1_11_(batchnorm58_1_1_11_)
        conv59_1_1_11_padding = self.conv59_1_1_11_padding(relu58_1_1_11_)
        conv59_1_1_11_ = self.conv59_1_1_11_(conv59_1_1_11_padding)
        batchnorm59_1_1_11_ = self.batchnorm59_1_1_11_(conv59_1_1_11_)
        relu59_1_1_11_ = self.relu59_1_1_11_(batchnorm59_1_1_11_)
        conv60_1_1_11_ = self.conv60_1_1_11_(relu59_1_1_11_)
        batchnorm60_1_1_11_ = self.batchnorm60_1_1_11_(conv60_1_1_11_)
        conv58_1_1_12_ = self.conv58_1_1_12_(relu56_1_)
        batchnorm58_1_1_12_ = self.batchnorm58_1_1_12_(conv58_1_1_12_)
        relu58_1_1_12_ = self.relu58_1_1_12_(batchnorm58_1_1_12_)
        conv59_1_1_12_padding = self.conv59_1_1_12_padding(relu58_1_1_12_)
        conv59_1_1_12_ = self.conv59_1_1_12_(conv59_1_1_12_padding)
        batchnorm59_1_1_12_ = self.batchnorm59_1_1_12_(conv59_1_1_12_)
        relu59_1_1_12_ = self.relu59_1_1_12_(batchnorm59_1_1_12_)
        conv60_1_1_12_ = self.conv60_1_1_12_(relu59_1_1_12_)
        batchnorm60_1_1_12_ = self.batchnorm60_1_1_12_(conv60_1_1_12_)
        conv58_1_1_13_ = self.conv58_1_1_13_(relu56_1_)
        batchnorm58_1_1_13_ = self.batchnorm58_1_1_13_(conv58_1_1_13_)
        relu58_1_1_13_ = self.relu58_1_1_13_(batchnorm58_1_1_13_)
        conv59_1_1_13_padding = self.conv59_1_1_13_padding(relu58_1_1_13_)
        conv59_1_1_13_ = self.conv59_1_1_13_(conv59_1_1_13_padding)
        batchnorm59_1_1_13_ = self.batchnorm59_1_1_13_(conv59_1_1_13_)
        relu59_1_1_13_ = self.relu59_1_1_13_(batchnorm59_1_1_13_)
        conv60_1_1_13_ = self.conv60_1_1_13_(relu59_1_1_13_)
        batchnorm60_1_1_13_ = self.batchnorm60_1_1_13_(conv60_1_1_13_)
        conv58_1_1_14_ = self.conv58_1_1_14_(relu56_1_)
        batchnorm58_1_1_14_ = self.batchnorm58_1_1_14_(conv58_1_1_14_)
        relu58_1_1_14_ = self.relu58_1_1_14_(batchnorm58_1_1_14_)
        conv59_1_1_14_padding = self.conv59_1_1_14_padding(relu58_1_1_14_)
        conv59_1_1_14_ = self.conv59_1_1_14_(conv59_1_1_14_padding)
        batchnorm59_1_1_14_ = self.batchnorm59_1_1_14_(conv59_1_1_14_)
        relu59_1_1_14_ = self.relu59_1_1_14_(batchnorm59_1_1_14_)
        conv60_1_1_14_ = self.conv60_1_1_14_(relu59_1_1_14_)
        batchnorm60_1_1_14_ = self.batchnorm60_1_1_14_(conv60_1_1_14_)
        conv58_1_1_15_ = self.conv58_1_1_15_(relu56_1_)
        batchnorm58_1_1_15_ = self.batchnorm58_1_1_15_(conv58_1_1_15_)
        relu58_1_1_15_ = self.relu58_1_1_15_(batchnorm58_1_1_15_)
        conv59_1_1_15_padding = self.conv59_1_1_15_padding(relu58_1_1_15_)
        conv59_1_1_15_ = self.conv59_1_1_15_(conv59_1_1_15_padding)
        batchnorm59_1_1_15_ = self.batchnorm59_1_1_15_(conv59_1_1_15_)
        relu59_1_1_15_ = self.relu59_1_1_15_(batchnorm59_1_1_15_)
        conv60_1_1_15_ = self.conv60_1_1_15_(relu59_1_1_15_)
        batchnorm60_1_1_15_ = self.batchnorm60_1_1_15_(conv60_1_1_15_)
        conv58_1_1_16_ = self.conv58_1_1_16_(relu56_1_)
        batchnorm58_1_1_16_ = self.batchnorm58_1_1_16_(conv58_1_1_16_)
        relu58_1_1_16_ = self.relu58_1_1_16_(batchnorm58_1_1_16_)
        conv59_1_1_16_padding = self.conv59_1_1_16_padding(relu58_1_1_16_)
        conv59_1_1_16_ = self.conv59_1_1_16_(conv59_1_1_16_padding)
        batchnorm59_1_1_16_ = self.batchnorm59_1_1_16_(conv59_1_1_16_)
        relu59_1_1_16_ = self.relu59_1_1_16_(batchnorm59_1_1_16_)
        conv60_1_1_16_ = self.conv60_1_1_16_(relu59_1_1_16_)
        batchnorm60_1_1_16_ = self.batchnorm60_1_1_16_(conv60_1_1_16_)
        conv58_1_1_17_ = self.conv58_1_1_17_(relu56_1_)
        batchnorm58_1_1_17_ = self.batchnorm58_1_1_17_(conv58_1_1_17_)
        relu58_1_1_17_ = self.relu58_1_1_17_(batchnorm58_1_1_17_)
        conv59_1_1_17_padding = self.conv59_1_1_17_padding(relu58_1_1_17_)
        conv59_1_1_17_ = self.conv59_1_1_17_(conv59_1_1_17_padding)
        batchnorm59_1_1_17_ = self.batchnorm59_1_1_17_(conv59_1_1_17_)
        relu59_1_1_17_ = self.relu59_1_1_17_(batchnorm59_1_1_17_)
        conv60_1_1_17_ = self.conv60_1_1_17_(relu59_1_1_17_)
        batchnorm60_1_1_17_ = self.batchnorm60_1_1_17_(conv60_1_1_17_)
        conv58_1_1_18_ = self.conv58_1_1_18_(relu56_1_)
        batchnorm58_1_1_18_ = self.batchnorm58_1_1_18_(conv58_1_1_18_)
        relu58_1_1_18_ = self.relu58_1_1_18_(batchnorm58_1_1_18_)
        conv59_1_1_18_padding = self.conv59_1_1_18_padding(relu58_1_1_18_)
        conv59_1_1_18_ = self.conv59_1_1_18_(conv59_1_1_18_padding)
        batchnorm59_1_1_18_ = self.batchnorm59_1_1_18_(conv59_1_1_18_)
        relu59_1_1_18_ = self.relu59_1_1_18_(batchnorm59_1_1_18_)
        conv60_1_1_18_ = self.conv60_1_1_18_(relu59_1_1_18_)
        batchnorm60_1_1_18_ = self.batchnorm60_1_1_18_(conv60_1_1_18_)
        conv58_1_1_19_ = self.conv58_1_1_19_(relu56_1_)
        batchnorm58_1_1_19_ = self.batchnorm58_1_1_19_(conv58_1_1_19_)
        relu58_1_1_19_ = self.relu58_1_1_19_(batchnorm58_1_1_19_)
        conv59_1_1_19_padding = self.conv59_1_1_19_padding(relu58_1_1_19_)
        conv59_1_1_19_ = self.conv59_1_1_19_(conv59_1_1_19_padding)
        batchnorm59_1_1_19_ = self.batchnorm59_1_1_19_(conv59_1_1_19_)
        relu59_1_1_19_ = self.relu59_1_1_19_(batchnorm59_1_1_19_)
        conv60_1_1_19_ = self.conv60_1_1_19_(relu59_1_1_19_)
        batchnorm60_1_1_19_ = self.batchnorm60_1_1_19_(conv60_1_1_19_)
        conv58_1_1_20_ = self.conv58_1_1_20_(relu56_1_)
        batchnorm58_1_1_20_ = self.batchnorm58_1_1_20_(conv58_1_1_20_)
        relu58_1_1_20_ = self.relu58_1_1_20_(batchnorm58_1_1_20_)
        conv59_1_1_20_padding = self.conv59_1_1_20_padding(relu58_1_1_20_)
        conv59_1_1_20_ = self.conv59_1_1_20_(conv59_1_1_20_padding)
        batchnorm59_1_1_20_ = self.batchnorm59_1_1_20_(conv59_1_1_20_)
        relu59_1_1_20_ = self.relu59_1_1_20_(batchnorm59_1_1_20_)
        conv60_1_1_20_ = self.conv60_1_1_20_(relu59_1_1_20_)
        batchnorm60_1_1_20_ = self.batchnorm60_1_1_20_(conv60_1_1_20_)
        conv58_1_1_21_ = self.conv58_1_1_21_(relu56_1_)
        batchnorm58_1_1_21_ = self.batchnorm58_1_1_21_(conv58_1_1_21_)
        relu58_1_1_21_ = self.relu58_1_1_21_(batchnorm58_1_1_21_)
        conv59_1_1_21_padding = self.conv59_1_1_21_padding(relu58_1_1_21_)
        conv59_1_1_21_ = self.conv59_1_1_21_(conv59_1_1_21_padding)
        batchnorm59_1_1_21_ = self.batchnorm59_1_1_21_(conv59_1_1_21_)
        relu59_1_1_21_ = self.relu59_1_1_21_(batchnorm59_1_1_21_)
        conv60_1_1_21_ = self.conv60_1_1_21_(relu59_1_1_21_)
        batchnorm60_1_1_21_ = self.batchnorm60_1_1_21_(conv60_1_1_21_)
        conv58_1_1_22_ = self.conv58_1_1_22_(relu56_1_)
        batchnorm58_1_1_22_ = self.batchnorm58_1_1_22_(conv58_1_1_22_)
        relu58_1_1_22_ = self.relu58_1_1_22_(batchnorm58_1_1_22_)
        conv59_1_1_22_padding = self.conv59_1_1_22_padding(relu58_1_1_22_)
        conv59_1_1_22_ = self.conv59_1_1_22_(conv59_1_1_22_padding)
        batchnorm59_1_1_22_ = self.batchnorm59_1_1_22_(conv59_1_1_22_)
        relu59_1_1_22_ = self.relu59_1_1_22_(batchnorm59_1_1_22_)
        conv60_1_1_22_ = self.conv60_1_1_22_(relu59_1_1_22_)
        batchnorm60_1_1_22_ = self.batchnorm60_1_1_22_(conv60_1_1_22_)
        conv58_1_1_23_ = self.conv58_1_1_23_(relu56_1_)
        batchnorm58_1_1_23_ = self.batchnorm58_1_1_23_(conv58_1_1_23_)
        relu58_1_1_23_ = self.relu58_1_1_23_(batchnorm58_1_1_23_)
        conv59_1_1_23_padding = self.conv59_1_1_23_padding(relu58_1_1_23_)
        conv59_1_1_23_ = self.conv59_1_1_23_(conv59_1_1_23_padding)
        batchnorm59_1_1_23_ = self.batchnorm59_1_1_23_(conv59_1_1_23_)
        relu59_1_1_23_ = self.relu59_1_1_23_(batchnorm59_1_1_23_)
        conv60_1_1_23_ = self.conv60_1_1_23_(relu59_1_1_23_)
        batchnorm60_1_1_23_ = self.batchnorm60_1_1_23_(conv60_1_1_23_)
        conv58_1_1_24_ = self.conv58_1_1_24_(relu56_1_)
        batchnorm58_1_1_24_ = self.batchnorm58_1_1_24_(conv58_1_1_24_)
        relu58_1_1_24_ = self.relu58_1_1_24_(batchnorm58_1_1_24_)
        conv59_1_1_24_padding = self.conv59_1_1_24_padding(relu58_1_1_24_)
        conv59_1_1_24_ = self.conv59_1_1_24_(conv59_1_1_24_padding)
        batchnorm59_1_1_24_ = self.batchnorm59_1_1_24_(conv59_1_1_24_)
        relu59_1_1_24_ = self.relu59_1_1_24_(batchnorm59_1_1_24_)
        conv60_1_1_24_ = self.conv60_1_1_24_(relu59_1_1_24_)
        batchnorm60_1_1_24_ = self.batchnorm60_1_1_24_(conv60_1_1_24_)
        conv58_1_1_25_ = self.conv58_1_1_25_(relu56_1_)
        batchnorm58_1_1_25_ = self.batchnorm58_1_1_25_(conv58_1_1_25_)
        relu58_1_1_25_ = self.relu58_1_1_25_(batchnorm58_1_1_25_)
        conv59_1_1_25_padding = self.conv59_1_1_25_padding(relu58_1_1_25_)
        conv59_1_1_25_ = self.conv59_1_1_25_(conv59_1_1_25_padding)
        batchnorm59_1_1_25_ = self.batchnorm59_1_1_25_(conv59_1_1_25_)
        relu59_1_1_25_ = self.relu59_1_1_25_(batchnorm59_1_1_25_)
        conv60_1_1_25_ = self.conv60_1_1_25_(relu59_1_1_25_)
        batchnorm60_1_1_25_ = self.batchnorm60_1_1_25_(conv60_1_1_25_)
        conv58_1_1_26_ = self.conv58_1_1_26_(relu56_1_)
        batchnorm58_1_1_26_ = self.batchnorm58_1_1_26_(conv58_1_1_26_)
        relu58_1_1_26_ = self.relu58_1_1_26_(batchnorm58_1_1_26_)
        conv59_1_1_26_padding = self.conv59_1_1_26_padding(relu58_1_1_26_)
        conv59_1_1_26_ = self.conv59_1_1_26_(conv59_1_1_26_padding)
        batchnorm59_1_1_26_ = self.batchnorm59_1_1_26_(conv59_1_1_26_)
        relu59_1_1_26_ = self.relu59_1_1_26_(batchnorm59_1_1_26_)
        conv60_1_1_26_ = self.conv60_1_1_26_(relu59_1_1_26_)
        batchnorm60_1_1_26_ = self.batchnorm60_1_1_26_(conv60_1_1_26_)
        conv58_1_1_27_ = self.conv58_1_1_27_(relu56_1_)
        batchnorm58_1_1_27_ = self.batchnorm58_1_1_27_(conv58_1_1_27_)
        relu58_1_1_27_ = self.relu58_1_1_27_(batchnorm58_1_1_27_)
        conv59_1_1_27_padding = self.conv59_1_1_27_padding(relu58_1_1_27_)
        conv59_1_1_27_ = self.conv59_1_1_27_(conv59_1_1_27_padding)
        batchnorm59_1_1_27_ = self.batchnorm59_1_1_27_(conv59_1_1_27_)
        relu59_1_1_27_ = self.relu59_1_1_27_(batchnorm59_1_1_27_)
        conv60_1_1_27_ = self.conv60_1_1_27_(relu59_1_1_27_)
        batchnorm60_1_1_27_ = self.batchnorm60_1_1_27_(conv60_1_1_27_)
        conv58_1_1_28_ = self.conv58_1_1_28_(relu56_1_)
        batchnorm58_1_1_28_ = self.batchnorm58_1_1_28_(conv58_1_1_28_)
        relu58_1_1_28_ = self.relu58_1_1_28_(batchnorm58_1_1_28_)
        conv59_1_1_28_padding = self.conv59_1_1_28_padding(relu58_1_1_28_)
        conv59_1_1_28_ = self.conv59_1_1_28_(conv59_1_1_28_padding)
        batchnorm59_1_1_28_ = self.batchnorm59_1_1_28_(conv59_1_1_28_)
        relu59_1_1_28_ = self.relu59_1_1_28_(batchnorm59_1_1_28_)
        conv60_1_1_28_ = self.conv60_1_1_28_(relu59_1_1_28_)
        batchnorm60_1_1_28_ = self.batchnorm60_1_1_28_(conv60_1_1_28_)
        conv58_1_1_29_ = self.conv58_1_1_29_(relu56_1_)
        batchnorm58_1_1_29_ = self.batchnorm58_1_1_29_(conv58_1_1_29_)
        relu58_1_1_29_ = self.relu58_1_1_29_(batchnorm58_1_1_29_)
        conv59_1_1_29_padding = self.conv59_1_1_29_padding(relu58_1_1_29_)
        conv59_1_1_29_ = self.conv59_1_1_29_(conv59_1_1_29_padding)
        batchnorm59_1_1_29_ = self.batchnorm59_1_1_29_(conv59_1_1_29_)
        relu59_1_1_29_ = self.relu59_1_1_29_(batchnorm59_1_1_29_)
        conv60_1_1_29_ = self.conv60_1_1_29_(relu59_1_1_29_)
        batchnorm60_1_1_29_ = self.batchnorm60_1_1_29_(conv60_1_1_29_)
        conv58_1_1_30_ = self.conv58_1_1_30_(relu56_1_)
        batchnorm58_1_1_30_ = self.batchnorm58_1_1_30_(conv58_1_1_30_)
        relu58_1_1_30_ = self.relu58_1_1_30_(batchnorm58_1_1_30_)
        conv59_1_1_30_padding = self.conv59_1_1_30_padding(relu58_1_1_30_)
        conv59_1_1_30_ = self.conv59_1_1_30_(conv59_1_1_30_padding)
        batchnorm59_1_1_30_ = self.batchnorm59_1_1_30_(conv59_1_1_30_)
        relu59_1_1_30_ = self.relu59_1_1_30_(batchnorm59_1_1_30_)
        conv60_1_1_30_ = self.conv60_1_1_30_(relu59_1_1_30_)
        batchnorm60_1_1_30_ = self.batchnorm60_1_1_30_(conv60_1_1_30_)
        conv58_1_1_31_ = self.conv58_1_1_31_(relu56_1_)
        batchnorm58_1_1_31_ = self.batchnorm58_1_1_31_(conv58_1_1_31_)
        relu58_1_1_31_ = self.relu58_1_1_31_(batchnorm58_1_1_31_)
        conv59_1_1_31_padding = self.conv59_1_1_31_padding(relu58_1_1_31_)
        conv59_1_1_31_ = self.conv59_1_1_31_(conv59_1_1_31_padding)
        batchnorm59_1_1_31_ = self.batchnorm59_1_1_31_(conv59_1_1_31_)
        relu59_1_1_31_ = self.relu59_1_1_31_(batchnorm59_1_1_31_)
        conv60_1_1_31_ = self.conv60_1_1_31_(relu59_1_1_31_)
        batchnorm60_1_1_31_ = self.batchnorm60_1_1_31_(conv60_1_1_31_)
        conv58_1_1_32_ = self.conv58_1_1_32_(relu56_1_)
        batchnorm58_1_1_32_ = self.batchnorm58_1_1_32_(conv58_1_1_32_)
        relu58_1_1_32_ = self.relu58_1_1_32_(batchnorm58_1_1_32_)
        conv59_1_1_32_padding = self.conv59_1_1_32_padding(relu58_1_1_32_)
        conv59_1_1_32_ = self.conv59_1_1_32_(conv59_1_1_32_padding)
        batchnorm59_1_1_32_ = self.batchnorm59_1_1_32_(conv59_1_1_32_)
        relu59_1_1_32_ = self.relu59_1_1_32_(batchnorm59_1_1_32_)
        conv60_1_1_32_ = self.conv60_1_1_32_(relu59_1_1_32_)
        batchnorm60_1_1_32_ = self.batchnorm60_1_1_32_(conv60_1_1_32_)
        add61_1_1_ = batchnorm60_1_1_1_ + batchnorm60_1_1_2_ + batchnorm60_1_1_3_ + batchnorm60_1_1_4_ + batchnorm60_1_1_5_ + batchnorm60_1_1_6_ + batchnorm60_1_1_7_ + batchnorm60_1_1_8_ + batchnorm60_1_1_9_ + batchnorm60_1_1_10_ + batchnorm60_1_1_11_ + batchnorm60_1_1_12_ + batchnorm60_1_1_13_ + batchnorm60_1_1_14_ + batchnorm60_1_1_15_ + batchnorm60_1_1_16_ + batchnorm60_1_1_17_ + batchnorm60_1_1_18_ + batchnorm60_1_1_19_ + batchnorm60_1_1_20_ + batchnorm60_1_1_21_ + batchnorm60_1_1_22_ + batchnorm60_1_1_23_ + batchnorm60_1_1_24_ + batchnorm60_1_1_25_ + batchnorm60_1_1_26_ + batchnorm60_1_1_27_ + batchnorm60_1_1_28_ + batchnorm60_1_1_29_ + batchnorm60_1_1_30_ + batchnorm60_1_1_31_ + batchnorm60_1_1_32_
        add62_1_ = add61_1_1_ + relu56_1_
        relu62_1_ = self.relu62_1_(add62_1_)
        conv64_1_1_1_ = self.conv64_1_1_1_(relu62_1_)
        batchnorm64_1_1_1_ = self.batchnorm64_1_1_1_(conv64_1_1_1_)
        relu64_1_1_1_ = self.relu64_1_1_1_(batchnorm64_1_1_1_)
        conv65_1_1_1_padding = self.conv65_1_1_1_padding(relu64_1_1_1_)
        conv65_1_1_1_ = self.conv65_1_1_1_(conv65_1_1_1_padding)
        batchnorm65_1_1_1_ = self.batchnorm65_1_1_1_(conv65_1_1_1_)
        relu65_1_1_1_ = self.relu65_1_1_1_(batchnorm65_1_1_1_)
        conv66_1_1_1_ = self.conv66_1_1_1_(relu65_1_1_1_)
        batchnorm66_1_1_1_ = self.batchnorm66_1_1_1_(conv66_1_1_1_)
        conv64_1_1_2_ = self.conv64_1_1_2_(relu62_1_)
        batchnorm64_1_1_2_ = self.batchnorm64_1_1_2_(conv64_1_1_2_)
        relu64_1_1_2_ = self.relu64_1_1_2_(batchnorm64_1_1_2_)
        conv65_1_1_2_padding = self.conv65_1_1_2_padding(relu64_1_1_2_)
        conv65_1_1_2_ = self.conv65_1_1_2_(conv65_1_1_2_padding)
        batchnorm65_1_1_2_ = self.batchnorm65_1_1_2_(conv65_1_1_2_)
        relu65_1_1_2_ = self.relu65_1_1_2_(batchnorm65_1_1_2_)
        conv66_1_1_2_ = self.conv66_1_1_2_(relu65_1_1_2_)
        batchnorm66_1_1_2_ = self.batchnorm66_1_1_2_(conv66_1_1_2_)
        conv64_1_1_3_ = self.conv64_1_1_3_(relu62_1_)
        batchnorm64_1_1_3_ = self.batchnorm64_1_1_3_(conv64_1_1_3_)
        relu64_1_1_3_ = self.relu64_1_1_3_(batchnorm64_1_1_3_)
        conv65_1_1_3_padding = self.conv65_1_1_3_padding(relu64_1_1_3_)
        conv65_1_1_3_ = self.conv65_1_1_3_(conv65_1_1_3_padding)
        batchnorm65_1_1_3_ = self.batchnorm65_1_1_3_(conv65_1_1_3_)
        relu65_1_1_3_ = self.relu65_1_1_3_(batchnorm65_1_1_3_)
        conv66_1_1_3_ = self.conv66_1_1_3_(relu65_1_1_3_)
        batchnorm66_1_1_3_ = self.batchnorm66_1_1_3_(conv66_1_1_3_)
        conv64_1_1_4_ = self.conv64_1_1_4_(relu62_1_)
        batchnorm64_1_1_4_ = self.batchnorm64_1_1_4_(conv64_1_1_4_)
        relu64_1_1_4_ = self.relu64_1_1_4_(batchnorm64_1_1_4_)
        conv65_1_1_4_padding = self.conv65_1_1_4_padding(relu64_1_1_4_)
        conv65_1_1_4_ = self.conv65_1_1_4_(conv65_1_1_4_padding)
        batchnorm65_1_1_4_ = self.batchnorm65_1_1_4_(conv65_1_1_4_)
        relu65_1_1_4_ = self.relu65_1_1_4_(batchnorm65_1_1_4_)
        conv66_1_1_4_ = self.conv66_1_1_4_(relu65_1_1_4_)
        batchnorm66_1_1_4_ = self.batchnorm66_1_1_4_(conv66_1_1_4_)
        conv64_1_1_5_ = self.conv64_1_1_5_(relu62_1_)
        batchnorm64_1_1_5_ = self.batchnorm64_1_1_5_(conv64_1_1_5_)
        relu64_1_1_5_ = self.relu64_1_1_5_(batchnorm64_1_1_5_)
        conv65_1_1_5_padding = self.conv65_1_1_5_padding(relu64_1_1_5_)
        conv65_1_1_5_ = self.conv65_1_1_5_(conv65_1_1_5_padding)
        batchnorm65_1_1_5_ = self.batchnorm65_1_1_5_(conv65_1_1_5_)
        relu65_1_1_5_ = self.relu65_1_1_5_(batchnorm65_1_1_5_)
        conv66_1_1_5_ = self.conv66_1_1_5_(relu65_1_1_5_)
        batchnorm66_1_1_5_ = self.batchnorm66_1_1_5_(conv66_1_1_5_)
        conv64_1_1_6_ = self.conv64_1_1_6_(relu62_1_)
        batchnorm64_1_1_6_ = self.batchnorm64_1_1_6_(conv64_1_1_6_)
        relu64_1_1_6_ = self.relu64_1_1_6_(batchnorm64_1_1_6_)
        conv65_1_1_6_padding = self.conv65_1_1_6_padding(relu64_1_1_6_)
        conv65_1_1_6_ = self.conv65_1_1_6_(conv65_1_1_6_padding)
        batchnorm65_1_1_6_ = self.batchnorm65_1_1_6_(conv65_1_1_6_)
        relu65_1_1_6_ = self.relu65_1_1_6_(batchnorm65_1_1_6_)
        conv66_1_1_6_ = self.conv66_1_1_6_(relu65_1_1_6_)
        batchnorm66_1_1_6_ = self.batchnorm66_1_1_6_(conv66_1_1_6_)
        conv64_1_1_7_ = self.conv64_1_1_7_(relu62_1_)
        batchnorm64_1_1_7_ = self.batchnorm64_1_1_7_(conv64_1_1_7_)
        relu64_1_1_7_ = self.relu64_1_1_7_(batchnorm64_1_1_7_)
        conv65_1_1_7_padding = self.conv65_1_1_7_padding(relu64_1_1_7_)
        conv65_1_1_7_ = self.conv65_1_1_7_(conv65_1_1_7_padding)
        batchnorm65_1_1_7_ = self.batchnorm65_1_1_7_(conv65_1_1_7_)
        relu65_1_1_7_ = self.relu65_1_1_7_(batchnorm65_1_1_7_)
        conv66_1_1_7_ = self.conv66_1_1_7_(relu65_1_1_7_)
        batchnorm66_1_1_7_ = self.batchnorm66_1_1_7_(conv66_1_1_7_)
        conv64_1_1_8_ = self.conv64_1_1_8_(relu62_1_)
        batchnorm64_1_1_8_ = self.batchnorm64_1_1_8_(conv64_1_1_8_)
        relu64_1_1_8_ = self.relu64_1_1_8_(batchnorm64_1_1_8_)
        conv65_1_1_8_padding = self.conv65_1_1_8_padding(relu64_1_1_8_)
        conv65_1_1_8_ = self.conv65_1_1_8_(conv65_1_1_8_padding)
        batchnorm65_1_1_8_ = self.batchnorm65_1_1_8_(conv65_1_1_8_)
        relu65_1_1_8_ = self.relu65_1_1_8_(batchnorm65_1_1_8_)
        conv66_1_1_8_ = self.conv66_1_1_8_(relu65_1_1_8_)
        batchnorm66_1_1_8_ = self.batchnorm66_1_1_8_(conv66_1_1_8_)
        conv64_1_1_9_ = self.conv64_1_1_9_(relu62_1_)
        batchnorm64_1_1_9_ = self.batchnorm64_1_1_9_(conv64_1_1_9_)
        relu64_1_1_9_ = self.relu64_1_1_9_(batchnorm64_1_1_9_)
        conv65_1_1_9_padding = self.conv65_1_1_9_padding(relu64_1_1_9_)
        conv65_1_1_9_ = self.conv65_1_1_9_(conv65_1_1_9_padding)
        batchnorm65_1_1_9_ = self.batchnorm65_1_1_9_(conv65_1_1_9_)
        relu65_1_1_9_ = self.relu65_1_1_9_(batchnorm65_1_1_9_)
        conv66_1_1_9_ = self.conv66_1_1_9_(relu65_1_1_9_)
        batchnorm66_1_1_9_ = self.batchnorm66_1_1_9_(conv66_1_1_9_)
        conv64_1_1_10_ = self.conv64_1_1_10_(relu62_1_)
        batchnorm64_1_1_10_ = self.batchnorm64_1_1_10_(conv64_1_1_10_)
        relu64_1_1_10_ = self.relu64_1_1_10_(batchnorm64_1_1_10_)
        conv65_1_1_10_padding = self.conv65_1_1_10_padding(relu64_1_1_10_)
        conv65_1_1_10_ = self.conv65_1_1_10_(conv65_1_1_10_padding)
        batchnorm65_1_1_10_ = self.batchnorm65_1_1_10_(conv65_1_1_10_)
        relu65_1_1_10_ = self.relu65_1_1_10_(batchnorm65_1_1_10_)
        conv66_1_1_10_ = self.conv66_1_1_10_(relu65_1_1_10_)
        batchnorm66_1_1_10_ = self.batchnorm66_1_1_10_(conv66_1_1_10_)
        conv64_1_1_11_ = self.conv64_1_1_11_(relu62_1_)
        batchnorm64_1_1_11_ = self.batchnorm64_1_1_11_(conv64_1_1_11_)
        relu64_1_1_11_ = self.relu64_1_1_11_(batchnorm64_1_1_11_)
        conv65_1_1_11_padding = self.conv65_1_1_11_padding(relu64_1_1_11_)
        conv65_1_1_11_ = self.conv65_1_1_11_(conv65_1_1_11_padding)
        batchnorm65_1_1_11_ = self.batchnorm65_1_1_11_(conv65_1_1_11_)
        relu65_1_1_11_ = self.relu65_1_1_11_(batchnorm65_1_1_11_)
        conv66_1_1_11_ = self.conv66_1_1_11_(relu65_1_1_11_)
        batchnorm66_1_1_11_ = self.batchnorm66_1_1_11_(conv66_1_1_11_)
        conv64_1_1_12_ = self.conv64_1_1_12_(relu62_1_)
        batchnorm64_1_1_12_ = self.batchnorm64_1_1_12_(conv64_1_1_12_)
        relu64_1_1_12_ = self.relu64_1_1_12_(batchnorm64_1_1_12_)
        conv65_1_1_12_padding = self.conv65_1_1_12_padding(relu64_1_1_12_)
        conv65_1_1_12_ = self.conv65_1_1_12_(conv65_1_1_12_padding)
        batchnorm65_1_1_12_ = self.batchnorm65_1_1_12_(conv65_1_1_12_)
        relu65_1_1_12_ = self.relu65_1_1_12_(batchnorm65_1_1_12_)
        conv66_1_1_12_ = self.conv66_1_1_12_(relu65_1_1_12_)
        batchnorm66_1_1_12_ = self.batchnorm66_1_1_12_(conv66_1_1_12_)
        conv64_1_1_13_ = self.conv64_1_1_13_(relu62_1_)
        batchnorm64_1_1_13_ = self.batchnorm64_1_1_13_(conv64_1_1_13_)
        relu64_1_1_13_ = self.relu64_1_1_13_(batchnorm64_1_1_13_)
        conv65_1_1_13_padding = self.conv65_1_1_13_padding(relu64_1_1_13_)
        conv65_1_1_13_ = self.conv65_1_1_13_(conv65_1_1_13_padding)
        batchnorm65_1_1_13_ = self.batchnorm65_1_1_13_(conv65_1_1_13_)
        relu65_1_1_13_ = self.relu65_1_1_13_(batchnorm65_1_1_13_)
        conv66_1_1_13_ = self.conv66_1_1_13_(relu65_1_1_13_)
        batchnorm66_1_1_13_ = self.batchnorm66_1_1_13_(conv66_1_1_13_)
        conv64_1_1_14_ = self.conv64_1_1_14_(relu62_1_)
        batchnorm64_1_1_14_ = self.batchnorm64_1_1_14_(conv64_1_1_14_)
        relu64_1_1_14_ = self.relu64_1_1_14_(batchnorm64_1_1_14_)
        conv65_1_1_14_padding = self.conv65_1_1_14_padding(relu64_1_1_14_)
        conv65_1_1_14_ = self.conv65_1_1_14_(conv65_1_1_14_padding)
        batchnorm65_1_1_14_ = self.batchnorm65_1_1_14_(conv65_1_1_14_)
        relu65_1_1_14_ = self.relu65_1_1_14_(batchnorm65_1_1_14_)
        conv66_1_1_14_ = self.conv66_1_1_14_(relu65_1_1_14_)
        batchnorm66_1_1_14_ = self.batchnorm66_1_1_14_(conv66_1_1_14_)
        conv64_1_1_15_ = self.conv64_1_1_15_(relu62_1_)
        batchnorm64_1_1_15_ = self.batchnorm64_1_1_15_(conv64_1_1_15_)
        relu64_1_1_15_ = self.relu64_1_1_15_(batchnorm64_1_1_15_)
        conv65_1_1_15_padding = self.conv65_1_1_15_padding(relu64_1_1_15_)
        conv65_1_1_15_ = self.conv65_1_1_15_(conv65_1_1_15_padding)
        batchnorm65_1_1_15_ = self.batchnorm65_1_1_15_(conv65_1_1_15_)
        relu65_1_1_15_ = self.relu65_1_1_15_(batchnorm65_1_1_15_)
        conv66_1_1_15_ = self.conv66_1_1_15_(relu65_1_1_15_)
        batchnorm66_1_1_15_ = self.batchnorm66_1_1_15_(conv66_1_1_15_)
        conv64_1_1_16_ = self.conv64_1_1_16_(relu62_1_)
        batchnorm64_1_1_16_ = self.batchnorm64_1_1_16_(conv64_1_1_16_)
        relu64_1_1_16_ = self.relu64_1_1_16_(batchnorm64_1_1_16_)
        conv65_1_1_16_padding = self.conv65_1_1_16_padding(relu64_1_1_16_)
        conv65_1_1_16_ = self.conv65_1_1_16_(conv65_1_1_16_padding)
        batchnorm65_1_1_16_ = self.batchnorm65_1_1_16_(conv65_1_1_16_)
        relu65_1_1_16_ = self.relu65_1_1_16_(batchnorm65_1_1_16_)
        conv66_1_1_16_ = self.conv66_1_1_16_(relu65_1_1_16_)
        batchnorm66_1_1_16_ = self.batchnorm66_1_1_16_(conv66_1_1_16_)
        conv64_1_1_17_ = self.conv64_1_1_17_(relu62_1_)
        batchnorm64_1_1_17_ = self.batchnorm64_1_1_17_(conv64_1_1_17_)
        relu64_1_1_17_ = self.relu64_1_1_17_(batchnorm64_1_1_17_)
        conv65_1_1_17_padding = self.conv65_1_1_17_padding(relu64_1_1_17_)
        conv65_1_1_17_ = self.conv65_1_1_17_(conv65_1_1_17_padding)
        batchnorm65_1_1_17_ = self.batchnorm65_1_1_17_(conv65_1_1_17_)
        relu65_1_1_17_ = self.relu65_1_1_17_(batchnorm65_1_1_17_)
        conv66_1_1_17_ = self.conv66_1_1_17_(relu65_1_1_17_)
        batchnorm66_1_1_17_ = self.batchnorm66_1_1_17_(conv66_1_1_17_)
        conv64_1_1_18_ = self.conv64_1_1_18_(relu62_1_)
        batchnorm64_1_1_18_ = self.batchnorm64_1_1_18_(conv64_1_1_18_)
        relu64_1_1_18_ = self.relu64_1_1_18_(batchnorm64_1_1_18_)
        conv65_1_1_18_padding = self.conv65_1_1_18_padding(relu64_1_1_18_)
        conv65_1_1_18_ = self.conv65_1_1_18_(conv65_1_1_18_padding)
        batchnorm65_1_1_18_ = self.batchnorm65_1_1_18_(conv65_1_1_18_)
        relu65_1_1_18_ = self.relu65_1_1_18_(batchnorm65_1_1_18_)
        conv66_1_1_18_ = self.conv66_1_1_18_(relu65_1_1_18_)
        batchnorm66_1_1_18_ = self.batchnorm66_1_1_18_(conv66_1_1_18_)
        conv64_1_1_19_ = self.conv64_1_1_19_(relu62_1_)
        batchnorm64_1_1_19_ = self.batchnorm64_1_1_19_(conv64_1_1_19_)
        relu64_1_1_19_ = self.relu64_1_1_19_(batchnorm64_1_1_19_)
        conv65_1_1_19_padding = self.conv65_1_1_19_padding(relu64_1_1_19_)
        conv65_1_1_19_ = self.conv65_1_1_19_(conv65_1_1_19_padding)
        batchnorm65_1_1_19_ = self.batchnorm65_1_1_19_(conv65_1_1_19_)
        relu65_1_1_19_ = self.relu65_1_1_19_(batchnorm65_1_1_19_)
        conv66_1_1_19_ = self.conv66_1_1_19_(relu65_1_1_19_)
        batchnorm66_1_1_19_ = self.batchnorm66_1_1_19_(conv66_1_1_19_)
        conv64_1_1_20_ = self.conv64_1_1_20_(relu62_1_)
        batchnorm64_1_1_20_ = self.batchnorm64_1_1_20_(conv64_1_1_20_)
        relu64_1_1_20_ = self.relu64_1_1_20_(batchnorm64_1_1_20_)
        conv65_1_1_20_padding = self.conv65_1_1_20_padding(relu64_1_1_20_)
        conv65_1_1_20_ = self.conv65_1_1_20_(conv65_1_1_20_padding)
        batchnorm65_1_1_20_ = self.batchnorm65_1_1_20_(conv65_1_1_20_)
        relu65_1_1_20_ = self.relu65_1_1_20_(batchnorm65_1_1_20_)
        conv66_1_1_20_ = self.conv66_1_1_20_(relu65_1_1_20_)
        batchnorm66_1_1_20_ = self.batchnorm66_1_1_20_(conv66_1_1_20_)
        conv64_1_1_21_ = self.conv64_1_1_21_(relu62_1_)
        batchnorm64_1_1_21_ = self.batchnorm64_1_1_21_(conv64_1_1_21_)
        relu64_1_1_21_ = self.relu64_1_1_21_(batchnorm64_1_1_21_)
        conv65_1_1_21_padding = self.conv65_1_1_21_padding(relu64_1_1_21_)
        conv65_1_1_21_ = self.conv65_1_1_21_(conv65_1_1_21_padding)
        batchnorm65_1_1_21_ = self.batchnorm65_1_1_21_(conv65_1_1_21_)
        relu65_1_1_21_ = self.relu65_1_1_21_(batchnorm65_1_1_21_)
        conv66_1_1_21_ = self.conv66_1_1_21_(relu65_1_1_21_)
        batchnorm66_1_1_21_ = self.batchnorm66_1_1_21_(conv66_1_1_21_)
        conv64_1_1_22_ = self.conv64_1_1_22_(relu62_1_)
        batchnorm64_1_1_22_ = self.batchnorm64_1_1_22_(conv64_1_1_22_)
        relu64_1_1_22_ = self.relu64_1_1_22_(batchnorm64_1_1_22_)
        conv65_1_1_22_padding = self.conv65_1_1_22_padding(relu64_1_1_22_)
        conv65_1_1_22_ = self.conv65_1_1_22_(conv65_1_1_22_padding)
        batchnorm65_1_1_22_ = self.batchnorm65_1_1_22_(conv65_1_1_22_)
        relu65_1_1_22_ = self.relu65_1_1_22_(batchnorm65_1_1_22_)
        conv66_1_1_22_ = self.conv66_1_1_22_(relu65_1_1_22_)
        batchnorm66_1_1_22_ = self.batchnorm66_1_1_22_(conv66_1_1_22_)
        conv64_1_1_23_ = self.conv64_1_1_23_(relu62_1_)
        batchnorm64_1_1_23_ = self.batchnorm64_1_1_23_(conv64_1_1_23_)
        relu64_1_1_23_ = self.relu64_1_1_23_(batchnorm64_1_1_23_)
        conv65_1_1_23_padding = self.conv65_1_1_23_padding(relu64_1_1_23_)
        conv65_1_1_23_ = self.conv65_1_1_23_(conv65_1_1_23_padding)
        batchnorm65_1_1_23_ = self.batchnorm65_1_1_23_(conv65_1_1_23_)
        relu65_1_1_23_ = self.relu65_1_1_23_(batchnorm65_1_1_23_)
        conv66_1_1_23_ = self.conv66_1_1_23_(relu65_1_1_23_)
        batchnorm66_1_1_23_ = self.batchnorm66_1_1_23_(conv66_1_1_23_)
        conv64_1_1_24_ = self.conv64_1_1_24_(relu62_1_)
        batchnorm64_1_1_24_ = self.batchnorm64_1_1_24_(conv64_1_1_24_)
        relu64_1_1_24_ = self.relu64_1_1_24_(batchnorm64_1_1_24_)
        conv65_1_1_24_padding = self.conv65_1_1_24_padding(relu64_1_1_24_)
        conv65_1_1_24_ = self.conv65_1_1_24_(conv65_1_1_24_padding)
        batchnorm65_1_1_24_ = self.batchnorm65_1_1_24_(conv65_1_1_24_)
        relu65_1_1_24_ = self.relu65_1_1_24_(batchnorm65_1_1_24_)
        conv66_1_1_24_ = self.conv66_1_1_24_(relu65_1_1_24_)
        batchnorm66_1_1_24_ = self.batchnorm66_1_1_24_(conv66_1_1_24_)
        conv64_1_1_25_ = self.conv64_1_1_25_(relu62_1_)
        batchnorm64_1_1_25_ = self.batchnorm64_1_1_25_(conv64_1_1_25_)
        relu64_1_1_25_ = self.relu64_1_1_25_(batchnorm64_1_1_25_)
        conv65_1_1_25_padding = self.conv65_1_1_25_padding(relu64_1_1_25_)
        conv65_1_1_25_ = self.conv65_1_1_25_(conv65_1_1_25_padding)
        batchnorm65_1_1_25_ = self.batchnorm65_1_1_25_(conv65_1_1_25_)
        relu65_1_1_25_ = self.relu65_1_1_25_(batchnorm65_1_1_25_)
        conv66_1_1_25_ = self.conv66_1_1_25_(relu65_1_1_25_)
        batchnorm66_1_1_25_ = self.batchnorm66_1_1_25_(conv66_1_1_25_)
        conv64_1_1_26_ = self.conv64_1_1_26_(relu62_1_)
        batchnorm64_1_1_26_ = self.batchnorm64_1_1_26_(conv64_1_1_26_)
        relu64_1_1_26_ = self.relu64_1_1_26_(batchnorm64_1_1_26_)
        conv65_1_1_26_padding = self.conv65_1_1_26_padding(relu64_1_1_26_)
        conv65_1_1_26_ = self.conv65_1_1_26_(conv65_1_1_26_padding)
        batchnorm65_1_1_26_ = self.batchnorm65_1_1_26_(conv65_1_1_26_)
        relu65_1_1_26_ = self.relu65_1_1_26_(batchnorm65_1_1_26_)
        conv66_1_1_26_ = self.conv66_1_1_26_(relu65_1_1_26_)
        batchnorm66_1_1_26_ = self.batchnorm66_1_1_26_(conv66_1_1_26_)
        conv64_1_1_27_ = self.conv64_1_1_27_(relu62_1_)
        batchnorm64_1_1_27_ = self.batchnorm64_1_1_27_(conv64_1_1_27_)
        relu64_1_1_27_ = self.relu64_1_1_27_(batchnorm64_1_1_27_)
        conv65_1_1_27_padding = self.conv65_1_1_27_padding(relu64_1_1_27_)
        conv65_1_1_27_ = self.conv65_1_1_27_(conv65_1_1_27_padding)
        batchnorm65_1_1_27_ = self.batchnorm65_1_1_27_(conv65_1_1_27_)
        relu65_1_1_27_ = self.relu65_1_1_27_(batchnorm65_1_1_27_)
        conv66_1_1_27_ = self.conv66_1_1_27_(relu65_1_1_27_)
        batchnorm66_1_1_27_ = self.batchnorm66_1_1_27_(conv66_1_1_27_)
        conv64_1_1_28_ = self.conv64_1_1_28_(relu62_1_)
        batchnorm64_1_1_28_ = self.batchnorm64_1_1_28_(conv64_1_1_28_)
        relu64_1_1_28_ = self.relu64_1_1_28_(batchnorm64_1_1_28_)
        conv65_1_1_28_padding = self.conv65_1_1_28_padding(relu64_1_1_28_)
        conv65_1_1_28_ = self.conv65_1_1_28_(conv65_1_1_28_padding)
        batchnorm65_1_1_28_ = self.batchnorm65_1_1_28_(conv65_1_1_28_)
        relu65_1_1_28_ = self.relu65_1_1_28_(batchnorm65_1_1_28_)
        conv66_1_1_28_ = self.conv66_1_1_28_(relu65_1_1_28_)
        batchnorm66_1_1_28_ = self.batchnorm66_1_1_28_(conv66_1_1_28_)
        conv64_1_1_29_ = self.conv64_1_1_29_(relu62_1_)
        batchnorm64_1_1_29_ = self.batchnorm64_1_1_29_(conv64_1_1_29_)
        relu64_1_1_29_ = self.relu64_1_1_29_(batchnorm64_1_1_29_)
        conv65_1_1_29_padding = self.conv65_1_1_29_padding(relu64_1_1_29_)
        conv65_1_1_29_ = self.conv65_1_1_29_(conv65_1_1_29_padding)
        batchnorm65_1_1_29_ = self.batchnorm65_1_1_29_(conv65_1_1_29_)
        relu65_1_1_29_ = self.relu65_1_1_29_(batchnorm65_1_1_29_)
        conv66_1_1_29_ = self.conv66_1_1_29_(relu65_1_1_29_)
        batchnorm66_1_1_29_ = self.batchnorm66_1_1_29_(conv66_1_1_29_)
        conv64_1_1_30_ = self.conv64_1_1_30_(relu62_1_)
        batchnorm64_1_1_30_ = self.batchnorm64_1_1_30_(conv64_1_1_30_)
        relu64_1_1_30_ = self.relu64_1_1_30_(batchnorm64_1_1_30_)
        conv65_1_1_30_padding = self.conv65_1_1_30_padding(relu64_1_1_30_)
        conv65_1_1_30_ = self.conv65_1_1_30_(conv65_1_1_30_padding)
        batchnorm65_1_1_30_ = self.batchnorm65_1_1_30_(conv65_1_1_30_)
        relu65_1_1_30_ = self.relu65_1_1_30_(batchnorm65_1_1_30_)
        conv66_1_1_30_ = self.conv66_1_1_30_(relu65_1_1_30_)
        batchnorm66_1_1_30_ = self.batchnorm66_1_1_30_(conv66_1_1_30_)
        conv64_1_1_31_ = self.conv64_1_1_31_(relu62_1_)
        batchnorm64_1_1_31_ = self.batchnorm64_1_1_31_(conv64_1_1_31_)
        relu64_1_1_31_ = self.relu64_1_1_31_(batchnorm64_1_1_31_)
        conv65_1_1_31_padding = self.conv65_1_1_31_padding(relu64_1_1_31_)
        conv65_1_1_31_ = self.conv65_1_1_31_(conv65_1_1_31_padding)
        batchnorm65_1_1_31_ = self.batchnorm65_1_1_31_(conv65_1_1_31_)
        relu65_1_1_31_ = self.relu65_1_1_31_(batchnorm65_1_1_31_)
        conv66_1_1_31_ = self.conv66_1_1_31_(relu65_1_1_31_)
        batchnorm66_1_1_31_ = self.batchnorm66_1_1_31_(conv66_1_1_31_)
        conv64_1_1_32_ = self.conv64_1_1_32_(relu62_1_)
        batchnorm64_1_1_32_ = self.batchnorm64_1_1_32_(conv64_1_1_32_)
        relu64_1_1_32_ = self.relu64_1_1_32_(batchnorm64_1_1_32_)
        conv65_1_1_32_padding = self.conv65_1_1_32_padding(relu64_1_1_32_)
        conv65_1_1_32_ = self.conv65_1_1_32_(conv65_1_1_32_padding)
        batchnorm65_1_1_32_ = self.batchnorm65_1_1_32_(conv65_1_1_32_)
        relu65_1_1_32_ = self.relu65_1_1_32_(batchnorm65_1_1_32_)
        conv66_1_1_32_ = self.conv66_1_1_32_(relu65_1_1_32_)
        batchnorm66_1_1_32_ = self.batchnorm66_1_1_32_(conv66_1_1_32_)
        add67_1_1_ = batchnorm66_1_1_1_ + batchnorm66_1_1_2_ + batchnorm66_1_1_3_ + batchnorm66_1_1_4_ + batchnorm66_1_1_5_ + batchnorm66_1_1_6_ + batchnorm66_1_1_7_ + batchnorm66_1_1_8_ + batchnorm66_1_1_9_ + batchnorm66_1_1_10_ + batchnorm66_1_1_11_ + batchnorm66_1_1_12_ + batchnorm66_1_1_13_ + batchnorm66_1_1_14_ + batchnorm66_1_1_15_ + batchnorm66_1_1_16_ + batchnorm66_1_1_17_ + batchnorm66_1_1_18_ + batchnorm66_1_1_19_ + batchnorm66_1_1_20_ + batchnorm66_1_1_21_ + batchnorm66_1_1_22_ + batchnorm66_1_1_23_ + batchnorm66_1_1_24_ + batchnorm66_1_1_25_ + batchnorm66_1_1_26_ + batchnorm66_1_1_27_ + batchnorm66_1_1_28_ + batchnorm66_1_1_29_ + batchnorm66_1_1_30_ + batchnorm66_1_1_31_ + batchnorm66_1_1_32_
        add68_1_ = add67_1_1_ + relu62_1_
        relu68_1_ = self.relu68_1_(add68_1_)
        conv70_1_1_1_ = self.conv70_1_1_1_(relu68_1_)
        batchnorm70_1_1_1_ = self.batchnorm70_1_1_1_(conv70_1_1_1_)
        relu70_1_1_1_ = self.relu70_1_1_1_(batchnorm70_1_1_1_)
        conv71_1_1_1_padding = self.conv71_1_1_1_padding(relu70_1_1_1_)
        conv71_1_1_1_ = self.conv71_1_1_1_(conv71_1_1_1_padding)
        batchnorm71_1_1_1_ = self.batchnorm71_1_1_1_(conv71_1_1_1_)
        relu71_1_1_1_ = self.relu71_1_1_1_(batchnorm71_1_1_1_)
        conv72_1_1_1_ = self.conv72_1_1_1_(relu71_1_1_1_)
        batchnorm72_1_1_1_ = self.batchnorm72_1_1_1_(conv72_1_1_1_)
        conv70_1_1_2_ = self.conv70_1_1_2_(relu68_1_)
        batchnorm70_1_1_2_ = self.batchnorm70_1_1_2_(conv70_1_1_2_)
        relu70_1_1_2_ = self.relu70_1_1_2_(batchnorm70_1_1_2_)
        conv71_1_1_2_padding = self.conv71_1_1_2_padding(relu70_1_1_2_)
        conv71_1_1_2_ = self.conv71_1_1_2_(conv71_1_1_2_padding)
        batchnorm71_1_1_2_ = self.batchnorm71_1_1_2_(conv71_1_1_2_)
        relu71_1_1_2_ = self.relu71_1_1_2_(batchnorm71_1_1_2_)
        conv72_1_1_2_ = self.conv72_1_1_2_(relu71_1_1_2_)
        batchnorm72_1_1_2_ = self.batchnorm72_1_1_2_(conv72_1_1_2_)
        conv70_1_1_3_ = self.conv70_1_1_3_(relu68_1_)
        batchnorm70_1_1_3_ = self.batchnorm70_1_1_3_(conv70_1_1_3_)
        relu70_1_1_3_ = self.relu70_1_1_3_(batchnorm70_1_1_3_)
        conv71_1_1_3_padding = self.conv71_1_1_3_padding(relu70_1_1_3_)
        conv71_1_1_3_ = self.conv71_1_1_3_(conv71_1_1_3_padding)
        batchnorm71_1_1_3_ = self.batchnorm71_1_1_3_(conv71_1_1_3_)
        relu71_1_1_3_ = self.relu71_1_1_3_(batchnorm71_1_1_3_)
        conv72_1_1_3_ = self.conv72_1_1_3_(relu71_1_1_3_)
        batchnorm72_1_1_3_ = self.batchnorm72_1_1_3_(conv72_1_1_3_)
        conv70_1_1_4_ = self.conv70_1_1_4_(relu68_1_)
        batchnorm70_1_1_4_ = self.batchnorm70_1_1_4_(conv70_1_1_4_)
        relu70_1_1_4_ = self.relu70_1_1_4_(batchnorm70_1_1_4_)
        conv71_1_1_4_padding = self.conv71_1_1_4_padding(relu70_1_1_4_)
        conv71_1_1_4_ = self.conv71_1_1_4_(conv71_1_1_4_padding)
        batchnorm71_1_1_4_ = self.batchnorm71_1_1_4_(conv71_1_1_4_)
        relu71_1_1_4_ = self.relu71_1_1_4_(batchnorm71_1_1_4_)
        conv72_1_1_4_ = self.conv72_1_1_4_(relu71_1_1_4_)
        batchnorm72_1_1_4_ = self.batchnorm72_1_1_4_(conv72_1_1_4_)
        conv70_1_1_5_ = self.conv70_1_1_5_(relu68_1_)
        batchnorm70_1_1_5_ = self.batchnorm70_1_1_5_(conv70_1_1_5_)
        relu70_1_1_5_ = self.relu70_1_1_5_(batchnorm70_1_1_5_)
        conv71_1_1_5_padding = self.conv71_1_1_5_padding(relu70_1_1_5_)
        conv71_1_1_5_ = self.conv71_1_1_5_(conv71_1_1_5_padding)
        batchnorm71_1_1_5_ = self.batchnorm71_1_1_5_(conv71_1_1_5_)
        relu71_1_1_5_ = self.relu71_1_1_5_(batchnorm71_1_1_5_)
        conv72_1_1_5_ = self.conv72_1_1_5_(relu71_1_1_5_)
        batchnorm72_1_1_5_ = self.batchnorm72_1_1_5_(conv72_1_1_5_)
        conv70_1_1_6_ = self.conv70_1_1_6_(relu68_1_)
        batchnorm70_1_1_6_ = self.batchnorm70_1_1_6_(conv70_1_1_6_)
        relu70_1_1_6_ = self.relu70_1_1_6_(batchnorm70_1_1_6_)
        conv71_1_1_6_padding = self.conv71_1_1_6_padding(relu70_1_1_6_)
        conv71_1_1_6_ = self.conv71_1_1_6_(conv71_1_1_6_padding)
        batchnorm71_1_1_6_ = self.batchnorm71_1_1_6_(conv71_1_1_6_)
        relu71_1_1_6_ = self.relu71_1_1_6_(batchnorm71_1_1_6_)
        conv72_1_1_6_ = self.conv72_1_1_6_(relu71_1_1_6_)
        batchnorm72_1_1_6_ = self.batchnorm72_1_1_6_(conv72_1_1_6_)
        conv70_1_1_7_ = self.conv70_1_1_7_(relu68_1_)
        batchnorm70_1_1_7_ = self.batchnorm70_1_1_7_(conv70_1_1_7_)
        relu70_1_1_7_ = self.relu70_1_1_7_(batchnorm70_1_1_7_)
        conv71_1_1_7_padding = self.conv71_1_1_7_padding(relu70_1_1_7_)
        conv71_1_1_7_ = self.conv71_1_1_7_(conv71_1_1_7_padding)
        batchnorm71_1_1_7_ = self.batchnorm71_1_1_7_(conv71_1_1_7_)
        relu71_1_1_7_ = self.relu71_1_1_7_(batchnorm71_1_1_7_)
        conv72_1_1_7_ = self.conv72_1_1_7_(relu71_1_1_7_)
        batchnorm72_1_1_7_ = self.batchnorm72_1_1_7_(conv72_1_1_7_)
        conv70_1_1_8_ = self.conv70_1_1_8_(relu68_1_)
        batchnorm70_1_1_8_ = self.batchnorm70_1_1_8_(conv70_1_1_8_)
        relu70_1_1_8_ = self.relu70_1_1_8_(batchnorm70_1_1_8_)
        conv71_1_1_8_padding = self.conv71_1_1_8_padding(relu70_1_1_8_)
        conv71_1_1_8_ = self.conv71_1_1_8_(conv71_1_1_8_padding)
        batchnorm71_1_1_8_ = self.batchnorm71_1_1_8_(conv71_1_1_8_)
        relu71_1_1_8_ = self.relu71_1_1_8_(batchnorm71_1_1_8_)
        conv72_1_1_8_ = self.conv72_1_1_8_(relu71_1_1_8_)
        batchnorm72_1_1_8_ = self.batchnorm72_1_1_8_(conv72_1_1_8_)
        conv70_1_1_9_ = self.conv70_1_1_9_(relu68_1_)
        batchnorm70_1_1_9_ = self.batchnorm70_1_1_9_(conv70_1_1_9_)
        relu70_1_1_9_ = self.relu70_1_1_9_(batchnorm70_1_1_9_)
        conv71_1_1_9_padding = self.conv71_1_1_9_padding(relu70_1_1_9_)
        conv71_1_1_9_ = self.conv71_1_1_9_(conv71_1_1_9_padding)
        batchnorm71_1_1_9_ = self.batchnorm71_1_1_9_(conv71_1_1_9_)
        relu71_1_1_9_ = self.relu71_1_1_9_(batchnorm71_1_1_9_)
        conv72_1_1_9_ = self.conv72_1_1_9_(relu71_1_1_9_)
        batchnorm72_1_1_9_ = self.batchnorm72_1_1_9_(conv72_1_1_9_)
        conv70_1_1_10_ = self.conv70_1_1_10_(relu68_1_)
        batchnorm70_1_1_10_ = self.batchnorm70_1_1_10_(conv70_1_1_10_)
        relu70_1_1_10_ = self.relu70_1_1_10_(batchnorm70_1_1_10_)
        conv71_1_1_10_padding = self.conv71_1_1_10_padding(relu70_1_1_10_)
        conv71_1_1_10_ = self.conv71_1_1_10_(conv71_1_1_10_padding)
        batchnorm71_1_1_10_ = self.batchnorm71_1_1_10_(conv71_1_1_10_)
        relu71_1_1_10_ = self.relu71_1_1_10_(batchnorm71_1_1_10_)
        conv72_1_1_10_ = self.conv72_1_1_10_(relu71_1_1_10_)
        batchnorm72_1_1_10_ = self.batchnorm72_1_1_10_(conv72_1_1_10_)
        conv70_1_1_11_ = self.conv70_1_1_11_(relu68_1_)
        batchnorm70_1_1_11_ = self.batchnorm70_1_1_11_(conv70_1_1_11_)
        relu70_1_1_11_ = self.relu70_1_1_11_(batchnorm70_1_1_11_)
        conv71_1_1_11_padding = self.conv71_1_1_11_padding(relu70_1_1_11_)
        conv71_1_1_11_ = self.conv71_1_1_11_(conv71_1_1_11_padding)
        batchnorm71_1_1_11_ = self.batchnorm71_1_1_11_(conv71_1_1_11_)
        relu71_1_1_11_ = self.relu71_1_1_11_(batchnorm71_1_1_11_)
        conv72_1_1_11_ = self.conv72_1_1_11_(relu71_1_1_11_)
        batchnorm72_1_1_11_ = self.batchnorm72_1_1_11_(conv72_1_1_11_)
        conv70_1_1_12_ = self.conv70_1_1_12_(relu68_1_)
        batchnorm70_1_1_12_ = self.batchnorm70_1_1_12_(conv70_1_1_12_)
        relu70_1_1_12_ = self.relu70_1_1_12_(batchnorm70_1_1_12_)
        conv71_1_1_12_padding = self.conv71_1_1_12_padding(relu70_1_1_12_)
        conv71_1_1_12_ = self.conv71_1_1_12_(conv71_1_1_12_padding)
        batchnorm71_1_1_12_ = self.batchnorm71_1_1_12_(conv71_1_1_12_)
        relu71_1_1_12_ = self.relu71_1_1_12_(batchnorm71_1_1_12_)
        conv72_1_1_12_ = self.conv72_1_1_12_(relu71_1_1_12_)
        batchnorm72_1_1_12_ = self.batchnorm72_1_1_12_(conv72_1_1_12_)
        conv70_1_1_13_ = self.conv70_1_1_13_(relu68_1_)
        batchnorm70_1_1_13_ = self.batchnorm70_1_1_13_(conv70_1_1_13_)
        relu70_1_1_13_ = self.relu70_1_1_13_(batchnorm70_1_1_13_)
        conv71_1_1_13_padding = self.conv71_1_1_13_padding(relu70_1_1_13_)
        conv71_1_1_13_ = self.conv71_1_1_13_(conv71_1_1_13_padding)
        batchnorm71_1_1_13_ = self.batchnorm71_1_1_13_(conv71_1_1_13_)
        relu71_1_1_13_ = self.relu71_1_1_13_(batchnorm71_1_1_13_)
        conv72_1_1_13_ = self.conv72_1_1_13_(relu71_1_1_13_)
        batchnorm72_1_1_13_ = self.batchnorm72_1_1_13_(conv72_1_1_13_)
        conv70_1_1_14_ = self.conv70_1_1_14_(relu68_1_)
        batchnorm70_1_1_14_ = self.batchnorm70_1_1_14_(conv70_1_1_14_)
        relu70_1_1_14_ = self.relu70_1_1_14_(batchnorm70_1_1_14_)
        conv71_1_1_14_padding = self.conv71_1_1_14_padding(relu70_1_1_14_)
        conv71_1_1_14_ = self.conv71_1_1_14_(conv71_1_1_14_padding)
        batchnorm71_1_1_14_ = self.batchnorm71_1_1_14_(conv71_1_1_14_)
        relu71_1_1_14_ = self.relu71_1_1_14_(batchnorm71_1_1_14_)
        conv72_1_1_14_ = self.conv72_1_1_14_(relu71_1_1_14_)
        batchnorm72_1_1_14_ = self.batchnorm72_1_1_14_(conv72_1_1_14_)
        conv70_1_1_15_ = self.conv70_1_1_15_(relu68_1_)
        batchnorm70_1_1_15_ = self.batchnorm70_1_1_15_(conv70_1_1_15_)
        relu70_1_1_15_ = self.relu70_1_1_15_(batchnorm70_1_1_15_)
        conv71_1_1_15_padding = self.conv71_1_1_15_padding(relu70_1_1_15_)
        conv71_1_1_15_ = self.conv71_1_1_15_(conv71_1_1_15_padding)
        batchnorm71_1_1_15_ = self.batchnorm71_1_1_15_(conv71_1_1_15_)
        relu71_1_1_15_ = self.relu71_1_1_15_(batchnorm71_1_1_15_)
        conv72_1_1_15_ = self.conv72_1_1_15_(relu71_1_1_15_)
        batchnorm72_1_1_15_ = self.batchnorm72_1_1_15_(conv72_1_1_15_)
        conv70_1_1_16_ = self.conv70_1_1_16_(relu68_1_)
        batchnorm70_1_1_16_ = self.batchnorm70_1_1_16_(conv70_1_1_16_)
        relu70_1_1_16_ = self.relu70_1_1_16_(batchnorm70_1_1_16_)
        conv71_1_1_16_padding = self.conv71_1_1_16_padding(relu70_1_1_16_)
        conv71_1_1_16_ = self.conv71_1_1_16_(conv71_1_1_16_padding)
        batchnorm71_1_1_16_ = self.batchnorm71_1_1_16_(conv71_1_1_16_)
        relu71_1_1_16_ = self.relu71_1_1_16_(batchnorm71_1_1_16_)
        conv72_1_1_16_ = self.conv72_1_1_16_(relu71_1_1_16_)
        batchnorm72_1_1_16_ = self.batchnorm72_1_1_16_(conv72_1_1_16_)
        conv70_1_1_17_ = self.conv70_1_1_17_(relu68_1_)
        batchnorm70_1_1_17_ = self.batchnorm70_1_1_17_(conv70_1_1_17_)
        relu70_1_1_17_ = self.relu70_1_1_17_(batchnorm70_1_1_17_)
        conv71_1_1_17_padding = self.conv71_1_1_17_padding(relu70_1_1_17_)
        conv71_1_1_17_ = self.conv71_1_1_17_(conv71_1_1_17_padding)
        batchnorm71_1_1_17_ = self.batchnorm71_1_1_17_(conv71_1_1_17_)
        relu71_1_1_17_ = self.relu71_1_1_17_(batchnorm71_1_1_17_)
        conv72_1_1_17_ = self.conv72_1_1_17_(relu71_1_1_17_)
        batchnorm72_1_1_17_ = self.batchnorm72_1_1_17_(conv72_1_1_17_)
        conv70_1_1_18_ = self.conv70_1_1_18_(relu68_1_)
        batchnorm70_1_1_18_ = self.batchnorm70_1_1_18_(conv70_1_1_18_)
        relu70_1_1_18_ = self.relu70_1_1_18_(batchnorm70_1_1_18_)
        conv71_1_1_18_padding = self.conv71_1_1_18_padding(relu70_1_1_18_)
        conv71_1_1_18_ = self.conv71_1_1_18_(conv71_1_1_18_padding)
        batchnorm71_1_1_18_ = self.batchnorm71_1_1_18_(conv71_1_1_18_)
        relu71_1_1_18_ = self.relu71_1_1_18_(batchnorm71_1_1_18_)
        conv72_1_1_18_ = self.conv72_1_1_18_(relu71_1_1_18_)
        batchnorm72_1_1_18_ = self.batchnorm72_1_1_18_(conv72_1_1_18_)
        conv70_1_1_19_ = self.conv70_1_1_19_(relu68_1_)
        batchnorm70_1_1_19_ = self.batchnorm70_1_1_19_(conv70_1_1_19_)
        relu70_1_1_19_ = self.relu70_1_1_19_(batchnorm70_1_1_19_)
        conv71_1_1_19_padding = self.conv71_1_1_19_padding(relu70_1_1_19_)
        conv71_1_1_19_ = self.conv71_1_1_19_(conv71_1_1_19_padding)
        batchnorm71_1_1_19_ = self.batchnorm71_1_1_19_(conv71_1_1_19_)
        relu71_1_1_19_ = self.relu71_1_1_19_(batchnorm71_1_1_19_)
        conv72_1_1_19_ = self.conv72_1_1_19_(relu71_1_1_19_)
        batchnorm72_1_1_19_ = self.batchnorm72_1_1_19_(conv72_1_1_19_)
        conv70_1_1_20_ = self.conv70_1_1_20_(relu68_1_)
        batchnorm70_1_1_20_ = self.batchnorm70_1_1_20_(conv70_1_1_20_)
        relu70_1_1_20_ = self.relu70_1_1_20_(batchnorm70_1_1_20_)
        conv71_1_1_20_padding = self.conv71_1_1_20_padding(relu70_1_1_20_)
        conv71_1_1_20_ = self.conv71_1_1_20_(conv71_1_1_20_padding)
        batchnorm71_1_1_20_ = self.batchnorm71_1_1_20_(conv71_1_1_20_)
        relu71_1_1_20_ = self.relu71_1_1_20_(batchnorm71_1_1_20_)
        conv72_1_1_20_ = self.conv72_1_1_20_(relu71_1_1_20_)
        batchnorm72_1_1_20_ = self.batchnorm72_1_1_20_(conv72_1_1_20_)
        conv70_1_1_21_ = self.conv70_1_1_21_(relu68_1_)
        batchnorm70_1_1_21_ = self.batchnorm70_1_1_21_(conv70_1_1_21_)
        relu70_1_1_21_ = self.relu70_1_1_21_(batchnorm70_1_1_21_)
        conv71_1_1_21_padding = self.conv71_1_1_21_padding(relu70_1_1_21_)
        conv71_1_1_21_ = self.conv71_1_1_21_(conv71_1_1_21_padding)
        batchnorm71_1_1_21_ = self.batchnorm71_1_1_21_(conv71_1_1_21_)
        relu71_1_1_21_ = self.relu71_1_1_21_(batchnorm71_1_1_21_)
        conv72_1_1_21_ = self.conv72_1_1_21_(relu71_1_1_21_)
        batchnorm72_1_1_21_ = self.batchnorm72_1_1_21_(conv72_1_1_21_)
        conv70_1_1_22_ = self.conv70_1_1_22_(relu68_1_)
        batchnorm70_1_1_22_ = self.batchnorm70_1_1_22_(conv70_1_1_22_)
        relu70_1_1_22_ = self.relu70_1_1_22_(batchnorm70_1_1_22_)
        conv71_1_1_22_padding = self.conv71_1_1_22_padding(relu70_1_1_22_)
        conv71_1_1_22_ = self.conv71_1_1_22_(conv71_1_1_22_padding)
        batchnorm71_1_1_22_ = self.batchnorm71_1_1_22_(conv71_1_1_22_)
        relu71_1_1_22_ = self.relu71_1_1_22_(batchnorm71_1_1_22_)
        conv72_1_1_22_ = self.conv72_1_1_22_(relu71_1_1_22_)
        batchnorm72_1_1_22_ = self.batchnorm72_1_1_22_(conv72_1_1_22_)
        conv70_1_1_23_ = self.conv70_1_1_23_(relu68_1_)
        batchnorm70_1_1_23_ = self.batchnorm70_1_1_23_(conv70_1_1_23_)
        relu70_1_1_23_ = self.relu70_1_1_23_(batchnorm70_1_1_23_)
        conv71_1_1_23_padding = self.conv71_1_1_23_padding(relu70_1_1_23_)
        conv71_1_1_23_ = self.conv71_1_1_23_(conv71_1_1_23_padding)
        batchnorm71_1_1_23_ = self.batchnorm71_1_1_23_(conv71_1_1_23_)
        relu71_1_1_23_ = self.relu71_1_1_23_(batchnorm71_1_1_23_)
        conv72_1_1_23_ = self.conv72_1_1_23_(relu71_1_1_23_)
        batchnorm72_1_1_23_ = self.batchnorm72_1_1_23_(conv72_1_1_23_)
        conv70_1_1_24_ = self.conv70_1_1_24_(relu68_1_)
        batchnorm70_1_1_24_ = self.batchnorm70_1_1_24_(conv70_1_1_24_)
        relu70_1_1_24_ = self.relu70_1_1_24_(batchnorm70_1_1_24_)
        conv71_1_1_24_padding = self.conv71_1_1_24_padding(relu70_1_1_24_)
        conv71_1_1_24_ = self.conv71_1_1_24_(conv71_1_1_24_padding)
        batchnorm71_1_1_24_ = self.batchnorm71_1_1_24_(conv71_1_1_24_)
        relu71_1_1_24_ = self.relu71_1_1_24_(batchnorm71_1_1_24_)
        conv72_1_1_24_ = self.conv72_1_1_24_(relu71_1_1_24_)
        batchnorm72_1_1_24_ = self.batchnorm72_1_1_24_(conv72_1_1_24_)
        conv70_1_1_25_ = self.conv70_1_1_25_(relu68_1_)
        batchnorm70_1_1_25_ = self.batchnorm70_1_1_25_(conv70_1_1_25_)
        relu70_1_1_25_ = self.relu70_1_1_25_(batchnorm70_1_1_25_)
        conv71_1_1_25_padding = self.conv71_1_1_25_padding(relu70_1_1_25_)
        conv71_1_1_25_ = self.conv71_1_1_25_(conv71_1_1_25_padding)
        batchnorm71_1_1_25_ = self.batchnorm71_1_1_25_(conv71_1_1_25_)
        relu71_1_1_25_ = self.relu71_1_1_25_(batchnorm71_1_1_25_)
        conv72_1_1_25_ = self.conv72_1_1_25_(relu71_1_1_25_)
        batchnorm72_1_1_25_ = self.batchnorm72_1_1_25_(conv72_1_1_25_)
        conv70_1_1_26_ = self.conv70_1_1_26_(relu68_1_)
        batchnorm70_1_1_26_ = self.batchnorm70_1_1_26_(conv70_1_1_26_)
        relu70_1_1_26_ = self.relu70_1_1_26_(batchnorm70_1_1_26_)
        conv71_1_1_26_padding = self.conv71_1_1_26_padding(relu70_1_1_26_)
        conv71_1_1_26_ = self.conv71_1_1_26_(conv71_1_1_26_padding)
        batchnorm71_1_1_26_ = self.batchnorm71_1_1_26_(conv71_1_1_26_)
        relu71_1_1_26_ = self.relu71_1_1_26_(batchnorm71_1_1_26_)
        conv72_1_1_26_ = self.conv72_1_1_26_(relu71_1_1_26_)
        batchnorm72_1_1_26_ = self.batchnorm72_1_1_26_(conv72_1_1_26_)
        conv70_1_1_27_ = self.conv70_1_1_27_(relu68_1_)
        batchnorm70_1_1_27_ = self.batchnorm70_1_1_27_(conv70_1_1_27_)
        relu70_1_1_27_ = self.relu70_1_1_27_(batchnorm70_1_1_27_)
        conv71_1_1_27_padding = self.conv71_1_1_27_padding(relu70_1_1_27_)
        conv71_1_1_27_ = self.conv71_1_1_27_(conv71_1_1_27_padding)
        batchnorm71_1_1_27_ = self.batchnorm71_1_1_27_(conv71_1_1_27_)
        relu71_1_1_27_ = self.relu71_1_1_27_(batchnorm71_1_1_27_)
        conv72_1_1_27_ = self.conv72_1_1_27_(relu71_1_1_27_)
        batchnorm72_1_1_27_ = self.batchnorm72_1_1_27_(conv72_1_1_27_)
        conv70_1_1_28_ = self.conv70_1_1_28_(relu68_1_)
        batchnorm70_1_1_28_ = self.batchnorm70_1_1_28_(conv70_1_1_28_)
        relu70_1_1_28_ = self.relu70_1_1_28_(batchnorm70_1_1_28_)
        conv71_1_1_28_padding = self.conv71_1_1_28_padding(relu70_1_1_28_)
        conv71_1_1_28_ = self.conv71_1_1_28_(conv71_1_1_28_padding)
        batchnorm71_1_1_28_ = self.batchnorm71_1_1_28_(conv71_1_1_28_)
        relu71_1_1_28_ = self.relu71_1_1_28_(batchnorm71_1_1_28_)
        conv72_1_1_28_ = self.conv72_1_1_28_(relu71_1_1_28_)
        batchnorm72_1_1_28_ = self.batchnorm72_1_1_28_(conv72_1_1_28_)
        conv70_1_1_29_ = self.conv70_1_1_29_(relu68_1_)
        batchnorm70_1_1_29_ = self.batchnorm70_1_1_29_(conv70_1_1_29_)
        relu70_1_1_29_ = self.relu70_1_1_29_(batchnorm70_1_1_29_)
        conv71_1_1_29_padding = self.conv71_1_1_29_padding(relu70_1_1_29_)
        conv71_1_1_29_ = self.conv71_1_1_29_(conv71_1_1_29_padding)
        batchnorm71_1_1_29_ = self.batchnorm71_1_1_29_(conv71_1_1_29_)
        relu71_1_1_29_ = self.relu71_1_1_29_(batchnorm71_1_1_29_)
        conv72_1_1_29_ = self.conv72_1_1_29_(relu71_1_1_29_)
        batchnorm72_1_1_29_ = self.batchnorm72_1_1_29_(conv72_1_1_29_)
        conv70_1_1_30_ = self.conv70_1_1_30_(relu68_1_)
        batchnorm70_1_1_30_ = self.batchnorm70_1_1_30_(conv70_1_1_30_)
        relu70_1_1_30_ = self.relu70_1_1_30_(batchnorm70_1_1_30_)
        conv71_1_1_30_padding = self.conv71_1_1_30_padding(relu70_1_1_30_)
        conv71_1_1_30_ = self.conv71_1_1_30_(conv71_1_1_30_padding)
        batchnorm71_1_1_30_ = self.batchnorm71_1_1_30_(conv71_1_1_30_)
        relu71_1_1_30_ = self.relu71_1_1_30_(batchnorm71_1_1_30_)
        conv72_1_1_30_ = self.conv72_1_1_30_(relu71_1_1_30_)
        batchnorm72_1_1_30_ = self.batchnorm72_1_1_30_(conv72_1_1_30_)
        conv70_1_1_31_ = self.conv70_1_1_31_(relu68_1_)
        batchnorm70_1_1_31_ = self.batchnorm70_1_1_31_(conv70_1_1_31_)
        relu70_1_1_31_ = self.relu70_1_1_31_(batchnorm70_1_1_31_)
        conv71_1_1_31_padding = self.conv71_1_1_31_padding(relu70_1_1_31_)
        conv71_1_1_31_ = self.conv71_1_1_31_(conv71_1_1_31_padding)
        batchnorm71_1_1_31_ = self.batchnorm71_1_1_31_(conv71_1_1_31_)
        relu71_1_1_31_ = self.relu71_1_1_31_(batchnorm71_1_1_31_)
        conv72_1_1_31_ = self.conv72_1_1_31_(relu71_1_1_31_)
        batchnorm72_1_1_31_ = self.batchnorm72_1_1_31_(conv72_1_1_31_)
        conv70_1_1_32_ = self.conv70_1_1_32_(relu68_1_)
        batchnorm70_1_1_32_ = self.batchnorm70_1_1_32_(conv70_1_1_32_)
        relu70_1_1_32_ = self.relu70_1_1_32_(batchnorm70_1_1_32_)
        conv71_1_1_32_padding = self.conv71_1_1_32_padding(relu70_1_1_32_)
        conv71_1_1_32_ = self.conv71_1_1_32_(conv71_1_1_32_padding)
        batchnorm71_1_1_32_ = self.batchnorm71_1_1_32_(conv71_1_1_32_)
        relu71_1_1_32_ = self.relu71_1_1_32_(batchnorm71_1_1_32_)
        conv72_1_1_32_ = self.conv72_1_1_32_(relu71_1_1_32_)
        batchnorm72_1_1_32_ = self.batchnorm72_1_1_32_(conv72_1_1_32_)
        add73_1_1_ = batchnorm72_1_1_1_ + batchnorm72_1_1_2_ + batchnorm72_1_1_3_ + batchnorm72_1_1_4_ + batchnorm72_1_1_5_ + batchnorm72_1_1_6_ + batchnorm72_1_1_7_ + batchnorm72_1_1_8_ + batchnorm72_1_1_9_ + batchnorm72_1_1_10_ + batchnorm72_1_1_11_ + batchnorm72_1_1_12_ + batchnorm72_1_1_13_ + batchnorm72_1_1_14_ + batchnorm72_1_1_15_ + batchnorm72_1_1_16_ + batchnorm72_1_1_17_ + batchnorm72_1_1_18_ + batchnorm72_1_1_19_ + batchnorm72_1_1_20_ + batchnorm72_1_1_21_ + batchnorm72_1_1_22_ + batchnorm72_1_1_23_ + batchnorm72_1_1_24_ + batchnorm72_1_1_25_ + batchnorm72_1_1_26_ + batchnorm72_1_1_27_ + batchnorm72_1_1_28_ + batchnorm72_1_1_29_ + batchnorm72_1_1_30_ + batchnorm72_1_1_31_ + batchnorm72_1_1_32_
        add74_1_ = add73_1_1_ + relu68_1_
        relu74_1_ = self.relu74_1_(add74_1_)
        conv76_1_1_1_ = self.conv76_1_1_1_(relu74_1_)
        batchnorm76_1_1_1_ = self.batchnorm76_1_1_1_(conv76_1_1_1_)
        relu76_1_1_1_ = self.relu76_1_1_1_(batchnorm76_1_1_1_)
        conv77_1_1_1_padding = self.conv77_1_1_1_padding(relu76_1_1_1_)
        conv77_1_1_1_ = self.conv77_1_1_1_(conv77_1_1_1_padding)
        batchnorm77_1_1_1_ = self.batchnorm77_1_1_1_(conv77_1_1_1_)
        relu77_1_1_1_ = self.relu77_1_1_1_(batchnorm77_1_1_1_)
        conv78_1_1_1_ = self.conv78_1_1_1_(relu77_1_1_1_)
        batchnorm78_1_1_1_ = self.batchnorm78_1_1_1_(conv78_1_1_1_)
        conv76_1_1_2_ = self.conv76_1_1_2_(relu74_1_)
        batchnorm76_1_1_2_ = self.batchnorm76_1_1_2_(conv76_1_1_2_)
        relu76_1_1_2_ = self.relu76_1_1_2_(batchnorm76_1_1_2_)
        conv77_1_1_2_padding = self.conv77_1_1_2_padding(relu76_1_1_2_)
        conv77_1_1_2_ = self.conv77_1_1_2_(conv77_1_1_2_padding)
        batchnorm77_1_1_2_ = self.batchnorm77_1_1_2_(conv77_1_1_2_)
        relu77_1_1_2_ = self.relu77_1_1_2_(batchnorm77_1_1_2_)
        conv78_1_1_2_ = self.conv78_1_1_2_(relu77_1_1_2_)
        batchnorm78_1_1_2_ = self.batchnorm78_1_1_2_(conv78_1_1_2_)
        conv76_1_1_3_ = self.conv76_1_1_3_(relu74_1_)
        batchnorm76_1_1_3_ = self.batchnorm76_1_1_3_(conv76_1_1_3_)
        relu76_1_1_3_ = self.relu76_1_1_3_(batchnorm76_1_1_3_)
        conv77_1_1_3_padding = self.conv77_1_1_3_padding(relu76_1_1_3_)
        conv77_1_1_3_ = self.conv77_1_1_3_(conv77_1_1_3_padding)
        batchnorm77_1_1_3_ = self.batchnorm77_1_1_3_(conv77_1_1_3_)
        relu77_1_1_3_ = self.relu77_1_1_3_(batchnorm77_1_1_3_)
        conv78_1_1_3_ = self.conv78_1_1_3_(relu77_1_1_3_)
        batchnorm78_1_1_3_ = self.batchnorm78_1_1_3_(conv78_1_1_3_)
        conv76_1_1_4_ = self.conv76_1_1_4_(relu74_1_)
        batchnorm76_1_1_4_ = self.batchnorm76_1_1_4_(conv76_1_1_4_)
        relu76_1_1_4_ = self.relu76_1_1_4_(batchnorm76_1_1_4_)
        conv77_1_1_4_padding = self.conv77_1_1_4_padding(relu76_1_1_4_)
        conv77_1_1_4_ = self.conv77_1_1_4_(conv77_1_1_4_padding)
        batchnorm77_1_1_4_ = self.batchnorm77_1_1_4_(conv77_1_1_4_)
        relu77_1_1_4_ = self.relu77_1_1_4_(batchnorm77_1_1_4_)
        conv78_1_1_4_ = self.conv78_1_1_4_(relu77_1_1_4_)
        batchnorm78_1_1_4_ = self.batchnorm78_1_1_4_(conv78_1_1_4_)
        conv76_1_1_5_ = self.conv76_1_1_5_(relu74_1_)
        batchnorm76_1_1_5_ = self.batchnorm76_1_1_5_(conv76_1_1_5_)
        relu76_1_1_5_ = self.relu76_1_1_5_(batchnorm76_1_1_5_)
        conv77_1_1_5_padding = self.conv77_1_1_5_padding(relu76_1_1_5_)
        conv77_1_1_5_ = self.conv77_1_1_5_(conv77_1_1_5_padding)
        batchnorm77_1_1_5_ = self.batchnorm77_1_1_5_(conv77_1_1_5_)
        relu77_1_1_5_ = self.relu77_1_1_5_(batchnorm77_1_1_5_)
        conv78_1_1_5_ = self.conv78_1_1_5_(relu77_1_1_5_)
        batchnorm78_1_1_5_ = self.batchnorm78_1_1_5_(conv78_1_1_5_)
        conv76_1_1_6_ = self.conv76_1_1_6_(relu74_1_)
        batchnorm76_1_1_6_ = self.batchnorm76_1_1_6_(conv76_1_1_6_)
        relu76_1_1_6_ = self.relu76_1_1_6_(batchnorm76_1_1_6_)
        conv77_1_1_6_padding = self.conv77_1_1_6_padding(relu76_1_1_6_)
        conv77_1_1_6_ = self.conv77_1_1_6_(conv77_1_1_6_padding)
        batchnorm77_1_1_6_ = self.batchnorm77_1_1_6_(conv77_1_1_6_)
        relu77_1_1_6_ = self.relu77_1_1_6_(batchnorm77_1_1_6_)
        conv78_1_1_6_ = self.conv78_1_1_6_(relu77_1_1_6_)
        batchnorm78_1_1_6_ = self.batchnorm78_1_1_6_(conv78_1_1_6_)
        conv76_1_1_7_ = self.conv76_1_1_7_(relu74_1_)
        batchnorm76_1_1_7_ = self.batchnorm76_1_1_7_(conv76_1_1_7_)
        relu76_1_1_7_ = self.relu76_1_1_7_(batchnorm76_1_1_7_)
        conv77_1_1_7_padding = self.conv77_1_1_7_padding(relu76_1_1_7_)
        conv77_1_1_7_ = self.conv77_1_1_7_(conv77_1_1_7_padding)
        batchnorm77_1_1_7_ = self.batchnorm77_1_1_7_(conv77_1_1_7_)
        relu77_1_1_7_ = self.relu77_1_1_7_(batchnorm77_1_1_7_)
        conv78_1_1_7_ = self.conv78_1_1_7_(relu77_1_1_7_)
        batchnorm78_1_1_7_ = self.batchnorm78_1_1_7_(conv78_1_1_7_)
        conv76_1_1_8_ = self.conv76_1_1_8_(relu74_1_)
        batchnorm76_1_1_8_ = self.batchnorm76_1_1_8_(conv76_1_1_8_)
        relu76_1_1_8_ = self.relu76_1_1_8_(batchnorm76_1_1_8_)
        conv77_1_1_8_padding = self.conv77_1_1_8_padding(relu76_1_1_8_)
        conv77_1_1_8_ = self.conv77_1_1_8_(conv77_1_1_8_padding)
        batchnorm77_1_1_8_ = self.batchnorm77_1_1_8_(conv77_1_1_8_)
        relu77_1_1_8_ = self.relu77_1_1_8_(batchnorm77_1_1_8_)
        conv78_1_1_8_ = self.conv78_1_1_8_(relu77_1_1_8_)
        batchnorm78_1_1_8_ = self.batchnorm78_1_1_8_(conv78_1_1_8_)
        conv76_1_1_9_ = self.conv76_1_1_9_(relu74_1_)
        batchnorm76_1_1_9_ = self.batchnorm76_1_1_9_(conv76_1_1_9_)
        relu76_1_1_9_ = self.relu76_1_1_9_(batchnorm76_1_1_9_)
        conv77_1_1_9_padding = self.conv77_1_1_9_padding(relu76_1_1_9_)
        conv77_1_1_9_ = self.conv77_1_1_9_(conv77_1_1_9_padding)
        batchnorm77_1_1_9_ = self.batchnorm77_1_1_9_(conv77_1_1_9_)
        relu77_1_1_9_ = self.relu77_1_1_9_(batchnorm77_1_1_9_)
        conv78_1_1_9_ = self.conv78_1_1_9_(relu77_1_1_9_)
        batchnorm78_1_1_9_ = self.batchnorm78_1_1_9_(conv78_1_1_9_)
        conv76_1_1_10_ = self.conv76_1_1_10_(relu74_1_)
        batchnorm76_1_1_10_ = self.batchnorm76_1_1_10_(conv76_1_1_10_)
        relu76_1_1_10_ = self.relu76_1_1_10_(batchnorm76_1_1_10_)
        conv77_1_1_10_padding = self.conv77_1_1_10_padding(relu76_1_1_10_)
        conv77_1_1_10_ = self.conv77_1_1_10_(conv77_1_1_10_padding)
        batchnorm77_1_1_10_ = self.batchnorm77_1_1_10_(conv77_1_1_10_)
        relu77_1_1_10_ = self.relu77_1_1_10_(batchnorm77_1_1_10_)
        conv78_1_1_10_ = self.conv78_1_1_10_(relu77_1_1_10_)
        batchnorm78_1_1_10_ = self.batchnorm78_1_1_10_(conv78_1_1_10_)
        conv76_1_1_11_ = self.conv76_1_1_11_(relu74_1_)
        batchnorm76_1_1_11_ = self.batchnorm76_1_1_11_(conv76_1_1_11_)
        relu76_1_1_11_ = self.relu76_1_1_11_(batchnorm76_1_1_11_)
        conv77_1_1_11_padding = self.conv77_1_1_11_padding(relu76_1_1_11_)
        conv77_1_1_11_ = self.conv77_1_1_11_(conv77_1_1_11_padding)
        batchnorm77_1_1_11_ = self.batchnorm77_1_1_11_(conv77_1_1_11_)
        relu77_1_1_11_ = self.relu77_1_1_11_(batchnorm77_1_1_11_)
        conv78_1_1_11_ = self.conv78_1_1_11_(relu77_1_1_11_)
        batchnorm78_1_1_11_ = self.batchnorm78_1_1_11_(conv78_1_1_11_)
        conv76_1_1_12_ = self.conv76_1_1_12_(relu74_1_)
        batchnorm76_1_1_12_ = self.batchnorm76_1_1_12_(conv76_1_1_12_)
        relu76_1_1_12_ = self.relu76_1_1_12_(batchnorm76_1_1_12_)
        conv77_1_1_12_padding = self.conv77_1_1_12_padding(relu76_1_1_12_)
        conv77_1_1_12_ = self.conv77_1_1_12_(conv77_1_1_12_padding)
        batchnorm77_1_1_12_ = self.batchnorm77_1_1_12_(conv77_1_1_12_)
        relu77_1_1_12_ = self.relu77_1_1_12_(batchnorm77_1_1_12_)
        conv78_1_1_12_ = self.conv78_1_1_12_(relu77_1_1_12_)
        batchnorm78_1_1_12_ = self.batchnorm78_1_1_12_(conv78_1_1_12_)
        conv76_1_1_13_ = self.conv76_1_1_13_(relu74_1_)
        batchnorm76_1_1_13_ = self.batchnorm76_1_1_13_(conv76_1_1_13_)
        relu76_1_1_13_ = self.relu76_1_1_13_(batchnorm76_1_1_13_)
        conv77_1_1_13_padding = self.conv77_1_1_13_padding(relu76_1_1_13_)
        conv77_1_1_13_ = self.conv77_1_1_13_(conv77_1_1_13_padding)
        batchnorm77_1_1_13_ = self.batchnorm77_1_1_13_(conv77_1_1_13_)
        relu77_1_1_13_ = self.relu77_1_1_13_(batchnorm77_1_1_13_)
        conv78_1_1_13_ = self.conv78_1_1_13_(relu77_1_1_13_)
        batchnorm78_1_1_13_ = self.batchnorm78_1_1_13_(conv78_1_1_13_)
        conv76_1_1_14_ = self.conv76_1_1_14_(relu74_1_)
        batchnorm76_1_1_14_ = self.batchnorm76_1_1_14_(conv76_1_1_14_)
        relu76_1_1_14_ = self.relu76_1_1_14_(batchnorm76_1_1_14_)
        conv77_1_1_14_padding = self.conv77_1_1_14_padding(relu76_1_1_14_)
        conv77_1_1_14_ = self.conv77_1_1_14_(conv77_1_1_14_padding)
        batchnorm77_1_1_14_ = self.batchnorm77_1_1_14_(conv77_1_1_14_)
        relu77_1_1_14_ = self.relu77_1_1_14_(batchnorm77_1_1_14_)
        conv78_1_1_14_ = self.conv78_1_1_14_(relu77_1_1_14_)
        batchnorm78_1_1_14_ = self.batchnorm78_1_1_14_(conv78_1_1_14_)
        conv76_1_1_15_ = self.conv76_1_1_15_(relu74_1_)
        batchnorm76_1_1_15_ = self.batchnorm76_1_1_15_(conv76_1_1_15_)
        relu76_1_1_15_ = self.relu76_1_1_15_(batchnorm76_1_1_15_)
        conv77_1_1_15_padding = self.conv77_1_1_15_padding(relu76_1_1_15_)
        conv77_1_1_15_ = self.conv77_1_1_15_(conv77_1_1_15_padding)
        batchnorm77_1_1_15_ = self.batchnorm77_1_1_15_(conv77_1_1_15_)
        relu77_1_1_15_ = self.relu77_1_1_15_(batchnorm77_1_1_15_)
        conv78_1_1_15_ = self.conv78_1_1_15_(relu77_1_1_15_)
        batchnorm78_1_1_15_ = self.batchnorm78_1_1_15_(conv78_1_1_15_)
        conv76_1_1_16_ = self.conv76_1_1_16_(relu74_1_)
        batchnorm76_1_1_16_ = self.batchnorm76_1_1_16_(conv76_1_1_16_)
        relu76_1_1_16_ = self.relu76_1_1_16_(batchnorm76_1_1_16_)
        conv77_1_1_16_padding = self.conv77_1_1_16_padding(relu76_1_1_16_)
        conv77_1_1_16_ = self.conv77_1_1_16_(conv77_1_1_16_padding)
        batchnorm77_1_1_16_ = self.batchnorm77_1_1_16_(conv77_1_1_16_)
        relu77_1_1_16_ = self.relu77_1_1_16_(batchnorm77_1_1_16_)
        conv78_1_1_16_ = self.conv78_1_1_16_(relu77_1_1_16_)
        batchnorm78_1_1_16_ = self.batchnorm78_1_1_16_(conv78_1_1_16_)
        conv76_1_1_17_ = self.conv76_1_1_17_(relu74_1_)
        batchnorm76_1_1_17_ = self.batchnorm76_1_1_17_(conv76_1_1_17_)
        relu76_1_1_17_ = self.relu76_1_1_17_(batchnorm76_1_1_17_)
        conv77_1_1_17_padding = self.conv77_1_1_17_padding(relu76_1_1_17_)
        conv77_1_1_17_ = self.conv77_1_1_17_(conv77_1_1_17_padding)
        batchnorm77_1_1_17_ = self.batchnorm77_1_1_17_(conv77_1_1_17_)
        relu77_1_1_17_ = self.relu77_1_1_17_(batchnorm77_1_1_17_)
        conv78_1_1_17_ = self.conv78_1_1_17_(relu77_1_1_17_)
        batchnorm78_1_1_17_ = self.batchnorm78_1_1_17_(conv78_1_1_17_)
        conv76_1_1_18_ = self.conv76_1_1_18_(relu74_1_)
        batchnorm76_1_1_18_ = self.batchnorm76_1_1_18_(conv76_1_1_18_)
        relu76_1_1_18_ = self.relu76_1_1_18_(batchnorm76_1_1_18_)
        conv77_1_1_18_padding = self.conv77_1_1_18_padding(relu76_1_1_18_)
        conv77_1_1_18_ = self.conv77_1_1_18_(conv77_1_1_18_padding)
        batchnorm77_1_1_18_ = self.batchnorm77_1_1_18_(conv77_1_1_18_)
        relu77_1_1_18_ = self.relu77_1_1_18_(batchnorm77_1_1_18_)
        conv78_1_1_18_ = self.conv78_1_1_18_(relu77_1_1_18_)
        batchnorm78_1_1_18_ = self.batchnorm78_1_1_18_(conv78_1_1_18_)
        conv76_1_1_19_ = self.conv76_1_1_19_(relu74_1_)
        batchnorm76_1_1_19_ = self.batchnorm76_1_1_19_(conv76_1_1_19_)
        relu76_1_1_19_ = self.relu76_1_1_19_(batchnorm76_1_1_19_)
        conv77_1_1_19_padding = self.conv77_1_1_19_padding(relu76_1_1_19_)
        conv77_1_1_19_ = self.conv77_1_1_19_(conv77_1_1_19_padding)
        batchnorm77_1_1_19_ = self.batchnorm77_1_1_19_(conv77_1_1_19_)
        relu77_1_1_19_ = self.relu77_1_1_19_(batchnorm77_1_1_19_)
        conv78_1_1_19_ = self.conv78_1_1_19_(relu77_1_1_19_)
        batchnorm78_1_1_19_ = self.batchnorm78_1_1_19_(conv78_1_1_19_)
        conv76_1_1_20_ = self.conv76_1_1_20_(relu74_1_)
        batchnorm76_1_1_20_ = self.batchnorm76_1_1_20_(conv76_1_1_20_)
        relu76_1_1_20_ = self.relu76_1_1_20_(batchnorm76_1_1_20_)
        conv77_1_1_20_padding = self.conv77_1_1_20_padding(relu76_1_1_20_)
        conv77_1_1_20_ = self.conv77_1_1_20_(conv77_1_1_20_padding)
        batchnorm77_1_1_20_ = self.batchnorm77_1_1_20_(conv77_1_1_20_)
        relu77_1_1_20_ = self.relu77_1_1_20_(batchnorm77_1_1_20_)
        conv78_1_1_20_ = self.conv78_1_1_20_(relu77_1_1_20_)
        batchnorm78_1_1_20_ = self.batchnorm78_1_1_20_(conv78_1_1_20_)
        conv76_1_1_21_ = self.conv76_1_1_21_(relu74_1_)
        batchnorm76_1_1_21_ = self.batchnorm76_1_1_21_(conv76_1_1_21_)
        relu76_1_1_21_ = self.relu76_1_1_21_(batchnorm76_1_1_21_)
        conv77_1_1_21_padding = self.conv77_1_1_21_padding(relu76_1_1_21_)
        conv77_1_1_21_ = self.conv77_1_1_21_(conv77_1_1_21_padding)
        batchnorm77_1_1_21_ = self.batchnorm77_1_1_21_(conv77_1_1_21_)
        relu77_1_1_21_ = self.relu77_1_1_21_(batchnorm77_1_1_21_)
        conv78_1_1_21_ = self.conv78_1_1_21_(relu77_1_1_21_)
        batchnorm78_1_1_21_ = self.batchnorm78_1_1_21_(conv78_1_1_21_)
        conv76_1_1_22_ = self.conv76_1_1_22_(relu74_1_)
        batchnorm76_1_1_22_ = self.batchnorm76_1_1_22_(conv76_1_1_22_)
        relu76_1_1_22_ = self.relu76_1_1_22_(batchnorm76_1_1_22_)
        conv77_1_1_22_padding = self.conv77_1_1_22_padding(relu76_1_1_22_)
        conv77_1_1_22_ = self.conv77_1_1_22_(conv77_1_1_22_padding)
        batchnorm77_1_1_22_ = self.batchnorm77_1_1_22_(conv77_1_1_22_)
        relu77_1_1_22_ = self.relu77_1_1_22_(batchnorm77_1_1_22_)
        conv78_1_1_22_ = self.conv78_1_1_22_(relu77_1_1_22_)
        batchnorm78_1_1_22_ = self.batchnorm78_1_1_22_(conv78_1_1_22_)
        conv76_1_1_23_ = self.conv76_1_1_23_(relu74_1_)
        batchnorm76_1_1_23_ = self.batchnorm76_1_1_23_(conv76_1_1_23_)
        relu76_1_1_23_ = self.relu76_1_1_23_(batchnorm76_1_1_23_)
        conv77_1_1_23_padding = self.conv77_1_1_23_padding(relu76_1_1_23_)
        conv77_1_1_23_ = self.conv77_1_1_23_(conv77_1_1_23_padding)
        batchnorm77_1_1_23_ = self.batchnorm77_1_1_23_(conv77_1_1_23_)
        relu77_1_1_23_ = self.relu77_1_1_23_(batchnorm77_1_1_23_)
        conv78_1_1_23_ = self.conv78_1_1_23_(relu77_1_1_23_)
        batchnorm78_1_1_23_ = self.batchnorm78_1_1_23_(conv78_1_1_23_)
        conv76_1_1_24_ = self.conv76_1_1_24_(relu74_1_)
        batchnorm76_1_1_24_ = self.batchnorm76_1_1_24_(conv76_1_1_24_)
        relu76_1_1_24_ = self.relu76_1_1_24_(batchnorm76_1_1_24_)
        conv77_1_1_24_padding = self.conv77_1_1_24_padding(relu76_1_1_24_)
        conv77_1_1_24_ = self.conv77_1_1_24_(conv77_1_1_24_padding)
        batchnorm77_1_1_24_ = self.batchnorm77_1_1_24_(conv77_1_1_24_)
        relu77_1_1_24_ = self.relu77_1_1_24_(batchnorm77_1_1_24_)
        conv78_1_1_24_ = self.conv78_1_1_24_(relu77_1_1_24_)
        batchnorm78_1_1_24_ = self.batchnorm78_1_1_24_(conv78_1_1_24_)
        conv76_1_1_25_ = self.conv76_1_1_25_(relu74_1_)
        batchnorm76_1_1_25_ = self.batchnorm76_1_1_25_(conv76_1_1_25_)
        relu76_1_1_25_ = self.relu76_1_1_25_(batchnorm76_1_1_25_)
        conv77_1_1_25_padding = self.conv77_1_1_25_padding(relu76_1_1_25_)
        conv77_1_1_25_ = self.conv77_1_1_25_(conv77_1_1_25_padding)
        batchnorm77_1_1_25_ = self.batchnorm77_1_1_25_(conv77_1_1_25_)
        relu77_1_1_25_ = self.relu77_1_1_25_(batchnorm77_1_1_25_)
        conv78_1_1_25_ = self.conv78_1_1_25_(relu77_1_1_25_)
        batchnorm78_1_1_25_ = self.batchnorm78_1_1_25_(conv78_1_1_25_)
        conv76_1_1_26_ = self.conv76_1_1_26_(relu74_1_)
        batchnorm76_1_1_26_ = self.batchnorm76_1_1_26_(conv76_1_1_26_)
        relu76_1_1_26_ = self.relu76_1_1_26_(batchnorm76_1_1_26_)
        conv77_1_1_26_padding = self.conv77_1_1_26_padding(relu76_1_1_26_)
        conv77_1_1_26_ = self.conv77_1_1_26_(conv77_1_1_26_padding)
        batchnorm77_1_1_26_ = self.batchnorm77_1_1_26_(conv77_1_1_26_)
        relu77_1_1_26_ = self.relu77_1_1_26_(batchnorm77_1_1_26_)
        conv78_1_1_26_ = self.conv78_1_1_26_(relu77_1_1_26_)
        batchnorm78_1_1_26_ = self.batchnorm78_1_1_26_(conv78_1_1_26_)
        conv76_1_1_27_ = self.conv76_1_1_27_(relu74_1_)
        batchnorm76_1_1_27_ = self.batchnorm76_1_1_27_(conv76_1_1_27_)
        relu76_1_1_27_ = self.relu76_1_1_27_(batchnorm76_1_1_27_)
        conv77_1_1_27_padding = self.conv77_1_1_27_padding(relu76_1_1_27_)
        conv77_1_1_27_ = self.conv77_1_1_27_(conv77_1_1_27_padding)
        batchnorm77_1_1_27_ = self.batchnorm77_1_1_27_(conv77_1_1_27_)
        relu77_1_1_27_ = self.relu77_1_1_27_(batchnorm77_1_1_27_)
        conv78_1_1_27_ = self.conv78_1_1_27_(relu77_1_1_27_)
        batchnorm78_1_1_27_ = self.batchnorm78_1_1_27_(conv78_1_1_27_)
        conv76_1_1_28_ = self.conv76_1_1_28_(relu74_1_)
        batchnorm76_1_1_28_ = self.batchnorm76_1_1_28_(conv76_1_1_28_)
        relu76_1_1_28_ = self.relu76_1_1_28_(batchnorm76_1_1_28_)
        conv77_1_1_28_padding = self.conv77_1_1_28_padding(relu76_1_1_28_)
        conv77_1_1_28_ = self.conv77_1_1_28_(conv77_1_1_28_padding)
        batchnorm77_1_1_28_ = self.batchnorm77_1_1_28_(conv77_1_1_28_)
        relu77_1_1_28_ = self.relu77_1_1_28_(batchnorm77_1_1_28_)
        conv78_1_1_28_ = self.conv78_1_1_28_(relu77_1_1_28_)
        batchnorm78_1_1_28_ = self.batchnorm78_1_1_28_(conv78_1_1_28_)
        conv76_1_1_29_ = self.conv76_1_1_29_(relu74_1_)
        batchnorm76_1_1_29_ = self.batchnorm76_1_1_29_(conv76_1_1_29_)
        relu76_1_1_29_ = self.relu76_1_1_29_(batchnorm76_1_1_29_)
        conv77_1_1_29_padding = self.conv77_1_1_29_padding(relu76_1_1_29_)
        conv77_1_1_29_ = self.conv77_1_1_29_(conv77_1_1_29_padding)
        batchnorm77_1_1_29_ = self.batchnorm77_1_1_29_(conv77_1_1_29_)
        relu77_1_1_29_ = self.relu77_1_1_29_(batchnorm77_1_1_29_)
        conv78_1_1_29_ = self.conv78_1_1_29_(relu77_1_1_29_)
        batchnorm78_1_1_29_ = self.batchnorm78_1_1_29_(conv78_1_1_29_)
        conv76_1_1_30_ = self.conv76_1_1_30_(relu74_1_)
        batchnorm76_1_1_30_ = self.batchnorm76_1_1_30_(conv76_1_1_30_)
        relu76_1_1_30_ = self.relu76_1_1_30_(batchnorm76_1_1_30_)
        conv77_1_1_30_padding = self.conv77_1_1_30_padding(relu76_1_1_30_)
        conv77_1_1_30_ = self.conv77_1_1_30_(conv77_1_1_30_padding)
        batchnorm77_1_1_30_ = self.batchnorm77_1_1_30_(conv77_1_1_30_)
        relu77_1_1_30_ = self.relu77_1_1_30_(batchnorm77_1_1_30_)
        conv78_1_1_30_ = self.conv78_1_1_30_(relu77_1_1_30_)
        batchnorm78_1_1_30_ = self.batchnorm78_1_1_30_(conv78_1_1_30_)
        conv76_1_1_31_ = self.conv76_1_1_31_(relu74_1_)
        batchnorm76_1_1_31_ = self.batchnorm76_1_1_31_(conv76_1_1_31_)
        relu76_1_1_31_ = self.relu76_1_1_31_(batchnorm76_1_1_31_)
        conv77_1_1_31_padding = self.conv77_1_1_31_padding(relu76_1_1_31_)
        conv77_1_1_31_ = self.conv77_1_1_31_(conv77_1_1_31_padding)
        batchnorm77_1_1_31_ = self.batchnorm77_1_1_31_(conv77_1_1_31_)
        relu77_1_1_31_ = self.relu77_1_1_31_(batchnorm77_1_1_31_)
        conv78_1_1_31_ = self.conv78_1_1_31_(relu77_1_1_31_)
        batchnorm78_1_1_31_ = self.batchnorm78_1_1_31_(conv78_1_1_31_)
        conv76_1_1_32_ = self.conv76_1_1_32_(relu74_1_)
        batchnorm76_1_1_32_ = self.batchnorm76_1_1_32_(conv76_1_1_32_)
        relu76_1_1_32_ = self.relu76_1_1_32_(batchnorm76_1_1_32_)
        conv77_1_1_32_padding = self.conv77_1_1_32_padding(relu76_1_1_32_)
        conv77_1_1_32_ = self.conv77_1_1_32_(conv77_1_1_32_padding)
        batchnorm77_1_1_32_ = self.batchnorm77_1_1_32_(conv77_1_1_32_)
        relu77_1_1_32_ = self.relu77_1_1_32_(batchnorm77_1_1_32_)
        conv78_1_1_32_ = self.conv78_1_1_32_(relu77_1_1_32_)
        batchnorm78_1_1_32_ = self.batchnorm78_1_1_32_(conv78_1_1_32_)
        add79_1_1_ = batchnorm78_1_1_1_ + batchnorm78_1_1_2_ + batchnorm78_1_1_3_ + batchnorm78_1_1_4_ + batchnorm78_1_1_5_ + batchnorm78_1_1_6_ + batchnorm78_1_1_7_ + batchnorm78_1_1_8_ + batchnorm78_1_1_9_ + batchnorm78_1_1_10_ + batchnorm78_1_1_11_ + batchnorm78_1_1_12_ + batchnorm78_1_1_13_ + batchnorm78_1_1_14_ + batchnorm78_1_1_15_ + batchnorm78_1_1_16_ + batchnorm78_1_1_17_ + batchnorm78_1_1_18_ + batchnorm78_1_1_19_ + batchnorm78_1_1_20_ + batchnorm78_1_1_21_ + batchnorm78_1_1_22_ + batchnorm78_1_1_23_ + batchnorm78_1_1_24_ + batchnorm78_1_1_25_ + batchnorm78_1_1_26_ + batchnorm78_1_1_27_ + batchnorm78_1_1_28_ + batchnorm78_1_1_29_ + batchnorm78_1_1_30_ + batchnorm78_1_1_31_ + batchnorm78_1_1_32_
        add80_1_ = add79_1_1_ + relu74_1_
        relu80_1_ = self.relu80_1_(add80_1_)
        conv82_1_1_1_ = self.conv82_1_1_1_(relu80_1_)
        batchnorm82_1_1_1_ = self.batchnorm82_1_1_1_(conv82_1_1_1_)
        relu82_1_1_1_ = self.relu82_1_1_1_(batchnorm82_1_1_1_)
        conv83_1_1_1_padding = self.conv83_1_1_1_padding(relu82_1_1_1_)
        conv83_1_1_1_ = self.conv83_1_1_1_(conv83_1_1_1_padding)
        batchnorm83_1_1_1_ = self.batchnorm83_1_1_1_(conv83_1_1_1_)
        relu83_1_1_1_ = self.relu83_1_1_1_(batchnorm83_1_1_1_)
        conv84_1_1_1_ = self.conv84_1_1_1_(relu83_1_1_1_)
        batchnorm84_1_1_1_ = self.batchnorm84_1_1_1_(conv84_1_1_1_)
        conv82_1_1_2_ = self.conv82_1_1_2_(relu80_1_)
        batchnorm82_1_1_2_ = self.batchnorm82_1_1_2_(conv82_1_1_2_)
        relu82_1_1_2_ = self.relu82_1_1_2_(batchnorm82_1_1_2_)
        conv83_1_1_2_padding = self.conv83_1_1_2_padding(relu82_1_1_2_)
        conv83_1_1_2_ = self.conv83_1_1_2_(conv83_1_1_2_padding)
        batchnorm83_1_1_2_ = self.batchnorm83_1_1_2_(conv83_1_1_2_)
        relu83_1_1_2_ = self.relu83_1_1_2_(batchnorm83_1_1_2_)
        conv84_1_1_2_ = self.conv84_1_1_2_(relu83_1_1_2_)
        batchnorm84_1_1_2_ = self.batchnorm84_1_1_2_(conv84_1_1_2_)
        conv82_1_1_3_ = self.conv82_1_1_3_(relu80_1_)
        batchnorm82_1_1_3_ = self.batchnorm82_1_1_3_(conv82_1_1_3_)
        relu82_1_1_3_ = self.relu82_1_1_3_(batchnorm82_1_1_3_)
        conv83_1_1_3_padding = self.conv83_1_1_3_padding(relu82_1_1_3_)
        conv83_1_1_3_ = self.conv83_1_1_3_(conv83_1_1_3_padding)
        batchnorm83_1_1_3_ = self.batchnorm83_1_1_3_(conv83_1_1_3_)
        relu83_1_1_3_ = self.relu83_1_1_3_(batchnorm83_1_1_3_)
        conv84_1_1_3_ = self.conv84_1_1_3_(relu83_1_1_3_)
        batchnorm84_1_1_3_ = self.batchnorm84_1_1_3_(conv84_1_1_3_)
        conv82_1_1_4_ = self.conv82_1_1_4_(relu80_1_)
        batchnorm82_1_1_4_ = self.batchnorm82_1_1_4_(conv82_1_1_4_)
        relu82_1_1_4_ = self.relu82_1_1_4_(batchnorm82_1_1_4_)
        conv83_1_1_4_padding = self.conv83_1_1_4_padding(relu82_1_1_4_)
        conv83_1_1_4_ = self.conv83_1_1_4_(conv83_1_1_4_padding)
        batchnorm83_1_1_4_ = self.batchnorm83_1_1_4_(conv83_1_1_4_)
        relu83_1_1_4_ = self.relu83_1_1_4_(batchnorm83_1_1_4_)
        conv84_1_1_4_ = self.conv84_1_1_4_(relu83_1_1_4_)
        batchnorm84_1_1_4_ = self.batchnorm84_1_1_4_(conv84_1_1_4_)
        conv82_1_1_5_ = self.conv82_1_1_5_(relu80_1_)
        batchnorm82_1_1_5_ = self.batchnorm82_1_1_5_(conv82_1_1_5_)
        relu82_1_1_5_ = self.relu82_1_1_5_(batchnorm82_1_1_5_)
        conv83_1_1_5_padding = self.conv83_1_1_5_padding(relu82_1_1_5_)
        conv83_1_1_5_ = self.conv83_1_1_5_(conv83_1_1_5_padding)
        batchnorm83_1_1_5_ = self.batchnorm83_1_1_5_(conv83_1_1_5_)
        relu83_1_1_5_ = self.relu83_1_1_5_(batchnorm83_1_1_5_)
        conv84_1_1_5_ = self.conv84_1_1_5_(relu83_1_1_5_)
        batchnorm84_1_1_5_ = self.batchnorm84_1_1_5_(conv84_1_1_5_)
        conv82_1_1_6_ = self.conv82_1_1_6_(relu80_1_)
        batchnorm82_1_1_6_ = self.batchnorm82_1_1_6_(conv82_1_1_6_)
        relu82_1_1_6_ = self.relu82_1_1_6_(batchnorm82_1_1_6_)
        conv83_1_1_6_padding = self.conv83_1_1_6_padding(relu82_1_1_6_)
        conv83_1_1_6_ = self.conv83_1_1_6_(conv83_1_1_6_padding)
        batchnorm83_1_1_6_ = self.batchnorm83_1_1_6_(conv83_1_1_6_)
        relu83_1_1_6_ = self.relu83_1_1_6_(batchnorm83_1_1_6_)
        conv84_1_1_6_ = self.conv84_1_1_6_(relu83_1_1_6_)
        batchnorm84_1_1_6_ = self.batchnorm84_1_1_6_(conv84_1_1_6_)
        conv82_1_1_7_ = self.conv82_1_1_7_(relu80_1_)
        batchnorm82_1_1_7_ = self.batchnorm82_1_1_7_(conv82_1_1_7_)
        relu82_1_1_7_ = self.relu82_1_1_7_(batchnorm82_1_1_7_)
        conv83_1_1_7_padding = self.conv83_1_1_7_padding(relu82_1_1_7_)
        conv83_1_1_7_ = self.conv83_1_1_7_(conv83_1_1_7_padding)
        batchnorm83_1_1_7_ = self.batchnorm83_1_1_7_(conv83_1_1_7_)
        relu83_1_1_7_ = self.relu83_1_1_7_(batchnorm83_1_1_7_)
        conv84_1_1_7_ = self.conv84_1_1_7_(relu83_1_1_7_)
        batchnorm84_1_1_7_ = self.batchnorm84_1_1_7_(conv84_1_1_7_)
        conv82_1_1_8_ = self.conv82_1_1_8_(relu80_1_)
        batchnorm82_1_1_8_ = self.batchnorm82_1_1_8_(conv82_1_1_8_)
        relu82_1_1_8_ = self.relu82_1_1_8_(batchnorm82_1_1_8_)
        conv83_1_1_8_padding = self.conv83_1_1_8_padding(relu82_1_1_8_)
        conv83_1_1_8_ = self.conv83_1_1_8_(conv83_1_1_8_padding)
        batchnorm83_1_1_8_ = self.batchnorm83_1_1_8_(conv83_1_1_8_)
        relu83_1_1_8_ = self.relu83_1_1_8_(batchnorm83_1_1_8_)
        conv84_1_1_8_ = self.conv84_1_1_8_(relu83_1_1_8_)
        batchnorm84_1_1_8_ = self.batchnorm84_1_1_8_(conv84_1_1_8_)
        conv82_1_1_9_ = self.conv82_1_1_9_(relu80_1_)
        batchnorm82_1_1_9_ = self.batchnorm82_1_1_9_(conv82_1_1_9_)
        relu82_1_1_9_ = self.relu82_1_1_9_(batchnorm82_1_1_9_)
        conv83_1_1_9_padding = self.conv83_1_1_9_padding(relu82_1_1_9_)
        conv83_1_1_9_ = self.conv83_1_1_9_(conv83_1_1_9_padding)
        batchnorm83_1_1_9_ = self.batchnorm83_1_1_9_(conv83_1_1_9_)
        relu83_1_1_9_ = self.relu83_1_1_9_(batchnorm83_1_1_9_)
        conv84_1_1_9_ = self.conv84_1_1_9_(relu83_1_1_9_)
        batchnorm84_1_1_9_ = self.batchnorm84_1_1_9_(conv84_1_1_9_)
        conv82_1_1_10_ = self.conv82_1_1_10_(relu80_1_)
        batchnorm82_1_1_10_ = self.batchnorm82_1_1_10_(conv82_1_1_10_)
        relu82_1_1_10_ = self.relu82_1_1_10_(batchnorm82_1_1_10_)
        conv83_1_1_10_padding = self.conv83_1_1_10_padding(relu82_1_1_10_)
        conv83_1_1_10_ = self.conv83_1_1_10_(conv83_1_1_10_padding)
        batchnorm83_1_1_10_ = self.batchnorm83_1_1_10_(conv83_1_1_10_)
        relu83_1_1_10_ = self.relu83_1_1_10_(batchnorm83_1_1_10_)
        conv84_1_1_10_ = self.conv84_1_1_10_(relu83_1_1_10_)
        batchnorm84_1_1_10_ = self.batchnorm84_1_1_10_(conv84_1_1_10_)
        conv82_1_1_11_ = self.conv82_1_1_11_(relu80_1_)
        batchnorm82_1_1_11_ = self.batchnorm82_1_1_11_(conv82_1_1_11_)
        relu82_1_1_11_ = self.relu82_1_1_11_(batchnorm82_1_1_11_)
        conv83_1_1_11_padding = self.conv83_1_1_11_padding(relu82_1_1_11_)
        conv83_1_1_11_ = self.conv83_1_1_11_(conv83_1_1_11_padding)
        batchnorm83_1_1_11_ = self.batchnorm83_1_1_11_(conv83_1_1_11_)
        relu83_1_1_11_ = self.relu83_1_1_11_(batchnorm83_1_1_11_)
        conv84_1_1_11_ = self.conv84_1_1_11_(relu83_1_1_11_)
        batchnorm84_1_1_11_ = self.batchnorm84_1_1_11_(conv84_1_1_11_)
        conv82_1_1_12_ = self.conv82_1_1_12_(relu80_1_)
        batchnorm82_1_1_12_ = self.batchnorm82_1_1_12_(conv82_1_1_12_)
        relu82_1_1_12_ = self.relu82_1_1_12_(batchnorm82_1_1_12_)
        conv83_1_1_12_padding = self.conv83_1_1_12_padding(relu82_1_1_12_)
        conv83_1_1_12_ = self.conv83_1_1_12_(conv83_1_1_12_padding)
        batchnorm83_1_1_12_ = self.batchnorm83_1_1_12_(conv83_1_1_12_)
        relu83_1_1_12_ = self.relu83_1_1_12_(batchnorm83_1_1_12_)
        conv84_1_1_12_ = self.conv84_1_1_12_(relu83_1_1_12_)
        batchnorm84_1_1_12_ = self.batchnorm84_1_1_12_(conv84_1_1_12_)
        conv82_1_1_13_ = self.conv82_1_1_13_(relu80_1_)
        batchnorm82_1_1_13_ = self.batchnorm82_1_1_13_(conv82_1_1_13_)
        relu82_1_1_13_ = self.relu82_1_1_13_(batchnorm82_1_1_13_)
        conv83_1_1_13_padding = self.conv83_1_1_13_padding(relu82_1_1_13_)
        conv83_1_1_13_ = self.conv83_1_1_13_(conv83_1_1_13_padding)
        batchnorm83_1_1_13_ = self.batchnorm83_1_1_13_(conv83_1_1_13_)
        relu83_1_1_13_ = self.relu83_1_1_13_(batchnorm83_1_1_13_)
        conv84_1_1_13_ = self.conv84_1_1_13_(relu83_1_1_13_)
        batchnorm84_1_1_13_ = self.batchnorm84_1_1_13_(conv84_1_1_13_)
        conv82_1_1_14_ = self.conv82_1_1_14_(relu80_1_)
        batchnorm82_1_1_14_ = self.batchnorm82_1_1_14_(conv82_1_1_14_)
        relu82_1_1_14_ = self.relu82_1_1_14_(batchnorm82_1_1_14_)
        conv83_1_1_14_padding = self.conv83_1_1_14_padding(relu82_1_1_14_)
        conv83_1_1_14_ = self.conv83_1_1_14_(conv83_1_1_14_padding)
        batchnorm83_1_1_14_ = self.batchnorm83_1_1_14_(conv83_1_1_14_)
        relu83_1_1_14_ = self.relu83_1_1_14_(batchnorm83_1_1_14_)
        conv84_1_1_14_ = self.conv84_1_1_14_(relu83_1_1_14_)
        batchnorm84_1_1_14_ = self.batchnorm84_1_1_14_(conv84_1_1_14_)
        conv82_1_1_15_ = self.conv82_1_1_15_(relu80_1_)
        batchnorm82_1_1_15_ = self.batchnorm82_1_1_15_(conv82_1_1_15_)
        relu82_1_1_15_ = self.relu82_1_1_15_(batchnorm82_1_1_15_)
        conv83_1_1_15_padding = self.conv83_1_1_15_padding(relu82_1_1_15_)
        conv83_1_1_15_ = self.conv83_1_1_15_(conv83_1_1_15_padding)
        batchnorm83_1_1_15_ = self.batchnorm83_1_1_15_(conv83_1_1_15_)
        relu83_1_1_15_ = self.relu83_1_1_15_(batchnorm83_1_1_15_)
        conv84_1_1_15_ = self.conv84_1_1_15_(relu83_1_1_15_)
        batchnorm84_1_1_15_ = self.batchnorm84_1_1_15_(conv84_1_1_15_)
        conv82_1_1_16_ = self.conv82_1_1_16_(relu80_1_)
        batchnorm82_1_1_16_ = self.batchnorm82_1_1_16_(conv82_1_1_16_)
        relu82_1_1_16_ = self.relu82_1_1_16_(batchnorm82_1_1_16_)
        conv83_1_1_16_padding = self.conv83_1_1_16_padding(relu82_1_1_16_)
        conv83_1_1_16_ = self.conv83_1_1_16_(conv83_1_1_16_padding)
        batchnorm83_1_1_16_ = self.batchnorm83_1_1_16_(conv83_1_1_16_)
        relu83_1_1_16_ = self.relu83_1_1_16_(batchnorm83_1_1_16_)
        conv84_1_1_16_ = self.conv84_1_1_16_(relu83_1_1_16_)
        batchnorm84_1_1_16_ = self.batchnorm84_1_1_16_(conv84_1_1_16_)
        conv82_1_1_17_ = self.conv82_1_1_17_(relu80_1_)
        batchnorm82_1_1_17_ = self.batchnorm82_1_1_17_(conv82_1_1_17_)
        relu82_1_1_17_ = self.relu82_1_1_17_(batchnorm82_1_1_17_)
        conv83_1_1_17_padding = self.conv83_1_1_17_padding(relu82_1_1_17_)
        conv83_1_1_17_ = self.conv83_1_1_17_(conv83_1_1_17_padding)
        batchnorm83_1_1_17_ = self.batchnorm83_1_1_17_(conv83_1_1_17_)
        relu83_1_1_17_ = self.relu83_1_1_17_(batchnorm83_1_1_17_)
        conv84_1_1_17_ = self.conv84_1_1_17_(relu83_1_1_17_)
        batchnorm84_1_1_17_ = self.batchnorm84_1_1_17_(conv84_1_1_17_)
        conv82_1_1_18_ = self.conv82_1_1_18_(relu80_1_)
        batchnorm82_1_1_18_ = self.batchnorm82_1_1_18_(conv82_1_1_18_)
        relu82_1_1_18_ = self.relu82_1_1_18_(batchnorm82_1_1_18_)
        conv83_1_1_18_padding = self.conv83_1_1_18_padding(relu82_1_1_18_)
        conv83_1_1_18_ = self.conv83_1_1_18_(conv83_1_1_18_padding)
        batchnorm83_1_1_18_ = self.batchnorm83_1_1_18_(conv83_1_1_18_)
        relu83_1_1_18_ = self.relu83_1_1_18_(batchnorm83_1_1_18_)
        conv84_1_1_18_ = self.conv84_1_1_18_(relu83_1_1_18_)
        batchnorm84_1_1_18_ = self.batchnorm84_1_1_18_(conv84_1_1_18_)
        conv82_1_1_19_ = self.conv82_1_1_19_(relu80_1_)
        batchnorm82_1_1_19_ = self.batchnorm82_1_1_19_(conv82_1_1_19_)
        relu82_1_1_19_ = self.relu82_1_1_19_(batchnorm82_1_1_19_)
        conv83_1_1_19_padding = self.conv83_1_1_19_padding(relu82_1_1_19_)
        conv83_1_1_19_ = self.conv83_1_1_19_(conv83_1_1_19_padding)
        batchnorm83_1_1_19_ = self.batchnorm83_1_1_19_(conv83_1_1_19_)
        relu83_1_1_19_ = self.relu83_1_1_19_(batchnorm83_1_1_19_)
        conv84_1_1_19_ = self.conv84_1_1_19_(relu83_1_1_19_)
        batchnorm84_1_1_19_ = self.batchnorm84_1_1_19_(conv84_1_1_19_)
        conv82_1_1_20_ = self.conv82_1_1_20_(relu80_1_)
        batchnorm82_1_1_20_ = self.batchnorm82_1_1_20_(conv82_1_1_20_)
        relu82_1_1_20_ = self.relu82_1_1_20_(batchnorm82_1_1_20_)
        conv83_1_1_20_padding = self.conv83_1_1_20_padding(relu82_1_1_20_)
        conv83_1_1_20_ = self.conv83_1_1_20_(conv83_1_1_20_padding)
        batchnorm83_1_1_20_ = self.batchnorm83_1_1_20_(conv83_1_1_20_)
        relu83_1_1_20_ = self.relu83_1_1_20_(batchnorm83_1_1_20_)
        conv84_1_1_20_ = self.conv84_1_1_20_(relu83_1_1_20_)
        batchnorm84_1_1_20_ = self.batchnorm84_1_1_20_(conv84_1_1_20_)
        conv82_1_1_21_ = self.conv82_1_1_21_(relu80_1_)
        batchnorm82_1_1_21_ = self.batchnorm82_1_1_21_(conv82_1_1_21_)
        relu82_1_1_21_ = self.relu82_1_1_21_(batchnorm82_1_1_21_)
        conv83_1_1_21_padding = self.conv83_1_1_21_padding(relu82_1_1_21_)
        conv83_1_1_21_ = self.conv83_1_1_21_(conv83_1_1_21_padding)
        batchnorm83_1_1_21_ = self.batchnorm83_1_1_21_(conv83_1_1_21_)
        relu83_1_1_21_ = self.relu83_1_1_21_(batchnorm83_1_1_21_)
        conv84_1_1_21_ = self.conv84_1_1_21_(relu83_1_1_21_)
        batchnorm84_1_1_21_ = self.batchnorm84_1_1_21_(conv84_1_1_21_)
        conv82_1_1_22_ = self.conv82_1_1_22_(relu80_1_)
        batchnorm82_1_1_22_ = self.batchnorm82_1_1_22_(conv82_1_1_22_)
        relu82_1_1_22_ = self.relu82_1_1_22_(batchnorm82_1_1_22_)
        conv83_1_1_22_padding = self.conv83_1_1_22_padding(relu82_1_1_22_)
        conv83_1_1_22_ = self.conv83_1_1_22_(conv83_1_1_22_padding)
        batchnorm83_1_1_22_ = self.batchnorm83_1_1_22_(conv83_1_1_22_)
        relu83_1_1_22_ = self.relu83_1_1_22_(batchnorm83_1_1_22_)
        conv84_1_1_22_ = self.conv84_1_1_22_(relu83_1_1_22_)
        batchnorm84_1_1_22_ = self.batchnorm84_1_1_22_(conv84_1_1_22_)
        conv82_1_1_23_ = self.conv82_1_1_23_(relu80_1_)
        batchnorm82_1_1_23_ = self.batchnorm82_1_1_23_(conv82_1_1_23_)
        relu82_1_1_23_ = self.relu82_1_1_23_(batchnorm82_1_1_23_)
        conv83_1_1_23_padding = self.conv83_1_1_23_padding(relu82_1_1_23_)
        conv83_1_1_23_ = self.conv83_1_1_23_(conv83_1_1_23_padding)
        batchnorm83_1_1_23_ = self.batchnorm83_1_1_23_(conv83_1_1_23_)
        relu83_1_1_23_ = self.relu83_1_1_23_(batchnorm83_1_1_23_)
        conv84_1_1_23_ = self.conv84_1_1_23_(relu83_1_1_23_)
        batchnorm84_1_1_23_ = self.batchnorm84_1_1_23_(conv84_1_1_23_)
        conv82_1_1_24_ = self.conv82_1_1_24_(relu80_1_)
        batchnorm82_1_1_24_ = self.batchnorm82_1_1_24_(conv82_1_1_24_)
        relu82_1_1_24_ = self.relu82_1_1_24_(batchnorm82_1_1_24_)
        conv83_1_1_24_padding = self.conv83_1_1_24_padding(relu82_1_1_24_)
        conv83_1_1_24_ = self.conv83_1_1_24_(conv83_1_1_24_padding)
        batchnorm83_1_1_24_ = self.batchnorm83_1_1_24_(conv83_1_1_24_)
        relu83_1_1_24_ = self.relu83_1_1_24_(batchnorm83_1_1_24_)
        conv84_1_1_24_ = self.conv84_1_1_24_(relu83_1_1_24_)
        batchnorm84_1_1_24_ = self.batchnorm84_1_1_24_(conv84_1_1_24_)
        conv82_1_1_25_ = self.conv82_1_1_25_(relu80_1_)
        batchnorm82_1_1_25_ = self.batchnorm82_1_1_25_(conv82_1_1_25_)
        relu82_1_1_25_ = self.relu82_1_1_25_(batchnorm82_1_1_25_)
        conv83_1_1_25_padding = self.conv83_1_1_25_padding(relu82_1_1_25_)
        conv83_1_1_25_ = self.conv83_1_1_25_(conv83_1_1_25_padding)
        batchnorm83_1_1_25_ = self.batchnorm83_1_1_25_(conv83_1_1_25_)
        relu83_1_1_25_ = self.relu83_1_1_25_(batchnorm83_1_1_25_)
        conv84_1_1_25_ = self.conv84_1_1_25_(relu83_1_1_25_)
        batchnorm84_1_1_25_ = self.batchnorm84_1_1_25_(conv84_1_1_25_)
        conv82_1_1_26_ = self.conv82_1_1_26_(relu80_1_)
        batchnorm82_1_1_26_ = self.batchnorm82_1_1_26_(conv82_1_1_26_)
        relu82_1_1_26_ = self.relu82_1_1_26_(batchnorm82_1_1_26_)
        conv83_1_1_26_padding = self.conv83_1_1_26_padding(relu82_1_1_26_)
        conv83_1_1_26_ = self.conv83_1_1_26_(conv83_1_1_26_padding)
        batchnorm83_1_1_26_ = self.batchnorm83_1_1_26_(conv83_1_1_26_)
        relu83_1_1_26_ = self.relu83_1_1_26_(batchnorm83_1_1_26_)
        conv84_1_1_26_ = self.conv84_1_1_26_(relu83_1_1_26_)
        batchnorm84_1_1_26_ = self.batchnorm84_1_1_26_(conv84_1_1_26_)
        conv82_1_1_27_ = self.conv82_1_1_27_(relu80_1_)
        batchnorm82_1_1_27_ = self.batchnorm82_1_1_27_(conv82_1_1_27_)
        relu82_1_1_27_ = self.relu82_1_1_27_(batchnorm82_1_1_27_)
        conv83_1_1_27_padding = self.conv83_1_1_27_padding(relu82_1_1_27_)
        conv83_1_1_27_ = self.conv83_1_1_27_(conv83_1_1_27_padding)
        batchnorm83_1_1_27_ = self.batchnorm83_1_1_27_(conv83_1_1_27_)
        relu83_1_1_27_ = self.relu83_1_1_27_(batchnorm83_1_1_27_)
        conv84_1_1_27_ = self.conv84_1_1_27_(relu83_1_1_27_)
        batchnorm84_1_1_27_ = self.batchnorm84_1_1_27_(conv84_1_1_27_)
        conv82_1_1_28_ = self.conv82_1_1_28_(relu80_1_)
        batchnorm82_1_1_28_ = self.batchnorm82_1_1_28_(conv82_1_1_28_)
        relu82_1_1_28_ = self.relu82_1_1_28_(batchnorm82_1_1_28_)
        conv83_1_1_28_padding = self.conv83_1_1_28_padding(relu82_1_1_28_)
        conv83_1_1_28_ = self.conv83_1_1_28_(conv83_1_1_28_padding)
        batchnorm83_1_1_28_ = self.batchnorm83_1_1_28_(conv83_1_1_28_)
        relu83_1_1_28_ = self.relu83_1_1_28_(batchnorm83_1_1_28_)
        conv84_1_1_28_ = self.conv84_1_1_28_(relu83_1_1_28_)
        batchnorm84_1_1_28_ = self.batchnorm84_1_1_28_(conv84_1_1_28_)
        conv82_1_1_29_ = self.conv82_1_1_29_(relu80_1_)
        batchnorm82_1_1_29_ = self.batchnorm82_1_1_29_(conv82_1_1_29_)
        relu82_1_1_29_ = self.relu82_1_1_29_(batchnorm82_1_1_29_)
        conv83_1_1_29_padding = self.conv83_1_1_29_padding(relu82_1_1_29_)
        conv83_1_1_29_ = self.conv83_1_1_29_(conv83_1_1_29_padding)
        batchnorm83_1_1_29_ = self.batchnorm83_1_1_29_(conv83_1_1_29_)
        relu83_1_1_29_ = self.relu83_1_1_29_(batchnorm83_1_1_29_)
        conv84_1_1_29_ = self.conv84_1_1_29_(relu83_1_1_29_)
        batchnorm84_1_1_29_ = self.batchnorm84_1_1_29_(conv84_1_1_29_)
        conv82_1_1_30_ = self.conv82_1_1_30_(relu80_1_)
        batchnorm82_1_1_30_ = self.batchnorm82_1_1_30_(conv82_1_1_30_)
        relu82_1_1_30_ = self.relu82_1_1_30_(batchnorm82_1_1_30_)
        conv83_1_1_30_padding = self.conv83_1_1_30_padding(relu82_1_1_30_)
        conv83_1_1_30_ = self.conv83_1_1_30_(conv83_1_1_30_padding)
        batchnorm83_1_1_30_ = self.batchnorm83_1_1_30_(conv83_1_1_30_)
        relu83_1_1_30_ = self.relu83_1_1_30_(batchnorm83_1_1_30_)
        conv84_1_1_30_ = self.conv84_1_1_30_(relu83_1_1_30_)
        batchnorm84_1_1_30_ = self.batchnorm84_1_1_30_(conv84_1_1_30_)
        conv82_1_1_31_ = self.conv82_1_1_31_(relu80_1_)
        batchnorm82_1_1_31_ = self.batchnorm82_1_1_31_(conv82_1_1_31_)
        relu82_1_1_31_ = self.relu82_1_1_31_(batchnorm82_1_1_31_)
        conv83_1_1_31_padding = self.conv83_1_1_31_padding(relu82_1_1_31_)
        conv83_1_1_31_ = self.conv83_1_1_31_(conv83_1_1_31_padding)
        batchnorm83_1_1_31_ = self.batchnorm83_1_1_31_(conv83_1_1_31_)
        relu83_1_1_31_ = self.relu83_1_1_31_(batchnorm83_1_1_31_)
        conv84_1_1_31_ = self.conv84_1_1_31_(relu83_1_1_31_)
        batchnorm84_1_1_31_ = self.batchnorm84_1_1_31_(conv84_1_1_31_)
        conv82_1_1_32_ = self.conv82_1_1_32_(relu80_1_)
        batchnorm82_1_1_32_ = self.batchnorm82_1_1_32_(conv82_1_1_32_)
        relu82_1_1_32_ = self.relu82_1_1_32_(batchnorm82_1_1_32_)
        conv83_1_1_32_padding = self.conv83_1_1_32_padding(relu82_1_1_32_)
        conv83_1_1_32_ = self.conv83_1_1_32_(conv83_1_1_32_padding)
        batchnorm83_1_1_32_ = self.batchnorm83_1_1_32_(conv83_1_1_32_)
        relu83_1_1_32_ = self.relu83_1_1_32_(batchnorm83_1_1_32_)
        conv84_1_1_32_ = self.conv84_1_1_32_(relu83_1_1_32_)
        batchnorm84_1_1_32_ = self.batchnorm84_1_1_32_(conv84_1_1_32_)
        add85_1_1_ = batchnorm84_1_1_1_ + batchnorm84_1_1_2_ + batchnorm84_1_1_3_ + batchnorm84_1_1_4_ + batchnorm84_1_1_5_ + batchnorm84_1_1_6_ + batchnorm84_1_1_7_ + batchnorm84_1_1_8_ + batchnorm84_1_1_9_ + batchnorm84_1_1_10_ + batchnorm84_1_1_11_ + batchnorm84_1_1_12_ + batchnorm84_1_1_13_ + batchnorm84_1_1_14_ + batchnorm84_1_1_15_ + batchnorm84_1_1_16_ + batchnorm84_1_1_17_ + batchnorm84_1_1_18_ + batchnorm84_1_1_19_ + batchnorm84_1_1_20_ + batchnorm84_1_1_21_ + batchnorm84_1_1_22_ + batchnorm84_1_1_23_ + batchnorm84_1_1_24_ + batchnorm84_1_1_25_ + batchnorm84_1_1_26_ + batchnorm84_1_1_27_ + batchnorm84_1_1_28_ + batchnorm84_1_1_29_ + batchnorm84_1_1_30_ + batchnorm84_1_1_31_ + batchnorm84_1_1_32_
        conv81_1_2_ = self.conv81_1_2_(relu80_1_)
        batchnorm81_1_2_ = self.batchnorm81_1_2_(conv81_1_2_)
        add86_1_ = add85_1_1_ + batchnorm81_1_2_
        relu86_1_ = self.relu86_1_(add86_1_)
        conv89_1_1_1_1_ = self.conv89_1_1_1_1_(relu86_1_)
        batchnorm89_1_1_1_1_ = self.batchnorm89_1_1_1_1_(conv89_1_1_1_1_)
        relu89_1_1_1_1_ = self.relu89_1_1_1_1_(batchnorm89_1_1_1_1_)
        conv90_1_1_1_1_padding = self.conv90_1_1_1_1_padding(relu89_1_1_1_1_)
        conv90_1_1_1_1_ = self.conv90_1_1_1_1_(conv90_1_1_1_1_padding)
        batchnorm90_1_1_1_1_ = self.batchnorm90_1_1_1_1_(conv90_1_1_1_1_)
        relu90_1_1_1_1_ = self.relu90_1_1_1_1_(batchnorm90_1_1_1_1_)
        conv91_1_1_1_1_ = self.conv91_1_1_1_1_(relu90_1_1_1_1_)
        batchnorm91_1_1_1_1_ = self.batchnorm91_1_1_1_1_(conv91_1_1_1_1_)
        conv89_1_1_1_2_ = self.conv89_1_1_1_2_(relu86_1_)
        batchnorm89_1_1_1_2_ = self.batchnorm89_1_1_1_2_(conv89_1_1_1_2_)
        relu89_1_1_1_2_ = self.relu89_1_1_1_2_(batchnorm89_1_1_1_2_)
        conv90_1_1_1_2_padding = self.conv90_1_1_1_2_padding(relu89_1_1_1_2_)
        conv90_1_1_1_2_ = self.conv90_1_1_1_2_(conv90_1_1_1_2_padding)
        batchnorm90_1_1_1_2_ = self.batchnorm90_1_1_1_2_(conv90_1_1_1_2_)
        relu90_1_1_1_2_ = self.relu90_1_1_1_2_(batchnorm90_1_1_1_2_)
        conv91_1_1_1_2_ = self.conv91_1_1_1_2_(relu90_1_1_1_2_)
        batchnorm91_1_1_1_2_ = self.batchnorm91_1_1_1_2_(conv91_1_1_1_2_)
        conv89_1_1_1_3_ = self.conv89_1_1_1_3_(relu86_1_)
        batchnorm89_1_1_1_3_ = self.batchnorm89_1_1_1_3_(conv89_1_1_1_3_)
        relu89_1_1_1_3_ = self.relu89_1_1_1_3_(batchnorm89_1_1_1_3_)
        conv90_1_1_1_3_padding = self.conv90_1_1_1_3_padding(relu89_1_1_1_3_)
        conv90_1_1_1_3_ = self.conv90_1_1_1_3_(conv90_1_1_1_3_padding)
        batchnorm90_1_1_1_3_ = self.batchnorm90_1_1_1_3_(conv90_1_1_1_3_)
        relu90_1_1_1_3_ = self.relu90_1_1_1_3_(batchnorm90_1_1_1_3_)
        conv91_1_1_1_3_ = self.conv91_1_1_1_3_(relu90_1_1_1_3_)
        batchnorm91_1_1_1_3_ = self.batchnorm91_1_1_1_3_(conv91_1_1_1_3_)
        conv89_1_1_1_4_ = self.conv89_1_1_1_4_(relu86_1_)
        batchnorm89_1_1_1_4_ = self.batchnorm89_1_1_1_4_(conv89_1_1_1_4_)
        relu89_1_1_1_4_ = self.relu89_1_1_1_4_(batchnorm89_1_1_1_4_)
        conv90_1_1_1_4_padding = self.conv90_1_1_1_4_padding(relu89_1_1_1_4_)
        conv90_1_1_1_4_ = self.conv90_1_1_1_4_(conv90_1_1_1_4_padding)
        batchnorm90_1_1_1_4_ = self.batchnorm90_1_1_1_4_(conv90_1_1_1_4_)
        relu90_1_1_1_4_ = self.relu90_1_1_1_4_(batchnorm90_1_1_1_4_)
        conv91_1_1_1_4_ = self.conv91_1_1_1_4_(relu90_1_1_1_4_)
        batchnorm91_1_1_1_4_ = self.batchnorm91_1_1_1_4_(conv91_1_1_1_4_)
        conv89_1_1_1_5_ = self.conv89_1_1_1_5_(relu86_1_)
        batchnorm89_1_1_1_5_ = self.batchnorm89_1_1_1_5_(conv89_1_1_1_5_)
        relu89_1_1_1_5_ = self.relu89_1_1_1_5_(batchnorm89_1_1_1_5_)
        conv90_1_1_1_5_padding = self.conv90_1_1_1_5_padding(relu89_1_1_1_5_)
        conv90_1_1_1_5_ = self.conv90_1_1_1_5_(conv90_1_1_1_5_padding)
        batchnorm90_1_1_1_5_ = self.batchnorm90_1_1_1_5_(conv90_1_1_1_5_)
        relu90_1_1_1_5_ = self.relu90_1_1_1_5_(batchnorm90_1_1_1_5_)
        conv91_1_1_1_5_ = self.conv91_1_1_1_5_(relu90_1_1_1_5_)
        batchnorm91_1_1_1_5_ = self.batchnorm91_1_1_1_5_(conv91_1_1_1_5_)
        conv89_1_1_1_6_ = self.conv89_1_1_1_6_(relu86_1_)
        batchnorm89_1_1_1_6_ = self.batchnorm89_1_1_1_6_(conv89_1_1_1_6_)
        relu89_1_1_1_6_ = self.relu89_1_1_1_6_(batchnorm89_1_1_1_6_)
        conv90_1_1_1_6_padding = self.conv90_1_1_1_6_padding(relu89_1_1_1_6_)
        conv90_1_1_1_6_ = self.conv90_1_1_1_6_(conv90_1_1_1_6_padding)
        batchnorm90_1_1_1_6_ = self.batchnorm90_1_1_1_6_(conv90_1_1_1_6_)
        relu90_1_1_1_6_ = self.relu90_1_1_1_6_(batchnorm90_1_1_1_6_)
        conv91_1_1_1_6_ = self.conv91_1_1_1_6_(relu90_1_1_1_6_)
        batchnorm91_1_1_1_6_ = self.batchnorm91_1_1_1_6_(conv91_1_1_1_6_)
        conv89_1_1_1_7_ = self.conv89_1_1_1_7_(relu86_1_)
        batchnorm89_1_1_1_7_ = self.batchnorm89_1_1_1_7_(conv89_1_1_1_7_)
        relu89_1_1_1_7_ = self.relu89_1_1_1_7_(batchnorm89_1_1_1_7_)
        conv90_1_1_1_7_padding = self.conv90_1_1_1_7_padding(relu89_1_1_1_7_)
        conv90_1_1_1_7_ = self.conv90_1_1_1_7_(conv90_1_1_1_7_padding)
        batchnorm90_1_1_1_7_ = self.batchnorm90_1_1_1_7_(conv90_1_1_1_7_)
        relu90_1_1_1_7_ = self.relu90_1_1_1_7_(batchnorm90_1_1_1_7_)
        conv91_1_1_1_7_ = self.conv91_1_1_1_7_(relu90_1_1_1_7_)
        batchnorm91_1_1_1_7_ = self.batchnorm91_1_1_1_7_(conv91_1_1_1_7_)
        conv89_1_1_1_8_ = self.conv89_1_1_1_8_(relu86_1_)
        batchnorm89_1_1_1_8_ = self.batchnorm89_1_1_1_8_(conv89_1_1_1_8_)
        relu89_1_1_1_8_ = self.relu89_1_1_1_8_(batchnorm89_1_1_1_8_)
        conv90_1_1_1_8_padding = self.conv90_1_1_1_8_padding(relu89_1_1_1_8_)
        conv90_1_1_1_8_ = self.conv90_1_1_1_8_(conv90_1_1_1_8_padding)
        batchnorm90_1_1_1_8_ = self.batchnorm90_1_1_1_8_(conv90_1_1_1_8_)
        relu90_1_1_1_8_ = self.relu90_1_1_1_8_(batchnorm90_1_1_1_8_)
        conv91_1_1_1_8_ = self.conv91_1_1_1_8_(relu90_1_1_1_8_)
        batchnorm91_1_1_1_8_ = self.batchnorm91_1_1_1_8_(conv91_1_1_1_8_)
        conv89_1_1_1_9_ = self.conv89_1_1_1_9_(relu86_1_)
        batchnorm89_1_1_1_9_ = self.batchnorm89_1_1_1_9_(conv89_1_1_1_9_)
        relu89_1_1_1_9_ = self.relu89_1_1_1_9_(batchnorm89_1_1_1_9_)
        conv90_1_1_1_9_padding = self.conv90_1_1_1_9_padding(relu89_1_1_1_9_)
        conv90_1_1_1_9_ = self.conv90_1_1_1_9_(conv90_1_1_1_9_padding)
        batchnorm90_1_1_1_9_ = self.batchnorm90_1_1_1_9_(conv90_1_1_1_9_)
        relu90_1_1_1_9_ = self.relu90_1_1_1_9_(batchnorm90_1_1_1_9_)
        conv91_1_1_1_9_ = self.conv91_1_1_1_9_(relu90_1_1_1_9_)
        batchnorm91_1_1_1_9_ = self.batchnorm91_1_1_1_9_(conv91_1_1_1_9_)
        conv89_1_1_1_10_ = self.conv89_1_1_1_10_(relu86_1_)
        batchnorm89_1_1_1_10_ = self.batchnorm89_1_1_1_10_(conv89_1_1_1_10_)
        relu89_1_1_1_10_ = self.relu89_1_1_1_10_(batchnorm89_1_1_1_10_)
        conv90_1_1_1_10_padding = self.conv90_1_1_1_10_padding(relu89_1_1_1_10_)
        conv90_1_1_1_10_ = self.conv90_1_1_1_10_(conv90_1_1_1_10_padding)
        batchnorm90_1_1_1_10_ = self.batchnorm90_1_1_1_10_(conv90_1_1_1_10_)
        relu90_1_1_1_10_ = self.relu90_1_1_1_10_(batchnorm90_1_1_1_10_)
        conv91_1_1_1_10_ = self.conv91_1_1_1_10_(relu90_1_1_1_10_)
        batchnorm91_1_1_1_10_ = self.batchnorm91_1_1_1_10_(conv91_1_1_1_10_)
        conv89_1_1_1_11_ = self.conv89_1_1_1_11_(relu86_1_)
        batchnorm89_1_1_1_11_ = self.batchnorm89_1_1_1_11_(conv89_1_1_1_11_)
        relu89_1_1_1_11_ = self.relu89_1_1_1_11_(batchnorm89_1_1_1_11_)
        conv90_1_1_1_11_padding = self.conv90_1_1_1_11_padding(relu89_1_1_1_11_)
        conv90_1_1_1_11_ = self.conv90_1_1_1_11_(conv90_1_1_1_11_padding)
        batchnorm90_1_1_1_11_ = self.batchnorm90_1_1_1_11_(conv90_1_1_1_11_)
        relu90_1_1_1_11_ = self.relu90_1_1_1_11_(batchnorm90_1_1_1_11_)
        conv91_1_1_1_11_ = self.conv91_1_1_1_11_(relu90_1_1_1_11_)
        batchnorm91_1_1_1_11_ = self.batchnorm91_1_1_1_11_(conv91_1_1_1_11_)
        conv89_1_1_1_12_ = self.conv89_1_1_1_12_(relu86_1_)
        batchnorm89_1_1_1_12_ = self.batchnorm89_1_1_1_12_(conv89_1_1_1_12_)
        relu89_1_1_1_12_ = self.relu89_1_1_1_12_(batchnorm89_1_1_1_12_)
        conv90_1_1_1_12_padding = self.conv90_1_1_1_12_padding(relu89_1_1_1_12_)
        conv90_1_1_1_12_ = self.conv90_1_1_1_12_(conv90_1_1_1_12_padding)
        batchnorm90_1_1_1_12_ = self.batchnorm90_1_1_1_12_(conv90_1_1_1_12_)
        relu90_1_1_1_12_ = self.relu90_1_1_1_12_(batchnorm90_1_1_1_12_)
        conv91_1_1_1_12_ = self.conv91_1_1_1_12_(relu90_1_1_1_12_)
        batchnorm91_1_1_1_12_ = self.batchnorm91_1_1_1_12_(conv91_1_1_1_12_)
        conv89_1_1_1_13_ = self.conv89_1_1_1_13_(relu86_1_)
        batchnorm89_1_1_1_13_ = self.batchnorm89_1_1_1_13_(conv89_1_1_1_13_)
        relu89_1_1_1_13_ = self.relu89_1_1_1_13_(batchnorm89_1_1_1_13_)
        conv90_1_1_1_13_padding = self.conv90_1_1_1_13_padding(relu89_1_1_1_13_)
        conv90_1_1_1_13_ = self.conv90_1_1_1_13_(conv90_1_1_1_13_padding)
        batchnorm90_1_1_1_13_ = self.batchnorm90_1_1_1_13_(conv90_1_1_1_13_)
        relu90_1_1_1_13_ = self.relu90_1_1_1_13_(batchnorm90_1_1_1_13_)
        conv91_1_1_1_13_ = self.conv91_1_1_1_13_(relu90_1_1_1_13_)
        batchnorm91_1_1_1_13_ = self.batchnorm91_1_1_1_13_(conv91_1_1_1_13_)
        conv89_1_1_1_14_ = self.conv89_1_1_1_14_(relu86_1_)
        batchnorm89_1_1_1_14_ = self.batchnorm89_1_1_1_14_(conv89_1_1_1_14_)
        relu89_1_1_1_14_ = self.relu89_1_1_1_14_(batchnorm89_1_1_1_14_)
        conv90_1_1_1_14_padding = self.conv90_1_1_1_14_padding(relu89_1_1_1_14_)
        conv90_1_1_1_14_ = self.conv90_1_1_1_14_(conv90_1_1_1_14_padding)
        batchnorm90_1_1_1_14_ = self.batchnorm90_1_1_1_14_(conv90_1_1_1_14_)
        relu90_1_1_1_14_ = self.relu90_1_1_1_14_(batchnorm90_1_1_1_14_)
        conv91_1_1_1_14_ = self.conv91_1_1_1_14_(relu90_1_1_1_14_)
        batchnorm91_1_1_1_14_ = self.batchnorm91_1_1_1_14_(conv91_1_1_1_14_)
        conv89_1_1_1_15_ = self.conv89_1_1_1_15_(relu86_1_)
        batchnorm89_1_1_1_15_ = self.batchnorm89_1_1_1_15_(conv89_1_1_1_15_)
        relu89_1_1_1_15_ = self.relu89_1_1_1_15_(batchnorm89_1_1_1_15_)
        conv90_1_1_1_15_padding = self.conv90_1_1_1_15_padding(relu89_1_1_1_15_)
        conv90_1_1_1_15_ = self.conv90_1_1_1_15_(conv90_1_1_1_15_padding)
        batchnorm90_1_1_1_15_ = self.batchnorm90_1_1_1_15_(conv90_1_1_1_15_)
        relu90_1_1_1_15_ = self.relu90_1_1_1_15_(batchnorm90_1_1_1_15_)
        conv91_1_1_1_15_ = self.conv91_1_1_1_15_(relu90_1_1_1_15_)
        batchnorm91_1_1_1_15_ = self.batchnorm91_1_1_1_15_(conv91_1_1_1_15_)
        conv89_1_1_1_16_ = self.conv89_1_1_1_16_(relu86_1_)
        batchnorm89_1_1_1_16_ = self.batchnorm89_1_1_1_16_(conv89_1_1_1_16_)
        relu89_1_1_1_16_ = self.relu89_1_1_1_16_(batchnorm89_1_1_1_16_)
        conv90_1_1_1_16_padding = self.conv90_1_1_1_16_padding(relu89_1_1_1_16_)
        conv90_1_1_1_16_ = self.conv90_1_1_1_16_(conv90_1_1_1_16_padding)
        batchnorm90_1_1_1_16_ = self.batchnorm90_1_1_1_16_(conv90_1_1_1_16_)
        relu90_1_1_1_16_ = self.relu90_1_1_1_16_(batchnorm90_1_1_1_16_)
        conv91_1_1_1_16_ = self.conv91_1_1_1_16_(relu90_1_1_1_16_)
        batchnorm91_1_1_1_16_ = self.batchnorm91_1_1_1_16_(conv91_1_1_1_16_)
        conv89_1_1_1_17_ = self.conv89_1_1_1_17_(relu86_1_)
        batchnorm89_1_1_1_17_ = self.batchnorm89_1_1_1_17_(conv89_1_1_1_17_)
        relu89_1_1_1_17_ = self.relu89_1_1_1_17_(batchnorm89_1_1_1_17_)
        conv90_1_1_1_17_padding = self.conv90_1_1_1_17_padding(relu89_1_1_1_17_)
        conv90_1_1_1_17_ = self.conv90_1_1_1_17_(conv90_1_1_1_17_padding)
        batchnorm90_1_1_1_17_ = self.batchnorm90_1_1_1_17_(conv90_1_1_1_17_)
        relu90_1_1_1_17_ = self.relu90_1_1_1_17_(batchnorm90_1_1_1_17_)
        conv91_1_1_1_17_ = self.conv91_1_1_1_17_(relu90_1_1_1_17_)
        batchnorm91_1_1_1_17_ = self.batchnorm91_1_1_1_17_(conv91_1_1_1_17_)
        conv89_1_1_1_18_ = self.conv89_1_1_1_18_(relu86_1_)
        batchnorm89_1_1_1_18_ = self.batchnorm89_1_1_1_18_(conv89_1_1_1_18_)
        relu89_1_1_1_18_ = self.relu89_1_1_1_18_(batchnorm89_1_1_1_18_)
        conv90_1_1_1_18_padding = self.conv90_1_1_1_18_padding(relu89_1_1_1_18_)
        conv90_1_1_1_18_ = self.conv90_1_1_1_18_(conv90_1_1_1_18_padding)
        batchnorm90_1_1_1_18_ = self.batchnorm90_1_1_1_18_(conv90_1_1_1_18_)
        relu90_1_1_1_18_ = self.relu90_1_1_1_18_(batchnorm90_1_1_1_18_)
        conv91_1_1_1_18_ = self.conv91_1_1_1_18_(relu90_1_1_1_18_)
        batchnorm91_1_1_1_18_ = self.batchnorm91_1_1_1_18_(conv91_1_1_1_18_)
        conv89_1_1_1_19_ = self.conv89_1_1_1_19_(relu86_1_)
        batchnorm89_1_1_1_19_ = self.batchnorm89_1_1_1_19_(conv89_1_1_1_19_)
        relu89_1_1_1_19_ = self.relu89_1_1_1_19_(batchnorm89_1_1_1_19_)
        conv90_1_1_1_19_padding = self.conv90_1_1_1_19_padding(relu89_1_1_1_19_)
        conv90_1_1_1_19_ = self.conv90_1_1_1_19_(conv90_1_1_1_19_padding)
        batchnorm90_1_1_1_19_ = self.batchnorm90_1_1_1_19_(conv90_1_1_1_19_)
        relu90_1_1_1_19_ = self.relu90_1_1_1_19_(batchnorm90_1_1_1_19_)
        conv91_1_1_1_19_ = self.conv91_1_1_1_19_(relu90_1_1_1_19_)
        batchnorm91_1_1_1_19_ = self.batchnorm91_1_1_1_19_(conv91_1_1_1_19_)
        conv89_1_1_1_20_ = self.conv89_1_1_1_20_(relu86_1_)
        batchnorm89_1_1_1_20_ = self.batchnorm89_1_1_1_20_(conv89_1_1_1_20_)
        relu89_1_1_1_20_ = self.relu89_1_1_1_20_(batchnorm89_1_1_1_20_)
        conv90_1_1_1_20_padding = self.conv90_1_1_1_20_padding(relu89_1_1_1_20_)
        conv90_1_1_1_20_ = self.conv90_1_1_1_20_(conv90_1_1_1_20_padding)
        batchnorm90_1_1_1_20_ = self.batchnorm90_1_1_1_20_(conv90_1_1_1_20_)
        relu90_1_1_1_20_ = self.relu90_1_1_1_20_(batchnorm90_1_1_1_20_)
        conv91_1_1_1_20_ = self.conv91_1_1_1_20_(relu90_1_1_1_20_)
        batchnorm91_1_1_1_20_ = self.batchnorm91_1_1_1_20_(conv91_1_1_1_20_)
        conv89_1_1_1_21_ = self.conv89_1_1_1_21_(relu86_1_)
        batchnorm89_1_1_1_21_ = self.batchnorm89_1_1_1_21_(conv89_1_1_1_21_)
        relu89_1_1_1_21_ = self.relu89_1_1_1_21_(batchnorm89_1_1_1_21_)
        conv90_1_1_1_21_padding = self.conv90_1_1_1_21_padding(relu89_1_1_1_21_)
        conv90_1_1_1_21_ = self.conv90_1_1_1_21_(conv90_1_1_1_21_padding)
        batchnorm90_1_1_1_21_ = self.batchnorm90_1_1_1_21_(conv90_1_1_1_21_)
        relu90_1_1_1_21_ = self.relu90_1_1_1_21_(batchnorm90_1_1_1_21_)
        conv91_1_1_1_21_ = self.conv91_1_1_1_21_(relu90_1_1_1_21_)
        batchnorm91_1_1_1_21_ = self.batchnorm91_1_1_1_21_(conv91_1_1_1_21_)
        conv89_1_1_1_22_ = self.conv89_1_1_1_22_(relu86_1_)
        batchnorm89_1_1_1_22_ = self.batchnorm89_1_1_1_22_(conv89_1_1_1_22_)
        relu89_1_1_1_22_ = self.relu89_1_1_1_22_(batchnorm89_1_1_1_22_)
        conv90_1_1_1_22_padding = self.conv90_1_1_1_22_padding(relu89_1_1_1_22_)
        conv90_1_1_1_22_ = self.conv90_1_1_1_22_(conv90_1_1_1_22_padding)
        batchnorm90_1_1_1_22_ = self.batchnorm90_1_1_1_22_(conv90_1_1_1_22_)
        relu90_1_1_1_22_ = self.relu90_1_1_1_22_(batchnorm90_1_1_1_22_)
        conv91_1_1_1_22_ = self.conv91_1_1_1_22_(relu90_1_1_1_22_)
        batchnorm91_1_1_1_22_ = self.batchnorm91_1_1_1_22_(conv91_1_1_1_22_)
        conv89_1_1_1_23_ = self.conv89_1_1_1_23_(relu86_1_)
        batchnorm89_1_1_1_23_ = self.batchnorm89_1_1_1_23_(conv89_1_1_1_23_)
        relu89_1_1_1_23_ = self.relu89_1_1_1_23_(batchnorm89_1_1_1_23_)
        conv90_1_1_1_23_padding = self.conv90_1_1_1_23_padding(relu89_1_1_1_23_)
        conv90_1_1_1_23_ = self.conv90_1_1_1_23_(conv90_1_1_1_23_padding)
        batchnorm90_1_1_1_23_ = self.batchnorm90_1_1_1_23_(conv90_1_1_1_23_)
        relu90_1_1_1_23_ = self.relu90_1_1_1_23_(batchnorm90_1_1_1_23_)
        conv91_1_1_1_23_ = self.conv91_1_1_1_23_(relu90_1_1_1_23_)
        batchnorm91_1_1_1_23_ = self.batchnorm91_1_1_1_23_(conv91_1_1_1_23_)
        conv89_1_1_1_24_ = self.conv89_1_1_1_24_(relu86_1_)
        batchnorm89_1_1_1_24_ = self.batchnorm89_1_1_1_24_(conv89_1_1_1_24_)
        relu89_1_1_1_24_ = self.relu89_1_1_1_24_(batchnorm89_1_1_1_24_)
        conv90_1_1_1_24_padding = self.conv90_1_1_1_24_padding(relu89_1_1_1_24_)
        conv90_1_1_1_24_ = self.conv90_1_1_1_24_(conv90_1_1_1_24_padding)
        batchnorm90_1_1_1_24_ = self.batchnorm90_1_1_1_24_(conv90_1_1_1_24_)
        relu90_1_1_1_24_ = self.relu90_1_1_1_24_(batchnorm90_1_1_1_24_)
        conv91_1_1_1_24_ = self.conv91_1_1_1_24_(relu90_1_1_1_24_)
        batchnorm91_1_1_1_24_ = self.batchnorm91_1_1_1_24_(conv91_1_1_1_24_)
        conv89_1_1_1_25_ = self.conv89_1_1_1_25_(relu86_1_)
        batchnorm89_1_1_1_25_ = self.batchnorm89_1_1_1_25_(conv89_1_1_1_25_)
        relu89_1_1_1_25_ = self.relu89_1_1_1_25_(batchnorm89_1_1_1_25_)
        conv90_1_1_1_25_padding = self.conv90_1_1_1_25_padding(relu89_1_1_1_25_)
        conv90_1_1_1_25_ = self.conv90_1_1_1_25_(conv90_1_1_1_25_padding)
        batchnorm90_1_1_1_25_ = self.batchnorm90_1_1_1_25_(conv90_1_1_1_25_)
        relu90_1_1_1_25_ = self.relu90_1_1_1_25_(batchnorm90_1_1_1_25_)
        conv91_1_1_1_25_ = self.conv91_1_1_1_25_(relu90_1_1_1_25_)
        batchnorm91_1_1_1_25_ = self.batchnorm91_1_1_1_25_(conv91_1_1_1_25_)
        conv89_1_1_1_26_ = self.conv89_1_1_1_26_(relu86_1_)
        batchnorm89_1_1_1_26_ = self.batchnorm89_1_1_1_26_(conv89_1_1_1_26_)
        relu89_1_1_1_26_ = self.relu89_1_1_1_26_(batchnorm89_1_1_1_26_)
        conv90_1_1_1_26_padding = self.conv90_1_1_1_26_padding(relu89_1_1_1_26_)
        conv90_1_1_1_26_ = self.conv90_1_1_1_26_(conv90_1_1_1_26_padding)
        batchnorm90_1_1_1_26_ = self.batchnorm90_1_1_1_26_(conv90_1_1_1_26_)
        relu90_1_1_1_26_ = self.relu90_1_1_1_26_(batchnorm90_1_1_1_26_)
        conv91_1_1_1_26_ = self.conv91_1_1_1_26_(relu90_1_1_1_26_)
        batchnorm91_1_1_1_26_ = self.batchnorm91_1_1_1_26_(conv91_1_1_1_26_)
        conv89_1_1_1_27_ = self.conv89_1_1_1_27_(relu86_1_)
        batchnorm89_1_1_1_27_ = self.batchnorm89_1_1_1_27_(conv89_1_1_1_27_)
        relu89_1_1_1_27_ = self.relu89_1_1_1_27_(batchnorm89_1_1_1_27_)
        conv90_1_1_1_27_padding = self.conv90_1_1_1_27_padding(relu89_1_1_1_27_)
        conv90_1_1_1_27_ = self.conv90_1_1_1_27_(conv90_1_1_1_27_padding)
        batchnorm90_1_1_1_27_ = self.batchnorm90_1_1_1_27_(conv90_1_1_1_27_)
        relu90_1_1_1_27_ = self.relu90_1_1_1_27_(batchnorm90_1_1_1_27_)
        conv91_1_1_1_27_ = self.conv91_1_1_1_27_(relu90_1_1_1_27_)
        batchnorm91_1_1_1_27_ = self.batchnorm91_1_1_1_27_(conv91_1_1_1_27_)
        conv89_1_1_1_28_ = self.conv89_1_1_1_28_(relu86_1_)
        batchnorm89_1_1_1_28_ = self.batchnorm89_1_1_1_28_(conv89_1_1_1_28_)
        relu89_1_1_1_28_ = self.relu89_1_1_1_28_(batchnorm89_1_1_1_28_)
        conv90_1_1_1_28_padding = self.conv90_1_1_1_28_padding(relu89_1_1_1_28_)
        conv90_1_1_1_28_ = self.conv90_1_1_1_28_(conv90_1_1_1_28_padding)
        batchnorm90_1_1_1_28_ = self.batchnorm90_1_1_1_28_(conv90_1_1_1_28_)
        relu90_1_1_1_28_ = self.relu90_1_1_1_28_(batchnorm90_1_1_1_28_)
        conv91_1_1_1_28_ = self.conv91_1_1_1_28_(relu90_1_1_1_28_)
        batchnorm91_1_1_1_28_ = self.batchnorm91_1_1_1_28_(conv91_1_1_1_28_)
        conv89_1_1_1_29_ = self.conv89_1_1_1_29_(relu86_1_)
        batchnorm89_1_1_1_29_ = self.batchnorm89_1_1_1_29_(conv89_1_1_1_29_)
        relu89_1_1_1_29_ = self.relu89_1_1_1_29_(batchnorm89_1_1_1_29_)
        conv90_1_1_1_29_padding = self.conv90_1_1_1_29_padding(relu89_1_1_1_29_)
        conv90_1_1_1_29_ = self.conv90_1_1_1_29_(conv90_1_1_1_29_padding)
        batchnorm90_1_1_1_29_ = self.batchnorm90_1_1_1_29_(conv90_1_1_1_29_)
        relu90_1_1_1_29_ = self.relu90_1_1_1_29_(batchnorm90_1_1_1_29_)
        conv91_1_1_1_29_ = self.conv91_1_1_1_29_(relu90_1_1_1_29_)
        batchnorm91_1_1_1_29_ = self.batchnorm91_1_1_1_29_(conv91_1_1_1_29_)
        conv89_1_1_1_30_ = self.conv89_1_1_1_30_(relu86_1_)
        batchnorm89_1_1_1_30_ = self.batchnorm89_1_1_1_30_(conv89_1_1_1_30_)
        relu89_1_1_1_30_ = self.relu89_1_1_1_30_(batchnorm89_1_1_1_30_)
        conv90_1_1_1_30_padding = self.conv90_1_1_1_30_padding(relu89_1_1_1_30_)
        conv90_1_1_1_30_ = self.conv90_1_1_1_30_(conv90_1_1_1_30_padding)
        batchnorm90_1_1_1_30_ = self.batchnorm90_1_1_1_30_(conv90_1_1_1_30_)
        relu90_1_1_1_30_ = self.relu90_1_1_1_30_(batchnorm90_1_1_1_30_)
        conv91_1_1_1_30_ = self.conv91_1_1_1_30_(relu90_1_1_1_30_)
        batchnorm91_1_1_1_30_ = self.batchnorm91_1_1_1_30_(conv91_1_1_1_30_)
        conv89_1_1_1_31_ = self.conv89_1_1_1_31_(relu86_1_)
        batchnorm89_1_1_1_31_ = self.batchnorm89_1_1_1_31_(conv89_1_1_1_31_)
        relu89_1_1_1_31_ = self.relu89_1_1_1_31_(batchnorm89_1_1_1_31_)
        conv90_1_1_1_31_padding = self.conv90_1_1_1_31_padding(relu89_1_1_1_31_)
        conv90_1_1_1_31_ = self.conv90_1_1_1_31_(conv90_1_1_1_31_padding)
        batchnorm90_1_1_1_31_ = self.batchnorm90_1_1_1_31_(conv90_1_1_1_31_)
        relu90_1_1_1_31_ = self.relu90_1_1_1_31_(batchnorm90_1_1_1_31_)
        conv91_1_1_1_31_ = self.conv91_1_1_1_31_(relu90_1_1_1_31_)
        batchnorm91_1_1_1_31_ = self.batchnorm91_1_1_1_31_(conv91_1_1_1_31_)
        conv89_1_1_1_32_ = self.conv89_1_1_1_32_(relu86_1_)
        batchnorm89_1_1_1_32_ = self.batchnorm89_1_1_1_32_(conv89_1_1_1_32_)
        relu89_1_1_1_32_ = self.relu89_1_1_1_32_(batchnorm89_1_1_1_32_)
        conv90_1_1_1_32_padding = self.conv90_1_1_1_32_padding(relu89_1_1_1_32_)
        conv90_1_1_1_32_ = self.conv90_1_1_1_32_(conv90_1_1_1_32_padding)
        batchnorm90_1_1_1_32_ = self.batchnorm90_1_1_1_32_(conv90_1_1_1_32_)
        relu90_1_1_1_32_ = self.relu90_1_1_1_32_(batchnorm90_1_1_1_32_)
        conv91_1_1_1_32_ = self.conv91_1_1_1_32_(relu90_1_1_1_32_)
        batchnorm91_1_1_1_32_ = self.batchnorm91_1_1_1_32_(conv91_1_1_1_32_)
        add92_1_1_1_ = batchnorm91_1_1_1_1_ + batchnorm91_1_1_1_2_ + batchnorm91_1_1_1_3_ + batchnorm91_1_1_1_4_ + batchnorm91_1_1_1_5_ + batchnorm91_1_1_1_6_ + batchnorm91_1_1_1_7_ + batchnorm91_1_1_1_8_ + batchnorm91_1_1_1_9_ + batchnorm91_1_1_1_10_ + batchnorm91_1_1_1_11_ + batchnorm91_1_1_1_12_ + batchnorm91_1_1_1_13_ + batchnorm91_1_1_1_14_ + batchnorm91_1_1_1_15_ + batchnorm91_1_1_1_16_ + batchnorm91_1_1_1_17_ + batchnorm91_1_1_1_18_ + batchnorm91_1_1_1_19_ + batchnorm91_1_1_1_20_ + batchnorm91_1_1_1_21_ + batchnorm91_1_1_1_22_ + batchnorm91_1_1_1_23_ + batchnorm91_1_1_1_24_ + batchnorm91_1_1_1_25_ + batchnorm91_1_1_1_26_ + batchnorm91_1_1_1_27_ + batchnorm91_1_1_1_28_ + batchnorm91_1_1_1_29_ + batchnorm91_1_1_1_30_ + batchnorm91_1_1_1_31_ + batchnorm91_1_1_1_32_
        add93_1_1_ = add92_1_1_1_ + relu86_1_
        relu93_1_1_ = self.relu93_1_1_(add93_1_1_)
        conv95_1_1_1_1_ = self.conv95_1_1_1_1_(relu93_1_1_)
        batchnorm95_1_1_1_1_ = self.batchnorm95_1_1_1_1_(conv95_1_1_1_1_)
        relu95_1_1_1_1_ = self.relu95_1_1_1_1_(batchnorm95_1_1_1_1_)
        conv96_1_1_1_1_padding = self.conv96_1_1_1_1_padding(relu95_1_1_1_1_)
        conv96_1_1_1_1_ = self.conv96_1_1_1_1_(conv96_1_1_1_1_padding)
        batchnorm96_1_1_1_1_ = self.batchnorm96_1_1_1_1_(conv96_1_1_1_1_)
        relu96_1_1_1_1_ = self.relu96_1_1_1_1_(batchnorm96_1_1_1_1_)
        conv97_1_1_1_1_ = self.conv97_1_1_1_1_(relu96_1_1_1_1_)
        batchnorm97_1_1_1_1_ = self.batchnorm97_1_1_1_1_(conv97_1_1_1_1_)
        conv95_1_1_1_2_ = self.conv95_1_1_1_2_(relu93_1_1_)
        batchnorm95_1_1_1_2_ = self.batchnorm95_1_1_1_2_(conv95_1_1_1_2_)
        relu95_1_1_1_2_ = self.relu95_1_1_1_2_(batchnorm95_1_1_1_2_)
        conv96_1_1_1_2_padding = self.conv96_1_1_1_2_padding(relu95_1_1_1_2_)
        conv96_1_1_1_2_ = self.conv96_1_1_1_2_(conv96_1_1_1_2_padding)
        batchnorm96_1_1_1_2_ = self.batchnorm96_1_1_1_2_(conv96_1_1_1_2_)
        relu96_1_1_1_2_ = self.relu96_1_1_1_2_(batchnorm96_1_1_1_2_)
        conv97_1_1_1_2_ = self.conv97_1_1_1_2_(relu96_1_1_1_2_)
        batchnorm97_1_1_1_2_ = self.batchnorm97_1_1_1_2_(conv97_1_1_1_2_)
        conv95_1_1_1_3_ = self.conv95_1_1_1_3_(relu93_1_1_)
        batchnorm95_1_1_1_3_ = self.batchnorm95_1_1_1_3_(conv95_1_1_1_3_)
        relu95_1_1_1_3_ = self.relu95_1_1_1_3_(batchnorm95_1_1_1_3_)
        conv96_1_1_1_3_padding = self.conv96_1_1_1_3_padding(relu95_1_1_1_3_)
        conv96_1_1_1_3_ = self.conv96_1_1_1_3_(conv96_1_1_1_3_padding)
        batchnorm96_1_1_1_3_ = self.batchnorm96_1_1_1_3_(conv96_1_1_1_3_)
        relu96_1_1_1_3_ = self.relu96_1_1_1_3_(batchnorm96_1_1_1_3_)
        conv97_1_1_1_3_ = self.conv97_1_1_1_3_(relu96_1_1_1_3_)
        batchnorm97_1_1_1_3_ = self.batchnorm97_1_1_1_3_(conv97_1_1_1_3_)
        conv95_1_1_1_4_ = self.conv95_1_1_1_4_(relu93_1_1_)
        batchnorm95_1_1_1_4_ = self.batchnorm95_1_1_1_4_(conv95_1_1_1_4_)
        relu95_1_1_1_4_ = self.relu95_1_1_1_4_(batchnorm95_1_1_1_4_)
        conv96_1_1_1_4_padding = self.conv96_1_1_1_4_padding(relu95_1_1_1_4_)
        conv96_1_1_1_4_ = self.conv96_1_1_1_4_(conv96_1_1_1_4_padding)
        batchnorm96_1_1_1_4_ = self.batchnorm96_1_1_1_4_(conv96_1_1_1_4_)
        relu96_1_1_1_4_ = self.relu96_1_1_1_4_(batchnorm96_1_1_1_4_)
        conv97_1_1_1_4_ = self.conv97_1_1_1_4_(relu96_1_1_1_4_)
        batchnorm97_1_1_1_4_ = self.batchnorm97_1_1_1_4_(conv97_1_1_1_4_)
        conv95_1_1_1_5_ = self.conv95_1_1_1_5_(relu93_1_1_)
        batchnorm95_1_1_1_5_ = self.batchnorm95_1_1_1_5_(conv95_1_1_1_5_)
        relu95_1_1_1_5_ = self.relu95_1_1_1_5_(batchnorm95_1_1_1_5_)
        conv96_1_1_1_5_padding = self.conv96_1_1_1_5_padding(relu95_1_1_1_5_)
        conv96_1_1_1_5_ = self.conv96_1_1_1_5_(conv96_1_1_1_5_padding)
        batchnorm96_1_1_1_5_ = self.batchnorm96_1_1_1_5_(conv96_1_1_1_5_)
        relu96_1_1_1_5_ = self.relu96_1_1_1_5_(batchnorm96_1_1_1_5_)
        conv97_1_1_1_5_ = self.conv97_1_1_1_5_(relu96_1_1_1_5_)
        batchnorm97_1_1_1_5_ = self.batchnorm97_1_1_1_5_(conv97_1_1_1_5_)
        conv95_1_1_1_6_ = self.conv95_1_1_1_6_(relu93_1_1_)
        batchnorm95_1_1_1_6_ = self.batchnorm95_1_1_1_6_(conv95_1_1_1_6_)
        relu95_1_1_1_6_ = self.relu95_1_1_1_6_(batchnorm95_1_1_1_6_)
        conv96_1_1_1_6_padding = self.conv96_1_1_1_6_padding(relu95_1_1_1_6_)
        conv96_1_1_1_6_ = self.conv96_1_1_1_6_(conv96_1_1_1_6_padding)
        batchnorm96_1_1_1_6_ = self.batchnorm96_1_1_1_6_(conv96_1_1_1_6_)
        relu96_1_1_1_6_ = self.relu96_1_1_1_6_(batchnorm96_1_1_1_6_)
        conv97_1_1_1_6_ = self.conv97_1_1_1_6_(relu96_1_1_1_6_)
        batchnorm97_1_1_1_6_ = self.batchnorm97_1_1_1_6_(conv97_1_1_1_6_)
        conv95_1_1_1_7_ = self.conv95_1_1_1_7_(relu93_1_1_)
        batchnorm95_1_1_1_7_ = self.batchnorm95_1_1_1_7_(conv95_1_1_1_7_)
        relu95_1_1_1_7_ = self.relu95_1_1_1_7_(batchnorm95_1_1_1_7_)
        conv96_1_1_1_7_padding = self.conv96_1_1_1_7_padding(relu95_1_1_1_7_)
        conv96_1_1_1_7_ = self.conv96_1_1_1_7_(conv96_1_1_1_7_padding)
        batchnorm96_1_1_1_7_ = self.batchnorm96_1_1_1_7_(conv96_1_1_1_7_)
        relu96_1_1_1_7_ = self.relu96_1_1_1_7_(batchnorm96_1_1_1_7_)
        conv97_1_1_1_7_ = self.conv97_1_1_1_7_(relu96_1_1_1_7_)
        batchnorm97_1_1_1_7_ = self.batchnorm97_1_1_1_7_(conv97_1_1_1_7_)
        conv95_1_1_1_8_ = self.conv95_1_1_1_8_(relu93_1_1_)
        batchnorm95_1_1_1_8_ = self.batchnorm95_1_1_1_8_(conv95_1_1_1_8_)
        relu95_1_1_1_8_ = self.relu95_1_1_1_8_(batchnorm95_1_1_1_8_)
        conv96_1_1_1_8_padding = self.conv96_1_1_1_8_padding(relu95_1_1_1_8_)
        conv96_1_1_1_8_ = self.conv96_1_1_1_8_(conv96_1_1_1_8_padding)
        batchnorm96_1_1_1_8_ = self.batchnorm96_1_1_1_8_(conv96_1_1_1_8_)
        relu96_1_1_1_8_ = self.relu96_1_1_1_8_(batchnorm96_1_1_1_8_)
        conv97_1_1_1_8_ = self.conv97_1_1_1_8_(relu96_1_1_1_8_)
        batchnorm97_1_1_1_8_ = self.batchnorm97_1_1_1_8_(conv97_1_1_1_8_)
        conv95_1_1_1_9_ = self.conv95_1_1_1_9_(relu93_1_1_)
        batchnorm95_1_1_1_9_ = self.batchnorm95_1_1_1_9_(conv95_1_1_1_9_)
        relu95_1_1_1_9_ = self.relu95_1_1_1_9_(batchnorm95_1_1_1_9_)
        conv96_1_1_1_9_padding = self.conv96_1_1_1_9_padding(relu95_1_1_1_9_)
        conv96_1_1_1_9_ = self.conv96_1_1_1_9_(conv96_1_1_1_9_padding)
        batchnorm96_1_1_1_9_ = self.batchnorm96_1_1_1_9_(conv96_1_1_1_9_)
        relu96_1_1_1_9_ = self.relu96_1_1_1_9_(batchnorm96_1_1_1_9_)
        conv97_1_1_1_9_ = self.conv97_1_1_1_9_(relu96_1_1_1_9_)
        batchnorm97_1_1_1_9_ = self.batchnorm97_1_1_1_9_(conv97_1_1_1_9_)
        conv95_1_1_1_10_ = self.conv95_1_1_1_10_(relu93_1_1_)
        batchnorm95_1_1_1_10_ = self.batchnorm95_1_1_1_10_(conv95_1_1_1_10_)
        relu95_1_1_1_10_ = self.relu95_1_1_1_10_(batchnorm95_1_1_1_10_)
        conv96_1_1_1_10_padding = self.conv96_1_1_1_10_padding(relu95_1_1_1_10_)
        conv96_1_1_1_10_ = self.conv96_1_1_1_10_(conv96_1_1_1_10_padding)
        batchnorm96_1_1_1_10_ = self.batchnorm96_1_1_1_10_(conv96_1_1_1_10_)
        relu96_1_1_1_10_ = self.relu96_1_1_1_10_(batchnorm96_1_1_1_10_)
        conv97_1_1_1_10_ = self.conv97_1_1_1_10_(relu96_1_1_1_10_)
        batchnorm97_1_1_1_10_ = self.batchnorm97_1_1_1_10_(conv97_1_1_1_10_)
        conv95_1_1_1_11_ = self.conv95_1_1_1_11_(relu93_1_1_)
        batchnorm95_1_1_1_11_ = self.batchnorm95_1_1_1_11_(conv95_1_1_1_11_)
        relu95_1_1_1_11_ = self.relu95_1_1_1_11_(batchnorm95_1_1_1_11_)
        conv96_1_1_1_11_padding = self.conv96_1_1_1_11_padding(relu95_1_1_1_11_)
        conv96_1_1_1_11_ = self.conv96_1_1_1_11_(conv96_1_1_1_11_padding)
        batchnorm96_1_1_1_11_ = self.batchnorm96_1_1_1_11_(conv96_1_1_1_11_)
        relu96_1_1_1_11_ = self.relu96_1_1_1_11_(batchnorm96_1_1_1_11_)
        conv97_1_1_1_11_ = self.conv97_1_1_1_11_(relu96_1_1_1_11_)
        batchnorm97_1_1_1_11_ = self.batchnorm97_1_1_1_11_(conv97_1_1_1_11_)
        conv95_1_1_1_12_ = self.conv95_1_1_1_12_(relu93_1_1_)
        batchnorm95_1_1_1_12_ = self.batchnorm95_1_1_1_12_(conv95_1_1_1_12_)
        relu95_1_1_1_12_ = self.relu95_1_1_1_12_(batchnorm95_1_1_1_12_)
        conv96_1_1_1_12_padding = self.conv96_1_1_1_12_padding(relu95_1_1_1_12_)
        conv96_1_1_1_12_ = self.conv96_1_1_1_12_(conv96_1_1_1_12_padding)
        batchnorm96_1_1_1_12_ = self.batchnorm96_1_1_1_12_(conv96_1_1_1_12_)
        relu96_1_1_1_12_ = self.relu96_1_1_1_12_(batchnorm96_1_1_1_12_)
        conv97_1_1_1_12_ = self.conv97_1_1_1_12_(relu96_1_1_1_12_)
        batchnorm97_1_1_1_12_ = self.batchnorm97_1_1_1_12_(conv97_1_1_1_12_)
        conv95_1_1_1_13_ = self.conv95_1_1_1_13_(relu93_1_1_)
        batchnorm95_1_1_1_13_ = self.batchnorm95_1_1_1_13_(conv95_1_1_1_13_)
        relu95_1_1_1_13_ = self.relu95_1_1_1_13_(batchnorm95_1_1_1_13_)
        conv96_1_1_1_13_padding = self.conv96_1_1_1_13_padding(relu95_1_1_1_13_)
        conv96_1_1_1_13_ = self.conv96_1_1_1_13_(conv96_1_1_1_13_padding)
        batchnorm96_1_1_1_13_ = self.batchnorm96_1_1_1_13_(conv96_1_1_1_13_)
        relu96_1_1_1_13_ = self.relu96_1_1_1_13_(batchnorm96_1_1_1_13_)
        conv97_1_1_1_13_ = self.conv97_1_1_1_13_(relu96_1_1_1_13_)
        batchnorm97_1_1_1_13_ = self.batchnorm97_1_1_1_13_(conv97_1_1_1_13_)
        conv95_1_1_1_14_ = self.conv95_1_1_1_14_(relu93_1_1_)
        batchnorm95_1_1_1_14_ = self.batchnorm95_1_1_1_14_(conv95_1_1_1_14_)
        relu95_1_1_1_14_ = self.relu95_1_1_1_14_(batchnorm95_1_1_1_14_)
        conv96_1_1_1_14_padding = self.conv96_1_1_1_14_padding(relu95_1_1_1_14_)
        conv96_1_1_1_14_ = self.conv96_1_1_1_14_(conv96_1_1_1_14_padding)
        batchnorm96_1_1_1_14_ = self.batchnorm96_1_1_1_14_(conv96_1_1_1_14_)
        relu96_1_1_1_14_ = self.relu96_1_1_1_14_(batchnorm96_1_1_1_14_)
        conv97_1_1_1_14_ = self.conv97_1_1_1_14_(relu96_1_1_1_14_)
        batchnorm97_1_1_1_14_ = self.batchnorm97_1_1_1_14_(conv97_1_1_1_14_)
        conv95_1_1_1_15_ = self.conv95_1_1_1_15_(relu93_1_1_)
        batchnorm95_1_1_1_15_ = self.batchnorm95_1_1_1_15_(conv95_1_1_1_15_)
        relu95_1_1_1_15_ = self.relu95_1_1_1_15_(batchnorm95_1_1_1_15_)
        conv96_1_1_1_15_padding = self.conv96_1_1_1_15_padding(relu95_1_1_1_15_)
        conv96_1_1_1_15_ = self.conv96_1_1_1_15_(conv96_1_1_1_15_padding)
        batchnorm96_1_1_1_15_ = self.batchnorm96_1_1_1_15_(conv96_1_1_1_15_)
        relu96_1_1_1_15_ = self.relu96_1_1_1_15_(batchnorm96_1_1_1_15_)
        conv97_1_1_1_15_ = self.conv97_1_1_1_15_(relu96_1_1_1_15_)
        batchnorm97_1_1_1_15_ = self.batchnorm97_1_1_1_15_(conv97_1_1_1_15_)
        conv95_1_1_1_16_ = self.conv95_1_1_1_16_(relu93_1_1_)
        batchnorm95_1_1_1_16_ = self.batchnorm95_1_1_1_16_(conv95_1_1_1_16_)
        relu95_1_1_1_16_ = self.relu95_1_1_1_16_(batchnorm95_1_1_1_16_)
        conv96_1_1_1_16_padding = self.conv96_1_1_1_16_padding(relu95_1_1_1_16_)
        conv96_1_1_1_16_ = self.conv96_1_1_1_16_(conv96_1_1_1_16_padding)
        batchnorm96_1_1_1_16_ = self.batchnorm96_1_1_1_16_(conv96_1_1_1_16_)
        relu96_1_1_1_16_ = self.relu96_1_1_1_16_(batchnorm96_1_1_1_16_)
        conv97_1_1_1_16_ = self.conv97_1_1_1_16_(relu96_1_1_1_16_)
        batchnorm97_1_1_1_16_ = self.batchnorm97_1_1_1_16_(conv97_1_1_1_16_)
        conv95_1_1_1_17_ = self.conv95_1_1_1_17_(relu93_1_1_)
        batchnorm95_1_1_1_17_ = self.batchnorm95_1_1_1_17_(conv95_1_1_1_17_)
        relu95_1_1_1_17_ = self.relu95_1_1_1_17_(batchnorm95_1_1_1_17_)
        conv96_1_1_1_17_padding = self.conv96_1_1_1_17_padding(relu95_1_1_1_17_)
        conv96_1_1_1_17_ = self.conv96_1_1_1_17_(conv96_1_1_1_17_padding)
        batchnorm96_1_1_1_17_ = self.batchnorm96_1_1_1_17_(conv96_1_1_1_17_)
        relu96_1_1_1_17_ = self.relu96_1_1_1_17_(batchnorm96_1_1_1_17_)
        conv97_1_1_1_17_ = self.conv97_1_1_1_17_(relu96_1_1_1_17_)
        batchnorm97_1_1_1_17_ = self.batchnorm97_1_1_1_17_(conv97_1_1_1_17_)
        conv95_1_1_1_18_ = self.conv95_1_1_1_18_(relu93_1_1_)
        batchnorm95_1_1_1_18_ = self.batchnorm95_1_1_1_18_(conv95_1_1_1_18_)
        relu95_1_1_1_18_ = self.relu95_1_1_1_18_(batchnorm95_1_1_1_18_)
        conv96_1_1_1_18_padding = self.conv96_1_1_1_18_padding(relu95_1_1_1_18_)
        conv96_1_1_1_18_ = self.conv96_1_1_1_18_(conv96_1_1_1_18_padding)
        batchnorm96_1_1_1_18_ = self.batchnorm96_1_1_1_18_(conv96_1_1_1_18_)
        relu96_1_1_1_18_ = self.relu96_1_1_1_18_(batchnorm96_1_1_1_18_)
        conv97_1_1_1_18_ = self.conv97_1_1_1_18_(relu96_1_1_1_18_)
        batchnorm97_1_1_1_18_ = self.batchnorm97_1_1_1_18_(conv97_1_1_1_18_)
        conv95_1_1_1_19_ = self.conv95_1_1_1_19_(relu93_1_1_)
        batchnorm95_1_1_1_19_ = self.batchnorm95_1_1_1_19_(conv95_1_1_1_19_)
        relu95_1_1_1_19_ = self.relu95_1_1_1_19_(batchnorm95_1_1_1_19_)
        conv96_1_1_1_19_padding = self.conv96_1_1_1_19_padding(relu95_1_1_1_19_)
        conv96_1_1_1_19_ = self.conv96_1_1_1_19_(conv96_1_1_1_19_padding)
        batchnorm96_1_1_1_19_ = self.batchnorm96_1_1_1_19_(conv96_1_1_1_19_)
        relu96_1_1_1_19_ = self.relu96_1_1_1_19_(batchnorm96_1_1_1_19_)
        conv97_1_1_1_19_ = self.conv97_1_1_1_19_(relu96_1_1_1_19_)
        batchnorm97_1_1_1_19_ = self.batchnorm97_1_1_1_19_(conv97_1_1_1_19_)
        conv95_1_1_1_20_ = self.conv95_1_1_1_20_(relu93_1_1_)
        batchnorm95_1_1_1_20_ = self.batchnorm95_1_1_1_20_(conv95_1_1_1_20_)
        relu95_1_1_1_20_ = self.relu95_1_1_1_20_(batchnorm95_1_1_1_20_)
        conv96_1_1_1_20_padding = self.conv96_1_1_1_20_padding(relu95_1_1_1_20_)
        conv96_1_1_1_20_ = self.conv96_1_1_1_20_(conv96_1_1_1_20_padding)
        batchnorm96_1_1_1_20_ = self.batchnorm96_1_1_1_20_(conv96_1_1_1_20_)
        relu96_1_1_1_20_ = self.relu96_1_1_1_20_(batchnorm96_1_1_1_20_)
        conv97_1_1_1_20_ = self.conv97_1_1_1_20_(relu96_1_1_1_20_)
        batchnorm97_1_1_1_20_ = self.batchnorm97_1_1_1_20_(conv97_1_1_1_20_)
        conv95_1_1_1_21_ = self.conv95_1_1_1_21_(relu93_1_1_)
        batchnorm95_1_1_1_21_ = self.batchnorm95_1_1_1_21_(conv95_1_1_1_21_)
        relu95_1_1_1_21_ = self.relu95_1_1_1_21_(batchnorm95_1_1_1_21_)
        conv96_1_1_1_21_padding = self.conv96_1_1_1_21_padding(relu95_1_1_1_21_)
        conv96_1_1_1_21_ = self.conv96_1_1_1_21_(conv96_1_1_1_21_padding)
        batchnorm96_1_1_1_21_ = self.batchnorm96_1_1_1_21_(conv96_1_1_1_21_)
        relu96_1_1_1_21_ = self.relu96_1_1_1_21_(batchnorm96_1_1_1_21_)
        conv97_1_1_1_21_ = self.conv97_1_1_1_21_(relu96_1_1_1_21_)
        batchnorm97_1_1_1_21_ = self.batchnorm97_1_1_1_21_(conv97_1_1_1_21_)
        conv95_1_1_1_22_ = self.conv95_1_1_1_22_(relu93_1_1_)
        batchnorm95_1_1_1_22_ = self.batchnorm95_1_1_1_22_(conv95_1_1_1_22_)
        relu95_1_1_1_22_ = self.relu95_1_1_1_22_(batchnorm95_1_1_1_22_)
        conv96_1_1_1_22_padding = self.conv96_1_1_1_22_padding(relu95_1_1_1_22_)
        conv96_1_1_1_22_ = self.conv96_1_1_1_22_(conv96_1_1_1_22_padding)
        batchnorm96_1_1_1_22_ = self.batchnorm96_1_1_1_22_(conv96_1_1_1_22_)
        relu96_1_1_1_22_ = self.relu96_1_1_1_22_(batchnorm96_1_1_1_22_)
        conv97_1_1_1_22_ = self.conv97_1_1_1_22_(relu96_1_1_1_22_)
        batchnorm97_1_1_1_22_ = self.batchnorm97_1_1_1_22_(conv97_1_1_1_22_)
        conv95_1_1_1_23_ = self.conv95_1_1_1_23_(relu93_1_1_)
        batchnorm95_1_1_1_23_ = self.batchnorm95_1_1_1_23_(conv95_1_1_1_23_)
        relu95_1_1_1_23_ = self.relu95_1_1_1_23_(batchnorm95_1_1_1_23_)
        conv96_1_1_1_23_padding = self.conv96_1_1_1_23_padding(relu95_1_1_1_23_)
        conv96_1_1_1_23_ = self.conv96_1_1_1_23_(conv96_1_1_1_23_padding)
        batchnorm96_1_1_1_23_ = self.batchnorm96_1_1_1_23_(conv96_1_1_1_23_)
        relu96_1_1_1_23_ = self.relu96_1_1_1_23_(batchnorm96_1_1_1_23_)
        conv97_1_1_1_23_ = self.conv97_1_1_1_23_(relu96_1_1_1_23_)
        batchnorm97_1_1_1_23_ = self.batchnorm97_1_1_1_23_(conv97_1_1_1_23_)
        conv95_1_1_1_24_ = self.conv95_1_1_1_24_(relu93_1_1_)
        batchnorm95_1_1_1_24_ = self.batchnorm95_1_1_1_24_(conv95_1_1_1_24_)
        relu95_1_1_1_24_ = self.relu95_1_1_1_24_(batchnorm95_1_1_1_24_)
        conv96_1_1_1_24_padding = self.conv96_1_1_1_24_padding(relu95_1_1_1_24_)
        conv96_1_1_1_24_ = self.conv96_1_1_1_24_(conv96_1_1_1_24_padding)
        batchnorm96_1_1_1_24_ = self.batchnorm96_1_1_1_24_(conv96_1_1_1_24_)
        relu96_1_1_1_24_ = self.relu96_1_1_1_24_(batchnorm96_1_1_1_24_)
        conv97_1_1_1_24_ = self.conv97_1_1_1_24_(relu96_1_1_1_24_)
        batchnorm97_1_1_1_24_ = self.batchnorm97_1_1_1_24_(conv97_1_1_1_24_)
        conv95_1_1_1_25_ = self.conv95_1_1_1_25_(relu93_1_1_)
        batchnorm95_1_1_1_25_ = self.batchnorm95_1_1_1_25_(conv95_1_1_1_25_)
        relu95_1_1_1_25_ = self.relu95_1_1_1_25_(batchnorm95_1_1_1_25_)
        conv96_1_1_1_25_padding = self.conv96_1_1_1_25_padding(relu95_1_1_1_25_)
        conv96_1_1_1_25_ = self.conv96_1_1_1_25_(conv96_1_1_1_25_padding)
        batchnorm96_1_1_1_25_ = self.batchnorm96_1_1_1_25_(conv96_1_1_1_25_)
        relu96_1_1_1_25_ = self.relu96_1_1_1_25_(batchnorm96_1_1_1_25_)
        conv97_1_1_1_25_ = self.conv97_1_1_1_25_(relu96_1_1_1_25_)
        batchnorm97_1_1_1_25_ = self.batchnorm97_1_1_1_25_(conv97_1_1_1_25_)
        conv95_1_1_1_26_ = self.conv95_1_1_1_26_(relu93_1_1_)
        batchnorm95_1_1_1_26_ = self.batchnorm95_1_1_1_26_(conv95_1_1_1_26_)
        relu95_1_1_1_26_ = self.relu95_1_1_1_26_(batchnorm95_1_1_1_26_)
        conv96_1_1_1_26_padding = self.conv96_1_1_1_26_padding(relu95_1_1_1_26_)
        conv96_1_1_1_26_ = self.conv96_1_1_1_26_(conv96_1_1_1_26_padding)
        batchnorm96_1_1_1_26_ = self.batchnorm96_1_1_1_26_(conv96_1_1_1_26_)
        relu96_1_1_1_26_ = self.relu96_1_1_1_26_(batchnorm96_1_1_1_26_)
        conv97_1_1_1_26_ = self.conv97_1_1_1_26_(relu96_1_1_1_26_)
        batchnorm97_1_1_1_26_ = self.batchnorm97_1_1_1_26_(conv97_1_1_1_26_)
        conv95_1_1_1_27_ = self.conv95_1_1_1_27_(relu93_1_1_)
        batchnorm95_1_1_1_27_ = self.batchnorm95_1_1_1_27_(conv95_1_1_1_27_)
        relu95_1_1_1_27_ = self.relu95_1_1_1_27_(batchnorm95_1_1_1_27_)
        conv96_1_1_1_27_padding = self.conv96_1_1_1_27_padding(relu95_1_1_1_27_)
        conv96_1_1_1_27_ = self.conv96_1_1_1_27_(conv96_1_1_1_27_padding)
        batchnorm96_1_1_1_27_ = self.batchnorm96_1_1_1_27_(conv96_1_1_1_27_)
        relu96_1_1_1_27_ = self.relu96_1_1_1_27_(batchnorm96_1_1_1_27_)
        conv97_1_1_1_27_ = self.conv97_1_1_1_27_(relu96_1_1_1_27_)
        batchnorm97_1_1_1_27_ = self.batchnorm97_1_1_1_27_(conv97_1_1_1_27_)
        conv95_1_1_1_28_ = self.conv95_1_1_1_28_(relu93_1_1_)
        batchnorm95_1_1_1_28_ = self.batchnorm95_1_1_1_28_(conv95_1_1_1_28_)
        relu95_1_1_1_28_ = self.relu95_1_1_1_28_(batchnorm95_1_1_1_28_)
        conv96_1_1_1_28_padding = self.conv96_1_1_1_28_padding(relu95_1_1_1_28_)
        conv96_1_1_1_28_ = self.conv96_1_1_1_28_(conv96_1_1_1_28_padding)
        batchnorm96_1_1_1_28_ = self.batchnorm96_1_1_1_28_(conv96_1_1_1_28_)
        relu96_1_1_1_28_ = self.relu96_1_1_1_28_(batchnorm96_1_1_1_28_)
        conv97_1_1_1_28_ = self.conv97_1_1_1_28_(relu96_1_1_1_28_)
        batchnorm97_1_1_1_28_ = self.batchnorm97_1_1_1_28_(conv97_1_1_1_28_)
        conv95_1_1_1_29_ = self.conv95_1_1_1_29_(relu93_1_1_)
        batchnorm95_1_1_1_29_ = self.batchnorm95_1_1_1_29_(conv95_1_1_1_29_)
        relu95_1_1_1_29_ = self.relu95_1_1_1_29_(batchnorm95_1_1_1_29_)
        conv96_1_1_1_29_padding = self.conv96_1_1_1_29_padding(relu95_1_1_1_29_)
        conv96_1_1_1_29_ = self.conv96_1_1_1_29_(conv96_1_1_1_29_padding)
        batchnorm96_1_1_1_29_ = self.batchnorm96_1_1_1_29_(conv96_1_1_1_29_)
        relu96_1_1_1_29_ = self.relu96_1_1_1_29_(batchnorm96_1_1_1_29_)
        conv97_1_1_1_29_ = self.conv97_1_1_1_29_(relu96_1_1_1_29_)
        batchnorm97_1_1_1_29_ = self.batchnorm97_1_1_1_29_(conv97_1_1_1_29_)
        conv95_1_1_1_30_ = self.conv95_1_1_1_30_(relu93_1_1_)
        batchnorm95_1_1_1_30_ = self.batchnorm95_1_1_1_30_(conv95_1_1_1_30_)
        relu95_1_1_1_30_ = self.relu95_1_1_1_30_(batchnorm95_1_1_1_30_)
        conv96_1_1_1_30_padding = self.conv96_1_1_1_30_padding(relu95_1_1_1_30_)
        conv96_1_1_1_30_ = self.conv96_1_1_1_30_(conv96_1_1_1_30_padding)
        batchnorm96_1_1_1_30_ = self.batchnorm96_1_1_1_30_(conv96_1_1_1_30_)
        relu96_1_1_1_30_ = self.relu96_1_1_1_30_(batchnorm96_1_1_1_30_)
        conv97_1_1_1_30_ = self.conv97_1_1_1_30_(relu96_1_1_1_30_)
        batchnorm97_1_1_1_30_ = self.batchnorm97_1_1_1_30_(conv97_1_1_1_30_)
        conv95_1_1_1_31_ = self.conv95_1_1_1_31_(relu93_1_1_)
        batchnorm95_1_1_1_31_ = self.batchnorm95_1_1_1_31_(conv95_1_1_1_31_)
        relu95_1_1_1_31_ = self.relu95_1_1_1_31_(batchnorm95_1_1_1_31_)
        conv96_1_1_1_31_padding = self.conv96_1_1_1_31_padding(relu95_1_1_1_31_)
        conv96_1_1_1_31_ = self.conv96_1_1_1_31_(conv96_1_1_1_31_padding)
        batchnorm96_1_1_1_31_ = self.batchnorm96_1_1_1_31_(conv96_1_1_1_31_)
        relu96_1_1_1_31_ = self.relu96_1_1_1_31_(batchnorm96_1_1_1_31_)
        conv97_1_1_1_31_ = self.conv97_1_1_1_31_(relu96_1_1_1_31_)
        batchnorm97_1_1_1_31_ = self.batchnorm97_1_1_1_31_(conv97_1_1_1_31_)
        conv95_1_1_1_32_ = self.conv95_1_1_1_32_(relu93_1_1_)
        batchnorm95_1_1_1_32_ = self.batchnorm95_1_1_1_32_(conv95_1_1_1_32_)
        relu95_1_1_1_32_ = self.relu95_1_1_1_32_(batchnorm95_1_1_1_32_)
        conv96_1_1_1_32_padding = self.conv96_1_1_1_32_padding(relu95_1_1_1_32_)
        conv96_1_1_1_32_ = self.conv96_1_1_1_32_(conv96_1_1_1_32_padding)
        batchnorm96_1_1_1_32_ = self.batchnorm96_1_1_1_32_(conv96_1_1_1_32_)
        relu96_1_1_1_32_ = self.relu96_1_1_1_32_(batchnorm96_1_1_1_32_)
        conv97_1_1_1_32_ = self.conv97_1_1_1_32_(relu96_1_1_1_32_)
        batchnorm97_1_1_1_32_ = self.batchnorm97_1_1_1_32_(conv97_1_1_1_32_)
        add98_1_1_1_ = batchnorm97_1_1_1_1_ + batchnorm97_1_1_1_2_ + batchnorm97_1_1_1_3_ + batchnorm97_1_1_1_4_ + batchnorm97_1_1_1_5_ + batchnorm97_1_1_1_6_ + batchnorm97_1_1_1_7_ + batchnorm97_1_1_1_8_ + batchnorm97_1_1_1_9_ + batchnorm97_1_1_1_10_ + batchnorm97_1_1_1_11_ + batchnorm97_1_1_1_12_ + batchnorm97_1_1_1_13_ + batchnorm97_1_1_1_14_ + batchnorm97_1_1_1_15_ + batchnorm97_1_1_1_16_ + batchnorm97_1_1_1_17_ + batchnorm97_1_1_1_18_ + batchnorm97_1_1_1_19_ + batchnorm97_1_1_1_20_ + batchnorm97_1_1_1_21_ + batchnorm97_1_1_1_22_ + batchnorm97_1_1_1_23_ + batchnorm97_1_1_1_24_ + batchnorm97_1_1_1_25_ + batchnorm97_1_1_1_26_ + batchnorm97_1_1_1_27_ + batchnorm97_1_1_1_28_ + batchnorm97_1_1_1_29_ + batchnorm97_1_1_1_30_ + batchnorm97_1_1_1_31_ + batchnorm97_1_1_1_32_
        add99_1_1_ = add98_1_1_1_ + relu93_1_1_
        relu99_1_1_ = self.relu99_1_1_(add99_1_1_)
        conv99_1_1_padding = self.conv99_1_1_padding(relu99_1_1_)
        conv99_1_1_ = self.conv99_1_1_(conv99_1_1_padding)
        batchnorm99_1_1_ = self.batchnorm99_1_1_(conv99_1_1_)
        relu100_1_1_ = self.relu100_1_1_(batchnorm99_1_1_)
        relu101_1_1_ = self.relu101_1_1_(relu100_1_1_)
        dropout101_1_1_ = self.dropout101_1_1_(relu101_1_1_)
        conv101_1_1_padding = self.conv101_1_1_padding(dropout101_1_1_)
        conv101_1_1_ = self.conv101_1_1_(conv101_1_1_padding)
        batchnorm101_1_1_ = self.batchnorm101_1_1_(conv101_1_1_)
        relu102_1_1_ = self.relu102_1_1_(batchnorm101_1_1_)
        relu103_1_1_ = self.relu103_1_1_(relu102_1_1_)
        dropout103_1_1_ = self.dropout103_1_1_(relu103_1_1_)
        conv103_1_1_ = self.conv103_1_1_(dropout103_1_1_)
        transconv103_1_1_ = self.transconv103_1_1_(conv103_1_1_)
        conv87_1_2_padding = self.conv87_1_2_padding(relu86_1_)
        conv87_1_2_ = self.conv87_1_2_(conv87_1_2_padding)
        concatenate104_1_ = F.concat(transconv103_1_1_, conv87_1_2_, dim=1)
        transconv104_1_ = self.transconv104_1_(concatenate104_1_)
        conv50_2_ = self.conv50_2_(relu49_)
        concatenate105_ = F.concat(transconv104_1_, conv50_2_, dim=1)
        transconv105_ = self.transconv105_(concatenate105_)
        softmax_ = F.identity(transconv105_)

        return softmax_

    def getInputs(self):
        inputs = {}
        input_dimensions = (3,480,480)
        input_domains = (float,-1.0,1.0)
        inputs["data_"] = input_domains + (input_dimensions,)
        return inputs

    def getOutputs(self):
        outputs = {}
        output_dimensions = (21,480,480)
        output_domains = (float,float('-inf'),float('inf'))
        outputs["softmax_"] = output_domains + (output_dimensions,)
        return outputs
