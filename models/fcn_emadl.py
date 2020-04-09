from __future__ import absolute_import

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


class Net_0(gluon.HybridBlock):
    def __init__(self, data_mean=None, data_std=None, **kwargs):
        super(Net_0, self).__init__(**kwargs)
        with self.name_scope():
            # if data_mean:
            if None:
                print(data_mean, data_std)
                assert(data_std)
                self.input_normalization_data_ = ZScoreNormalization(data_mean=data_mean['data_'],
                                                                               data_std=data_std['data_'])
            else:
                self.input_normalization_data_ = NoNormalization()

            self.conv1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv1_ = gluon.nn.Conv2D(channels=64,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv1_, output shape: {[64,480,480]}

            self.relu1_ = gluon.nn.Activation(activation='relu')
            self.conv2_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv2_ = gluon.nn.Conv2D(channels=64,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv2_, output shape: {[64,480,480]}

            self.relu2_ = gluon.nn.Activation(activation='relu')
            self.pool2_ = gluon.nn.MaxPool2D(
                pool_size=(2,2),
                strides=(2,2))
            # pool2_, output shape: {[64,240,240]}

            self.conv3_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv3_ = gluon.nn.Conv2D(channels=128,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv3_, output shape: {[128,240,240]}

            self.relu3_ = gluon.nn.Activation(activation='relu')
            self.conv4_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_ = gluon.nn.Conv2D(channels=128,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv4_, output shape: {[128,240,240]}

            self.relu4_ = gluon.nn.Activation(activation='relu')
            self.pool4_ = gluon.nn.MaxPool2D(
                pool_size=(2,2),
                strides=(2,2))
            # pool4_, output shape: {[128,120,120]}

            self.conv5_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv5_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv5_, output shape: {[256,120,120]}

            self.relu5_ = gluon.nn.Activation(activation='relu')
            self.conv6_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv6_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv6_, output shape: {[256,120,120]}

            self.relu6_ = gluon.nn.Activation(activation='relu')
            self.conv7_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv7_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv7_, output shape: {[256,120,120]}

            self.relu7_ = gluon.nn.Activation(activation='relu')
            self.pool7_ = gluon.nn.MaxPool2D(
                pool_size=(2,2),
                strides=(2,2))
            # pool7_, output shape: {[256,60,60]}

            self.conv8_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv8_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv8_1_, output shape: {[512,60,60]}

            self.relu8_1_ = gluon.nn.Activation(activation='relu')
            self.conv9_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv9_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv9_1_, output shape: {[512,60,60]}

            self.relu9_1_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv10_1_, output shape: {[512,60,60]}

            self.relu10_1_ = gluon.nn.Activation(activation='relu')
            self.pool10_1_ = gluon.nn.MaxPool2D(
                pool_size=(2,2),
                strides=(2,2))
            # pool10_1_, output shape: {[512,30,30]}

            self.conv11_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv11_1_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv11_1_1_, output shape: {[512,30,30]}

            self.relu11_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv12_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv12_1_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv12_1_1_, output shape: {[512,30,30]}

            self.relu12_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv13_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv13_1_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv13_1_1_, output shape: {[512,30,30]}

            self.relu13_1_1_ = gluon.nn.Activation(activation='relu')
            self.pool13_1_1_ = gluon.nn.MaxPool2D(
                pool_size=(2,2),
                strides=(2,2))
            # pool13_1_1_, output shape: {[512,15,15]}

            self.conv14_1_1_padding = Padding(padding=(0,0,0,0,3,3,3,3))
            self.conv14_1_1_ = gluon.nn.Conv2D(channels=4096,
                kernel_size=(7,7),
                strides=(1,1),
                use_bias=True)
            # conv14_1_1_, output shape: {[4096,15,15]}

            self.relu14_1_1_ = gluon.nn.Activation(activation='relu')
            self.relu15_1_1_ = gluon.nn.Activation(activation='relu')
            self.dropout15_1_1_ = gluon.nn.Dropout(rate=0.5)
            self.conv15_1_1_padding = Padding(padding=(0,0,0,0,3,3,3,3))
            self.conv15_1_1_ = gluon.nn.Conv2D(channels=4096,
                kernel_size=(7,7),
                strides=(1,1),
                use_bias=True)
            # conv15_1_1_, output shape: {[4096,15,15]}

            self.relu16_1_1_ = gluon.nn.Activation(activation='relu')
            self.relu17_1_1_ = gluon.nn.Activation(activation='relu')
            self.dropout17_1_1_ = gluon.nn.Dropout(rate=0.5)
            self.conv17_1_1_ = gluon.nn.Conv2D(channels=21,
                kernel_size=(1,1),
                strides=(1,1),
                use_bias=True)
            # conv17_1_1_, output shape: {[21,15,15]}

            self.transconv17_1_1_padding = (1,1)
            self.transconv17_1_1_ = gluon.nn.Conv2DTranspose(channels=21,
                kernel_size=(4,4),
                strides=(2,2),
                padding=self.transconv17_1_1_padding,
                groups=21,
                use_bias=True)
            # transconv17_1_1_, output shape: {[21,30,30]}

            self.conv11_1_2_padding = Padding(padding=(0,0,0,0,2,1,2,1))
            self.conv11_1_2_ = gluon.nn.Conv2D(channels=21,
                kernel_size=(4,4),
                strides=(1,1),
                use_bias=True)
            # conv11_1_2_, output shape: {[21,30,30]}

            self.transconv18_1_padding = (1,1)
            self.transconv18_1_ = gluon.nn.Conv2DTranspose(channels=21,
                kernel_size=(4,4),
                strides=(2,2),
                padding=self.transconv18_1_padding,
                groups=21,
                use_bias=True)
            # transconv18_1_, output shape: {[21,60,60]}

            self.conv8_2_ = gluon.nn.Conv2D(channels=21,
                kernel_size=(1,1),
                strides=(1,1),
                use_bias=True)
            # conv8_2_, output shape: {[21,60,60]}

            self.transconv19_padding = (4,4)
            self.transconv19_ = gluon.nn.Conv2DTranspose(channels=21,
                kernel_size=(16,16),
                strides=(8,8),
                padding=self.transconv19_padding,
                groups=21,
                use_bias=True)
            # transconv19_, output shape: {[21,480,480]}


            pass

    def hybrid_forward(self, F, data_):
        data_ = self.input_normalization_data_(data_)
        conv1_padding = self.conv1_padding(data_)
        conv1_ = self.conv1_(conv1_padding)
        relu1_ = self.relu1_(conv1_)
        conv2_padding = self.conv2_padding(relu1_)
        conv2_ = self.conv2_(conv2_padding)
        relu2_ = self.relu2_(conv2_)
        pool2_ = self.pool2_(relu2_)
        conv3_padding = self.conv3_padding(pool2_)
        conv3_ = self.conv3_(conv3_padding)
        relu3_ = self.relu3_(conv3_)
        conv4_padding = self.conv4_padding(relu3_)
        conv4_ = self.conv4_(conv4_padding)
        relu4_ = self.relu4_(conv4_)
        pool4_ = self.pool4_(relu4_)
        conv5_padding = self.conv5_padding(pool4_)
        conv5_ = self.conv5_(conv5_padding)
        relu5_ = self.relu5_(conv5_)
        conv6_padding = self.conv6_padding(relu5_)
        conv6_ = self.conv6_(conv6_padding)
        relu6_ = self.relu6_(conv6_)
        conv7_padding = self.conv7_padding(relu6_)
        conv7_ = self.conv7_(conv7_padding)
        relu7_ = self.relu7_(conv7_)
        pool7_ = self.pool7_(relu7_)
        conv8_1_padding = self.conv8_1_padding(pool7_)
        conv8_1_ = self.conv8_1_(conv8_1_padding)
        relu8_1_ = self.relu8_1_(conv8_1_)
        conv9_1_padding = self.conv9_1_padding(relu8_1_)
        conv9_1_ = self.conv9_1_(conv9_1_padding)
        relu9_1_ = self.relu9_1_(conv9_1_)
        conv10_1_padding = self.conv10_1_padding(relu9_1_)
        conv10_1_ = self.conv10_1_(conv10_1_padding)
        relu10_1_ = self.relu10_1_(conv10_1_)
        pool10_1_ = self.pool10_1_(relu10_1_)
        conv11_1_1_padding = self.conv11_1_1_padding(pool10_1_)
        conv11_1_1_ = self.conv11_1_1_(conv11_1_1_padding)
        relu11_1_1_ = self.relu11_1_1_(conv11_1_1_)
        conv12_1_1_padding = self.conv12_1_1_padding(relu11_1_1_)
        conv12_1_1_ = self.conv12_1_1_(conv12_1_1_padding)
        relu12_1_1_ = self.relu12_1_1_(conv12_1_1_)
        conv13_1_1_padding = self.conv13_1_1_padding(relu12_1_1_)
        conv13_1_1_ = self.conv13_1_1_(conv13_1_1_padding)
        relu13_1_1_ = self.relu13_1_1_(conv13_1_1_)
        pool13_1_1_ = self.pool13_1_1_(relu13_1_1_)
        conv14_1_1_padding = self.conv14_1_1_padding(pool13_1_1_)
        conv14_1_1_ = self.conv14_1_1_(conv14_1_1_padding)
        relu14_1_1_ = self.relu14_1_1_(conv14_1_1_)
        relu15_1_1_ = self.relu15_1_1_(relu14_1_1_)
        dropout15_1_1_ = self.dropout15_1_1_(relu15_1_1_)
        conv15_1_1_padding = self.conv15_1_1_padding(dropout15_1_1_)
        conv15_1_1_ = self.conv15_1_1_(conv15_1_1_padding)
        relu16_1_1_ = self.relu16_1_1_(conv15_1_1_)
        relu17_1_1_ = self.relu17_1_1_(relu16_1_1_)
        dropout17_1_1_ = self.dropout17_1_1_(relu17_1_1_)
        conv17_1_1_ = self.conv17_1_1_(dropout17_1_1_)
        transconv17_1_1_ = self.transconv17_1_1_(conv17_1_1_)
        conv11_1_2_padding = self.conv11_1_2_padding(pool10_1_)
        conv11_1_2_ = self.conv11_1_2_(conv11_1_2_padding)
        concatenate18_1_ = F.concat(transconv17_1_1_, conv11_1_2_, dim=1)
        transconv18_1_ = self.transconv18_1_(concatenate18_1_)
        conv8_2_ = self.conv8_2_(pool7_)
        concatenate19_ = F.concat(transconv18_1_, conv8_2_, dim=1)
        transconv19_ = self.transconv19_(concatenate19_)
        softmax_ = F.identity(transconv19_)

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
