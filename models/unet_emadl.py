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


class UNET_EMADL(gluon.HybridBlock):
    def __init__(self, data_mean=None, data_std=None, **kwargs):
        super(UNET_EMADL, self).__init__(**kwargs)
        with self.name_scope():
            if False:
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
            self.pool3_1_ = gluon.nn.MaxPool2D(
                pool_size=(2,2),
                strides=(2,2))
            # pool3_1_, output shape: {[64,240,240]}

            self.conv3_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv3_1_ = gluon.nn.Conv2D(channels=128,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv3_1_, output shape: {[128,240,240]}

            self.relu3_1_ = gluon.nn.Activation(activation='relu')
            self.conv4_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv4_1_ = gluon.nn.Conv2D(channels=128,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv4_1_, output shape: {[128,240,240]}

            self.relu4_1_ = gluon.nn.Activation(activation='relu')
            self.pool5_1_1_ = gluon.nn.MaxPool2D(
                pool_size=(2,2),
                strides=(2,2))
            # pool5_1_1_, output shape: {[128,120,120]}

            self.conv5_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv5_1_1_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv5_1_1_, output shape: {[256,120,120]}

            self.relu5_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv6_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv6_1_1_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv6_1_1_, output shape: {[256,120,120]}

            self.relu6_1_1_ = gluon.nn.Activation(activation='relu')
            self.pool7_1_1_1_ = gluon.nn.MaxPool2D(
                pool_size=(2,2),
                strides=(2,2))
            # pool7_1_1_1_, output shape: {[256,60,60]}

            self.conv7_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv7_1_1_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv7_1_1_1_, output shape: {[512,60,60]}

            self.relu7_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv8_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv8_1_1_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv8_1_1_1_, output shape: {[512,60,60]}

            self.relu8_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.pool9_1_1_1_1_ = gluon.nn.MaxPool2D(
                pool_size=(2,2),
                strides=(2,2))
            # pool9_1_1_1_1_, output shape: {[512,30,30]}

            self.conv9_1_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv9_1_1_1_1_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv9_1_1_1_1_, output shape: {[1024,30,30]}

            self.relu9_1_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv10_1_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv10_1_1_1_1_ = gluon.nn.Conv2D(channels=1024,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv10_1_1_1_1_, output shape: {[1024,30,30]}

            self.relu10_1_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.transconv10_1_1_1_1_padding = (1,1)
            self.transconv10_1_1_1_1_ = gluon.nn.Conv2DTranspose(channels=1024,
                kernel_size=(4,4),
                strides=(2,2),
                padding=self.transconv10_1_1_1_1_padding,
                use_bias=True)
            # transconv10_1_1_1_1_, output shape: {[1024,60,60]}

            self.conv11_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv11_1_1_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv11_1_1_1_, output shape: {[512,60,60]}

            self.relu11_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv12_1_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv12_1_1_1_ = gluon.nn.Conv2D(channels=512,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv12_1_1_1_, output shape: {[512,60,60]}

            self.relu12_1_1_1_ = gluon.nn.Activation(activation='relu')
            self.transconv12_1_1_1_padding = (1,1)
            self.transconv12_1_1_1_ = gluon.nn.Conv2DTranspose(channels=512,
                kernel_size=(4,4),
                strides=(2,2),
                padding=self.transconv12_1_1_1_padding,
                use_bias=True)
            # transconv12_1_1_1_, output shape: {[512,120,120]}

            self.conv13_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv13_1_1_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv13_1_1_, output shape: {[256,120,120]}

            self.relu13_1_1_ = gluon.nn.Activation(activation='relu')
            self.conv14_1_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv14_1_1_ = gluon.nn.Conv2D(channels=256,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv14_1_1_, output shape: {[256,120,120]}

            self.relu14_1_1_ = gluon.nn.Activation(activation='relu')
            self.transconv14_1_1_padding = (1,1)
            self.transconv14_1_1_ = gluon.nn.Conv2DTranspose(channels=256,
                kernel_size=(4,4),
                strides=(2,2),
                padding=self.transconv14_1_1_padding,
                use_bias=True)
            # transconv14_1_1_, output shape: {[256,240,240]}

            self.conv15_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv15_1_ = gluon.nn.Conv2D(channels=128,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv15_1_, output shape: {[128,240,240]}

            self.relu15_1_ = gluon.nn.Activation(activation='relu')
            self.conv16_1_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv16_1_ = gluon.nn.Conv2D(channels=128,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv16_1_, output shape: {[128,240,240]}

            self.relu16_1_ = gluon.nn.Activation(activation='relu')
            self.transconv16_1_padding = (1,1)
            self.transconv16_1_ = gluon.nn.Conv2DTranspose(channels=128,
                kernel_size=(4,4),
                strides=(2,2),
                padding=self.transconv16_1_padding,
                use_bias=True)
            # transconv16_1_, output shape: {[128,480,480]}

            self.conv17_padding = Padding(padding=(0,0,0,0,1,1,1,1))
            self.conv17_ = gluon.nn.Conv2D(channels=64,
                kernel_size=(3,3),
                strides=(1,1),
                use_bias=True)
            # conv17_, output shape: {[64,480,480]}

            self.relu17_ = gluon.nn.Activation(activation='relu')
            self.conv18_ = gluon.nn.Conv2D(channels=21,
                kernel_size=(1,1),
                strides=(1,1),
                use_bias=True)
            # conv18_, output shape: {[21,480,480]}


            pass

    def hybrid_forward(self, F, data_):
        data_ = self.input_normalization_data_(data_)
        conv1_padding = self.conv1_padding(data_)
        conv1_ = self.conv1_(conv1_padding)
        relu1_ = self.relu1_(conv1_)
        conv2_padding = self.conv2_padding(relu1_)
        conv2_ = self.conv2_(conv2_padding)
        relu2_ = self.relu2_(conv2_)
        pool3_1_ = self.pool3_1_(relu2_)
        conv3_1_padding = self.conv3_1_padding(pool3_1_)
        conv3_1_ = self.conv3_1_(conv3_1_padding)
        relu3_1_ = self.relu3_1_(conv3_1_)
        conv4_1_padding = self.conv4_1_padding(relu3_1_)
        conv4_1_ = self.conv4_1_(conv4_1_padding)
        relu4_1_ = self.relu4_1_(conv4_1_)
        pool5_1_1_ = self.pool5_1_1_(relu4_1_)
        conv5_1_1_padding = self.conv5_1_1_padding(pool5_1_1_)
        conv5_1_1_ = self.conv5_1_1_(conv5_1_1_padding)
        relu5_1_1_ = self.relu5_1_1_(conv5_1_1_)
        conv6_1_1_padding = self.conv6_1_1_padding(relu5_1_1_)
        conv6_1_1_ = self.conv6_1_1_(conv6_1_1_padding)
        relu6_1_1_ = self.relu6_1_1_(conv6_1_1_)
        pool7_1_1_1_ = self.pool7_1_1_1_(relu6_1_1_)
        conv7_1_1_1_padding = self.conv7_1_1_1_padding(pool7_1_1_1_)
        conv7_1_1_1_ = self.conv7_1_1_1_(conv7_1_1_1_padding)
        relu7_1_1_1_ = self.relu7_1_1_1_(conv7_1_1_1_)
        conv8_1_1_1_padding = self.conv8_1_1_1_padding(relu7_1_1_1_)
        conv8_1_1_1_ = self.conv8_1_1_1_(conv8_1_1_1_padding)
        relu8_1_1_1_ = self.relu8_1_1_1_(conv8_1_1_1_)
        pool9_1_1_1_1_ = self.pool9_1_1_1_1_(relu8_1_1_1_)
        conv9_1_1_1_1_padding = self.conv9_1_1_1_1_padding(pool9_1_1_1_1_)
        conv9_1_1_1_1_ = self.conv9_1_1_1_1_(conv9_1_1_1_1_padding)
        relu9_1_1_1_1_ = self.relu9_1_1_1_1_(conv9_1_1_1_1_)
        conv10_1_1_1_1_padding = self.conv10_1_1_1_1_padding(relu9_1_1_1_1_)
        conv10_1_1_1_1_ = self.conv10_1_1_1_1_(conv10_1_1_1_1_padding)
        relu10_1_1_1_1_ = self.relu10_1_1_1_1_(conv10_1_1_1_1_)
        transconv10_1_1_1_1_ = self.transconv10_1_1_1_1_(relu10_1_1_1_1_)
        get9_1_1_1_2_ = relu8_1_1_1_
        concatenate11_1_1_1_ = F.concat(transconv10_1_1_1_1_, get9_1_1_1_2_, dim=1)
        conv11_1_1_1_padding = self.conv11_1_1_1_padding(concatenate11_1_1_1_)
        conv11_1_1_1_ = self.conv11_1_1_1_(conv11_1_1_1_padding)
        relu11_1_1_1_ = self.relu11_1_1_1_(conv11_1_1_1_)
        conv12_1_1_1_padding = self.conv12_1_1_1_padding(relu11_1_1_1_)
        conv12_1_1_1_ = self.conv12_1_1_1_(conv12_1_1_1_padding)
        relu12_1_1_1_ = self.relu12_1_1_1_(conv12_1_1_1_)
        transconv12_1_1_1_ = self.transconv12_1_1_1_(relu12_1_1_1_)
        get7_1_1_2_ = relu6_1_1_
        concatenate13_1_1_ = F.concat(transconv12_1_1_1_, get7_1_1_2_, dim=1)
        conv13_1_1_padding = self.conv13_1_1_padding(concatenate13_1_1_)
        conv13_1_1_ = self.conv13_1_1_(conv13_1_1_padding)
        relu13_1_1_ = self.relu13_1_1_(conv13_1_1_)
        conv14_1_1_padding = self.conv14_1_1_padding(relu13_1_1_)
        conv14_1_1_ = self.conv14_1_1_(conv14_1_1_padding)
        relu14_1_1_ = self.relu14_1_1_(conv14_1_1_)
        transconv14_1_1_ = self.transconv14_1_1_(relu14_1_1_)
        get5_1_2_ = relu4_1_
        concatenate15_1_ = F.concat(transconv14_1_1_, get5_1_2_, dim=1)
        conv15_1_padding = self.conv15_1_padding(concatenate15_1_)
        conv15_1_ = self.conv15_1_(conv15_1_padding)
        relu15_1_ = self.relu15_1_(conv15_1_)
        conv16_1_padding = self.conv16_1_padding(relu15_1_)
        conv16_1_ = self.conv16_1_(conv16_1_padding)
        relu16_1_ = self.relu16_1_(conv16_1_)
        transconv16_1_ = self.transconv16_1_(relu16_1_)
        get3_2_ = relu2_
        concatenate17_ = F.concat(transconv16_1_, get3_2_, dim=1)
        conv17_padding = self.conv17_padding(concatenate17_)
        conv17_ = self.conv17_(conv17_padding)
        relu17_ = self.relu17_(conv17_)
        conv18_ = self.conv18_(relu17_)
        softmax_ = F.identity(conv18_)

        return softmax_

    def getInputs(self):
        inputs = {}
        input_dimensions = (3,480,480)
        input_domains = (float,0.0,1.0)
        inputs["data_"] = input_domains + (input_dimensions,)
        return inputs

    def getOutputs(self):
        outputs = {}
        output_dimensions = (21,480,480)
        output_domains = (float,float('-inf'),float('inf'))
        outputs["softmax_"] = output_domains + (output_dimensions,)
        return outputs
