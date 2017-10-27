import theano.tensor as T
from layers import *
from LossCalculation import Dual_Loss
import layers as la


class DGCN(object):
    def __init__(self, rng, input, layer_sizes, diffusion, ppmi,
                 dropout_rate=0.3, activation=None, nell_dataset=False):

        self.a_layers = []
        self.ppmi_layers = []
        self.l1 = 0.
        self.l2 = 0.
        self.loss = None
        self.input = input  #### X

        # define the dual NN sharing the same weights Ws
        next_a_layer_input = input
        next_ppmi_layer_input = input

        for s in layer_sizes:
            _hiddenLayer_a = HiddenDenseLayer(
                rng=rng,
                input=next_a_layer_input,
                n_in=s[0],
                n_out=s[1],
                diffusion=diffusion)
            self.a_layers.append(_hiddenLayer_a)

            _hiddenLayer_ppmi = HiddenDenseLayer(
                rng=rng,
                input=next_ppmi_layer_input,
                n_in=s[0],
                n_out=s[1],
                diffusion=ppmi,
                W=_hiddenLayer_a.W)  #### share the same weight matrix W
            self.ppmi_layers.append(_hiddenLayer_ppmi)
            # drop out
            _layer_output_a = _hiddenLayer_a.output
            _layer_output_ppmi = _hiddenLayer_ppmi.output
            next_a_layer_input = la._dropout_from_layer(rng, _layer_output_a, dropout_rate)
            next_ppmi_layer_input = la._dropout_from_layer(rng, _layer_output_ppmi, dropout_rate)

        # record all the params to do training
        self.params = [param for layer in self.a_layers for param in layer.params]

        # define the NN output
        self.a_output = T.nnet.softmax(self.a_layers[-1].output)
        self.ppmi_output = T.nnet.softmax(self.ppmi_layers[-1].output)

        # define the regulizer
        for _W in self.params:
            self.l2 += (_W ** 2).sum() / 2.0
            self.l1 += abs(_W).sum()

        self.LossCal = Dual_Loss(self.a_output, self.ppmi_output)

        # define the supervised loss
        if nell_dataset:
            self.supervised_loss = self.LossCal.masked_cross_entropy
        else:
            self.supervised_loss = self.LossCal.masked_mean_square

        # define the unsupervised loss
        self.unsupervised_loss = self.LossCal.unsupervised_loss

        # define the test accuracy function
        self.acc = self.LossCal.acc
