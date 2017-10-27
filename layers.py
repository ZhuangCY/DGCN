import numpy as np
import theano
import theano.tensor as T


# define the diffusion based hidden layer
class HiddenDenseLayer(object):
    def __init__(self, rng, input, n_in, n_out, diffusion, W=None,
                 activation=T.nnet.relu):

        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W
        self.D = diffusion

        lin_output = theano.dot(theano.dot(diffusion, input), self.W)
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W]


# define a normal dense layer
class HiddenDenseLayer_normal(object):
    def __init__(self, rng, input, n_in, n_out, W=None,
                 activation=T.nnet.relu):

        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W

        lin_output = theano.dot(input, self.W)
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W]


# Use to do the dropout process when training
def _dropout_from_layer(rng, layer, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output
