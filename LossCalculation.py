import theano
import theano.tensor as T
import lasagne


class Dual_Loss(object):
    def __init__(self, a_input, ppmi_input):
        self.a_input = a_input
        self.ppmi_input = ppmi_input
        self.y_preds = T.argmax(self.a_input, axis=1)

    def masked_mean_square(self, y, mask):
        loss = ((self.a_input - y) ** 2).sum(axis=1)
        loss *= mask
        return T.sum(loss)

    def masked_cross_entropy(self, y, mask):
        y = T.cast(y, dtype='int32')
        loss = lasagne.objectives.categorical_crossentropy(self.a_input, y)
        return lasagne.objectives.aggregate(loss, weights=mask, mode='sum')

    def unsupervised_loss(self):
        return ((self.a_input - self.ppmi_input) ** 2).sum()

    def acc(self, y, mask):
        y_labels = T.argmax(y, axis=1)
        all_compares = T.eq(self.y_preds, y_labels)
        all_compares *= mask
        msum = T.cast(mask.sum(), theano.config.floatX)
        return (all_compares.sum()) / msum