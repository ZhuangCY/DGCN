import sys
import DataPreparation as dp
import numpy as np
from theano import sparse
import theano
import theano.tensor as T
import time

from Model import DGCN
from utilities import diffusion_fun_sparse, \
    diffusion_fun_improved_ppmi_dynamic_sparsity, get_scaled_unsup_weight_max, rampup


def test_DGCN(learning_rate=0.01, L1_reg=0.00, L2_reg=0.00,
              n_epochs=200, dataset='cora', dropout_rate=0.3,
              hidden_size=32, cons=1, tper=0.1):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = dp.load_graph_data(dataset)
    train_mask = np.reshape(train_mask, (-1, len(train_mask)))
    val_mask = np.reshape(val_mask, (-1, len(val_mask)))
    test_mask = np.reshape(test_mask, (-1, len(test_mask)))

    #### obtain diffusion and ppmi matrices
    diffusions = diffusion_fun_sparse(adj.tocsc())
    ppmi = diffusion_fun_improved_ppmi_dynamic_sparsity(adj, path_len=2, k=1.0)

    #### construct the classifier model ####
    rng = np.random.RandomState(1234)
    tX = sparse.csc_matrix(name='X', dtype=theano.config.floatX)  # sparse matrix features
    tD = sparse.csc_matrix(name='D', dtype=theano.config.floatX)  # sparse matrix diffusion
    tP = sparse.csc_matrix(name='PPMI', dtype=theano.config.floatX)  # sparse matrix ppmi
    tRU = T.scalar(name='ramp-up', dtype=theano.config.floatX)

    feature_size = features.shape[1]
    label_size = y_train.shape[1]
    layer_sizes = [(feature_size, hidden_size), (hidden_size, label_size)]

    print "Convolution Layers:" + str(layer_sizes)

    classifier = DGCN(
        rng=rng,
        input=tX,
        layer_sizes=layer_sizes,
        diffusion=tD,
        ppmi=tP,
        dropout_rate=dropout_rate,
        nell_dataset=False
    )

    #### loss function ####
    tY = T.matrix('Y', dtype=theano.config.floatX)
    tMask = T.matrix('Mask', dtype=theano.config.floatX)
    cost = (
        classifier.supervised_loss(tY, tMask)
        + tRU * classifier.unsupervised_loss()
        + L1_reg * classifier.l1
        + L2_reg * classifier.l2
    )

    #### update rules for NN params ####
    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    #### the train, validation and test models ####
    train_model = theano.function(
        inputs=[tY, tMask, tRU],
        outputs=cost,
        updates=updates,
        givens={
            tX: features.tocsc().astype(theano.config.floatX),
            tD: diffusions.astype(theano.config.floatX),
            tP: ppmi.astype(theano.config.floatX)
        },
        on_unused_input='warn'
    )

    validate_model = theano.function(
        inputs=[tY, tMask],
        outputs=classifier.acc(tY, tMask),
        givens={
            tX: features.tocsc().astype(theano.config.floatX),
            tD: diffusions.astype(theano.config.floatX),
            tP: ppmi.astype(theano.config.floatX)
        },
        on_unused_input='warn'
    )

    validate_model_cost = theano.function(
        inputs=[tY, tMask, tRU],
        outputs=cost,
        givens={
            tX: features.tocsc().astype(theano.config.floatX),
            tD: diffusions.astype(theano.config.floatX),
            tP: ppmi.astype(theano.config.floatX)
        },
        on_unused_input='warn'
    )

    test_model = theano.function(
        inputs=[tY, tMask],
        outputs=classifier.acc(tY, tMask),
        givens={
            tX: features.tocsc().astype(theano.config.floatX),
            tD: diffusions.astype(theano.config.floatX),
            tP: ppmi.astype(theano.config.floatX)
        },
        on_unused_input='warn'
    )

    #### train model ####
    print "...training..."
    num_labels = np.sum(train_mask)
    X_train_shape = features.shape[0]
    accuracys = []
    for epoch in range(n_epochs):
        t = time.time()

        scaled_unsup_weight_max = get_scaled_unsup_weight_max(
            num_labels, X_train_shape, unsup_weight_max=15.0)

        ramp_up = rampup(epoch, scaled_unsup_weight_max, exp=5.0, rampup_length=120)
        ramp_up = np.asarray(ramp_up, dtype=theano.config.floatX)

        _train_cost = train_model(y_train.astype(theano.config.floatX),
                                  train_mask.astype(theano.config.floatX),
                                  ramp_up)

        _valid_acc = validate_model(y_val.astype(theano.config.floatX),
                                    val_mask.astype(theano.config.floatX))

        _valid_cost = validate_model_cost(y_val.astype(theano.config.floatX),
                                          val_mask.astype(theano.config.floatX),
                                          ramp_up)

        _test_acc = test_model(y_test.astype(theano.config.floatX),
                               test_mask.astype(theano.config.floatX))

        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", '%.5f' % (_train_cost),
              "val_acc=", '%.5f' % (_valid_acc),
              "test_acc=", '%.5f' % (_test_acc),
              "val_cost=", '%.5f' % (_valid_cost),
              "time=", '%.5f' % (time.time() - t))

        # xs.append(epoch)
        accuracys.append(_test_acc)

    #### test the trained model ####
    test_acc = test_model(y_test.astype(theano.config.floatX),
                          test_mask.astype(theano.config.floatX))
    print("Test Acc:", "%.5f" % (test_acc))
    print("Best Test Acc:", "%.5f" % (max(accuracys)))
    return test_acc, max(accuracys)


if __name__ == "__main__":
    dataset = sys.argv[1]
    print "Testing on the dataset: " + dataset
    if dataset == 'cora' or dataset == 'citeseer' or dataset == 'pubmed':
        res = test_DGCN(dataset=dataset, learning_rate=0.05,
                        dropout_rate=0.1, n_epochs=300, hidden_size=32)
        print "Finished"
    else:
        print "No such a dataset: " + dataset
