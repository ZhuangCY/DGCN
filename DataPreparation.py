import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from random import random
import cPickle
import scipy.sparse as sp
from collections import defaultdict
import random


def import_karate_ckub_graph():
    G = nx.karate_club_graph()
    numOFnodes = len(G.nodes())
    att = nx.get_node_attributes(G, 'club')
    # the X: |nodes| multiply |features|
    X = np.identity(numOFnodes)
    # the Y: 1 multiply |nodes|
    label_map = {'Mr. Hi': 1.0,  # red
                 'Officer': 2.0}  # blue
    Y = [label_map.get(att.get(node)) for node in G.nodes()]
    # the A: |nodes| multiply |nodes|
    A = nx.adjacency_matrix(G)
    return X, Y, A


def draw_karate_ckub_graph():
    G = nx.karate_club_graph()
    att = nx.get_node_attributes(G, 'club')

    val_map = {'Mr. Hi': 'r',
               'Officer': 'g'}
    values = [val_map.get(att.get(node), 0.25) for node in G.nodes()]
    nx.draw(G, cmap=plt.get_cmap('jet'), node_color=values, with_labels=True)
    plt.show()


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_graph_data(DATASET='cora'):
    NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    OBJECTS = []
    for i in range(len(NAMES)):
        OBJECTS.append(cPickle.load(open('data/ind.{}.{}'.format(DATASET, NAMES[i]), 'rb')))
    x, y, tx, ty, allx, ally, graph = tuple(OBJECTS)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(DATASET))
    # the indices of test instances in graph
    test_idx_range = np.sort(test_idx_reorder)

    if DATASET == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # get the features:X
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    print "Feature matrix:" + str(features.shape)

    # get the labels: y
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    print "Label matrix:" + str(labels.shape)

    # get the adjcent matrix: A
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    print "Adjcent matrix:" + str(adj.shape)

    # test, validation, train
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def _load_nell_data(DATASET='nell'):
    NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    OBJECTS = []
    for i in range(len(NAMES)):
        OBJECTS.append(cPickle.load(open('data/ind.{}.{}'.format(DATASET, NAMES[i]), 'rb')))
    x, y, tx, ty, allx, ally, graph = tuple(OBJECTS)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(DATASET))
    exclu_rang = []
    for i in range(8922, 65755):
        if i not in test_idx_reorder:
            exclu_rang.append(i)

    # get the features:X
    allx_v_tx = sp.vstack((allx, tx)).tolil()
    _x = sp.lil_matrix(np.zeros((9891, 55864)))

    up_features = sp.hstack((allx_v_tx, _x))

    _x = sp.lil_matrix(np.zeros((55864, 5414)))
    _y = sp.identity(55864, format='lil')
    down_features = sp.hstack((_x, _y))
    features = sp.vstack((up_features, down_features)).tolil()
    features[test_idx_reorder + exclu_rang, :] = features[range(8922, 65755), :]
    print "Feature matrix:" + str(features.shape)

    # get the labels: y
    up_labels = np.vstack((ally, ty))
    down_labels = np.zeros((55864, 210))
    labels = np.vstack((up_labels, down_labels))
    labels[test_idx_reorder + exclu_rang, :] = labels[range(8922, 65755), :]
    print "Label matrix:" + str(labels.shape)

    # print np.sort(graph.get(17493))

    # get the adjcent matrix: A
    # adj = nx.to_numpy_matrix(nx.from_dict_of_lists(graph))
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    print "Adjcent matrix:" + str(adj.shape)

    # test, validation, train
    idx_test = test_idx_reorder
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # record the intermedia result for saving time
    cPickle.dump(adj, open('data/cell.adj.pkl', 'wb'))
    cPickle.dump(features, open('data/cell.features.pkl', 'wb'))
    cPickle.dump(y_train, open('data/cell.yTrain.pkl', 'wb'))
    cPickle.dump(y_val, open('data/cell.yVal.pkl', 'wb'))
    cPickle.dump(y_test, open('data/cell.yTest.pkl', 'wb'))
    cPickle.dump(train_mask, open('data/cell.trainMask.pkl', 'wb'))
    cPickle.dump(val_mask, open('data/cell.valMask.pkl', 'wb'))
    cPickle.dump(test_mask, open('data/cell.testMask.pkl', 'wb'))

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_nell_data():
    NAMES = ['adj', 'features', 'yTrain', 'yVal', 'yTest', 'trainMask', 'valMask', 'testMask']
    OBJECTS = []
    for i in range(len(NAMES)):
        OBJECTS.append(cPickle.load(open('data/cell.{}.pkl'.format(NAMES[i]), 'rb')))
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = tuple(OBJECTS)
    print "Feature matrix:" + str(features.shape)
    print "Adjcent matrix:" + str(adj.shape)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_nell_adj():
    NAMES = ['adj']
    OBJECTS = []
    for i in range(len(NAMES)):
        OBJECTS.append(cPickle.load(open('data/cell.{}.pkl'.format(NAMES[i]), 'rb')))
    adj = OBJECTS[0]
    print "Adjcent matrix:" + str(adj.shape)
    return adj


def load_nell_rw():
    NAMES = ['rw_len_2']
    OBJECTS = []
    for i in range(len(NAMES)):
        OBJECTS.append(cPickle.load(open('data/cell.{}.pkl'.format(NAMES[i]), 'rb')))
    rw = OBJECTS[0]
    print "Random walk matrix:" + str(rw.shape)
    return rw


def load_nell_concise_dataset_3nd(per=0.1, cons=5, DATASET='nell'):
    NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    OBJECTS = []
    for i in range(len(NAMES)):
        OBJECTS.append(cPickle.load(open('data/ind.{}.{}'.format(DATASET, NAMES[i]), 'rb')))
    x, y, tx, ty, allx, ally, graph = tuple(OBJECTS)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(DATASET))

    # get the features:X
    features = allx.tolil()
    print "Feature matrix:" + str(features.shape)

    # get the labels: y
    labels = ally
    print "Label matrix:" + str(labels.shape)

    b = allx.shape[0]
    a = features.shape[0]
    adj = sp.dok_matrix((a, a))
    index_value_dict = defaultdict(int)

    for i in range(b):
        _temp = []
        first_neighbors = set(graph.get(i))
        _temp += list(first_neighbors)
        for _n in first_neighbors:
            second_neighbors = set(graph.get(_n))
            _temp += list(second_neighbors)
        for t in _temp:
            if t < a and t != i:
                index_value_dict[(i, t)] += 1.0
    for key, value in index_value_dict.iteritems():
        if value > cons:
            adj[key[0], key[1]] = 1.0
            adj[key[1], key[0]] = 1.0

    print "Adjacence matrix:" + str(adj.shape)
    print adj.sum()

    # test, validation, train
    indeces = range(labels.shape[0])
    random.shuffle(indeces)

    train_num = int(labels.shape[0] * per)

    idx_test = indeces[train_num:train_num + 500]
    idx_train = indeces[:train_num]
    idx_val = indeces[train_num+500:train_num + 1000]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    print y_train.sum()

    return adj.tocsr(), features, y_train, y_val, y_test, train_mask, val_mask, test_mask
