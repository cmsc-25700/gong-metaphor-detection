"""
data_utils.py
 - process metaphor data
"""

import os
import sys
import numpy as np
import pickle


class InputExample(object):
    """
    A sentence example for token classification
    """
    def __init__(self, example_id, words, pos_list, labels=None):
        self.example_id = example_id
        self.words = words
        self.pos_list = pos_list
        self.labels = labels


class InputFeatures(object):
    """
    Features for an example
    """
    def __init__(self, input_ids, input_mask, segment_ids, pos_ids, label_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pos_ids = pos_ids
        self.label_ids = label_ids


def read_pos_tags(data_folder, pos_pad="POSPAD"):
    pos_set = set()
    pos_file = open(os.path.join(data_folder,"train_pos.txt", "r"))
    for pos_line in pos_file:
        for pos in pos_line.strip().split():
            pos_set.add(pos)
    print("# of POS tags: {}".format(len(pos_set)))
    print("POS: {}".format(pos_set))

    # the pos_pad has index 0 for <PAD> in input sequence
    pos_vocab = {0: pos_pad}
    pos_id = 1
    for pos in pos_set:
        pos_vocab[pos] = pos_id
        pos_id += 1
    return pos_vocab


def _parse_str_vector(line):
    return [[float(val) for val in str_vector.split(",")] for str_vector in line.strip().split()]


