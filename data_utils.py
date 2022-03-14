"""
data_utils.py from https://github.com/HongyuGong/MetaphorDetectionSharedTask
 - process metaphor data

CI: I removed references to external features
"""

import os


class InputExample(object):
    """
    A sentence example for token classification
    """
    def __init__(self, example_id, words, pos_list, labels=None, target_indicator = None):
        self.example_id = example_id
        self.words = words
        self.pos_list = pos_list
        self.labels = labels
        self.target_inticator = target_indicator


class InputFeatures(object):
    """
    Features for an example
    """
    def __init__(self, input_ids, input_mask, segment_ids, pos_ids, label_ids=None, target_indicator = None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pos_ids = pos_ids
        self.label_ids = label_ids
        self.target_indicator = target_indicator


def read_pos_tags(data_folder, pos_pad="POSPAD"):
    pos_set = set()
    # CI: for the POS vocab from train
    pos_file = open(os.path.join(data_folder,"train_pos.txt"),"r")
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


