"""
Process raw trofidata into gong format
"""
import os
import random
from core.data.gao_data import ExperimentData
from core.data.process_raw_data_util import convert_raw_cls_data, TRAIN_PERC, TEST_PERC

### DATA DIRECTORIES
input_data_path = os.path.join("resources", "metaphor-in-context", "data")
output_dir = os.path.join("data", "Trofi")

# TROFI DATA INDICES
WORD_IDX = 0
TOKENS_IDX = 1
LABEL_IDX = 3
VERB_IDX = 2

# READ DATA
gao_data = ExperimentData(input_data_path)
gao_data.read_trofi_data()
data = gao_data.trofi_formatted_all

# Split into training - val - test sets
random.shuffle(data)
train_index = int(len(data) * TRAIN_PERC)
train_data = data[0:train_index]
test_data = data[train_index:]

mode_data_dict = {
    "train": train_data,
    "test": test_data,
}

initial_id = 0
for subset, data in mode_data_dict.items():
    initial_id = convert_raw_cls_data(data, output_dir, subset, initial_id,
                                      WORD_IDX,
                                      TOKENS_IDX,
                                      LABEL_IDX,
                                      VERB_IDX,
                                      get_pos=True)
