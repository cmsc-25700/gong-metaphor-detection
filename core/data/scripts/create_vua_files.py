"""
Process raw vua data into gong format
"""
import os
from core.data.gao_data import ExperimentData
from core.data.process_raw_data_util import convert_raw_vua_data

### INPUT
input_data_path = os.path.join("resources", "metaphor-in-context", "data")
gao_data = ExperimentData(input_data_path)
gao_data.read_vua_seq_data()

#merging the val set into the train set
gao_data.vua_seq_formatted_train.extend(gao_data.vua_seq_formatted_val)

vua_seq_data = {
    "train": gao_data.vua_seq_formatted_train,
    "test": gao_data.vua_seq_formatted_test,
}

output_dir = os.path.join("data", "VUA")

initial_id = 0
for subset, data in vua_seq_data.items():
    # if the gong code starts every id at 0 (for train and test), we should change this
    # to do the same thing
    initial_id = convert_raw_vua_data(data, output_dir, subset, initial_id)
