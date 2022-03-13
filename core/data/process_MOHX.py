import os
import ast
import random



from gao_data import ExperimentData
from io_util import write_dict_to_json


TRAIN_PERC = 0.6
TEST_PERC = 0.25
VAL_PERC = 0.15

### INPUT
input_data_path = os.path.join("resources", "metaphor-in-context", "data")
gao_data = ExperimentData(input_data_path)
gao_data.read_moh_x_data()
moh_x = gao_data.moh_x_formatted_svo_cleaned

#Split into training - val - test sets
random.shuffle(moh_x)

train_index = int(len(moh_x)*TRAIN_PERC)
train_data = moh_x[0:train_index]

valid_index = int(len(moh_x)*VAL_PERC)
val_data = moh_x[train_index:train_index+valid_index]

test_data = moh_x[train_index+valid_index:]

trofi_class_data = {
    "train": train_data,
    "test": test_data,
    "val": val_data
}

output_dir = os.path.join("data", "MOH_X")

def function_convert_raw_vua_data(input_data_list, output_dir, data_subset, starting_id):
    """
    Convert csv vua data to Gong et all format
    """
    # files for gong et all
    id_file_name = os.path.join(output_dir, "{}_ids.txt".format(data_subset))
    tokens_file_name = os.path.join(output_dir, "{}_tokens.txt".format(data_subset))
    metaphor_pos_file_name = os.path.join(output_dir, "{}_metaphor_pos.txt".format(data_subset))
    metaphor_file_name = os.path.join(output_dir, "{}_metaphor.txt".format(data_subset)) # labels

    # extra files
    id_map_file_name = os.path.join(output_dir, "{}_id_map.json".format(data_subset))
    #genre_map_file_name = os.path.join(output_dir, "{}_genre.json".format(data_subset)) # map id to genre

    # indices for Trofi seq data
    #TEXT_ID_IDX = 0
    #SENT_ID_IDX = 1
    WORD_IDX = 0
    TOKENS_IDX = 3  # not sure what diff is from 5, but gao code uses 2
    LABEL_IDX = 5
    POS_IDX = 4
    #GENRE_IDX = 6

    # not sure if I need to use latin-1 encoding for output, but I am not.
    id_map = {}
    #genre_map = {}

    # this will cause last line in file to be blank
    # if that's a problem save data in list then write at once
    with open(id_file_name, 'w') as id_f, open(tokens_file_name, 'w') as tok_f, open(metaphor_pos_file_name, 'w') as pos_f, open(metaphor_file_name, 'w') as labels_f:
        for line in input_data_list:
            id_tuple = str(starting_id) + "-" + line[WORD_IDX]
            if id_tuple not in id_map:
                id_map[id_tuple] = starting_id
                starting_id += 1
            id = id_map[id_tuple]
            num_words = len(line[TOKENS_IDX].split())
            labels = [str(0)] * num_words
            labels[int(line[POS_IDX])] = line[LABEL_IDX]
            label = ' '.join(labels)
            pos = ' '.join(str(ast.literal_eval(line[POS_IDX])))

            # write files for gong et al format
            id_f.write(str(id)+'\n')
            tok_f.write(line[TOKENS_IDX]+'\n')
            pos_f.write(pos+'\n')
            labels_f.write(label +'\n')

        write_dict_to_json(id_map_file_name, id_map)

        return starting_id

initial_id = 0
for subset, data in trofi_class_data.items():
    initial_id = function_convert_raw_vua_data(data, output_dir, subset, initial_id)
