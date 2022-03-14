"""
Process vua data per repo instructions:
Obtain the datasets (VUA or TOEFL).

Extract from the train data the:
    token ids,
    tokens,
    POS tags,
    and metaphor labels,

and save them to:
    train_ids.txt,
    train_tokens.txt,
    train_pos.txt,
    and train_metaphor.txt (respectively)

These files are in such format that each row corresponds to one sentence,
and each id/token/POS/label is separated by space in a row.
Save VUA data to [DATA_DIR]/VUA/, and TOEFL data to [DATA_DIR]/TOEFL/.

Process the test data in the same way as the train data,
and save test_ids.txt, test_tokens.txt and test_pos.txt to the folder [DATA_DIR]/VUA/ or [DATA_DIR]/TOEFL/.

"""
import os
import ast
from core.data.gao_data import ExperimentData
from core.data.io_util import write_dict_to_json

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


def convert_raw_vua_data(input_data_list, output_dir, data_subset, starting_id):
    """
    Convert csv vua data to Gong et all format
    """
    # files for gong et all
    id_file_name = os.path.join(output_dir, "{}_ids.txt".format(data_subset))
    tokens_file_name = os.path.join(output_dir, "{}_tokens.txt".format(data_subset))
    pos_file_name = os.path.join(output_dir, "{}_pos.txt".format(data_subset))
    metaphor_file_name = os.path.join(output_dir, "{}_metaphor.txt".format(data_subset)) # labels

    # extra files
    id_map_file_name = os.path.join(output_dir, "{}_id_map.json".format(data_subset))
    genre_map_file_name = os.path.join(output_dir, "{}_genre.json".format(data_subset)) # map id to genre

    # indices for vua seq data
    TEXT_ID_IDX = 0
    SENT_ID_IDX = 1
    TOKENS_IDX = 2  # not sure what diff is from 5, but gao code uses 2
    LABEL_IDX = 3
    POS_IDX = 4
    GENRE_IDX = 6

    # not sure if I need to use latin-1 encoding for output, but I am not.
    id_map = {} # integer id
    genre_map = {}

    # this will cause last line in file to be blank
    # if that's a problem save data in list then write at once
    with open(id_file_name, 'w') as id_f, open(tokens_file_name, 'w') as tok_f, open(pos_file_name, 'w') as pos_f, open(
            metaphor_file_name, 'w') as labels_f:
        for line in input_data_list:
            id_tuple = line[TEXT_ID_IDX] + "-" + line[SENT_ID_IDX]
            if id_tuple not in id_map:
                id_map[id_tuple] = starting_id
                starting_id += 1
            id = id_map[id_tuple]
            labels = ' '.join([str(x) for x in ast.literal_eval(line[LABEL_IDX])])
            pos = ' '.join(ast.literal_eval(line[POS_IDX]))
            genre = line[GENRE_IDX]

            # write files for gong et al format
            id_f.write(str(id)+'\n')
            tok_f.write(line[TOKENS_IDX]+'\n')
            pos_f.write(pos+'\n')
            labels_f.write(labels +'\n')

            genre_map[id] = genre

        # write maps
        write_dict_to_json(genre_map_file_name, genre_map)
        write_dict_to_json(id_map_file_name, id_map)
        return starting_id


initial_id = 0
for subset, data in vua_seq_data.items():
    # if the gong code starts every id at 0 (for train and test), we should change this
    # to do the same thing
    initial_id = convert_raw_vua_data(data, output_dir, subset, initial_id)
