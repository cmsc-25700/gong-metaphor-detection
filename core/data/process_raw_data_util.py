"""
Process data per repo instructions:
Obtain the datasets

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
from core.data.io_util import write_dict_to_json

TRAIN_PERC = 0.75
TEST_PERC = 0.25


def convert_raw_cls_data(input_data_list, output_dir, data_subset, starting_id,
                                  word_idx,
                                  tokens_idx,
                                  label_idx,
                                  verb_idx):
    """
    CONVERT CLASSIFICATION DATA
    This data has no POS label. ONLY HAS LABEL FOR TARGET VERB.
    There is also no id for these observations.
    """
    # files for gong et all
    id_file_name = os.path.join(output_dir, "{}_ids.txt".format(data_subset))
    tokens_file_name = os.path.join(output_dir, "{}_tokens.txt".format(data_subset))
    metaphor_file_name = os.path.join(output_dir, "{}_metaphor.txt".format(data_subset)) # labels
    target_verb_file_name = os.path.join(output_dir, "{}_target_verb.txt".format(data_subset)) #binary indicator for target verb
    
    # extra files
    id_map_file_name = os.path.join(output_dir, "{}_id_map.json".format(data_subset))
    id_map = {}

    # this will cause last line in file to be blank
    # if that's a problem save data in list then write at once
    with open(id_file_name, 'w') as id_f, open(tokens_file_name, 'w') as tok_f, \
            open(metaphor_file_name, 'w') as labels_f,\
                open(target_verb_file_name, "w") as target_f:
        for i, line in enumerate(input_data_list):
            id_tuple = str(i) + '-' + line[word_idx]
            if id_tuple not in id_map:
                id_map[id_tuple] = starting_id
                starting_id += 1
            id = id_map[id_tuple]
            num_words = len(line[tokens_idx].split(" "))
            labels = [str(0)] * num_words
            labels[int(line[verb_idx])] = line[label_idx]
            label = ' '.join(labels)
            target_verb = [str(0)] * num_words
            target_verb[int(line[verb_idx])] = "1"
            target_verb = " ".join(target_verb)

            # write files for gong et al format
            id_f.write(str(id)+'\n')
            tok_f.write(line[tokens_idx]+'\n')
            labels_f.write(label +'\n')
            target_f.write(target_verb+ "\n")
        write_dict_to_json(id_map_file_name, id_map)

        return starting_id


def convert_raw_vua_data(input_data_list, output_dir, data_subset, starting_id):
    """
    Convert csv vua data to Gong et all format
    ALL WORDS ARE LABELED IN THIS DATA
    There are also pos tags for each word
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
