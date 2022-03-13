python3 run_metaphor_detection.py \
--data_dir data/VUA \
--model_type roberta \
--model_name_or_path roberta-large \
--output_dir output/VUA/modeltest/ \
--dataset VUA \
--max_seq_length 256 \
--do_predict \
--do_lower_case \
--use_pos