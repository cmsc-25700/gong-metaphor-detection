python3 python3 run_metaphor_detection.py \
--data_dir data/VUA \
--model_type roberta \
--model_name_or_path roberta-large \
--output_dir output/VUA/model/ \
--dataset VUA \
--max_seq_length 256 \
--do_train \
--evaluate_during_training \
--do_lower_case \
--per_gpu_train_batch_size 6 \
--per_gpu_eval_batch_size 18 \
--learning_rate 2e-5 \
--num_train_epochs 5.0 \
--warmup_steps 500 \
--use_pos # \
#--overwrite_output_dir