python3 run_metaphor_detection.py --data_dir data/VUA --model_type roberta --model_name_or_path roberta-large --output_dir output/VUA/model/ --dataset VUA --use_features= False
#--do_lower_case
#--per_gpu_train_batch_size 6
#--per_gpu_eval_batch_size 18
#--learning_rate 2e-5
#--num_train_epochs 5.0
#--warmup_steps 500
#--seed 311
#--use_pos
#--pos_vocab_size 43
#--pos_dim 50
#--use_features= False
#--feature_dim 696