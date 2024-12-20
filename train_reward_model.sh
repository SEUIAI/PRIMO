python train_reward_model.py \
    --model "bert-large-cased" \
    --train_path "dataset/reward_datasets/train.tsv" \
    --dev_path "dataset/reward_datasets/dev.tsv" \
    --save_dir "models/RM_checkpoints" \
    --img_log_dir "logs/reward_model" \
    --img_log_name "BERT Reward Model" \
    --batch_size 8 \
    --max_seq_len 256 \
    --learning_rate 1e-5 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:0"