#!/usr/bin/env bash
#BERT_BASE_DIR="data"
python run_pretraining.py  --input_file=data/tf_examples.tfrecord  --output_dir=data/pretraining_output  --do_train=True  --do_eval=True  --bert_config_file=data/bert_config.json  --train_batch_size=15  --max_seq_length=30  --max_predictions_per_seq=20  --num_train_steps=20  --num_warmup_steps=10  --learning_rate=2e-5
#--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
