#!/usr/bin/env bash

data_dir="data/FB15K-237-10"
model="point.rs.conve"
group_examples_by_query="True"
use_action_space_bucketing="True"

use_action_selection="True"
use_state_prediction="True"
mask_sim_relation="False"
max_dynamic_action_size=20
dynamic_split_bound=2
avg_entity_per_relation=1
strategy="top1"

bandwidth=400
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
bucket_interval=10
num_epochs=240
num_wait_epochs=30
num_peek_epochs=2
batch_size=96
train_batch_size=96
dev_batch_size=4
learning_rate=0.001
baseline="n/a"
grad_norm=0
emb_dropout_rate=0.3
ff_dropout_rate=0.1
action_dropout_rate=0.5
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0.02
relation_only="False"
beam_size=128
emb_2D_d1=10
emb_2D_d2=20

ptranse_state_dict_path="model/FB15K-237-10-PTransE-xavier-200-200-0.001-0.3-0.1/checkpoint-999.tar"
distmult_state_dict_path="model/FB15K-237-10-distmult-xavier-200-200-0.003-0.3-0.1/checkpoint-15.tar"
complex_state_dict_path="model/FB15K-237-10-complex-RV-xavier-200-200-0.003-0.3-0.1/checkpoint-999.tar"
conve_state_dict_path="model/FB15K-237-10-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/model_best.tar"
tucker_state_dict_path="model/FB15K-237-10-tucker-RV-xavier-200-200-0.0005-32-3-0.3-0.3-0.2-0.1/model_best.tar"

num_paths_per_entity=-1
margin=-1
