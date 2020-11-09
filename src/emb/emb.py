"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Embedding-based knowledge base completion baselines.
"""

import os
from tqdm import tqdm

import torch
import torch.nn as nn
import random

from src.learn_framework import LFramework
from src.data_utils import NO_OP_ENTITY_ID, DUMMY_ENTITY_ID
from src.utils.ops import var_cuda, int_var_cuda, int_fill_var_cuda
from torch.autograd import Variable

class EmbeddingBasedMethod(LFramework):
    def __init__(self, args, kg, mdl, secondary_kg=None, tertiary_kg=None):
        super(EmbeddingBasedMethod, self).__init__(args, kg, mdl)
        self.num_negative_samples = args.num_negative_samples
        self.label_smoothing_epsilon = args.label_smoothing_epsilon

        if self.model == '!TransE':
            self.loss_fun = nn.MarginRankingLoss(1, False)
        elif self.model == 'PTransE':
            self.loss_fun_TransE = nn.MarginRankingLoss(1, False)
            self.loss_fun_path = nn.MarginRankingLoss(1, False)
        else:
            self.loss_fun = nn.BCELoss()

        self.theta = args.theta
        self.secondary_kg = secondary_kg
        self.tertiary_kg = tertiary_kg

    def forward_fact(self, examples):
        kg, mdl = self.kg, self.mdl
        pred_scores = []
        for example_id in tqdm(range(0, len(examples), self.batch_size)):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)
            e1, e2, r = self.format_batch(mini_batch)
            pred_score = mdl.forward_fact(e1, r, e2, kg)
            pred_scores.append(pred_score[:mini_batch_size])
        return torch.cat(pred_scores)

    def get_corrupt_triple(self, e1, e2, r):
        batch_size = e1.size()[0]
        e1_list = []
        e2_list = []
        r_list = []
        decide_num = random.randint(0, 99)
        for i in range(batch_size):
            if decide_num < 25:
                rand_e1 = random.randint(0, len(self.kg.entity2id) - 1)
                try:
                    while int(e2[i]) in self.kg.train_objects[rand_e1][int(r[i])]:
                        rand_e1 = random.randint(0, len(self.kg.entity2id) - 1)
                except:
                    pass
                e1_list.append(rand_e1)
                e2_list.append(int(e2[i]))
                r_list.append(int(r[i]))
            elif decide_num < 50:
                rand_e2 = random.randint(0, len(self.kg.entity2id) - 1)
                try:
                    while rand_e2 in self.kg.train_objects[int(e1[i])][int(r[i])]:
                        rand_e2 = random.randint(0, len(self.kg.entity2id) - 1)
                except:
                    pass
                e1_list.append(int(e1[i]))
                e2_list.append(rand_e2)
                r_list.append(int(r[i]))
            else:
                rand_r = random.randint(0, len(self.kg.relation2id) - 1)
                try:
                    while int(e2[i]) in self.kg.train_objects[int(e1[i])][rand_r]:
                        rand_r = random.randint(0, len(self.kg.relation2id) - 1)
                except:
                    pass
                e1_list.append(int(e1[i]))
                e2_list.append(int(e2[i]))
                r_list.append(rand_r)
        return var_cuda(torch.LongTensor(e1_list), requires_grad=False), var_cuda(torch.LongTensor(e2_list), requires_grad=False), var_cuda(torch.LongTensor(r_list), requires_grad=False)

    def get_corrupt_relation(self, e1, e2, r):
        batch_size = e1.size()[0]
        r_list = []
        for i in range(batch_size):
            rand_r = random.randint(0, len(self.kg.relation2id) - 1)
            try:
                while int(e2[i]) in self.kg.train_objects[int(e1[i])][rand_r]:
                    rand_r = random.randint(0, len(self.kg.relation2id) - 1)
            except:
                pass
            r_list.append(rand_r)
        return var_cuda(torch.LongTensor(r_list), requires_grad=False)

    def loss(self, mini_batch):
        kg, mdl = self.kg, self.mdl
        # compute object training loss
        if self.model == '!TransE':
            e1, e2, r = self.format_batch(mini_batch)
            corrupt_e1, corrupt_e2, corrupt_r = self.get_corrupt_triple(e1, e2, r)
            pred_scores = mdl.forward_train(e1, e2, r, kg)
            pred_scores = torch.sum(pred_scores, 1)
            corrupt_scores = mdl.forward_train(corrupt_e1, corrupt_e2, corrupt_r, kg)
            corrupt_scores = torch.sum(corrupt_scores, 1)
            y = Variable(torch.Tensor([-1])).cuda()
            loss = self.loss_fun(pred_scores, corrupt_scores, y)
        else:
            e1, e2, r = self.format_batch(mini_batch, num_labels=kg.num_entities)
            e2_label = ((1 - self.label_smoothing_epsilon) * e2) + (1.0 / e2.size(1))
            pred_scores = mdl.forward(e1, r, kg)
            loss = self.loss_fun(pred_scores, e2_label)
        loss_dict = {}
        loss_dict['model_loss'] = loss
        loss_dict['print_loss'] = float(loss)
        return loss_dict

    def predict(self, mini_batch, verbose=False):
        kg, mdl = self.kg, self.mdl
        e1, e2, r = self.format_batch(mini_batch)
        if self.model == 'hypere':
            pred_scores = mdl.forward(e1, r, kg, [self.secondary_kg])
        elif self.model == 'triplee':
            pred_scores = mdl.forward(e1, r, kg, [self.secondary_kg, self.tertiary_kg])
        else:
            pred_scores = mdl.forward(e1, r, kg)
        return pred_scores

    def get_subject_mask(self, e1_space, e2, q):
        kg = self.kg
        if kg.args.mask_test_false_negatives:
            answer_vectors = kg.all_subject_vectors
        else:
            answer_vectors = kg.train_subject_vectors
        subject_masks = []
        for i in range(len(e1_space)):
            _e2, _q = int(e2[i]), int(q[i])
            if not _e2 in answer_vectors or not _q in answer_vectors[_e2]:
                answer_vector = var_cuda(torch.LongTensor([[kg.num_entities]]))
            else:
                answer_vector = answer_vectors[_e2][_q]
            subject_mask = torch.sum(e1_space[i].unsqueeze(0) == answer_vector, dim=0)
            subject_masks.append(subject_mask)
        subject_mask = torch.cat(subject_masks).view(len(e1_space), -1)
        return subject_mask

    def get_object_mask(self, e2_space, e1, q):
        kg = self.kg
        if kg.args.mask_test_false_negatives:
            answer_vectors = kg.all_object_vectors
        else:
            answer_vectors = kg.train_object_vectors
        object_masks = []
        for i in range(len(e2_space)):
            _e1, _q = int(e1[i]), int(q[i])
            if not e1 in answer_vectors or not q in answer_vectors[_e1]:
                answer_vector = var_cuda(torch.LongTensor([[kg.num_entities]]))
            else:
                answer_vector = answer_vectors[_e1][_q]
            object_mask = torch.sum(e2_space[i].unsqueeze(0) == answer_vector, dim=0)
            object_masks.append(object_mask)
        object_mask = torch.cat(object_masks).view(len(e2_space), -1)
        return object_mask

    def export_reward_shaping_parameters(self):
        """
        Export knowledge graph embeddings and fact network parameters for reward shaping models.
        """
        fn_state_dict_path = os.path.join(self.model_dir, 'fn_state_dict')
        fn_kg_state_dict_path = os.path.join(self.model_dir, 'fn_kg_state_dict')
        torch.save(self.mdl.state_dict(), fn_state_dict_path)
        print('Fact network parameters export to {}'.format(fn_state_dict_path))
        torch.save(self.kg.state_dict(), fn_kg_state_dict_path)
        print('Knowledge graph embeddings export to {}'.format(fn_kg_state_dict_path))

    def export_fuzzy_facts(self):
        """
        Export high confidence facts according to the model.
        """
        kg, mdl = self.kg, self.mdl

        # Gather all possible (subject, relation) and (relation, object) pairs
        sub_rel, rel_obj = {}, {}
        for file_name in ['raw.kb', 'train.triples', 'dev.triples', 'test.triples']:
            with open(os.path.join(self.data_dir, file_name)) as f:
                for line in f:
                    e1, e2, r = line.strip().split()
                    e1_id, e2_id, r_id = kg.triple2ids((e1, e2, r))
                    if not e1_id in sub_rel:
                        sub_rel[e1_id] = {}
                    if not r_id in sub_rel[e1_id]:
                        sub_rel[e1_id][r_id] = set()
                    sub_rel[e1_id][r_id].add(e2_id)
                    if not e2_id in rel_obj:
                        rel_obj[e2_id] = {}
                    if not r_id in rel_obj[e2_id]:
                        rel_obj[e2_id][r_id] = set()
                    rel_obj[e2_id][r_id].add(e1_id)

        o_f = open(os.path.join(self.data_dir, 'train.fuzzy.triples'), 'w')
        print('Saving fuzzy facts to {}'.format(os.path.join(self.data_dir, 'train.fuzzy.triples')))
        count = 0
        # Save recovered objects
        e1_ids, r_ids = [], []
        for e1_id in sub_rel:
            for r_id in sub_rel[e1_id]:
                e1_ids.append(e1_id)
                r_ids.append(r_id)
        for i in range(0, len(e1_ids), self.batch_size):
            e1_ids_b = e1_ids[i:i+self.batch_size]
            r_ids_b = r_ids[i:i+self.batch_size]
            e1 = var_cuda(torch.LongTensor(e1_ids_b))
            r = var_cuda(torch.LongTensor(r_ids_b))
            pred_scores = mdl.forward(e1, r, kg)
            for j in range(pred_scores.size(0)):
                for _e2 in range(pred_scores.size(1)):
                    if _e2 in [NO_OP_ENTITY_ID, DUMMY_ENTITY_ID]:
                        continue
                    if pred_scores[j, _e2] >= self.theta:
                        _e1 = int(e1[j])
                        _r = int(r[j])
                        o_f.write('{}\t{}\t{}\t{}\n'.format(
                            kg.id2entity[_e1], kg.id2entity[_e2], kg.id2relation[_r], float(pred_scores[j, _e2])))
                        count += 1
                        if count % 1000 == 0:
                            print('{} fuzzy facts exported'.format(count))
        # Save recovered subjects
        e2_ids, r_ids = [], []
        for e2_id in rel_obj:
            for r_id in rel_obj[e2_id]:
                e2_ids.append(e2_id)
                r_ids.append(r_id)
        e1 = int_var_cuda(torch.arange(kg.num_entities))
        for i in range(len(e2_ids)):
            r = int_fill_var_cuda(e1.size(), r_ids[i])
            e2 = int_fill_var_cuda(e1.size(), e2_ids[i])
            pred_scores = mdl.forward_fact(e1, r, e2, kg)
            for j in range(pred_scores.size(1)):
                if pred_scores[j] > self.theta:
                    _e1 = int(e1[j])
                    if _e1 in [NO_OP_ENTITY_ID, DUMMY_ENTITY_ID]:
                        continue
                    _r = int(r[j])
                    _e2 = int(e2[j])
                    if _e1 in sub_rel and _r in sub_rel[_e1]:
                        continue
                    o_f.write('{}\t{}\t{}\t{}\n'.format(
                        kg.id2entity[_e1], kg.id2entity[_e2], kg.id2relation[_r], float(pred_scores[j])))
                    count += 1
                    if count % 1000 == 0:
                        print('{} fuzzy facts exported'.format(count))
