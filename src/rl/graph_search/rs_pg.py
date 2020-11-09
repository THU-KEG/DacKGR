"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Policy gradient with reward shaping.
"""

from tqdm import tqdm

import torch

from src.emb.fact_network import get_conve_nn_state_dict, get_conve_kg_state_dict, \
    get_complex_kg_state_dict, get_distmult_kg_state_dict, get_ptranse_kg_state_dict, \
    get_tucker_nn_state_dict, get_tucker_kg_state_dict
from src.rl.graph_search.pg import PolicyGradient
import src.utils.ops as ops
import src.rl.graph_search.beam_search as search
from src.utils.ops import int_fill_var_cuda, var_cuda, zeros_var_cuda
import random


class RewardShapingPolicyGradient(PolicyGradient):
    def __init__(self, args, kg, pn, fn_kg, fn, fn_secondary_kg=None):
        super(RewardShapingPolicyGradient, self).__init__(args, kg, pn)
        self.reward_shaping_threshold = args.reward_shaping_threshold

        # Fact network modules
        self.fn_kg = fn_kg
        self.fn = fn
        self.fn_secondary_kg = fn_secondary_kg
        self.mu = args.mu
        self.use_state_prediction = args.use_state_prediction
        self.hits = 0.0
        self.num = 0.0
        self.strategy = args.strategy

        fn_model = self.fn_model
        if fn_model in ['conve']:
            fn_state_dict = torch.load(args.conve_state_dict_path, map_location=('cuda:' + str(args.gpu)))
            fn_nn_state_dict = get_conve_nn_state_dict(fn_state_dict)
            fn_kg_state_dict = get_conve_kg_state_dict(fn_state_dict)
            self.fn.load_state_dict(fn_nn_state_dict)
        elif fn_model == 'tucker':
            fn_state_dict = torch.load(args.tucker_state_dict_path, map_location=('cuda:' + str(args.gpu)))
            fn_nn_state_dict = get_tucker_nn_state_dict(fn_state_dict)
            fn_kg_state_dict = get_tucker_kg_state_dict(fn_state_dict)
            self.fn.load_state_dict(fn_nn_state_dict)
        elif fn_model == 'distmult':
            fn_state_dict = torch.load(args.distmult_state_dict_path, map_location=('cuda:' + str(args.gpu)))
            fn_kg_state_dict = get_distmult_kg_state_dict(fn_state_dict)
        elif fn_model == 'complex':
            fn_state_dict = torch.load(args.complex_state_dict_path, map_location=('cuda:' + str(args.gpu)))
            fn_kg_state_dict = get_complex_kg_state_dict(fn_state_dict)
        elif fn_model == 'hypere':
            fn_state_dict = torch.load(args.conve_state_dict_path)
            fn_kg_state_dict = get_conve_kg_state_dict(fn_state_dict)
        elif fn_model == 'PTransE':
            fn_state_dict = torch.load(args.ptranse_state_dict_path, map_location=('cuda:' + str(args.gpu)))
            fn_kg_state_dict = get_ptranse_kg_state_dict(fn_state_dict)
        else:
            raise NotImplementedError
        self.fn_kg.load_state_dict(fn_kg_state_dict)
        if fn_model == 'hypere':
            complex_state_dict = torch.load(args.complex_state_dict_path)
            complex_kg_state_dict = get_complex_kg_state_dict(complex_state_dict)
            self.fn_secondary_kg.load_state_dict(complex_kg_state_dict)

        self.fn.eval()
        self.fn_kg.eval()
        ops.detach_module(self.fn)
        ops.detach_module(self.fn_kg)
        if fn_model == 'hypere':
            self.fn_secondary_kg.eval()
            ops.detach_module(self.fn_secondary_kg)

    def reward_fun(self, e1, r, e2, pred_e2, path_trace):
        if self.model.endswith('.rso'):
            oracle_reward = forward_fact_oracle(e1, r, pred_e2, self.kg)
            return oracle_reward
        else:
            if self.fn_secondary_kg:
                real_reward = self.fn.forward_fact(e1, r, pred_e2, self.fn_kg, [self.fn_secondary_kg]).squeeze(1)
            elif self.fn_model == 'PTransE':
                real_reward = self.fn.forward_fact(e1, r, pred_e2, self.fn_kg, path_trace).squeeze(1)
            else:
                real_reward = self.fn.forward_fact(e1, r, pred_e2, self.fn_kg).squeeze(1)
            real_reward_mask = (real_reward > self.reward_shaping_threshold).float()
            real_reward *= real_reward_mask
            if self.model.endswith('rsc'):
                return real_reward
            else:
                # print(e2.shape)
                # print(pred_e2.shape)
                binary_reward = torch.gather(e2, 1, pred_e2.unsqueeze(-1)).squeeze(-1).float()

                # binary_reward = (pred_e2 == e2).float()
                return binary_reward + self.mu * (1 - binary_reward) * real_reward

    def loss(self, mini_batch):
        
        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r
    
        e1, e2, r, kg_pred = self.format_batch(mini_batch, num_tiles=self.num_rollouts, num_labels=self.kg.num_entities)
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps, kg_pred=kg_pred)

        # Compute policy gradient loss
        pred_e2 = output['pred_e2']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']
        path_trace = output['path_trace']

        # Compute discounted reward
        final_reward = self.reward_fun(e1, r, e2, pred_e2, path_trace)
        if self.baseline != 'n/a':
            final_reward = stablize_reward(final_reward)
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()

        loss_dict = {}
        loss_dict['model_loss'] = pg_loss
        loss_dict['print_loss'] = float(pt_loss)
        loss_dict['reward'] = final_reward
        loss_dict['entropy'] = float(entropy.mean())
        if self.run_analysis:
            fn = torch.zeros(final_reward.size())
            for i in range(len(final_reward)):
                if not final_reward[i]:
                    if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
                        fn[i] = 1
            loss_dict['fn'] = fn

        return loss_dict

    def format_batch(self, batch_data, num_labels=-1, num_tiles=1, inference=False):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """
        def convert_to_binary_multi_subject(e1):
            e1_label = zeros_var_cuda([len(e1), num_labels])
            for i in range(len(e1)):
                e1_label[i][e1[i]] = 1
            return e1_label

        def convert_to_binary_multi_object(e2):
            e2_label = zeros_var_cuda([len(e2), self.kg.num_entities])
            for i in range(len(e2)):
                e2_label[i][e2[i]] = 1
            return e2_label

        batch_e1, batch_e2, batch_r = [], [], []
        for i in range(len(batch_data)):
            e1, e2, r = batch_data[i]
            batch_e1.append(e1)
            batch_e2.append(e2)
            batch_r.append(r)
        if inference:
            batch_e2_new = []
            for i in range(len(batch_e1)):
                tmp = []
                if batch_e1[i] in self.kg.train_objects and batch_r[i] in self.kg.train_objects[batch_e1[i]]:
                    tmp += list(self.kg.train_objects[batch_e1[i]][batch_r[i]])
                batch_e2_new.append(tmp)
            batch_e2 = batch_e2_new
        batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
        batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
        if type(batch_e2[0]) is list:
            batch_e2 = convert_to_binary_multi_object(batch_e2)
        elif type(batch_e1[0]) is list:
            batch_e1 = convert_to_binary_multi_subject(batch_e1)
        else:
            batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
        # Rollout multiple times for each example

        # weight emb
        # S = self.fn.forward(batch_e1, batch_r, self.fn_kg).view(-1, self.kg.num_entities)
        # pred_kg = torch.matmul(S, self.kg.entity_embeddings.weight) / torch.sum(S, dim=1, keepdim=True)

        # sample emb
        if self.strategy == 'sample':
            # sample emb
            S = self.fn.forward(batch_e1, batch_r, self.fn_kg).view(-1, self.kg.num_entities)
            a = torch.multinomial(S, 1)
            pred_kg = self.kg.entity_embeddings(a.view(self.batch_size))
        elif self.strategy == 'avg':
            # weight emb
            S = self.fn.forward(batch_e1, batch_r, self.fn_kg).view(-1, self.kg.num_entities)
            pred_kg = torch.matmul(S, self.kg.entity_embeddings.weight) / torch.sum(S, dim=1, keepdim=True)
        elif self.strategy == 'top1':
            # Top K method
            S = self.fn.forward(batch_e1, batch_r, self.fn_kg)
            Sx, idx = torch.topk(S, k=1, dim=1)
            Sx = Sx.unsqueeze(-1)
            S = self.kg.entity_embeddings(idx) * Sx
            x = torch.sum(Sx, dim=1, keepdim=True)
            S = S / x
            pred_kg = torch.sum(S, dim=1)

        # hits = float(torch.sum(torch.gather(batch_e2, 1, a), dim=0))
        # self.hits += hits
        # self.num += float(a.shape[0])
        # print('Hits ratio: {}'.format(self.hits / self.num))

        # Top K method
        # S = self.fn.forward(batch_e1, batch_r, self.fn_kg)
        # Sx, idx = torch.topk(S, k=1, dim=1)
        # Sx = Sx.unsqueeze(-1)
        # S = self.kg.entity_embeddings(idx) * Sx
        # x = torch.sum(Sx, dim=1, keepdim=True)
        # S = S / x
        # pred_kg = torch.sum(S, dim=1)

        # Top K with fc
        # S = self.fn.forward(batch_e1, batch_r, self.fn_kg)
        # _, idx = torch.topk(S, k=10, dim=1)
        # pred_kg = self.fc1(self.kg.entity_embeddings(idx)).view(self.batch_size, -1)

        if num_tiles > 1:
            batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
            batch_r = ops.tile_along_beam(batch_r, num_tiles)
            batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
            pred_kg = ops.tile_along_beam(pred_kg, num_tiles)
        return batch_e1, batch_e2, batch_r, pred_kg

    def rollout(self, e_s, q, e_t, num_steps, kg_pred, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query embedding.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :param visualize_action_probs: If set, save action probabilities for visualization.
        :return pred_e2: Target entities reached at the end of rollout.
        :return log_path_prob: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn = self.kg, self.mdl

        # Initialization
        log_action_probs = []
        action_entropy = []
        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
        path_components = []

        path_trace = [(r_s, e_s)]
        pn.initialize_path((r_s, e_s), kg)

        for t in range(num_steps):
            last_r, e = path_trace[-1]
            obs = [e_s, q, e_t, t==(num_steps-1), last_r, seen_nodes]
            db_outcomes, inv_offset, policy_entropy = pn.transit(
                e, obs, kg, kg_pred=kg_pred, fn_kg=self.fn_kg, use_action_space_bucketing=self.use_action_space_bucketing, use_kg_pred=self.use_state_prediction)
            sample_outcome = self.sample_action(db_outcomes, inv_offset)
            action = sample_outcome['action_sample']
            pn.update_path(action, kg)
            action_prob = sample_outcome['action_prob']
            log_action_probs.append(ops.safe_log(action_prob))
            action_entropy.append(policy_entropy)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append(action)

            if visualize_action_probs:
                top_k_action = sample_outcome['top_actions']
                top_k_action_prob = sample_outcome['top_action_probs']
                path_components.append((e, top_k_action, top_k_action_prob))

        pred_e2 = path_trace[-1][1]
        self.record_path_trace(path_trace)

        return {
            'pred_e2': pred_e2,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace,
            'path_components': path_components
        }
    
    def predict(self, mini_batch, verbose=False):
        kg, pn = self.kg, self.mdl
        e1, e2, r, kg_pred = self.format_batch(mini_batch, inference=False)
        beam_search_output = search.beam_search(
            pn, e1, r, e2, kg, self.num_rollout_steps, self.beam_size, use_kg_pred=self.use_state_prediction, kg_pred=kg_pred, fn_kg=self.fn_kg)
        pred_e2s = beam_search_output['pred_e2s']
        pred_e2_scores = beam_search_output['pred_e2_scores']
        if verbose:
            # print inference paths
            MAX_PATH = 10
            search_traces = beam_search_output['search_traces']
            output_beam_size = min(self.beam_size, pred_e2_scores.shape[1], MAX_PATH)
            for i in range(len(e1)):
                h = kg.id2entity[int(e1[i])]
                rel = kg.id2relation[int(r[i])]
                t = kg.id2entity[int(e2[i])]
                print('({}, {}, {})'.format(h, rel, t))
                for j in range(output_beam_size):
                    ind = i * output_beam_size + j
                    if pred_e2s[i][j] == kg.dummy_e:
                        break
                    search_trace = []
                    for k in range(len(search_traces)):
                        search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
                    print('<PATH> {}'.format(ops.format_path(search_trace, kg)))
        with torch.no_grad():
            pred_scores = zeros_var_cuda([len(e1), kg.num_entities])
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
        return pred_scores

    def test_fn(self, examples):
        fn_kg, fn = self.fn_kg, self.fn
        pred_scores = []
        for example_id in tqdm(range(0, len(examples), self.batch_size)):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)
            e1, e2, r = self.format_batch(mini_batch)
            if self.fn_secondary_kg:
                pred_score = fn.forward_fact(e1, r, e2, fn_kg, [self.fn_secondary_kg])
            elif self.fn_model == 'PTransE':
                # TODO
                pred_score = fn.forward_fact(e1, r, e2, fn_kg)
            else:
                pred_score = fn.forward_fact(e1, r, e2, fn_kg)
            pred_scores.append(pred_score[:mini_batch_size])
        return torch.cat(pred_scores)

    @property
    def fn_model(self):
        return self.model.split('.')[2]

def forward_fact_oracle(e1, r, e2, kg):
    oracle = zeros_var_cuda([len(e1), kg.num_entities]).cuda()
    for i in range(len(e1)):
        _e1, _r = int(e1[i]), int(r[i])
        if _e1 in kg.all_object_vectors and _r in kg.all_object_vectors[_e1]:
            answer_vector = kg.all_object_vectors[_e1][_r]
            oracle[i][answer_vector] = 1
        else:
            raise ValueError('Query answer not found')
    oracle_e2 = ops.batch_lookup(oracle, e2.unsqueeze(1))
    return oracle_e2
