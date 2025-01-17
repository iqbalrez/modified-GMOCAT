import sys
import json
import os
import math
import yaml
import torch
import dgl
import random
import copy as cp
import numpy as np
import logging
from util import *
from sklearn.metrics import roc_auc_score
from .dataset import TrainDataset
from .Env import Env

class GCATEnv(Env):
    def __init__(self, args):
        super(GCATEnv, self).__init__(args)
        
        self.nov_reward_map = self.load_nov_reward()
        
        if args.target_concepts != [0]:
          self.target_concepts = set(args.target_concepts)
        else:
          self.target_concepts = set(concept for concepts in self.know_map.values() for concept in concepts)

    def reset_with_users(self, uids):
        self.state = {}
        self.avail_questions = {}
        self.used_questions = {}
        self.know_stat = {}
        self.concept_consistency = {}

        for uu in uids:
            self.avail_questions[uu] = {
                qid for qid in self.sup_rates[uu].keys() if any(concept in self.target_concepts for concept in self.know_map[qid])
            }
            self.used_questions[uu] = []
            self.concept_consistency[uu] = {concept: False for concept in self.target_concepts}
        
        self.uids = uids
        self.cnt_step = 0
        self.state['batch_question']= np.zeros((len(uids), len(self.target_concepts)*2))
        self.state['batch_answer']= np.zeros((len(uids), len(self.target_concepts)*2))
        self.last_div = 0

        self.know_stat = self.model.get_knowledge_status(torch.tensor(self.uids).to(self.device))
        self.previous_know_stat = self.know_stat * 0
        
        _, pred, correct_query = self.model.cal_loss(self.uids, self.query_rates, self.know_map)
        self.last_accuracy = np.zeros(len(uids))
        for i in range(len(pred)):
            pred_bin = np.where(pred[i] > 0.5, 1, 0)
            ACC = np.sum(np.equal(pred_bin, correct_query[i])) / len(pred_bin) 
            self.last_accuracy[i] = ACC

        return self.state

    def step(self, action, last_epoch):
        for i, uu in enumerate(self.uids):
            if action[i] not in self.avail_questions[uu]:
                self.logger.info("action exited")
                print(f"action exited at {i} on {uu}:", action[i], len(self.avail_questions[uu]), len(set(self.sup_rates[uu].keys())))
                exit()

        reward, pred, label, rate = self.reward(action)
        
        self.previous_know_stat = self.know_stat
        self.know_stat = self.model.get_knowledge_status(torch.tensor(self.uids).to(self.device))
        
        done = False

        coverages = []
        for i, uu in enumerate(self.uids):
                    
            self.avail_questions[uu].remove(action[i])
            self.used_questions[uu].append(action[i])
            
            for concept, prev_score, curr_score in zip(
                self.know_map[action[i]],
                self.previous_know_stat[i], 
                self.know_stat[i], 
            ):
                if concept in self.target_concepts:
                  error = abs(curr_score - prev_score)
                  print(f"\nerror[{concept}]: {error}")
                  if error < 0.010:  # Threshold stabilitas
                      self.concept_consistency[uu][concept] = True
                      print("consistency[{concept}]: TRUE")
                      stable_questions = []
                      for qid in self.avail_questions[uu]:
                          # Periksa apakah semua konsep dalam soal sudah stabil
                          concepts_in_question = self.know_map[qid]
                          all_concepts_stable = all(
                              concept in self.concept_consistency[uu] and self.concept_consistency[uu][concept]
                              for concept in concepts_in_question
                          )
                          
                          if all_concepts_stable:
                              stable_questions.append(qid)

                      # Hapus hanya soal yang sepenuhnya stabil
                      for qid in stable_questions:
                          self.avail_questions[uu].remove(qid)

                  # print(f"cc {self.concept_consistency[uu][concept]}")

            all_concepts = set()
            tested_concepts = set()
            for qid in self.rates[uu]: 
                all_concepts.update(set(self.know_map[qid]))
            for qid in self.used_questions[uu]:
                tested_concepts.update(set(self.know_map[qid]))
            coverage = len(tested_concepts) / len(all_concepts)
            coverages.append(coverage)

            all_stable = all(
                self.concept_consistency[uu][concept] 
                for concept in self.target_concepts
            )

            if (coverage == 1.0 and all_stable) or len(self.avail_questions[uu])<1:
              print(f"\nStopped at {self.cnt_step + 1} | Coverage: {coverage}, All stable: {all_stable}, Len: {len(self.avail_questions[uu])}")
              done = True
              
        cov = sum(coverages) / len(coverages)

        all_info = [{"pred": pred[i], "label": label[i], "rate":rate[i]} for i, uu in enumerate(self.uids)]

        self.cnt_step += 1

        self.state['batch_question'][:, self.cnt_step] = action
        self.state['batch_answer'][:, self.cnt_step] = np.array(rate)+1 # idx=0 is pad
        # self.state['batch_question'] = action
        # self.state['batch_answer']= np.array(rate)+1 # idx=0 is pad

        return self.state, reward, done, all_info, cov

    def reward(self, action):
        # update cdm
        records = [(uu, action[i], self.rates[uu][action[i]]) for i, uu in enumerate(self.uids)]
        self.dataset = TrainDataset(records, self.know_map, self.user_num, self.item_num, self.know_num)
        self.model.update(self.dataset, self.args.cdm_lr, epochs=self.args.cdm_epoch, batch_size=self.args.cdm_bs)

        # eval on query
        loss, pred, correct_query = self.model.cal_loss(self.uids, self.query_rates, self.know_map)
        final_rate = [self.rates[uu][action[i]] for i, uu in enumerate(self.uids)]

        new_accuracy = np.zeros(len(self.uids))
        for i in range(len(pred)):
            pred_bin = np.where(pred[i] > 0.5, 1, 0)
            ACC = np.sum(np.equal(pred_bin, correct_query[i])) / len(pred_bin) 
            new_accuracy[i] = ACC

        # tmp = new_accuracy - self.last_accuracy
        # tmp = np.where(tmp>0,1,tmp)
        # acc_rwd = np.where(tmp<0,-1, tmp)
        acc_rwd = new_accuracy - self.last_accuracy
        self.last_accuracy = new_accuracy

        div_rwd = np.array([self.compute_div_reward(list(self.sup_rates[uu].keys()), self.know_map, self.used_questions[uu], action[i], self.concept_consistency[uu]) for i, uu in enumerate(self.uids)])

        nov_rwd = np.array([self.nov_reward_map[action[i]] for i in range(len(action))])
        # print(acc_rwd.shape, div_rwd.shape, nov_rwd.shape) # (B,)
        rwd = np.concatenate((acc_rwd.reshape(-1,1), div_rwd.reshape(-1,1), nov_rwd.reshape(-1,1)),axis=-1) # (B,3)
        # print(rwd[:5])
        
        return rwd, pred, correct_query, final_rate
    
    def load_nov_reward(self):
        with open(f'data/nov_reward_{self.args.data_name}.json', encoding='utf8') as i_f:
            concept_data = json.load(i_f) 
        
        nov_reward_map = {}
        for k,v in concept_data.items():
            qid_pad = int(k) +1
            nov_reward_map[qid_pad] = v
            
        return nov_reward_map
    
    def compute_div_reward(self, all_questions, concept_map, tested_questions, qid, concept_consistency):
        concept_cnt = set()
        reward = 0
        
        for q in list(tested_questions):
            for c in concept_map[q]:
                concept_cnt.add(c)
        
        for c in concept_map[qid]:
            if c not in concept_cnt:
                reward = 1
            elif not concept_consistency.get(c, False):
                reward = 1

        return reward