import torch
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

import time
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import Adam, RMSprop, SGD
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader

class NetModifTime(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, prednet_len1=128, prednet_len2=64):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.prednet_input_len = self.knowledge_dim

        self.prednet_len1, self.prednet_len2 = prednet_len1, prednet_len2 

        self.stu_dim = self.knowledge_dim
        self.emb_dim = knowledge_n
        self.knowledge_n = knowledge_n
        #self.exercise_emb = exer_n

        super(NetModifTime, self).__init__()

        # prediction sub-net
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.k_diff_full = nn.Linear(2 * self.emb_dim, 1) #ncf1

        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.k_time_full = nn.Linear(1, self.knowledge_dim)  # emb_dim + 1 karena kita menambahkan 1 fitur dari tensor `time`

        self.prednet_full1 = nn.Linear(
            self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point , waktu=None):
        # before prednet
        exercise_emb = self.exercise_emb(input_exercise)
        waktu = waktu.float().unsqueeze(1)
        time_scalar = torch.sigmoid(waktu)
        #time_expanded = time_scalar.expand(-1, exercise_emb.size(1))  # Menjadi (32, 101)
        #combined_input = torch.cat((exercise_emb, time_scalar), dim=1)
         # Menerapkan lapisan linear
        k_time = self.k_time_full(time_scalar)


        stu_emb = self.student_emb(stu_id)
        batch, dim = stu_emb.size()

        exer_emb = exercise_emb.view(batch, 1, dim). repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        #k_time = k_time.view(batch, 1, dim). repeat(1, self.knowledge_n, 1)
        #print('exercise_emb after view', exer_emb.shape)
        #k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim= -1, keepdim=False))
        #k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((k_time, knowledge_emb), dim=-1))).view(batch, -1)#ncf1

        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10

        # Pernyataan debug untuk memeriksa bentuk
        #print(f"stu_emb shape: {stu_emb.shape}")
        #print(f"stat_emb shape: {stat_emb.shape}")
        #print(f"k_difficulty shape: {k_difficulty.shape}")
        #print(f"e_difficulty shape: {e_difficulty.shape}")
        # print(f"combined_input shape: {combined_input.shape}")
        #print(f"input_knowledge_point shape: {input_knowledge_point.shape}")
        # prednet
        input_x = e_difficulty * (stat_emb - k_time - (k_difficulty )) * input_knowledge_point


        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1

    def init_stu_emb(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'student' in name:
                nn.init.xavier_normal_(param)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

class NCDMModifTime():

  def __init__(self, args, student_n, exer_n, knowledge_n):
      super().__init__()
      self.args = args
      self.device = torch.device('cuda')
      self.num_knowledges = knowledge_n
      self.ncdm_net = NetModifTime(student_n, exer_n, knowledge_n).to(self.device)
      self.model = self.ncdm_net
      self.loss_function = nn.BCELoss()

  @property
  def name(self):
      return 'Neural Cognitive Diagnosis'

  def init_stu_emb(self):
      self.model.init_stu_emb()

  def cal_loss(self, sids, query_rates, concept_map):
      device = self.device
      real = []
      pred = []
      all_loss = np.zeros(len(sids))
      with torch.no_grad():
          self.model.eval()
          for idx, sid in enumerate(sids):
              question_ids = list(query_rates[sid].keys())
              student_ids = [sid] * len(question_ids)
              concepts_embs = []
              times = []
              for qid in question_ids:
                  concepts = concept_map[qid]
                  concepts_emb = [0.] * self.num_knowledges
                  for concept in concepts:
                      concepts_emb[concept] = 1.0
                  concepts_embs.append(concepts_emb)
                  times.append(query_rates[sid]["times"].get[qid])

              labels = [query_rates[sid][qid] for qid in question_ids]
              real.append(np.array(labels))

              student_ids = torch.LongTensor(student_ids).to(device)
              question_ids = torch.LongTensor(question_ids).to(device)
              concepts_embs = torch.Tensor(concepts_embs).to(device)
              labels = torch.LongTensor(labels).to(device)
              times = torch.FloatTensor(times).to(device)

              output = self.model(student_ids, question_ids, concepts_embs, times)
              loss = self._loss_function(output, labels).cpu().detach().numpy()
              all_loss[idx] = loss.item()
              pred.append(np.array(output.view(-1).tolist()))
          self.model.train()
      
      return all_loss, pred, real
          
  def train(self, train_data, lr, batch_size, epochs, path, silence=False, test_data=None): 
      device = self.device
      logger = logging.getLogger("Pretrain")
      logger.info('train on {}'.format(device))
      train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
      optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
      best_loss = 1000000

      stus = []
      epochl = []
      exercise_list = []
      userid_list = []

      epoch_acc_plot = []
      epoch_auc_plot = []
      epoch_loss_plot = []

      for epoch_i in range(epochs):
          epoch_losses = []

          y_true, y_pred = [], []
          batch_count = 0
          loss = 0.0
          for cnt, (user_id, item_id, knowledge_emb, y, time) in enumerate(train_loader):
              batch_count += 1
              user_id = user_id.to(device)
              item_id = item_id.to(device)
              knowledge_emb= knowledge_emb.to(device)
              y = y.to(device)
              time = time.to(device)
              pred = self.ncdm_net(
                  user_id, item_id, knowledge_emb , time)
              bz_loss = self._loss_function(pred, y)
              optimizer.zero_grad()
              bz_loss.backward()
              optimizer.step()
              self.model.apply_clipper()
              loss += bz_loss.data.float()
          loss /= len(train_loader) 
          logger.info('Epoch [{}]: loss={:.5f}'.format(epoch_i, loss)) 

          if loss < best_loss:
                best_loss = loss
                logger.info('Store model')
                self.adaptest_save(path) 

  def _loss_function(self, pred, real):
        pred_0 = torch.ones(pred.size()).to(self.device) - pred
        output = torch.cat((pred_0, pred), 1)
        criteria = nn.NLLLoss()
        return criteria(torch.log(output), real)
    
  def adaptest_save(self, path):
      model_dict = self.model.state_dict()
      model_dict = {k:v for k,v in model_dict.items() if 'student' not in k}
      torch.save(model_dict, path)
  
  def adaptest_load(self, path):
      self.model.load_state_dict(torch.load(path), strict=False)
      self.model.to(self.device)
  
  def update(self, tested_dataset, lr, epochs, batch_size):
      device = self.device
      optimizer = torch.optim.Adam(self.model.student_emb.parameters(), lr=lr)
      dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)

      for ep in range(1, epochs + 1):
          loss = 0.0
          # log_steps = 100
          for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(dataloader):
              student_ids = student_ids.to(device)
              question_ids = question_ids.to(device)
              labels = labels.to(device)
              concepts_emb = concepts_emb.to(device)
              pred = self.model(student_ids, question_ids, concepts_emb)
              bz_loss = self._loss_function(pred, labels)
              optimizer.zero_grad()
              bz_loss.backward()
              optimizer.step()
              self.model.apply_clipper()
              loss += bz_loss.data.float()
              # if cnt % log_steps == 0:
                  # print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, cnt, loss / cnt))
  
  def get_pred(self, user_ids, avail_questions, concept_map):
      device = self.device

      pred_all = {}
      with torch.no_grad():
          self.model.eval()
          for sid in user_ids:
              pred_all[sid] = {}
              question_ids =  list(avail_questions[sid])
              student_ids = [sid] * len(question_ids)
        
              concepts_embs = []
              for qid in question_ids:
                  concepts = concept_map[qid]
                  concepts_emb = [0.] * self.num_knowledges
                  for concept in concepts:
                      concepts_emb[concept] = 1.0
                  concepts_embs.append(concepts_emb)
              student_ids = torch.LongTensor(student_ids).to(device)
              question_ids = torch.LongTensor(question_ids).to(device)
              concepts_embs = torch.Tensor(concepts_embs).to(device)
              output = self.model(student_ids, question_ids, concepts_embs).view(-1).tolist()
              for i, qid in enumerate(list(avail_questions[sid])):
                  pred_all[sid][qid] = output[i]
          self.model.train()
      return pred_all

  def expected_model_change(self, sid: int, qid: int, pred_all: dict, concept_map):
      """ get expected model change
      Args:
          student_id: int, student id
          question_id: int, question id
      Returns:
          float, expected model change
      """
      # epochs = self.args.cdm_epoch
      epochs = 1
      # lr = self.args.cdm_lr
      lr = 0.01
      device = self.device
      optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

      for name, param in self.model.named_parameters():
          if 'student' not in name:
              param.requires_grad = False

      original_weights = self.model.student_emb.weight.data.clone()

      student_id = torch.LongTensor([sid]).to(device)
      question_id = torch.LongTensor([qid]).to(device)
      concepts = concept_map[qid]
      concepts_emb = [0.] * self.num_knowledges
      for concept in concepts:
          concepts_emb[concept] = 1.0
      concepts_emb = torch.Tensor([concepts_emb]).to(device)
      correct = torch.LongTensor([1]).to(device)
      wrong = torch.LongTensor([0]).to(device)

      for ep in range(epochs):
          optimizer.zero_grad()
          pred = self.model(student_id, question_id, concepts_emb)
          loss = self._loss_function(pred, correct)
          loss.backward()
          optimizer.step()

      pos_weights = self.model.student_emb.weight.data.clone()
      self.model.student_emb.weight.data.copy_(original_weights)

      for ep in range(epochs):
          optimizer.zero_grad()
          pred = self.model(student_id, question_id, concepts_emb)
          loss = self._loss_function(pred, wrong)
          loss.backward()
          optimizer.step()

      neg_weights = self.model.student_emb.weight.data.clone()
      self.model.student_emb.weight.data.copy_(original_weights)

      for param in self.model.parameters():
          param.requires_grad = True

      # pred = self.model(student_id, question_id, concepts_emb).item()
      pred = pred_all[sid][qid]
      return pred * torch.norm(pos_weights - original_weights).item() + \
              (1 - pred) * torch.norm(neg_weights - original_weights).item()