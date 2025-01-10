import os
from pickle import TRUE
import sys
import time
import logging
import numpy as np
import copy as cp
import torch
import dgl
import json
from datetime import datetime
from torch.distributions import Categorical
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import envs as all_envs
import agents as all_agents
import function as all_FA
from util import get_objects, set_global_seeds, arg_parser, check_path, tensor_to_numpy
from envs.ncd import NCDModel
from function.GCAT import GCAT
from agents.GCATAgent import GCATAgent  # Updated import for GCATAgent
from agents.dataset import Dataset, collate_fn

def str2bool(str=""):
    str = str.lower()
    return str in ["yes", "true", "y", "t"]

def common_arg_parser():
    parser = arg_parser()
    parser.add_argument('-seed', type=int, default=145)
    parser.add_argument('-environment', type=str, default="Env")
    parser.add_argument('-data_path', type=str, default="./data/")
    parser.add_argument('-data_name', type=str, default="name")
    parser.add_argument('-agent', type=str, default="GCATAgent")  # Updated default agent
    parser.add_argument('-FA', type=str, default="GCAT")
    parser.add_argument('-CDM', type=str, default='CDM', help="type of CDM")
    parser.add_argument('-T', type=int, default=3, help="time_step")
    parser.add_argument('-ST', type=eval, default="[10,30,60,120]", help="evaluation_time_step")
    parser.add_argument('-gpu_no', type=str, default="0", help='which gpu for usage')
    
    parser.add_argument('-learning_rate', type=float, default=0.01)
    parser.add_argument('-training_epoch', type=int, default=30000)
    parser.add_argument('-cdm_lr', type=float, default=0.01)
    parser.add_argument('-cdm_epoch', type=int, default=5)
    parser.add_argument('-cdm_bs', type=int, default=128)
    
    parser.add_argument('-train_bs', type=int, default=50)
    parser.add_argument('-test_bs', type=int, default=50)
    parser.add_argument('-batch', type=int, default=128)
    
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-latent_factor', type=int, default=256)
    parser.add_argument('-n_block', type=int, default=2)
    parser.add_argument('-graph_block', type=int, default=2)
    parser.add_argument('-n_head', type=int, default=1)
    parser.add_argument('-dropout_rate', type=float, default=0.0)
    parser.add_argument('-policy_epoch', type=int, default=4)
    parser.add_argument('-morl_weights', type=eval, default="[1,1,1]")
    parser.add_argument('-emb_dim', type=int, default=128)
    parser.add_argument('-use_graph', type=str2bool, default="True")
    parser.add_argument('-use_attention', type=str2bool, default="True")
    parser.add_argument('-store_action', type=str2bool, default="False")
    parser.add_argument('-student_ids', type=eval, default="[0]")
    parser.add_argument('-target_concepts', type=eval, default="[0]")
    return parser

def build_graph(type, node, path):
    g = dgl.graph(([],[]))
    g.add_nodes(node)
    edge_list = []
    file_map = {
        'direct': 'K_Directed.txt',
        'k_from_e': 'k_from_e.txt',
        'e_from_k': 'e_from_k.txt'
    }
    with open(path + file_map[type], 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            edge_list.append((int(line[0]), int(line[1])))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    return g

def construct_local_map(args, path):
    return {
        'directed_g': build_graph('direct', args.know_num, path),
        'k_from_e': build_graph('k_from_e', args.know_num + args.item_num, path),
        'e_from_k': build_graph('e_from_k', args.know_num + args.item_num, path),
    }

import torch
from torch.distributions import Categorical
import copy as cp
import numpy as np

def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data

def launch_adaptive_test(student_ids, target_concepts, agent, fa, env):

    user_ids = student_ids
    state = env.reset_with_users(user_ids) # Reset state
    done = False # Reset done parameter
    step_count = 0 # Reset step count
    seq_length = state['batch_question'].shape[1] # Menyesuaikan size sequence dengan batch question

    question_map = open_json('data/question_map_dbekt22')
    
    question_text_map = open_json('data/question_text_map_dbekt22.json')

    print("=== Tes dimulai ===")

    # Inisiasi metrik evaluasi
    infos = {int(step): [] for step in agent.args.ST}  # Ensure step keys are integers
    actions_list = []
    coverage = 0.0
    reward = [0. ,0. , 0.]
    

    while not done:
        step_count += 1
        print(f"\nStep {step_count}:")
        print(f"Coverage: {coverage}")
        print(f"Reward: {reward}")

        # Get the question for the current step
        question_id = int(state['batch_question'][0, step_count-1]) - 1  # Dikurang 1 sebagai padding
        
        question_data = question_text_map.get(str(question_id))
        if question_id == -1:
          print("Apakah Anda siap? \nA.Ya")
          input("Masukkan jawaban Anda: ")
          user_answer = 0
        elif not question_data:
            print(f"Question ID {question_id} not found in question text map.")
            return
        
        if question_data:
          print(f"Question ID: {question_id}")
          print(f"{question_data['question_text']}")
          for idx, choice in enumerate(question_data['choices']):
            print(f"  {chr(65 + idx)}. {choice}")  # Convert index to A, B, C, D, etc.
          
          valid = False
          while not valid:
            # Get user's answer
            user_choice = input("Masukkan jawaban Anda (A, B, C, atau D): ").strip().upper()
            valid_choices = [chr(65 + i) for i in range(len(question_data['choices']))]  # A, B, C, etc.

            if user_choice not in valid_choices:
                print("Invalid input!\n")
            else:
              valid = True
          
          # Map the user's choice back to an index
          user_choice_index = valid_choices.index(user_choice)
          correct_answer_index = question_data['choices'].index(question_data['correct_answer'])

          # Check if the user's answer is correct
          is_correct = 1 if user_choice_index == correct_answer_index else 0
          print("Benar!" if is_correct else "Salah.")

          # Jawaban
          user_answer = is_correct


        state['batch_answer'][0, step_count - 1] = user_answer + 1  # idx=0 is pad, so add 1 to answer

        p_rec = torch.tensor(state['batch_question'], dtype=torch.long)
        a_rec = torch.tensor(state['batch_answer'], dtype=torch.long)
        kn_rec, kn_num = agent.get_know_num(state['batch_question'])
        
        action_mask = torch.ones(1, env.item_num, dtype=torch.float32) # menandai action yang tidak available
        for i, uu in enumerate(user_ids):
            unavailable_questions = set(range(env.item_num)) - env.avail_questions[uu]
            for q in unavailable_questions:
                action_mask[i, q] = 0
        
        # Prediction and action selection
        data = {
            'p_rec': p_rec, #Problem Record
            'p_t': torch.tensor([step_count - 1], dtype=torch.long), #Step
            'a_rec': a_rec, #Answer Record
            'kn_rec': kn_rec, #Knowledge Record
            'kn_num': kn_num #Knowledge Number
        }
        logits = fa.policy_old.predict(data)

        # Add mask to logits to enforce valid actions
        inf_mask = torch.clamp(torch.log(action_mask.float()), min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        action_probs = F.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()  # Sample an action for adaptive response

        # Apply the action to the environment and observe outcome
        state, reward, done, all_info, coverage = env.step(action.tolist(), last_epoch=(step_count))

        # Display the outcome of the action
        # print(f"Info: {all_info}")
        # print(f"State: {state}")
        # Collect evaluation metrics
        if step_count in agent.args.ST:
          if all_info:
            infos[int(step_count)].extend(all_info)  # Explicitly cast step_count to int

    # After the test, evaluate metrics
    print("\n======")
    
    for step in agent.args.ST:
      if infos[int(step)]:
        pred, label = [], []
        for metric in infos[int(step)]:  # Ensure step is an int for dictionary key access
            pred.append(metric['pred'])
            label.append(metric['label'])
        pred = np.concatenate(pred, axis=0)
        label = np.concatenate(label, axis=0)
        
        pred_bin = np.where(pred > 0.5, 1, 0)
        accuracy = np.sum(np.equal(pred_bin, label)) / len(pred_bin) 
        try:
            auc = roc_auc_score(label, pred)
        except ValueError:
            auc = -1  # AUC is undefined if all labels are the same

        print(f"Step {step} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
  
    print("\nTes selesai")

def main(args):
    args_parser = common_arg_parser()
    args, unknown_args = args_parser.parse_known_args(args)
    args.model = "_".join([args.agent, args.FA, str(args.T)])
    set_global_seeds(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)

    env_classes = get_objects(all_envs)
    env = env_classes[args.environment](args)
    args.user_num = env.user_num
    args.item_num = env.item_num
    args.know_num = env.know_num

    local_map = construct_local_map(args, path=f'graph_data/{args.data_name}/')
    fa_classes = get_objects(all_FA)
    fa = fa_classes[args.FA].create_model(args, local_map)
    
    agent_class = get_objects(all_agents)
    agent = agent_class[args.agent](env, fa, args)  # Initialize GCATAgent with environment, function approximator, and arguments

    student_ids = args.student_ids
    target_concepts = args.target_concepts

    launch_adaptive_test(student_ids, target_concepts, agent, fa, env)

if __name__ == '__main__':
    main(sys.argv)
