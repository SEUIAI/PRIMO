# !/usr/bin/env python3
import os
import time
import random
import re
import csv

import torch
from rich import print
from tqdm import tqdm
import numpy as np

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from trl_rule.gpt2 import GPT2HeadWithValueModel
from trl_rule.ppo import PPOTrainer

from iTrainingLogger import iSummaryWriter


writer = iSummaryWriter(log_path='./logs', log_name='PPO-Sentiment')
config = {
    'G_MODEL_NAME': 'models/generate_model/',
    'E_MODEL_NAME': 'models/extract_model/',
    "steps": 1000,
    "batch_size": 4,
    "forward_batch_size": 4,
    "ppo_epochs": 4,
    "lr": 1e-5,
    "init_kl_coef": 0.2,
    "target": 6,
    "horizon": 10000,
    "gamma": 1,
    "lam": 0.95,
    "cliprange": .2,
    "cliprange_value": .2,
    "vf_coef": .1,
    "gen_len": 256,
    "save_freq": 1000,
    'save_dir': 'models/ppo_gpt_generate/'
}

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# Reward model
tokenizer = AutoTokenizer.from_pretrained(
    'models/reward_model')
R_model = torch.load('models/reward_model/model.pt')
R_model.to(device)

# 文本生成模型
G_gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['G_MODEL_NAME'])
G_gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(
    config['G_MODEL_NAME'])
G_gpt2_tokenizer = AutoTokenizer.from_pretrained(config['G_MODEL_NAME'])
G_gpt2_tokenizer.eos_token = G_gpt2_tokenizer.pad_token
# G_gpt2_tokenizer.pad_token = G_gpt2_tokenizer.eos_token
# G_gpt2_tokenizer.pad_token_id = G_gpt2_tokenizer.eos_token_id
# G_gpt2_tokenizer.padding_side = 'left'
G_gpt2_model.to(device)
G_gpt2_model_ref.to(device)

# 规则抽取模型
E_gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['E_MODEL_NAME'])
# E_gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(
#     config['E_MODEL_NAME'])
E_gpt2_tokenizer = AutoTokenizer.from_pretrained(config['E_MODEL_NAME'])
E_gpt2_tokenizer.eos_token = E_gpt2_tokenizer.pad_token
E_gpt2_model.to(device)
# E_gpt2_model_ref.to(device)


gen_len = 512
gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    # "pad_token_id": G_gpt2_tokenizer.eos_token_id
}

ext_len = 200
ext_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": E_gpt2_tokenizer.eos_token_id
}


class MyDataset(Dataset):
    def __init__(self, data_file_name, data_dir='.data/'):
        super().__init__()
        data_path = os.path.join(data_file_name)

        self.data_list = []

        with open(data_path, encoding='utf-8', errors='ignore') as csv_file:
            # csv_reader = pd.read_csv(csv_file)
            # colums = csv_reader.columns.tolist()
            reader = csv.reader(csv_file)
            header_row = next(reader)
            for line in reader:
                data = []
                q = line[0]
                segments = re.split(
                    'then what other relationships can we derive between A and B?', q)
                conditions = segments[0]
                data.append(q)
                data.append(conditions)
                self.data_list.append(data)

            # for index, row in csv_reader.iterrows():
            #     data = []
            #     q = row[colums[0]]
            #     segments = re.split('then what other relationships can we derive between A and B?', q)
            #     conditions = segments[0]
            #     data.append(q)
            #     data.append(conditions)
            #     self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]


def get_data_loader(data_file_name):
    dataset = MyDataset(data_file_name)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return data_loader


def inference(prompt: str, model, tokenizer, mode: str):
    """
    根据prompt生成内容。

    Args:
        prompt (str): _description_
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    if mode == 'generate':
        response = model.generate(inputs['input_ids'].to(device),
                                  max_new_tokens=gen_len, **gen_kwargs)
        r = response.squeeze()[-gen_len:]
        r_str = tokenizer.decode(r)
        pattern = re.compile(
            "then what other relationships can we derive between A and B\?" + r'(.*)$', re.DOTALL)
        substrings_list = pattern.findall(r_str)
        if substrings_list:
            result = ' '.join(substrings_list[0].split())
        else:
            result = r_str
    elif mode == 'extract':
        response = model.generate(inputs['input_ids'].to(device),
                                  max_new_tokens=ext_len, **ext_kwargs)
        r = response.squeeze()[-ext_len:]
        r_str = tokenizer.decode(r)
        result = r_str
    return result


def infer_processed(resp):
    rel1 = r'\((.*?)\)'
    rel1_result = re.findall(rel1, resp)
    re_result = []
    for word in rel1_result:
        if word != '' and 'A' in word and 'B' in word:
            re_result.append(word)
    return re_result


# RL Trainer
ppo_trainer = PPOTrainer(G_gpt2_model, G_gpt2_model_ref,
                         G_gpt2_tokenizer, **config)
total_ppo_epochs = int(np.ceil(config["steps"]/config['batch_size']))

# Dataloader
LOADER = get_data_loader(
    'dataset/generate_data.csv')


for epoch in tqdm(range(total_ppo_epochs)):
    logs, timing = dict(), dict()
    t0 = time.time()

    batch = {
        'tokens': [],
        'query': [],
        'conditions': [],
        'response': []
    }
    for idx, txt in enumerate(LOADER):
        tokens = G_gpt2_tokenizer.encode(txt[0][0])
        batch['tokens'].append(tokens)
        batch['query'].append(txt[0][0])
        batch['conditions'].append(txt[1][0])
        if (idx+1) % config['batch_size'] == 0:
            break
    query_tensors = [torch.tensor(t).long().to(device)
                     for t in batch["tokens"]]

    t = time.time()
    response_tensors = []
    rewards = []
    reward_sentences = []

    for i in range(config['batch_size']):
        G_prompt = batch['query'][i]
        passage1 = inference(G_prompt, G_gpt2_model,
                            G_gpt2_tokenizer, mode='generate')
        # E_prompt = "{},then which relationships between A and B can we extract from the passage'{}'You only need to output relations without other information, where the relation must be enclosed in brackets, such as (A is lover of B).".format(
        #     batch['conditions'][i],passage1)
        E_prompt = "Please extract relationships from the given passage:{}".format(
            passage1)
        passage2 = inference(E_prompt, E_gpt2_model,
                            E_gpt2_tokenizer, mode='extract')
        select_re = infer_processed(passage2)
        for word in select_re:
            reward_sentences.append(
                batch['conditions'][i] + ' we can get ' + word)
        if len(reward_sentences) == 0:
            reward_sentences.append(
                batch['conditions'][i] + " we can't get any other relationship between A and B ")
        inputs = tokenizer(
            reward_sentences,
            max_length=128,
            padding='max_length',           
            truncation=True,
            return_tensors='pt'
        )
        inputs.to(device)
        r = R_model(**inputs)
        r_list = r.tolist()
        # response_tensors.append(torch.tensor(G_gpt2_tokenizer.encode(
        #     reward_sentences[r_list.index(max(r_list))])).long())
        response_tensors.append(torch.tensor(G_gpt2_tokenizer.encode(batch['conditions'][i] + passage1)).long().to(device))
        # batch['response'].append(reward_sentences[r_list.index(max(r_list))])
        batch['response'].append(batch['conditions'][i] + passage1)
        rewards.append(np.max(r_list))


    # for i in range(config['batch_size']):
    #     gen_len = config['gen_len']
    #     response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0),       # generate()用于直接生成token_id
    #                                    max_new_tokens=gen_len, **gen_kwargs)
    #     response_tensors.append(response.squeeze()[-gen_len:])
    # batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing['time/get_response'] = time.time() - t
    #
    t = time.time()

    texts = [q + r for q, r in zip(batch['query'], batch['response'])]


    rewards = torch.tensor(rewards).to(device)
    timing['time/get_sentiment_preds'] = time.time() - t

    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors,
                             rewards, device, G_gpt2_tokenizer)          # PPO Update
    timing['time/optimization'] = time.time() - t

    # logging
    timing['time/epoch'] = time.time() - t0
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    print(f"epoch {epoch} mean-reward: {logs['env/reward_mean']}")

    print('Random Sample 5 text(s) of model output:')
    # 随机打5个生成的结果
    for i in range(5):
        print(f'{i+1}. {random.choice(texts)}')

    writer.add_scalar('train/reward', logs['env/reward_mean'], epoch)
    for k, v in timing.items():
        writer.add_scalar(k, v, epoch)
    writer.add_scalar('ppo/loss/policy', stats['ppo/loss/policy'], epoch)
    writer.add_scalar('ppo/loss/value', stats['ppo/loss/value'], epoch)
    writer.add_scalar('ppo/policy/entropy', stats['ppo/policy/entropy'], epoch)
    writer.add_scalar('ppo/policy/policykl',
                      stats['ppo/policy/policykl'], epoch)
    writer.record()

    if epoch % config['save_freq'] == 0:
        if not os.path.exists(config['save_dir']):
            os.makedirs(config['save_dir'])
        cur_save_path = os.path.join(
            config['save_dir'], f'model_{epoch}_{round(float(logs["env/reward_mean"]), 2)}'
        )
        ppo_trainer.model.save_pretrained(cur_save_path)
        ppo_trainer.tokenizer.save_pretrained(cur_save_path)
