import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from config import Config


from llm_sampler import data_sampler_CFRL
from data_loader import get_data_loader_BERT
from utils import Moment, gen_data
from encoder import MistralLMClassification
from peft import get_peft_model, LoraConfig, TaskType

from tqdm import tqdm
import logging
import pickle as pkl
import os

if os.path.exists('./representation') == False:
    os.makedirs('./representation')

device_map = "auto"

class Manager(object):
    def __init__(self, config, logger) -> None:
        super().__init__()
        self.config = config
        self.logger = logger
        
    def _edist(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        L2dist = nn.PairwiseDistance(p=2)
        dist = [] # B
        for i in range(b):
            dist_i = L2dist(x2, x1[i])
            dist.append(torch.unsqueeze(dist_i, 0)) # (N) --> (1,N)
        dist = torch.cat(dist, 0) # (B, N)
        return dist

    def get_memory_proto(self, encoder, dataset):
        '''
        only for one relation data
        '''
        data_loader = get_data_loader_BERT(config, dataset, shuffle=False, \
            drop_last=False,  batch_size=1) 
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to("cuda:0")
            hidden, lmhead_output = encoder(input_ids=instance['ids'], attention_mask=instance['mask']) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)    
        features = torch.cat(features, dim=0) # (M, H)
        proto = features.mean(0)

        return proto, features   

    def select_memory(self, encoder, dataset):
        '''
        only for one relation data
        '''
        N, M = len(dataset), self.config.memory_size
        data_loader = get_data_loader_BERT(self.config, dataset, shuffle=False, \
            drop_last= False, batch_size=1) # batch_size must = 1
        features = []
        encoder.eval()
        for step, (instance, label, idx) in tqdm(enumerate(data_loader)):
            for k in instance.keys():
                instance[k] = instance[k].to("cuda:0")
            hidden, lmhead_output = encoder(input_ids=instance['ids'], attention_mask=instance['mask']) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)

        features = np.concatenate(features) # tensor-->numpy array; (N, H)
        
        if N <= M: 
            return copy.deepcopy(dataset), torch.from_numpy(features)

        num_clusters = M # memory_size < len(dataset)
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features) # (N, M)

        mem_set = []
        mem_feas = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            sample = dataset[sel_index]
            mem_set.append(sample)
            mem_feas.append(features[sel_index])

        mem_feas = np.stack(mem_feas, axis=0) # (M, H)
        mem_feas = torch.from_numpy(mem_feas)
        # proto = memory mean
        # rel_proto = mem_feas.mean(0)
        # proto = all mean
        features = torch.from_numpy(features) # (N, H) tensor
        rel_proto = features.mean(0) # (H)

        return mem_set, mem_feas
        # return mem_set, features, rel_proto
        

    def train_model(self, encoder, training_data, is_memory=False):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)
        trainable_params = []
        for _, param in encoder.named_parameters():
            if param.requires_grad == True:
                trainable_params.append(param)

        optimizer = optim.Adam([{"params": trainable_params, "lr": self.config.lr}])      
        # optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)

        encoder.train()
        epoch = self.config.epoch_mem if is_memory else self.config.epoch
        softmax = nn.Softmax(dim=0)
        for i in range(epoch):
            total_loss = 0
            accum_iter = 8
            for batch_num, (instance, labels, ind) in tqdm(enumerate(data_loader)):
                for k in instance.keys():
                    instance[k] = instance[k].to("cuda:0")
                hidden, lmhead_output = encoder(input_ids=instance['ids'], attention_mask=instance['mask'])
                loss = self.moment.contrastive_loss(hidden, labels, is_memory)

                # compute infonceloss
                infoNCE_loss = 0
                list_labels = labels.cpu().numpy().tolist()

                for j in range(len(list_labels)):
                    negative_sample_indexs = np.where(np.array(list_labels) != list_labels[j])[0]
                    
                    positive_hidden = hidden[j].unsqueeze(0)
                    negative_hidden = hidden[negative_sample_indexs]

                    positive_lmhead_output = lmhead_output[j]#.unsqueeze(0)
                    negative_lmhead_output = lmhead_output[negative_sample_indexs]

                    f_pos = encoder.infoNCE_f(positive_lmhead_output, positive_hidden)
                    f_neg = encoder.infoNCE_f(positive_lmhead_output, negative_hidden)

                    f_concat = torch.cat([f_pos, f_neg], dim=1).squeeze()
                    f_concat = torch.log(torch.max(f_concat , torch.tensor(1e-9).to("cuda:0")))
                    try:
                        infoNCE_loss += -torch.log(softmax(f_concat)[0])
                    except:
                        print(f"cant callculate info here: {ind}")
                        self.logger.error(f"cant callculate info here: {ind}")

                infoNCE_loss = infoNCE_loss / len(list_labels)
                loss = 0.8*loss + infoNCE_loss

                if batch_num == 0:
                    optimizer.zero_grad()

                total_loss += loss.item()

                # loss = loss/accum_iter
                loss.backward()

                # if ((batch_num + 1) % accum_iter == 0) or (batch_num + 1 == len(data_loader)):
                optimizer.step()
                optimizer.zero_grad()
                # update moment
                if is_memory:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=True)
                    # self.moment.update_allmem(encoder)
                else:
                    self.moment.update(ind, hidden.detach().cpu().data, is_memory=False)
                
                
                # print
            total_loss = total_loss/len(data_loader)
            if is_memory:
                sys.stdout.write(f'MemoryTrain:  epoch {i} | loss: {total_loss:.4f}' + '\r')
                self.logger.info(f'MemoryTrain:  epoch {i} | loss: {total_loss:.4f}')
            else:
                sys.stdout.write(f'CurrentTrain:  epoch {i} | loss: {total_loss:.4f}' + '\r')
                self.logger.info(f'CurrentTrain:  epoch {i} | loss: {total_loss:.4f}')
            total_loss=0
            sys.stdout.flush() 
        print('')             

    def eval_encoder_proto(self, encoder, seen_proto, seen_relid, test_data):
        batch_size = 2
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)
        
        corrects = 0.0
        total = 0.0

        representation_dict = {} # store the representation and the prototype vector of each relation {rel: (proto , [rep1, rep2 ,...])}
        seen_proto_list = seen_proto.cpu().data.numpy()

        for i in range(len(seen_relid)):
            representation_dict[seen_relid[i]] = {
                "proto": seen_proto_list[i],
                "rep": []
            }

        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to("cuda:0")
            hidden, lmhead_output = encoder(input_ids=instance['ids'], attention_mask=instance['mask'])
            fea = hidden.cpu().data # place in cpu to eval

            label_list = label.cpu().data.tolist()
            for i in range(len(label_list)):
                representation_dict[label_list[i]]['rep'].append(fea[i].numpy())

            logits = -self._edist(fea, seen_proto) # (B, N) ;N is the number of seen relations

            cur_index = torch.argmax(logits, dim=1) # (B)
            pred =  []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size
        self.logger.info(f'total acc: {100 * (corrects / total):.2f}%   ')
        sys.stdout.write(f'total acc: {100 * (corrects / total):.2f}%   ' + '\r')
        sys.stdout.flush()        
        print('')
        return corrects / total, representation_dict

    def _get_sample_text(self, data_path, index):
        sample = {}
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    items = line.strip().split('\t')
                    sample['relation'] = self.id2rel[int(items[0])-1]
                    sample['tokens'] = items[2]
                    sample['h'] = items[3]
                    sample['t'] = items[5]
        return sample

    def _read_description(self, r_path):
        rset = {}
        with open(r_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                rset[items[1]] = items[2]
        return rset


    def train(self):
        # sampler 
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed, model="mistralai/Mistral-7B-v0.3")
        print('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        # encoder
        encoder = MistralLMClassification.from_pretrained("mistralai/Mistral-7B-v0.3",
                                                        # torch_dtype=torch.float16,
                                                        token="hf_token",
                                                        device_map=device_map)
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
                                r=16,
                                lora_alpha=32,
                                lora_dropout=0.1,
                                modules_to_save=["info_nce_fc"])
        
        encoder = get_peft_model(encoder, peft_config)

        # step is continual task number
        cur_acc, total_acc = [], []
        cur_acc_num, total_acc_num = [], []
        memory_samples = {}
        data_generation = []
        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations) in enumerate(sampler):
      
            # Initialization
            self.moment = Moment(self.config)

            # Train current task
            training_data_initialize = []
            for rel in current_relations:
                training_data_initialize += training_data[rel]
            self.moment.init_moment(encoder, training_data_initialize, is_memory=False)
            self.train_model(encoder, training_data_initialize)

            # Select memory samples
            for rel in current_relations:
                memory_samples[rel], _ = self.select_memory(encoder, training_data[rel])

            # Data gen
            if self.config.gen == 1:
                gen_text = []
                for rel in current_relations:
                    for sample in memory_samples[rel]:
                        sample_text = self._get_sample_text(self.config.training_data, sample['index'])
                        gen_samples = gen_data(self.r2desc, self.rel2id, sample_text, self.config.num_gen, self.config.gpt_temp, self.config.key)
                        gen_text += gen_samples
                for sample in gen_text:
                    data_generation.append(sampler.tokenize(sample))
                    
            # Train memory
            if step > 0:
                memory_data_initialize = []
                for rel in seen_relations:
                    memory_data_initialize += memory_samples[rel]
                memory_data_initialize += data_generation
                self.moment.init_moment(encoder, memory_data_initialize, is_memory=True) 
                self.train_model(encoder, memory_data_initialize, is_memory=True)

            # Update proto
            seen_proto = []  
            for rel in seen_relations:
                proto, _ = self.get_memory_proto(encoder, memory_samples[rel])
                seen_proto.append(proto)
            seen_proto = torch.stack(seen_proto, dim=0)

            # Eval current task and history task
            test_data_initialize_cur, test_data_initialize_seen = [], []
            for rel in current_relations:
                test_data_initialize_cur += test_data[rel]
            for rel in seen_relations:
                test_data_initialize_seen += historic_test_data[rel]
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])

            ac1, _ = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_cur)
            ac2, rep_dict_test = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_seen)

            # eval on training data and memory data
            train_and_memory = training_data_initialize[:]
            for rel in seen_relations:
                train_and_memory += memory_samples[rel]

            ac3 , rep_dict_train = self.eval_encoder_proto(encoder, seen_proto, seen_relid, train_and_memory)

            pkl.dump(rep_dict_train, open(f'./representation/seed_{str(config.seed)}_{self.config.task_name}_{self.config.num_k}-shot_{step}_train.pkl', 'wb'))
            pkl.dump(rep_dict_test, open(f'./representation/seed_{str(config.seed)}_{self.config.task_name}_{self.config.num_k}-shot_{step}_test.pkl', 'wb'))

            cur_acc_num.append(ac1)
            total_acc_num.append(ac2)
            cur_acc.append('{:.4f}'.format(ac1))
            total_acc.append('{:.4f}'.format(ac2))
            print('cur_acc: ', cur_acc)
            print('his_acc: ', total_acc)

        torch.cuda.empty_cache()
        return total_acc_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=2, type=int)
    args = parser.parse_args()
    config = Config('config.ini')
    config.task_name = args.task_name
    config.num_k = args.num_k
    config.num_gen = args.num_gen

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(f'mistral-mmi-logs-{config.task_name}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # config 
    print('#############params############')
    logger.info('#############params############')
    print(config.device)
    config.device = torch.device(config.device)
    print(f'Task={config.task_name}, {config.num_k}-shot')
    logger.info(f'Task={config.task_name}, {config.num_k}-shot')
    print(f'Encoding model: {config.model}')
    print(f'pattern={config.pattern}')
    print(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    logger.info(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    print('#############params############')
    logger.info('#############params############')

    if config.task_name == 'FewRel':
        config.rel_index = './data/CFRLFewRel/rel_index.npy'
        config.relation_name = './data/CFRLFewRel/relation_name.txt'
        config.relation_description = './data/CFRLFewRel/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/test_0.txt'
    else:
        config.rel_index = './data/CFRLTacred/rel_index.npy'
        config.relation_name = './data/CFRLTacred/relation_name.txt'
        config.relation_description = './data/CFRLTacred/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_5/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_5/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_10/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_10/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_10/test_0.txt'        

    # seed 
    random.seed(config.seed) 
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)   
    base_seed = config.seed

    acc_list = []
    for i in range(config.total_round):
        config.seed = base_seed + i * 100
        print('--------Round ', i)
        print('seed: ', config.seed)
        manager = Manager(config, logger)
        acc = manager.train()
        acc_list.append(acc)
        torch.cuda.empty_cache()
    
    accs = np.array(acc_list)
    ave = np.mean(accs, axis=0)
    print('----------END')
    print('his_acc mean: ', np.around(ave, 4))
    logger.info(f'his_acc mean: {np.around(ave, 4)}')



            
        
            
            


