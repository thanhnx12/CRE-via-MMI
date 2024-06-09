import pickle
import os 
import random
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer

class data_sampler_CFRL(object):
    def __init__(self, config=None, seed=None):
        self.config = config
        self.max_length = self.config.max_length
        self.task_length = self.config.task_length
        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_token = '[unused0]'
        if self.config.model == 'bert':
            self.mask_token = '[MASK]' 
            model_path = self.config.bert_path
            tokenizer_from_pretrained = BertTokenizer.from_pretrained
        elif self.config.model == 'roberta':
            self.mask_token = '<mask>' 
            model_path = self.config.roberta_path
            tokenizer_from_pretrained = RobertaTokenizer.from_pretrained

        # tokenizer
        if config.pattern == 'marker':
            self.tokenizer = tokenizer_from_pretrained(model_path, \
            additional_special_tokens=self.unused_tokens)
            self.config.h_ids = self.tokenizer.get_vocab()[self.unused_tokens[0]]
            self.config.t_ids = self.tokenizer.get_vocab()[self.unused_tokens[2]]
        elif config.pattern == 'hardprompt' or config.pattern == 'cls':
            self.tokenizer = tokenizer_from_pretrained(model_path)
        elif config.pattern == 'softprompt' or config.pattern == 'hybridprompt':
            self.tokenizer =tokenizer_from_pretrained(model_path, \
            additional_special_tokens=[self.unused_token])
            self.config.prompt_token_ids = self.tokenizer.get_vocab()[self.unused_token]

        self.config.vocab_size = len(self.tokenizer)
        self.config.sep_token_ids = self.tokenizer.get_vocab()[self.tokenizer.sep_token]
        self.config.mask_token_ids = self.tokenizer.get_vocab()[self.tokenizer.mask_token]
        self.sep_token_ids, self.mask_token_ids =  self.config.sep_token_ids, self.config.mask_token_ids

        # read relations
        self.id2rel, self.rel2id = self._read_relations(self.config.relation_name)
        self.config.num_of_relation = len(self.id2rel)

        # read data
        self.training_data = self._read_data(self.config.training_data, self._temp_datapath('train'))
        self.valid_data = self._read_data(self.config.valid_data, self._temp_datapath('valid'))
        self.test_data = self._read_data(self.config.test_data, self._temp_datapath('test'))

        # read relation order
        rel_index = np.load(self.config.rel_index)
        rel_cluster_label = np.load(self.config.rel_cluster_label)
        self.cluster_to_labels = {}
        for index, i in enumerate(rel_index):
            if rel_cluster_label[index] in self.cluster_to_labels.keys():
                self.cluster_to_labels[rel_cluster_label[index]].append(i-1)
            else:
                self.cluster_to_labels[rel_cluster_label[index]] = [i-1]

        # shuffle task order
        self.seed = seed
        if self.seed != None:
            self.set_seed(self.seed)

        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)        
        print(f'Task_order: {self.shuffle_index}')
        self.batch = 0

        # record relations
        self.seen_relations = []
        self.history_test_data = {}

    def set_seed(self, seed):
        self.seed = seed
        if self.seed != None:
            random.seed(self.seed)
        self.shuffle_index_old = list(range(self.task_length - 1))
        random.shuffle(self.shuffle_index_old)
        self.shuffle_index_old = np.argsort(self.shuffle_index_old)
        self.shuffle_index = np.insert(self.shuffle_index_old, 0, self.task_length - 1)


    def __iter__(self):
        return self

    def __next__(self):
        if self.batch == self.task_length:
            raise StopIteration()
        
        indexs = self.cluster_to_labels[self.shuffle_index[self.batch]]
        self.batch += 1

        current_relations = []
        cur_training_data = {}
        cur_valid_data = {}
        cur_test_data = {}

        for index in indexs:
            current_relations.append(self.id2rel[index])
            self.seen_relations.append(self.id2rel[index])

            cur_training_data[self.id2rel[index]] = self.training_data[index]
            cur_valid_data[self.id2rel[index]] = self.valid_data[index]
            cur_test_data[self.id2rel[index]] = self.test_data[index]
            self.history_test_data[self.id2rel[index]] = self.test_data[index]

        return cur_training_data, cur_valid_data, cur_test_data, current_relations,\
            self.history_test_data, self.seen_relations

    def _temp_datapath(self, data_type):
        '''
            data_type = 'train'/'valid'/'test'
        '''
        temp_name = [data_type]
        file_name = '{}.pkl'.format('-'.join([str(x) for x in temp_name]))
        prompt_len = self.config.prompt_len * self.config.prompt_num
        if self.config.model == 'bert':
            tp1 = '_process_BERT_'
        elif self.config.model == 'roberta':
            tp1 = '_process_Roberta_'
        if self.config.task_name == 'FewRel':
            tp2 = 'CFRLFewRel/CFRLdata_10_100_10_'
        else:
            tp2 = 'CFRLTacred/CFRLdata_6_100_5_'
        if self.config.pattern == 'hardprompt':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1  + self.config.pattern)
        elif self.config.pattern == 'softprompt' or self.config.pattern == 'hybridprompt':                
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1 + self.config.pattern + '_' + str(prompt_len) + 'token')
        elif self.config.pattern == 'cls':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k), \
            tp1 + self.config.pattern)            
        elif self.config.pattern == 'marker':
            mid_dir = os.path.join('data', tp2 + str(self.config.num_k),  \
            tp1 + self.config.pattern)      
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)
        save_data_path = os.path.join(mid_dir, file_name)   
        return save_data_path     

    def _read_data(self, file, save_data_path):
        if os.path.isfile(save_data_path):
            with open(save_data_path, 'rb') as f:
                datas = pickle.load(f)
                print(save_data_path)
            return datas
        else:
            samples = []
            with open(file) as f:
                for i, line in enumerate(f):
                    sample = {}
                    items = line.strip().split('\t')
                    if (len(items[0]) > 0):
                        sample['relation'] = int(items[0]) - 1
                        sample['index'] = i
                        if items[1] != 'noNegativeAnswer':
                            candidate_ixs = [int(ix) for ix in items[1].split()]
                            sample['tokens'] = items[2].split()
                            headent = items[3]
                            headidx = [[int(ix) for ix in items[4].split()]]
                            tailent = items[5]
                            tailidx = [[int(ix) for ix in items[6].split()]]
                            headid = items[7]
                            tailid = items[8]
                            sample['h'] = [headent, headid, headidx]
                            sample['t'] = [tailent, tailid, tailidx]
                            samples.append(sample)

            read_data = [[] for i in range(self.config.num_of_relation)]
            for sample in samples:
                tokenized_sample = self.tokenize(sample)
                read_data[tokenized_sample['relation']].append(tokenized_sample)
            with open(save_data_path, 'wb') as f:
                pickle.dump(read_data, f)
                print(save_data_path)
            return read_data

    def tokenize(self, sample):
        tokenized_sample = {}
        tokenized_sample['relation'] = sample['relation']
        tokenized_sample['index'] = sample['index']
        if self.config.pattern == 'hardprompt':
            ids, mask = self._tokenize_hardprompt(sample)
        elif self.config.pattern == 'softprompt':
            ids, mask = self._tokenize_softprompt(sample)   
        elif self.config.pattern == 'hybridprompt':
            ids, mask = self._tokenize_hybridprompt(sample)                     
        elif self.config.pattern == 'marker':
            ids, mask = self._tokenize_marker(sample)
        elif self.config.pattern == 'cls':
            ids, mask = self._tokenize_cls(sample)            
        tokenized_sample['ids'] = ids
        tokenized_sample['mask'] = mask    
        return tokenized_sample    


    def _read_relations(self, file):
        id2rel, rel2id = {}, {}
        with open(file) as f:
            for index, line in enumerate(f):
                rel = line.strip()
                id2rel[index] = rel
                rel2id[rel] = index
        return id2rel, rel2id

    def _tokenize_softprompt(self, sample):
        '''
        X [v] [v] [v] [v]
        [v] = [unused0] * prompt_len
        '''
        prompt_len = self.config.prompt_len
        raw_tokens = sample['tokens']
        prompt = raw_tokens + [self.unused_token] * prompt_len + [self.unused_token] * prompt_len \
                             + [self.unused_token] * prompt_len + [self.unused_token] * prompt_len  
        ids = self.tokenizer.encode(' '.join(prompt),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1 

        return ids, mask           

    def _tokenize_hybridprompt(self, sample):
        '''
        X [v] e1 [v] [MASK] [v] e2 [v] 
        [v] = [unused0] * prompt_len
        '''
        prompt_len = self.config.prompt_len
        raw_tokens = sample['tokens']
        h, t = sample['h'][0].split(' '),  sample['t'][0].split(' ')
        prompt = raw_tokens + [self.unused_token] * prompt_len + h + [self.unused_token] * prompt_len \
               + [self.mask_token] + [self.unused_token] * prompt_len + t + [self.unused_token] * prompt_len  
        ids = self.tokenizer.encode(' '.join(prompt),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1 

        return ids, mask        

    def _tokenize_hardprompt(self, sample):
        '''
        X e1 [MASK] e2 
        '''
        raw_tokens = sample['tokens']
        h, t = sample['h'][0].split(' '),  sample['t'][0].split(' ')
        prompt = raw_tokens +  h + [self.mask_token] + t
        ids = self.tokenizer.encode(' '.join(prompt),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)
        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1 

        return ids, mask

    def _tokenize_marker(self, sample):
        '''
        [unused]e[unused]
        '''
        raw_tokens = sample['tokens']
        h1, h2, t1, t2 =  sample['h'][2][0][0], sample['h'][2][0][-1], sample['t'][2][0][0], sample['t'][2][0][-1]
        new_tokens = []

        # add entities marker        
        for index, token in enumerate(raw_tokens):
            if index == h1:
                new_tokens.append(self.unused_tokens[0])
                new_tokens.append(token)
                if index == h2:
                    new_tokens.append(self.unused_tokens[1])
            elif index == h2:
                new_tokens.append(token)
                new_tokens.append(self.unused_tokens[1])
            elif index == t1:
                new_tokens.append(self.unused_tokens[2])
                new_tokens.append(token)
                if index == t2:
                    new_tokens.append(self.unused_tokens[3])
            elif index == t2:
                new_tokens.append(token)
                new_tokens.append(self.unused_tokens[3])
            else:
                new_tokens.append(token)
            
            ids = self.tokenizer.encode(' '.join(new_tokens),
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.max_length)
            
            # mask
            mask = np.zeros(self.max_length, dtype=np.int32)
            end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
            mask[:end_index + 1] = 1

        return ids, mask

    def _tokenize_cls(self, sample):
        '''
        [CLS] X
        '''
        raw_tokens = sample['tokens']
        ids = self.tokenizer.encode(' '.join(raw_tokens),
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_length)
        
        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        end_index = np.argwhere(np.array(ids) == self.sep_token_ids)[0][0]
        mask[:end_index + 1] = 1

        return ids, mask

