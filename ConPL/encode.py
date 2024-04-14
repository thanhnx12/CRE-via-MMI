import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import types
import numpy as np
import sys
# sys.path.append("/home4/chenxiudi/myfile/Continual_Fewshot_Relation_Learning_v2/transformers")
from model import base_model, embedding_layer, lstm_layer
from word_tokenizer import WordTokenizer
from transformers import BertTokenizer,BertModel
from transformers import BertForMaskedLM
from transformers.models.llama.modeling_llama import *


class LlamaClassification(GemmaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.model = GemmaModel(config)
        # self.ln = nn.Linear(config.hidden_size, self.config.hidden_size, bias=True)
        # self.dropout = nn.Dropout(0.1)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        get_feature: Optional[bool] = False,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # if self.config.pad_token_id is None and batch_size != 1:
        #     raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1

        e11 = []
        # for each sample in the batch, acquire the positions of its [E11] and [E21]
        for mask in attention_mask:
            e11.append(mask.sum().item() - 1)
        
        output = []

        # for each sample in the batch, acquire its representations for [E11] and [E21]
        for i in range(len(e11)):
            instance_output = torch.index_select(hidden_states, 0, torch.tensor(i).to(hidden_states.device))
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i]]).to(hidden_states.device))
            output.append(instance_output)  # [B,N,H] --> [B,2,H]
        
        output = torch.stack(output)
        output = output.view(output.shape[0],-1) # [B,1,H] --> [B,H]

        # logits = self.score(output)
        # return self.dropout(logits)
        return output
    
class LlamaLMClassification(GemmaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.info_nce_fc = nn.Linear(config.vocab_size, config.hidden_size , bias= False)
        self.hidden_size = config.hidden_size

        # Initialize weights and apply final processing
        self.post_init()

    def infoNCE_f(self,V,C, temperature):
        """
        V : 1 x dim_V
        C : 1 x dim_C

        """
        out = self.info_nce_fc(V) # N x dim_C
        C = C.to(out.device)
        out = torch.matmul(out, C.t()) # N x N
        return out

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        att_mask_0: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0] #B, N, H

        hidden_states = outputs[0] #B, N, H

        e11 = []
        for mask in att_mask_0:
            e11.append(mask.sum().item() - 1)
        
        output = []
        for i in range(len(e11)):
            instance_output = torch.index_select(hidden_states, 0, torch.tensor(i).to(hidden_states.device))
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i]]).to(hidden_states.device))
            output.append(instance_output)  # [B,N,H] --> [B,1,H]
        
        output = torch.stack(output)
        logit = self.lm_head(output) # B,1,V
        output = output.view(output.shape[0],-1) # [B,1,H] --> [B,H]

        mlm_loss = 0
        for i, hidden in enumerate(hidden_states):
            start = att_mask_0[i].sum().item()
            end = attention_mask[i].sum().item()
            if end > start:
                logits = self.lm_head(hidden[start:end])
                mlm_loss += F.cross_entropy(input=logits, target = input_ids[start:end])

        return output, logit.squeeze(1), mlm_loss/hidden_states.shape[0]
    
class base_encoder(base_model):

    def __init__(self,
                 token2id=None,
                 word2vec=None,
                 word_size=50,
                 max_length=128,
                 blank_padding=True):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy
        """
        # hyperparameters
        super(base_encoder, self).__init__()

        if isinstance(token2id, list):
            self.token2id = {}
            for index, token in enumerate(token2id):
                self.token2id[token] = index
        else:
            self.token2id = token2id

        self.max_length = max_length
        self.num_token = len(self.token2id)

        if isinstance(word2vec, type(None)):
            self.word_size = word_size
        else:
            self.word_size = word2vec.shape[-1]

        self.blank_padding = blank_padding

        if not '[UNK]' in self.token2id:
            self.token2id['[UNK]'] = len(self.token2id)
            self.num_token += 1
        if not '[PAD]' in self.token2id:
            self.token2id['[PAD]'] = len(self.token2id)
            self.num_token += 1

        if not isinstance(word2vec, type(None)):
            word2vec = torch.from_numpy(word2vec)
            if self.num_token == len(word2vec) + 2:
                unk = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                blk = torch.zeros(1, self.word_size)
                self.word2vec = (torch.cat([word2vec, unk, blk], 0)).numpy()
            else:
                self.word2vec = word2vec
        else:
            self.word2vec = None

        self.tokenizer = WordTokenizer(vocab=self.token2id, unk_token="[UNK]")

    def set_embedding_layer(self, embedding_layer):
        self.embedding_layer = embedding_layer

    def set_encoder_layer(self, encoder_layer):
        self.encoder_layer = encoder_layer

    def forward(self, token, pos1, pos2):
        pass

    def tokenize(self, sentence):
        """
        Args:
            item: input instance, including sentence, entity positions, etc.
            is_token: if is_token == True, sentence becomes an array of token
        Return:
            index number of tokens and positions
        """
        tokens = self.tokenizer.tokenize(sentence)
        length = min(len(tokens), self.max_length)
        # Token -> index

        if self.blank_padding:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'],
                                                                  self.token2id['[UNK]'])
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id=self.token2id['[UNK]'])

        if (len(indexed_tokens) > self.max_length):
            indexed_tokens = indexed_tokens[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
        length = torch.tensor([length]).long()
        return indexed_tokens, length

class lstm_encoder(base_encoder):

    def __init__(self, token2id = None, word2vec = None, word_size = 50, max_length = 128,
            pos_size = None, hidden_size = 230, dropout = 0, bidirectional = True, num_layers = 1, config = None):
        super(lstm_encoder, self).__init__(token2id, word2vec, word_size, max_length, blank_padding = False)
        self.config = config
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.word_size = word_size
        self.pos_size = pos_size
        self.input_size = word_size
        if bidirectional:
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size
        if pos_size != None:
            self.input_size += 2 * pos_size
        self.embedding_layer = embedding_layer(self.word2vec, max_length, word_size, None, False)
        self.encoder_layer = lstm_layer(max_length, self.input_size, hidden_size, dropout, bidirectional, num_layers, config)

    def forward(self, inputs, lengths = None):
        inputs, lengths, inputs_indexs = self.encoder_layer.pad_sequence(inputs, padding_value = self.token2id['[PAD]'])
        inputs = inputs.to(self.config['device'])
        x = self.embedding_layer(inputs)
        x = self.encoder_layer(x, lengths, inputs_indexs)
        return x

class BERTSentenceEncoderFreeze(nn.Module):
    def __init__(self, config,ckptpath=None):
        nn.Module.__init__(self)
        if ckptpath != None:
            ckpt = torch.load(ckptpath)
            self.bert = BertModel.from_pretrained(config["pretrained_model"],state_dict=ckpt["bert-base"])
        else:
            self.bert = BertModel.from_pretrained(config["pretrained_model"])
        print("aaaaaaaaaaaaaaaaaaaa")
        unfreeze_layers = ['layer.11', 'pooler.', 'output.']
        # # unfreeze_layers = []
        # print(unfreeze_layers)
        for name, param in self.bert.named_parameters():
            # param.requires_grad = True
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        print("freeze finished")
        #'''
        ###5 no freeze
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrained_model"])
        self.output_size = 768
    def forward(self, inputs, mask):
        outputs = self.bert(inputs, attention_mask=mask)
        #print("outputs[0].shape: ",outputs[0].shape)
        #print("outputs[1].shape: ",outputs[1].shape)
        return outputs[1]


class BERTSentenceEncoder(nn.Module):
    def __init__(self, config,ckptpath=None):
        nn.Module.__init__(self)
        if ckptpath != None:
            ckpt = torch.load(ckptpath)
            self.bert = BertModel.from_pretrained(config["pretrained_model"],state_dict=ckpt["bert-base"])
        else:
            self.bert = BertModel.from_pretrained(config["pretrained_model"])
        print("aaaaaaaaaaaaaaaaaaaa")
        unfreeze_layers = ['layer.11', 'pooler.', 'out.']
        # # unfreeze_layers = []
        # print(unfreeze_layers)
        for name, param in self.bert.named_parameters():
            param.requires_grad = True
            # param.requires_grad = False
            # for ele in unfreeze_layers:
            #     if ele in name:
            #         param.requires_grad = True
            #         break
        print("freeze finished")
        #'''
        ###5 no freeze
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrained_model"])
        self.output_size = 768
    def forward(self, inputs, mask):
        outputs = self.bert(inputs, attention_mask=mask)
        #print("outputs[0].shape: ",outputs[0].shape)
        #print("outputs[1].shape: ",outputs[1].shape)
        return outputs[1]

class BERTSentenceEncoderPrompt(nn.Module):
    def __init__(self, config,ckptpath=None):
        nn.Module.__init__(self)
        if ckptpath != None:
            ckpt = torch.load(ckptpath)
            self.bert = BertModel.from_pretrained(config["pretrained_model"],state_dict=ckpt["bert-base"])
        else:
            self.bert = BertModel.from_pretrained(config["pretrained_model"])
        for name, param in self.bert.named_parameters():
            param.requires_grad = True
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrained_model"])
        self.output_size = 768
    def forward(self, inputs, mask, mask_pos):
        outputs = self.bert(inputs, attention_mask=mask)
        tensor_range = torch.arange(inputs.size()[0])
        return outputs[0][tensor_range, mask_pos]

class BERTMLMSentenceEncoderPrompt(nn.Module):
    def __init__(self, config,ckptpath=None):
        nn.Module.__init__(self)
        additional_special_tokens = []
        additional_special_tokens.extend([f"[REL{i}]" for i in range(1, config['num_of_relation'] + 2)])
        
        # --- add special tokens --- # 
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrained_model"] , additional_special_tokens = additional_special_tokens)
        # --- add special tokens --- #
        
        if ckptpath != None:
            ckptpath = torch.load(ckptpath)
            self.bert = BertForMaskedLM.from_pretrained(config["pretrained_model"],state_dict=ckptpath["bert-base"])
        else:
            self.bert = BertForMaskedLM.from_pretrained(config["pretrained_model"])
        for name, param in self.bert.named_parameters():
            param.requires_grad = True
        # --- resize token embedding --- #
        self.bert.resize_token_embeddings(len(self.tokenizer)) # 30522 + num_relation
        # --- resize token embedding --- #


        self.output_size = 768
        self.info_nce_fc = nn.Linear(len(self.tokenizer) , self.output_size)
    def forward(self, inputs, mask, mask_pos):
        outputs = self.bert(inputs, attention_mask=mask , output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        lm_head_output = outputs.logits

        tensor_range = torch.arange(inputs.size()[0])
        return last_hidden_state[tensor_range, mask_pos] , lm_head_output[tensor_range,mask_pos]
    def infoNCE_f(self,V,C,temperature = 1000.0):
        """
        V : 1 x dim_V
        C : 1 x dim_C

        """
        try:
            out = self.info_nce_fc(V) # N x dim_C
            out = torch.matmul(out, C.t()) # N x N
            # out = torch.exp(out / temperature)
        except:
            print("V shape : ", V.shape)
            print("C shape : ", C.shape)
            raise Exception('Error in infoNCE_f')
        return out

class Transformer_Encoder(base_encoder):
    #def __init__(self, num_layers, d_model, vocab_size, h, dropout):
    def __init__(self, token2id=None, word2vec=None, word_size=300, max_length=128, dropout=0, head = 4, num_layers=1, config=None):
        super(Transformer_Encoder, self).__init__(token2id, word2vec, word_size, max_length, blank_padding=False)

        self.config = config
        self.max_length = max_length
        self.hidden_size = word_size  ####d_model
        self.output_size = word_size
        self.embedding_layer = EmbeddingLayer(self.word2vec, max_length, word_size, False)

        self.layers = nn.ModuleList([EncoderLayer(word_size, head, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(word_size)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()


    def forward(self, x, mask):
        #print("---------------------")
        #print(x.shape)
        #print(mask.shape)
        batch_size = x.size(0)
        max_enc_len = x.size(1)

        assert max_enc_len == self.max_length

        pos_idx = torch.arange(self.max_length).unsqueeze(0).expand((batch_size, -1))
        pos_idx = pos_idx.to(self.config['device'])

        y = self.embedding_layer(x, pos_idx[:, :max_enc_len])
        #print(y.shape)
        assert y.size(1) == mask.size(-1)
        mask = (mask[:, :max_enc_len] == 0)
        mask = mask.view(batch_size, 1, 1, max_enc_len)
        for layer in self.layers:
            y = layer(y, mask)

        encoder_outputs = self.norm(y)
        #print(encoder_outputs.shape)
        sequence_output = encoder_outputs[:,0]
        pooled_output = self.activation(self.dense(sequence_output))
        return pooled_output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.pw_ffn)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.head_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x, l in zip((query, key, value), self.head_projs)]

        attn_feature, _ = scaled_attention(query, key, value, mask)

        attn_concated = attn_feature.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.fc(attn_concated)


def scaled_attention(query, key, value, mask):
    d_k = query.size(-1)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
    scores.masked_fill_(mask, float('-inf'))
    attn_weight = F.softmax(scores, -1)
    attn_feature = attn_weight.matmul(value)

    return attn_feature, attn_weight


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.mlp = nn.Sequential(
            Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.mlp(x)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x + self.dropout(y)

    def incremental_forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x[:, -1:] + self.dropout(y)


def Linear(in_features, out_features, bias=True, uniform=True):
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

class EmbeddingLayer(nn.Module):
    #def __init__(self, n_words, d_model, max_length, pad_idx, learned_pos_embed, load_pretrained_embed):
    def __init__(self, word_vec_mat, max_length, word_embedding_dim=300, requires_grad=True):
        super(EmbeddingLayer, self).__init__()

        self.max_length = max_length

        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = word_embedding_dim
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.token_embed = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim,
                                           padding_idx=word_vec_mat.shape[0] - 1)
        self.token_embed.weight.data.copy_(word_vec_mat)
        self.token_embed.weight.requires_grad = requires_grad

        self.pos_embed = nn.Embedding(max_length, self.pos_embedding_dim, padding_idx=max_length - 1)

    def forward(self, x, pos):
        if len(x.size()) == 2:
            y = self.token_embed(x) + self.pos_embed(pos)
        return y

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim, eps=1e-6):
    m = nn.LayerNorm(embedding_dim, eps)
    return m