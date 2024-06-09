import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
from transformers import RobertaModel
from transformers import BertForMaskedLM
from transformers.models.llama.modeling_llama import *
from transformers.models.mistral.modeling_mistral import *

class EncodingModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        if config.model == 'bert':
            self.encoder = BertModel.from_pretrained(config.bert_path).to(config.device)
            self.lm_head = BertForMaskedLM.from_pretrained(config.bert_path).to(config.device).cls
        elif config.model == 'roberta':
            self.encoder = RobertaModel.from_pretrained(config.roberta_path).to(config.device)
            self.encoder.resize_token_embeddings(config.vocab_size)
        if config.tune == 'prompt':
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.bert_word_embedding = self.encoder.get_input_embeddings()
        self.embedding_dim = self.bert_word_embedding.embedding_dim
        self.prompt_lens = config.prompt_len * config.prompt_num
        self.softprompt_encoder = nn.Embedding(self.prompt_lens, self.embedding_dim).to(config.device)
        # initialize prompt embedding
        self._init_prompt()
        self.prompt_ids = torch.LongTensor(list(range(self.prompt_lens))).to(self.config.device)

        self.info_nce_fc = nn.Linear(config.vocab_size , self.embedding_dim).to(config.device)


    def infoNCE_f(self, V, C):
        """
        V : B x vocab_size
        C : B x embedding_dim
        """
        try:
            out = self.info_nce_fc(V) # B x embedding_dim
            out = torch.matmul(out , C.t()) # B x B

        except:
            print("V.shape: ", V.shape)
            print("C.shape: ", C.shape)
            print("info_nce_fc: ", self.info_nce_fc)
        return out
    def _init_prompt(self):
        # is is is [e1] is is is [MASK] is is is [e2] is is is
        if self.config.prompt_init == 1:
            prompt_embedding = torch.zeros_like(self.softprompt_encoder.weight).to(self.config.device)
            token_embedding = self.bert_word_embedding.weight[2003]
            prompt_embedding[list(range(self.prompt_lens)), :] = token_embedding.clone().detach()
            for param in self.softprompt_encoder.parameters():
                param.data = prompt_embedding # param.data
       
        # ! @ # [e1] he is as [MASK] * & % [e2] just do it  
        elif self.config.prompt_init == 2:
            prompt_embedding = torch.zeros_like(self.softprompt_encoder.weight).to(self.config.device)
            ids = [999, 1030, 1001, 2002, 2003, 2004, 1008, 1004, 1003, 2074, 2079,  2009]
            for i in range(self.prompt_lens):
                token_embedding = self.bert_word_embedding.weight[ids[i]]
                prompt_embedding[i, :] = token_embedding.clone().detach()
            for param in self.softprompt_encoder.parameters():
                param.data = prompt_embedding # param.data


    def embedding_input(self, input_ids): # (b, max_len)
        input_embedding = self.bert_word_embedding(input_ids) # (b, max_len, h)
        prompt_embedding = self.softprompt_encoder(self.prompt_ids) # (prompt_len, h)

        for i in range(input_ids.size()[0]):
            p = 0
            for j in range(input_ids.size()[1]):
                if input_ids[i][j] == self.config.prompt_token_ids:
                    input_embedding[i][j] = prompt_embedding[p]
                    p += 1

        return input_embedding


    def forward(self, inputs): # (b, max_length)
        batch_size = inputs['ids'].size()[0]
        tensor_range = torch.arange(batch_size) # (b)     
        pattern = self.config.pattern
        if pattern == 'softprompt' or pattern == 'hybridprompt':
            input_embedding = self.embedding_input(inputs['ids'])
            outputs_words = self.encoder(inputs_embeds=input_embedding, attention_mask=inputs['mask'])[0]
        else:
            outputs_words = self.encoder(inputs['ids'], attention_mask=inputs['mask'])[0] # (b, max_length, h)

        # return [CLS] hidden
        if pattern == 'cls' or pattern == 'softprompt':
            clss = torch.zeros(batch_size, dtype=torch.long)
            return outputs_words[tensor_range ,clss] # (b, h)

        # return [MASK] hidden
        elif pattern == 'hardprompt' or pattern == 'hybridprompt':
            masks = []
            for i in range(batch_size):
                ids = inputs['ids'][i].cpu().numpy()
                try:
                    mask = np.argwhere(ids == self.config.mask_token_ids)[0][0]
                except:
                    mask = 0
                
                masks.append(mask)
            mask_hidden = outputs_words[tensor_range, torch.tensor(masks)] # (b, h)
            lm_head_output = self.lm_head(mask_hidden) # (b, max_length, vocab_size)
            return mask_hidden , lm_head_output

        # return e1:e2 hidden
        elif pattern == 'marker':
            h1, t1 = [], []
            for i in range(batch_size):
                ids = inputs['ids'][i].cpu().numpy()
                h1_index, t1_index = np.argwhere(ids == self.config.h_ids), np.argwhere(ids == self.config.t_ids)
                h1.append(0) if h1_index.size == 0 else h1.append(h1_index[0][0])
                t1.append(0) if t1_index.size == 0 else t1.append(t1_index[0][0])

            h_state = outputs_words[tensor_range, torch.tensor(h1)] # (b, h)
            t_state = outputs_words[tensor_range, torch.tensor(t1)]

            concerate_h_t = (h_state + t_state) / 2 # (b, h)
            return concerate_h_t

class LlamaClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        # self.ln = nn.Linear(config.hidden_size, self.config.hidden_size, bias=True)
        # self.dropout = nn.Dropout(0.1)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
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
        

        e11 = []
        # for each sample in the batch, acquire the positions of its [E11] and [E21]
        for mask in attention_mask:
            e11.append(mask.sum().item() - 1)
        
        output = []
        for i in range(len(e11)):
            instance_output = torch.index_select(hidden_states, 0, torch.tensor(i).to(hidden_states.device))
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i]]).to(hidden_states.device))
            output.append(instance_output)  # [B,N,H] --> [B,1,H]
        
        output = torch.stack(output)
        output = output.view(output.shape[0],-1) # [B,1,H] --> [B,H]
        return output.to(torch.float32), None
    
class LlamaLMClassification(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config, bottle_neck_size=512)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bottle_neck = nn.Linear(config.vocab_size, bottle_neck_size, bias=False)
        self.info_nce_fc = nn.Linear(bottle_neck_size, config.hidden_size , bias= False)

        # Initialize weights and apply final processing
        self.post_init()

    def infoNCE_f(self,V,C):
        """
        V : 1 x dim_V
        C : 1 x dim_C

        """
        V = self.bottle_neck(V)
        out = self.info_nce_fc(V) # N x dim_C
        out = torch.matmul(out, C.t().to(out.device)) # N x N
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        e11 = []
        # for each sample in the batch, acquire the positions of its [E11] and [E21]
        for mask in attention_mask:
            e11.append(mask.sum().item() - 1)

        max_length = max(e11) + 1

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids[:max_length],
            attention_mask=attention_mask[:max_length],
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
        
        output = []
        # for each sample in the batch, acquire its representations for [E11] and [E21]
        for i in range(len(e11)):
            instance_output = torch.index_select(hidden_states, 0, torch.tensor(i).to(hidden_states.device))
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i]]).to(hidden_states.device))
            output.append(instance_output)  # [B,N,H] --> [B,1,H]
        
        output = torch.stack(output)
        logit = self.lm_head(output) # B,1,V
        output = output.view(output.shape[0],-1) # [B,1,H] --> [B,H]

        return output, logit.squeeze(1)

class MistralClassification(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.model = MistralModel(config)
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
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
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
        

        e11 = []
        # for each sample in the batch, acquire the positions of its [E11] and [E21]
        for mask in attention_mask:
            e11.append(mask.sum().item() - 1)
        
        output = []
        for i in range(len(e11)):
            instance_output = torch.index_select(hidden_states, 0, torch.tensor(i).to(hidden_states.device))
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i]]).to(hidden_states.device))
            output.append(instance_output)  # [B,N,H] --> [B,1,H]
        
        output = torch.stack(output)
        output = output.view(output.shape[0],-1) # [B,1,H] --> [B,H]
        return output.to(torch.float32), None


class MistralLMClassification(MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.info_nce_fc = nn.Linear(config.vocab_size, config.hidden_size , bias= False)

        # Initialize weights and apply final processing
        self.post_init()

    def infoNCE_f(self,V,C):
        """
        V : 1 x dim_V
        C : 1 x dim_C

        """
        out = self.info_nce_fc(V) # N x dim_C
        out = torch.matmul(out, C.t().to(out.device)) # N x N
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        e11 = []
        # for each sample in the batch, acquire the positions of its [E11] and [E21]
        for mask in attention_mask:
            e11.append(mask.sum().item() - 1)

        max_length = max(e11) + 1

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids[:max_length],
            attention_mask=attention_mask[:max_length],
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
        
        output = []
        # for each sample in the batch, acquire its representations for [E11] and [E21]
        for i in range(len(e11)):
            instance_output = torch.index_select(hidden_states, 0, torch.tensor(i).to(hidden_states.device))
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i]]).to(hidden_states.device))
            output.append(instance_output)  # [B,N,H] --> [B,1,H]
        
        output = torch.stack(output)
        logit = self.lm_head(output) # B,1,V
        output = output.view(output.shape[0],-1) # [B,1,H] --> [B,H]

        return output, logit.squeeze(1)