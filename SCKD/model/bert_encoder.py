import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_model import base_model
from transformers import BertModel, BertConfig
from transformers import BertForMaskedLM 
from transformers.models.llama.modeling_llama import *

class Bert_Encoder(base_model):

    def __init__(self, config):
        super(Bert_Encoder, self).__init__()

        # load model
        self.encoder = BertModel.from_pretrained(config.bert_path).cuda()
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # the dimension for the final outputs
        self.output_size = config.encoder_output_size

        self.drop = nn.Dropout(config.drop_out)

        # find which encoding is used
        if config.pattern in ['standard', 'entity_marker']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')
        config.hidden_size = self.bert_config.hidden_size
        config.output_size = config.encoder_output_size
        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size * 2, self.output_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)

        self.layer_normalization = nn.LayerNorm([self.output_size])

    def get_output_size(self):
        return self.output_size

    def forward(self, inputs):
        '''
        :param inputs: of dimension [B, N]
        :return: a result of size [B, H*2] or [B, H], according to different strategy
        '''
        # generate representation under a certain encoding strategy
        if self.pattern == 'standard':
            # in the standard mode, the representation is generated according to
            #  the representation of[CLS] mark.
            output = self.encoder(inputs)[1]
        else:
            e11 = []
            e21 = []
            # for each sample in the batch, acquire the positions of its [E11] and [E21]
            for i in range(inputs.size()[0]):
                tokens = inputs[i].cpu().numpy()
                try: # hot fix for just 1 sample (error when test)
                    e21.append(np.argwhere(tokens == 30524)[0][0]) #
                except:
                    e21.append(0)
                e11.append(np.argwhere(tokens == 30522)[0][0])

            # input the sample to BERT
            tokens_output = self.encoder(inputs)[0] # [B,N] --> [B,N,H]
            output = []

            # for each sample in the batch, acquire its representations for [E11] and [E21]
            for i in range(len(e11)):
                instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
                instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
                output.append(instance_output)  # [B,N] --> [B,2,H]

            # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
            output = torch.cat(output, dim=0)
            output = output.view(output.size()[0], -1)  # [B,N] --> [B,H*2]
        return output
    
    
    
class Bert_EncoderMLM(base_model):

    def __init__(self, config):
        super(Bert_EncoderMLM, self).__init__()

        # load model
        self.encoder = BertForMaskedLM.from_pretrained(config.bert_path).cuda()
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # the dimension for the final outputs
        self.output_size = config.encoder_output_size

        self.drop = nn.Dropout(config.drop_out)

        
        # find which encoding is used
        if config.pattern in ['standard', 'entity_marker', 'entity_marker_mask']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')
        config.hidden_size = self.bert_config.hidden_size
        config.output_size = config.encoder_output_size
        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size * 2, self.output_size, bias=True)
            raise Exception('Not implemented yet')
        elif self.pattern == 'entity_marker_mask':
            self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size + config.num_of_relation)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size * 2, self.output_size, bias=True)

            self.info_nce_fc_0 = nn.Sequential(
                nn.Linear(config.encoder_output_size, config.encoder_output_size),
                nn.ReLU(),
                nn.Dropout()
            )
            self.info_nce_fc = nn.Linear(config.vocab_size + config.marker_size + config.num_of_relation, config.encoder_output_size , bias= False)
            
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)

        self.layer_normalization = nn.LayerNorm([self.output_size])
    def infoNCE_f(self,V,C , temperature=1000.0):
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
    
    def get_output_size(self):
        return self.output_size

    def forward(self, inputs):
        '''
        :param inputs: of dimension [B, N]
        :return: a result of size [B, H*2] or [B, H], according to different strategy
        '''
        # generate representation under a certain encoding strategy
        if self.pattern == 'standard':
            # in the standard mode, the representation is generated according to
            #  the representation of[CLS] mark.
            outputs = self.encoder(inputs,output_hidden_states=True)
            output = outputs.hidden_states[-1][0] # last hidden state of the [CLS] token
            lm_head_output = outputs.logits
            
        elif self.pattern == 'entity_marker':
            e11 = []
            e21 = []
            # for each sample in the batch, acquire the positions of its [E11] and [E21]
            for i in range(inputs.size()[0]):
                tokens = inputs[i].cpu().numpy()
                try: # hot fix for just 1 sample (error when test)
                    e21.append(np.argwhere(tokens == 30524)[0][0]) #
                except:
                    e21.append(0)
                e11.append(np.argwhere(tokens == 30522)[0][0])

            # input the sample to BERT
            outputs = self.encoder(inputs,output_hidden_states=True) 
            last_hidden_states = outputs.hidden_states[-1] # [B,N,H]
            lm_head_output = outputs.logits
            output = []

            # for each sample in the batch, acquire its representations for [E11] and [E21]
            for i in range(len(e11)):
                instance_output = torch.index_select(last_hidden_states, 0, torch.tensor(i).cuda())
                instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
                output.append(instance_output)  # [B,N] --> [B,2,H]

            # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
            output = torch.cat(output, dim=0)
            output = output.view(output.size()[0], -1)  # [B,N] --> [B,H*2]
        else:
            # entity marker mask
            e11 = []
            e21 = []
            # for each sample in the batch, acquire the positions of its [E11] and [E21]
            for i in range(inputs.size()[0]):
                tokens = inputs[i].cpu().numpy()
                try: # hot fix for just 1 sample (error when test)
                    e21.append(np.argwhere(tokens == 30524)[0][0]) #
                except:
                    e21.append(0)
                    print("e21 not found" )
                try:
                    e11.append(np.argwhere(tokens == 30522)[0][0])
                except:
                    e11.append(0)
                    print("e11 not found" )

            # input the sample to BERT
            outputs = self.encoder(inputs,output_hidden_states=True) 
            last_hidden_states = outputs.hidden_states[-1] # [B,N,H]
            lm_head_output = outputs.logits
            output = []

            # for each sample in the batch, acquire its representations for [E11] and [E21]
            for i in range(len(e11)):
                instance_output = torch.index_select(last_hidden_states, 0, torch.tensor(i).cuda())
                instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
                output.append(instance_output)  # [B,N] --> [B,2,H]

            # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
            output = torch.cat(output, dim=0)
            output = output.view(output.size()[0], -1)  # [B,N] --> [B,H*2]
            
            # for each sample in the batch, acquire the representations for the [MASK] token
            mask_output = []
            for i in range(inputs.size()[0]):
                tokens = inputs[i].cpu().numpy()
                mask_ix = np.argwhere(tokens == 103)[0][0]
                instance_output = torch.index_select(lm_head_output, 0, torch.tensor(i).cuda())
                instance_output = torch.index_select(instance_output, 1, torch.tensor(mask_ix).cuda())
                mask_output.append(instance_output)
            mask_output = torch.cat(mask_output, dim=0)
        return output , mask_output
    
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
        

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

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
        for i in range(input_ids.shape[0]):
            tokens = input_ids[i].cpu().numpy()
            try:
                e11.append(np.argwhere(tokens == 2)[0][0] - 1)
            except:
                e11.append(len(tokens) - 1)
        
        output = []

        # for each sample in the batch, acquire its representations for [E11] and [E21]
        for i in range(len(e11)):
            instance_output = torch.index_select(hidden_states, 0, torch.tensor(i).to(hidden_states.device))
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i]]).to(hidden_states.device))
            output.append(instance_output)  # [B,N,H] --> [B,2,H]
        
        output = torch.stack(output)
        output = output.view(output.shape[0],-1) # [B,1,H] --> [B,H]

        if get_feature == True:
            # print("here")
            tokens = input_ids[0].cpu().numpy()
            x11 = np.argwhere(tokens == 376)[-2][0] + 1
            x21 = np.argwhere(tokens == 376)[-1][0] + 1
            try:
                x12 = np.argwhere(tokens == 29908)[-2][0]
                x22 = np.argwhere(tokens == 29908)[-1][0]
            except:
                try:
                    x12 = np.argwhere(tokens == 29908)[-1][0]
                    x22 = np.argwhere(tokens == 1213)[-1][0]
                except:
                    print(tokens)
                    x12 = x11+1
                    x22 = x21+1
            
            feature = torch.cat([torch.mean(hidden_states[:,x11:x12,:], dim=1), torch.mean(hidden_states[:,x21:x22,:], dim=1)], dim=1)
            return feature

        # logits = self.score(output)
        # return self.dropout(logits)
        return output
    
class LlamaLMClassification(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
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
        att_mask_0: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
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
