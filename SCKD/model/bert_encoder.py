import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_model import base_model
from transformers import BertModel, BertConfig
from transformers import BertForMaskedLM 

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
    
