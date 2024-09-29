# Preserving Generalization of LLM in FCRE

## Requirements
```
transformers==4.20.0
wordninja
wandb
scikit-learn
tqdm
numpy==1.23.0
peft
accelerate
sentencepiece
protobuf
```

## Datasets
We conduct our experiments on two public relation extraction datasets:
- [FewRel](https://github.com/thunlp/FewRel)
- [TACRED](https://nlp.stanford.edu/projects/tacred/)

## Train
To run our method, use these command:

### BERT
#### TacRed


```bash
>> cd CPL/bash
>> bash tacred_5shot.sh
```

```bash
>> cd SCKD
>> python main-mmi.py --task tacred --shot 5 
```

```bash
>> cd ConPL
>> python main.py --task tacred --shot 5  
```

#### FewRel


```bash
>> cd CPL/bash
>> bash fewrel_5shot.sh
```

```bash
>> cd SCKD
>> python main-mmi.py --task FewRel --shot 5 
```

```bash
>> cd ConPL
>> python main.py --task fewrel --shot 5  
```
### LLAMA2
* put `hf_token` to `main-llm.py` and `dataprocess.py` for `ConPL`
* put `hf_token` to `sampler.py`, `main-llm.py` and `main-llm-mmi.py` for `SCKD`

#### TacRed
```bash
>> cd SCKD
>> python main-llm-mmi.py --task tacred --shot 5 
```

```bash
>> cd ConPL
>> python main-llm.py --task tacred --shot 5  
```

#### FewRel
```bash
>> cd SCKD
>> python main-llm-mmi.py --task FewRel --shot 5 
```

```bash
>> cd ConPL
>> python main-llm.py --task fewrel --shot 5  
```

