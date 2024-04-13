# CRE via MMI

## Requirements
```
pip install -r requirements.txt
```

## Run scripts
### BERT
#### TacRed

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

