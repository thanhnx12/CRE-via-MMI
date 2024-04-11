# CRE via MMI

## Requirements
```
pip install -r requirements.txt
```

## Run scripts
### BERT
#### TacRed

[Wandb logs here](https://wandb.ai/banana1209/DATN/runs/hnaifl2y/logs?nw=nwuserthanhnx1209)
```bash
>> cd SCKD
>> python main-mmi.py --task tacred --shot 5 
```

[Wandb logs here](https://wandb.ai/banana1209/DATN/runs/qd1yel07/logs?nw=nwuserthanhnx1209)
```bash
>> cd ConPL
>> python main.py --task tacred --shot 5  
```

#### FewRel

[Wandb logs here](https://wandb.ai/banana1209/DATN/runs/ijm94ol7/logs?nw=nwuserthanhnx1209)
```bash
>> cd SCKD
>> python main-mmi.py --task FewRel --shot 5 
```

[Wandb logs here](https://wandb.ai/banana1209/DATN/runs/bs39lao1?nw=nwuserthanhnx1209)
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

