# Preserving Model Generalization in Few-shot Relation Extraction

## Official code repository for the work:
"Preserving Model Generalization in Few-shot Relation Extraction".

## Environment
```
pip install -r requirements.txt
```

## Run scripts bellow to reproduce the reported results: 

### On TacRed

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

### On FewRel

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

