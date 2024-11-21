# Preserving Generalization of Language Models in Few-shot Continual Relation Extraction

This repository provides the implementation for "Preserving Generalization of Language Models in Few-shot Continual Relation Extraction (EMNLP2024)."

## Requirements
To run the code, please install the following dependencies:
```bash
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
We perform experiments using two publicly available relation extraction datasets:

- **[FewRel](https://github.com/thunlp/FewRel)**: A large-scale few-shot relation extraction dataset.
- **[TACRED](https://nlp.stanford.edu/projects/tacred/)**: A widely-used dataset for relation classification.

## Training
### BERT-based Models

#### TACRED
To train BERT models on TACRED with 5-shot settings, follow these steps:

1. Navigate to the CPL scripts directory:
   ```bash
   cd CPL/bash
   ```
2. Run the 5-shot training script:
   ```bash
   bash tacred_5shot.sh
   ```

Alternatively, you can directly run the training script for different components:

1. For SCKD with Mutual Information Maximization (MMI):
   ```bash
   cd SCKD
   python main-mmi.py --task tacred --shot 5 
   ```

2. For ConPL:
   ```bash
   cd ConPL
   python main.py --task tacred --shot 5  
   ```

#### FewRel
To train BERT models on FewRel with 5-shot settings:

1. Run the 5-shot script from the CPL directory:
   ```bash
   cd CPL/bash
   bash fewrel_5shot.sh
   ```

2. Alternatively, run the training commands directly:

   - SCKD with MMI:
     ```bash
     cd SCKD
     python main-mmi.py --task fewrel --shot 5 
     ```

   - ConPL:
     ```bash
     cd ConPL
     python main.py --task fewrel --shot 5  
     ```

### LLAMA2-based Models

To train using LLAMA2, ensure that you have set up your Hugging Face token (`hf_token`) in the required scripts:

- For ConPL, add the token in `main-llm.py` and `dataprocess.py`.
- For SCKD, add the token in `sampler.py`, `main-llm.py`, and `main-llm-mmi.py`.

#### TACRED

1. To run SCKD with MMI:
   ```bash
   cd SCKD
   python main-llm-mmi.py --task tacred --shot 5 
   ```

2. To run ConPL:
   ```bash
   cd ConPL
   python main-llm.py --task tacred --shot 5  
   ```

#### FewRel

1. To run SCKD with MMI:
   ```bash
   cd SCKD
   python main-llm-mmi.py --task fewrel --shot 5 
   ```

2. To run ConPL:
   ```bash
   cd ConPL
   python main-llm.py --task fewrel --shot 5  
   ```
