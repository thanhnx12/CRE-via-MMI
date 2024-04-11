
# BERT
## FewlRel 5 shot
```bash
python main.py --task fewrel --shot 5
```
## TACRED 5 shot
```bash
python main.py --task tacred --shot 5
```
# LLM
* put `hf_token` to `main-llm.py` and `dataprocess.py`
## FewlRel 5 shot
```bash
python main-llm.py --task fewrel --shot 5
```
## TACRED 5 shot
```bash
python main-llm.py --task tacred --shot 5
```