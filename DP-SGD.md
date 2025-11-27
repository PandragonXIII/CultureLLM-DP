# Comparing General Model & Culture-specific Model under DP-SGD
With a bunch of data, we want to examine whether training a general model or culture-specific models yields better performance under differential privacy constraints using DP-SGD.
Use main method proposed in [CultureLLM: Incorporating Cultural Differences into Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/9a16935bf54c4af233e25d998b7f4a2c-Paper-Conference.pdf) 


## Dataset
World Values Survey (WVS) dataset (as seed data)

## Method
### Data Augmentation

For each item (query, answer), do: 

1. Rewrite the query $k$ times (after filtering).
2. For each rewritten query(template), fill in the blanks with synonyms or similar words to generate $m$ different versions.

### Fine-tuning with DP-SGD
Add system prompt "You are an [culture] chatbot that know [culture] very well."
Add template to query: `Give me the answer from {min(nums)} to {max(nums)}: Do you agree with {content}? {option}. You can only choose one option.` or `Give me the answer from {min(nums)} to {max(nums)}: Do you agree with {content}? {option}. You can only choose one option.`


## Metrics
- Arabic: 'offensive_detect', 'offensive_detect_osact4', 'offensive_detect_mp', 'offensive_detect_osact5', 'hate_detect_osact4', 'hate_detect_mp', 'hate_detect_osact5', 'vulgar_detect_mp', 'spam_detect', 'hate_detect_fine-grained'
- Bengali: 'hate_detect_religion', 'offensive_detect_1', 'offensive_detect_2', 'offensive_detect_3', 'racism_detect', 'threat_detect'
- Chinese: 'bias_on_gender_detect', 'spam_detect'
- English: 'hate_detect_2', 'hate_offens_detect', 'hostility_directness_detect', 'offensive_detect_easy', 'threat_detect', 'toxicity_detect'
- Germany: 'hate_detect', 'hate_off_detect', 'hate_detect_iwg_1', 'hate_detect_check', 'offensive_detect_eval'
- Korean: 'abusive_detect', 'abusive_detect_2', 'abusive_detect_4', 'hate_detect_3', 'hate_detect_6', 'hate_detect_7'
- Portuguese: 'homophobia_detect', 'insult_detect', 'misogyny_detect', 'offensive_detect_2', 'offensive_detect_3'
- Spanish: 'offensive_detect_ami', 'offensive_detect_mex_a3t', 'offensive_detect_mex_offend', 'hate_detect_eval', 'hate_detect_haterNet', 'stereotype_detect', 'mockery_detect', 'insult_detect', 'improper_detect', 'aggressiveness_detect' 'negative_stance_detect'
- Turkish: 'offensive_detect', 'offensive_detect_corpus', 'offensive_detect_finegrained', 'offensive_detect_kaggle', 'offensive_detect_kaggle2', 'abusive_detect', 'spam_detect'


## Pilot Study
Want to verify if culture-specific models outperform general models under DP-SGD.
General model should train on joint data from all cultures, while culture-specific models train on data from corresponding cultures.

Minimum viable experiment:
1. choose 2-3 culture with varient resources (and common evaluation metrics), train corresponding cultural-specific models & a general model(with all data). 
Local training with DP-SGD using Opacus (dp-Huggingface).先尝试在训练llama的代码上进行修改进行DP-SGD微调
2. Evaluate general model & culture-specific models on corresponding evaluation metrics and compare F1 score.


### Problems


# Experiment
Try run 
```shell
python llama_finetune.py --base_model "/home/qxy/models/Llama-3.2-1B-Instruct" --new_model "models/Germany/Llama-3.2-3B-Instruct-Germany" --data_files "./data/Germany/Finetune/WVQ_Germany_llama.jsonl"
```

then implement dp version

```shell
python llama_finetune.py --base_model "/home/qxy/models/Llama-3.2-1B-Instruct" --new_model "models/Germany/Llama-3.2-3B-Instruct-Germany-dp" --data_files "./data/Germany/Finetune/WVQ_Germany_llama.jsonl" --dp True
```

reinstall env:
1. `conda create -n dpsgd python=3.10`'
2. `python -m pip install torch torchvision`
3. `python -m pip install jsonlines fire scikit-learn  transformers bitsandbytes `
4. `python -m pip install datasets`
5. `pip install peft trl `
6. `pip install .` (dp-transformers)
7. `pip install tensorboardX`



### DUBUG
#### savemodel 
> RuntimeError: [enforce fail at inline_container.cc:664] . unexpected pos 2862770624 vs 2862770512
疑似硬盘空间不足导致

#### savemodel 2
>RuntimeError: 
>            Some tensors share memory, this will lead to duplicate memory on disk and potential differences when loading them again: [{'_module.model.embed_tokens.weight', '_module.lm_head.weight'}].
warrped model 使用trainer.save_model()有问题。查看其代码后发现其使用的也是model.save_pretrained()，所以获得warpper内部的原始LLama模型后，直接使用下面代码保存模型和tokenizer：
```python
    trainer.model_wrapped.model.save_pretrained(new_model, safe_serialization=True)
    trainer.data_collator.tokenizer.save_pretrained(new_model, safe_serialization=True)
```
not work:
```python
model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
model_to_save.save_pretrained(new_model_dir, safe_serialization=True)   # safe_serialization optional
tokenizer.save_pretrained(new_model_dir)
```
#### training OpacusDPTrainer arg
>TypeError: OpacusDPTrainer.training_step() takes 3 positional arguments but 4 were given
modify function `training_step` in dp-transformers pkg.

#### eval
> LoadFromFile
>    return _sentencepiece.SentencePieceProcessor_LoadFromFile(self, arg)
> TypeError: not a string
tokenizer load error, change to AutoTokenizer works.

#### device/map error
>ValueError: No activations detected for <class 'torch.nn.modules.linear.Linear'>, run forward after add_hooks(model)
On deleteing `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`, the error occurs.

## dp_pipeline.py
Load data, train, test on specific language and model w/wo DP
```shell
python dp_pipeline.py --country Germany --cuda 0 --dp
```

### TODO
[ ] evaluate original model and compare.
[ ] train different size model 1b/3b and compare
    [ ] Allow quantize fine-tuning. (3b model OOM)
[ ] train non-dp version and compare
[ ] train base model with full data (culturellm-all)
[ ] histo gram
