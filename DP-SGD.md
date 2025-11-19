# Comparing General Model & Culture-specific Model under DP-SGD
With a bunch of data, we want to examine whether training a general model or culture-specific models yields better performance under differential privacy constraints using DP-SGD.
Use main method raised in [CultureLLM: Incorporating Cultural Differences into Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/9a16935bf54c4af233e25d998b7f4a2c-Paper-Conference.pdf) 


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
