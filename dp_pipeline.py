"""
choose multiple languages
split train-test datasets
train model under DP
evaluate on testset & compare with base model

"""
import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM,LlamaForCausalLM,LlamaConfig,AutoTokenizer,BitsAndBytesConfig,TrainingArguments,pipeline
from trl import SFTTrainer
from typing import Union
import dp_transformers
import argparse
from sklearn.metrics import f1_score
import json
import numpy as np


Languages = ["Arabic", "English", "Korean", "Turkey","Bengali", "China", "Germany","Portugal", "Greece", "Spanish"] #High, middle, low resource languages
GENERATE_MAX_TOKENS = 5

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Llama Model on a Specific Language")
    parser.add_argument('--base_model', type=str, default="/home/qxy/models/Llama-3.2-1B-Instruct", help='Path to the base model')
    parser.add_argument('--dp', action='store_true', help='Enable differential privacy training')
    parser.add_argument('--country', type=str, choices=Languages, required=True, help='Country/Language for finetuning')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA device to use')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.country == 'Arabic':
        datapath = f"data/Arabic/Finetune/WVQ_arabic_Iraq_Jordan_llama.jsonl"
    else:
        datapath = f"data/{args.country}/Finetune/WVQ_{args.country}_llama.jsonl"
    dataset = load_dataset('json', data_files=datapath, split='train')
    # check dataset sizes
    print(f"{args.country} dataset size: {dataset.num_rows}")
    print(f"example:\n{dataset[0]}")
    # split train-test datasets
    splited = dataset.train_test_split(test_size=0.2,seed=42)
    print(f"train size: {splited['train'].num_rows}, test size: {splited['test'].num_rows}")

    ##### finetune models under DP #####
    base_model_name = args.base_model.split('/')[-1]
    md_pth = finetune_language_model(
        base_model=args.base_model,
        new_model=f"models/{args.country}/{base_model_name}-{args.country}-dp" if args.dp else f"models/{args.country}/{base_model_name}-{args.country}",
        dataset=splited['train'],
        country=args.country,
        dp=args.dp,
        device="auto"
    )

    ##### eval on testset & compare with base model #####
    weighted_f1 = eval(md_pth,"cuda:0",splited['test'])[-1]
    avg_distance = eval(md_pth,"cuda:0",splited['test'],metric="distance")
    # save to file
    if not os.path.exists("results/score.json"):
        with open("results/score.json", 'w') as fw:
            json.dump({}, fw)
            read_scores = {}
    else:
        with open(f"results/score.json", 'r') as fr:
            read_scores = json.load(fr)
    if md_pth not in read_scores.keys():
        read_scores[md_pth] = {}
    if args.country not in read_scores[md_pth].keys():
        read_scores[md_pth][args.country] = {}
    read_scores[md_pth][args.country]['weighted_f1'] = weighted_f1
    read_scores[md_pth][args.country]['avg_distance'] = avg_distance
    with open(f"results/score.json", 'w') as fw:
        json.dump(read_scores, fw, indent=4)

    
def eval(model_path,cuda_device,testset,metric='f1'):
    # load finetuned model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token # Basemodel may not have pad_token
    model = LlamaForCausalLM.from_pretrained(model_path, device_map=cuda_device)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    # get labels
    y_true = []
    y_pred = []
    for i in range(len(testset)):
        messages = testset[i]['text']
        # get true answer
        y_true.append(messages.split("### Answer:")[-1].strip())

    # preprocess testset
    testset = data_preprocess(testset,tokenizer,2048,eval_flag=True)
    # generate some samples
    print("Generating some sample responses:")
    for i in range(2):
        messages = testset[i]['text']
        # print(f"message:{messages}")
        # remove "assistant" part to evaluate
        # messages = "<|begin_of_text|>"+messages.split("### Answer:")[0].strip()+"### Answer:"
        print(f"Input messages:{messages}")
        outputs = pipe(
            messages,
            max_new_tokens=5,
            return_full_text=False
        )
        print(f"Response: {outputs[0]['generated_text']}\n")
    
    ##### compute f1 matrix on full testset #####
    # generate predictions with pipeline
    outputs = pipe(
        list(testset['text']),
        max_new_tokens=GENERATE_MAX_TOKENS,
        return_full_text=False
    )
    # post-process outputs to get predicted answers
    for i,output in enumerate(outputs):
        ans = post_process(output[0]['generated_text'],ground_truth=y_true[i])
        y_pred.append(ans)
    if metric == 'f1':
        # compute f1 score
        microf1 = f1_score(y_true, y_pred, average='micro')
        print(f"micro F1 score on testset: {microf1}")
        macrof1 = f1_score(y_true, y_pred, average='macro')
        print(f"macro F1 score on testset: {macrof1}")
        weightedf1 = f1_score(y_true, y_pred, average='weighted')
        print(f"weighted F1 score on testset: {weightedf1}")
        
        return (microf1,macrof1,weightedf1)
    elif metric == "distance":
        # clean data to float, ignore non-numeric answers
        num_y_true = []
        num_y_pred = []
        for y_t, y_p in zip(y_true[:], y_pred[:]):
            try:
                num_y_true.append(float(y_t))
                num_y_pred.append(float(y_p))
            except:
                continue
            
        # compute average distance
        dist = np.mean([abs(y_t - y_p) for y_t,y_p in zip(num_y_true,num_y_pred)])
        print(f"Average distance on testset: {dist}")
        return dist


def finetune_language_model(base_model,new_model,dataset:Dataset,country,dp=False,device='cuda:0'):

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map=device,
    )
    # model.config.use_cache = False # FIXME

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_params = TrainingArguments(
        output_dir=f"./cache/results-finetune/{new_model}",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        # save_steps=25, # disable ckpt save since ERROR: tensors share memory
        save_strategy='no',
        # logging_strategy='no'
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        remove_unused_columns=False,
        save_safetensors=True
    )

    if not dp:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            # peft_config=peft_params,
            processing_class=tokenizer,
            args=training_params,
        )
    else:
        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

        # Prepare Training Dataset
        # print(dataset)
        # print(dataset["text"][0])
        dataset = data_preprocess(dataset,tokenizer,2048,country)
        dataset = dataset.remove_columns("text")
        print("After preprocess:")
        print(dataset)
        print(model.dtype,end="\n\n")

        privacy_args = dp_transformers.PrivacyArguments(
            per_sample_max_grad_norm=1.0,
            # noise_multiplier=
            target_epsilon=8,
            # target_delta=
        )

        trainer = dp_transformers.dp_utils.OpacusDPTrainer(
            args=training_params,
            model=model,
            train_dataset=dataset,
            data_collator=data_collator,
            privacy_args=privacy_args,
        )

    trainer.train()

    if dp:
        trainer.model_wrapped.model.save_pretrained(new_model, safe_serialization=True)
        trainer.data_collator.tokenizer.save_pretrained(new_model, safe_serialization=True)
    else:
        trainer.save_model(new_model)
    print("Model save done.")
    return new_model

def data_preprocess(dataset,tokenizer,max_length,country=None,eval_flag=False):
    """
    Load dataset with "text" col, convert to Llama training format and tokenize
    split "text" field into "question" and "answer" parts and apply chat template,
    tokenize with tokenizer,
    params:
    dataset (llama format): {"text": "### Question: xxx \n ### Answer: x"}
    """
    llama_base_template = "<|begin_of_text|>{query} Answer Only by number; No Explanation \n{answer}<|end_of_text|>"
    llama_chat_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sysprompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eom_id|><|eot_id|><|end_of_text|>"
    def apply_base_template_tokenize(example,eval_flag):
        prompt = example["text"]
        query = prompt.split("### Answer:")[0].strip()
        answer = "### Answer: " + prompt.split("### Answer:")[-1].strip()
        example["text"] = llama_base_template.format(query=query, answer=answer)
        if eval_flag:
            # remove answer part and end token for evaluation
            example["text"] = example["text"].split("### Answer:")[0] + "### Answer:"
        # tokenize
        outputs = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_attention_mask=True,
        )
        example["input_ids"] = outputs["input_ids"]
        example["attention_mask"] = outputs["attention_mask"]
        return example
    def apply_chat_template_tokenize(example):
        messages = example["messages"]
        sys_prompt = ""
        user_prompt = ""
        assistant_answer = ""
        for msg in messages:
            if msg["role"] == "system":
                sys_prompt += msg["content"]
            elif msg["role"] == "user":
                user_prompt += msg["content"]
            elif msg["role"] == "assistant":
                assistant_answer += msg["content"]
        example["text"] = llama_chat_template.format(
            sysprompt=sys_prompt.strip(),
            query=user_prompt.strip(),
            answer=assistant_answer.strip()
        )
        # tokenize
        outputs = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_attention_mask=True,
        )
        example["input_ids"] = outputs["input_ids"]
        example["attention_mask"] = outputs["attention_mask"]
        return example
    dataset = dataset.map(apply_base_template_tokenize,
                          fn_kwargs={"eval_flag": eval_flag}) # ,remove_columns=["text"]
    print(dataset["text"][0])
    dataset.set_format(type="torch", columns=["text", "input_ids", "attention_mask"])
    return dataset

def post_process(output,ground_truth=None):
    if ground_truth is not None:
        if ground_truth.lower() in output.lower():
            label = ground_truth
            return label
    # find the only number in the output text
    import re
    matches = re.findall(r'\d', output)
    if len(matches) == 1:
        label = matches[0]
    else:
        label = output
    return label

if __name__=="__main__":
    main()