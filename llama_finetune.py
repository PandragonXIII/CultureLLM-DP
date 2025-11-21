import os, fire
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
from typing import Union
import dp_transformers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# _orig_training_step = transformers.OpacusDPTrainer.training_step

# def _patched_training_step(self, model, inputs, optimizer=None):
#     # Accept the optional optimizer argument that some transformers versions pass.
#     # Call the original implementation that expects (self, model, inputs).
#     return _orig_training_step(self, model, inputs)

# # monkeypatch
# dp_utils.OpacusDPTrainer.training_step = _patched_training_step


# Model from Hugging Face hub or Model path
base_model = ''

def run(base_model, new_model, data_files, dp=False):
    dataset = load_dataset('json', data_files=data_files, split='train[:10]')

    # compute_dtype = getattr(torch, "float16")

    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=False,
    # )

    # max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # quantization_config=quant_config,
        # device_map={"": 0}
        device_map={'':torch.cuda.current_device()},
        # max_memory=max_memory
    )
    # model.quantization_config = quant_config
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_params = TrainingArguments(
        output_dir="./cache/results-finetune",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        # save_steps=25, # disable ckpt save since ERROR: RuntimeError: Some tensors share memory, this will lead to duplicate memory on disk and potential differences when loading them again
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
            peft_config=peft_params,
            # tokenizer=tokenizer,
            processing_class=tokenizer,
            args=training_params,
        )
    else:
        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

        # dataset = prepare_dataset_dp(dataset,tokenizer,2048)
        print(dataset)
        print(dataset["text"][0])
        dataset = dataset.map(preprocess_batch, 
                              fn_kwargs={"tokenizer":tokenizer,"max_length":2048
                                },batched=True, remove_columns=["text"])
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        print(dataset.features,end="\n\n")
        print(model.dtype,end="\n\n")
        # dataset.

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

    trainer.model_wrapped.model.save_pretrained(new_model, safe_serialization=True)
    trainer.data_collator.tokenizer.save_pretrained(new_model, safe_serialization=True)
    # trainer.save_model(new_model)
    print("Model save done.")

def eval(model="models/Germany/Llama-3.2-3B-Instruct-Germany"):
    prompt = "Who is Leonardo Da Vinci?"
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(model, device_map="auto")
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n### Question: {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
    
    print(result[0]['generated_text'])

def prepare_dataset_dp(dataset,processing_class,max_length):
    # dataset = dataset.with_transform(remove_none_values)
    # map_kwargs = {}
    #     if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
    #         map_kwargs["num_proc"] = args.dataset_num_proc
    def add_eos(example, eos_token):
        if  not example["text"].endswith(eos_token):  # language modeling case
            example["text"] = example["text"] + eos_token
        return example

    dataset = dataset.map(
        add_eos,
        fn_kwargs={"eos_token": processing_class.eos_token},
    )

    def tokenize_fn(example, processing_class, dataset_text_field='text'):
        output = {"input_ids": processing_class(text=example[dataset_text_field],padding=True,)["input_ids"]}
        return output

    dataset = dataset.map(
        tokenize_fn,
        fn_kwargs={"processing_class": processing_class},
    )
    from trl.data_utils import truncate_dataset
    dataset = truncate_dataset(dataset, max_length)
    return dataset

def preprocess_batch(dataset,tokenizer,max_length):
        # apple chat template
        # separate dataset into "question" and "answer" parts
        
        outputs = tokenizer(
            dataset["text"],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_attention_mask=True,
        )
        return outputs

if __name__ == '__main__':
    fire.Fire(run)
    # eval()



