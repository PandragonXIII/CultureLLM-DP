import os, fire
import torch
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    logging,
)
import dp_transformers

raise RuntimeError

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model from Hugging Face hub or Model path
base_model = 'Llama-3.2-3B-Instruct'

def run(base_model, new_model, data_files):
    dataset = load_dataset('json', data_files=data_files, split='train')
    print(dataset)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map={'': torch.cuda.current_device()},
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # ensure pad token exists (causal LM often uses EOS as pad)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ===== Add preprocessing / tokenization step here =====
    # choose a sensible max_length — prefer tokenizer.model_max_length when valid
    try:
        max_length = tokenizer.model_max_length
        # some tokenizers set model_max_length to very large value like 1e30; cap it
        if max_length is None or max_length > 2**16:
            max_length = 2048
    except Exception:
        max_length = 1024
    print(max_length)

    def preprocess_batch(examples):
        # 'text' is the field in your jsonl
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            # do NOT pad here if you want dynamic padding in the data-collator.
            # padding=True would pad to max_length which might be unnecessary.
            # We'll leave padding to the data-collator (which supports DP).
            padding=True,
            return_attention_mask=True,
        )
        # for causal LM the labels are typically the same as input_ids
        # Some data-collators will create labels; but keeping them here is safe.
        # outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    # map tokenization over the dataset (batched)
    dataset = dataset.map(preprocess_batch, batched=True, remove_columns=["text"])
    # set dataset format to torch tensors (Trainer expects that)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print(dataset)

    training_params = TrainingArguments(
        output_dir="./cache/results-finetune",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
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
        remove_unused_columns=False
    )

    # Data collator for DP causal LM (expects tokenized inputs)
    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

    privacy_args = dp_transformers.PrivacyArguments(
        per_sample_max_grad_norm=1.0,
        # noise_multiplier=
        target_epsilon=8,
        # target_delta=
    )

    # Pass tokenizer to trainer so trainer.save_pretrained/trainer.tokenizer work
    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=training_params,
        model=model,
        train_dataset=dataset,
        data_collator=data_collator,
        privacy_args=privacy_args,
        tokenizer=tokenizer,
    )

    trainer.train()

    # save model + tokenizer
    # trainer.model.save_pretrained(new_model)
    # trainer.save_model(new_model)
    # trainer.tokenizer.save_pretrained(new_model)

    # ensure saving the underlying model and only on main process
    model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model

    # optionally use safe_serialization to avoid duplicate storage
    model_to_save.save_pretrained(new_model, safe_serialization=True)
    trainer.tokenizer.save_pretrained(new_model)

if __name__ == '__main__':
    fire.Fire(run)
