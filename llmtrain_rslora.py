import pickle

import os
import random
import pandas as pd
from datasets import Dataset
import json

from peft import PeftModel


def int2dow(int_day):
    tmp = {0:'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    return tmp[int_day]

def convert_to_12_hour_clock(minutes):
    if minutes < 0 or minutes >= 1440:
        return "Invalid input. Minutes should be between 0 and 1439."

    hours = minutes // 60
    minutes %= 60

    period = "AM"
    if hours >= 12:
        period = "PM"

    if hours == 0:
        hours = 12
    elif hours > 12:
        hours -= 12

    return f"{hours:02d}:{minutes:02d} {period}"

def data_format(inst, history, context, target, label, data_name):
    if data_name == 'fsq' or data_name == 'gl' or data_name == 'ori':
        pred_obj = 'place'
    if data_name == 'dest':
        pred_obj = 'destination'

    inst = inst + '\n' + f'Please organize your answer in a JSON object containing following keys: "prediction" ({pred_obj}).'

    data_fm = f"""<history>: {history} \n <context>: {context} \n <target>: {target} \n """
    data_label = 'Pred results: {\n' + f'''"prediction": {label}''' + '\n}'

    inst_data_fm = {
        "instruction": inst,
        "input": data_fm,
        "output": data_label
    }
    # data_label = f'prediction: {label}'

    return inst_data_fm

def organise_data(data_train, uid, data_inst, data_name='gl'):
    # Use another way of organising data
    # historical_data = []

    data_uid = data_train[uid]
    # predict_X = []
    # predict_y = []
    predict_xy = []

    for i in range(len(data_uid)):
        his_i, con_i, tar_i = data_uid[i]

        history, context = [], []

        if data_name == 'fsq':
            for act in his_i:
                history.append((convert_to_12_hour_clock(act[0]), int2dow(act[1]),
                                act[2]))

            for act in con_i:
                context.append((convert_to_12_hour_clock(act[0]), int2dow(act[1]),
                                act[2]))
            target = (
                convert_to_12_hour_clock(tar_i[0][0]),
                int2dow(tar_i[0][1]), "<next_place>")
            y = tar_i[0][2]

        if data_name == 'gl':
            for act in his_i:
                history.append((convert_to_12_hour_clock(act[0]), int2dow(act[1]),
                                act[2], act[3]))

            for act in con_i:
                context.append((convert_to_12_hour_clock(act[0]), int2dow(act[1]),
                                act[2], act[3]))
            target = (
                convert_to_12_hour_clock(tar_i[0][0]),
                int2dow(tar_i[0][1]), None, "<next_place>")
            y = tar_i[0][3]

        if data_name == 'ori':
            for act in his_i:
                history.append((convert_to_12_hour_clock(act[0]), int2dow(act[1]),
                                act[2], act[3], act[4]))

            for act in con_i:
                context.append((convert_to_12_hour_clock(act[0]), int2dow(act[1]),
                                act[2], act[3], act[4]))

            target = (
                convert_to_12_hour_clock(tar_i[0][0]),
                int2dow(tar_i[0][1]), None, tar_i[0][3], "<next_place>")
            y = tar_i[0][4]

        if data_name == 'dest':
            for act in his_i:
                history.append((convert_to_12_hour_clock(act[0]), int2dow(act[1]),
                                act[2], act[3], act[4]))

            for act in con_i:
                context.append((convert_to_12_hour_clock(act[0]), int2dow(act[1]),
                                act[2], act[3], act[4]))

            target = (
                convert_to_12_hour_clock(tar_i[0][0]),
                int2dow(tar_i[0][1]), tar_i[0][2], tar_i[0][3], "<next_destination>")
            y = tar_i[0][4]


        inst = random.choice(data_inst)
        inst_data_fm = data_format(inst, history, context, target, y, data_name)
        predict_xy.append(inst_data_fm)

    # print(f"Number of predict_data: {len(predict_X)}")
    # print(f"predict_y: {predict_y}")
    # print(f"Number of predict_y: {len(predict_y)}")

    return predict_xy



def load_dataset():
    seed = 111
    random.seed(seed)
    if not os.path.exists('data4FT/temp/ft_train.json'):
        trainset, valset = [], []
        # assemble instructions and data input
        for data_name in ['fsq', 'gl', 'ori', 'dest']:
            if data_name == 'fsq' or data_name == 'gl':
                with open(f'data4FT/ft_{data_name}.pkl', 'rb') as fr:
                    data_train, data_val, data_test = pickle.load(fr)
            else:
                with open(f'data4FT/ft_{data_name}.pkl', 'rb') as fr:
                    data_train, data_val, _, data_test = pickle.load(fr)
            with open('prompts_store/ft_traindata/prim_instructions.pkl', 'rb') as fr:
                data_inst = pickle.load(fr)[data_name]
            for uid in data_train:
                xy_train = organise_data(data_train, uid, data_inst, data_name)
                trainset.extend(xy_train)
            for uid in data_val:
                xy_val = organise_data(data_val, uid, data_inst, data_name)
                valset.extend(xy_val)
        trainset.extend(valset)
        with open('data4FT/temp/ft_train.json', 'w') as fw:
            json.dump(trainset, fw, indent=4)

    df = pd.read_json('data4FT/temp/ft_train.json')
    ds = Dataset.from_pandas(df)

    return ds


def process_func(example):
    MAX_LENGTH = 2048    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def train():
    global tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
    import torch
    from peft import LoraConfig, TaskType, get_peft_model


    # 1. create instruction FT dataset
    ds = load_dataset()
    print('Dataset is loaded successfully!')
    print('##############################################################################################')


    # 2. load model configurations
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct',
                                              use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct',
                                                 device_map="auto", torch_dtype=torch.bfloat16, use_cache=False)

    # 3. tokenize dataset
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    # 4. create lora config
    config = LoraConfig(
        use_rslora=True,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1  # Dropout 比例

    )

    # 5. load lora adapter
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 6. configure fine-tuning parameters and train model
    args = TrainingArguments(
        output_dir="modelSave/llama3_1_instruct_rslora",
        per_device_train_batch_size=3,
        gradient_accumulation_steps=3,
        logging_steps=10,
        num_train_epochs=1,
        save_steps=100,  # 为了快速演示，这里设置10，建议你设置成100
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    print('##############################################################################################')
    print('Fine-tuning has been finished!')


if __name__ == '__main__':

    train()
