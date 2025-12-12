from dataset import KOBART_MODEL_NAME, kobart_tokenizer, train_set, eval_set
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import time
import datetime


# # # # # # # # # # # # # # # # #

# conda env (LLMTEST), python=3.12

# # # # # # # # # # # # # # # # #
BATCH_SIZE = 80

if __name__ == '__main__':
    start = time.time()
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("device : ", device)

        model = AutoModelForSeq2SeqLM.from_pretrained(KOBART_MODEL_NAME)
        model.to(device)

        print(f"{KOBART_MODEL_NAME} 모델 로드 성공")
    except Exception as e:
        print(f"모델 로그 실패 : {e}")        
        exit()

    training_args = TrainingArguments(
        output_dir="./kobart_results",
        num_train_epochs=7,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.03,
        learning_rate=3e-5,
        logging_dir="./kobart_logs",
        logging_steps=500,
        save_strategy="epoch",
        eval_strategy="epoch", # 검증
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=kobart_tokenizer
    )

    print("학습 실행")
    trainer.train()
    save_path = "./final_kobart_chatbot_model"
    trainer.save_model(save_path)

    end = time.time()
    sec = (end - start)
    result = str(datetime.timedelta(seconds=sec)).split(".")
    print("소요 시간 ", result[0])
    # 3:14:39
    # Loss = 0.62
    