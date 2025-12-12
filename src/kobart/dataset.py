import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# # # # # # # # # # # # # # # # #

# conda env (LLMTEST), python=3.12

# # # # # # # # # # # # # # # # #

LOOPIDX = 0

def load_and_preprocess_csv_data(file_path):
    """
    CSV 파일 로드, 발화(c), 응답(s)를 Q&A 쌍으로 묶어 DataFrame 생성
    """
    global LOOPIDX
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error : 파일 {file_path} 를 찾을 수 없습니다.")
        return pd.DataFrame()

    processed_data = []
    
    for i in range(len(data) - 1):
        LOOPIDX +=1
        current_row = data.iloc[i]
        next_row = data.iloc[i+1]

        current_speaker = current_row['발화자'] if '발화자' in current_row else None
        next_speaker = next_row['발화자'] if '발화자' in next_row else None

        if current_speaker == 'c' and next_speaker =='s':
            processed_data.append({
                'Q': current_row['발화문'],
                'A': next_row['발화문'],
            })

    return pd.DataFrame(processed_data)


Chatbot_Traindata = load_and_preprocess_csv_data("../dataset_Training/라벨링데이터_train/의류_train.csv")
Chatbot_Evaldata = load_and_preprocess_csv_data("../dataset_Validation/라벨링데이터_validation/의류_validation.csv")


if Chatbot_Traindata.empty:
    print("훈련 데이터 로드 실패")
    exit()
else:
    print(f"훈련 전체 로우 데이터 : {len(Chatbot_Traindata)}")

if Chatbot_Evaldata.empty:
    print("검증 데이터 로드 실패")
    exit()
else:
    print(f"검증 전체 로우 데이터 : {len(Chatbot_Evaldata)}")

BOS = "<s>"
EOS = "</s>"
PAD = "<pad>"

MAX_LEN = 50
NUM_WORKERS = 0

try:
    KOBART_MODEL_NAME = "gogamza/kobart-base-v2"
    kobart_tokenizer = AutoTokenizer.from_pretrained(
        KOBART_MODEL_NAME,
        bos_token=BOS,
        eos_token=EOS,
        pad_token=PAD,
    )
    print(f"KoBART 토크나이저 로드 완료")
except Exception as e:
    print(f"KoBART 토크나이저 로드 실패 : {e}")
    exit()


class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=MAX_LEN):
        self._data = chats
        self.max_len = max_len
        self.eos = EOS
        self.tokenizer = kobart_tokenizer

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]

        q = turn['Q']
        a = turn['A']

        q = re.sub(r"([?.!,])", r" ", q)
        a = re.sub(r"([?.!,])", r" ", a)

        q_toked_str = f"{q}{self.eos}"
        a_toked_str = a + self.eos

        q_encoded = self.tokenizer.encode_plus(
            q_toked_str,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        a_encoded = self.tokenizer.encode_plus(
            a_toked_str,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # q_toked = self.tokenizer.tokenize(q_toked_str)
        # q_len = len(q_toked)

        # a_toked = self.tokenizer.tokenizer(a_toked_str)
        # a_len = len(a_toked)

        # if q_len > self.max_len:
        #     q_toked = q_toked[:self.max_len]
        #     q_len = len(q_toked)
        
        # if a_len > self.max_len:
        #     a_toked = a_toked[:self.max_len]
        #     a_len = len(a_toked)

        # token_ids_q = self.tokenizer.convert_tokens_to_ids(q_toked)
        # token_ids_a = self.tokenizer.convert.tokens_to_ids(a_toked)
        # labels_ids = token_ids_a

        # token_ids_q += [self.tokenizer.pad_token_id] * (self.max_len - q_len)
        # labels_ids += [self.tokenizer.pad_token_id] * (self.max_len - a_len)

        # attention_mask_q = [1] * q_len + [0] * (self.max_len - q_len)

        # return (token_ids_q, np.array(attention_mask_q), labels_ids)

        return {
            'input_ids': q_encoded['input_ids'].squeeze(),
            'attention_mask': q_encoded['attention_mask'].squeeze(),
            'labels':a_encoded['input_ids'].squeeze()
        }

    def collate_batch(self, batch):
        data_q = [item[0] for item in batch]
        attention_mask_q = [item[1] for item in batch]
        labels = [item[2] for item in batch]

        return torch.LongTensor(data_q), torch.LongTensor(attention_mask_q), torch.LongTensor(labels)

train_set = ChatbotDataset(Chatbot_Traindata, max_len=MAX_LEN)
eval_set = ChatbotDataset(Chatbot_Evaldata, max_len=MAX_LEN)
# train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, collate_fn=train_set.collate_batch)