import math
from typing_extensions import Sentinel
import numpy as np
import pandas as pd
import random
import re
import torch
import urllib.request
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast



# # # # # # # # # # # # # #

# conda env (LLM Test), python=3.12

# # # # # # # # # # # # # # 


# urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatbotData.csv")
Chatbot_Data = pd.read_csv("ChatbotData.csv")

print("전체 로우 데이터  = > ", len(Chatbot_Data))
Chatbot_Data.head()

# skt/kogpt2-base-v2의 토크나이저 사용
# 위 토크나이저에서 사용되는 특수 토큰 정의

# <usr> : 사용자 메시지
# <sys> : 시스템 메시지
# </s> : 문장 구분 토큰
# <pad> : 패딩 토큰
# <unused0> : 사용되지 않는 토큰 : 질문은 모델이 예측할 때 Loss에 포함되지않도록 MASKING 처리

# <usr>안녕?</s><sys>반가워</s><pad><pad><pad><pad>...

BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"
Q_TKN = "<usr>"
A_TKN = "<sys>"
SENT = "</s>"

MAX_LEN = 50
BATCH_SIZE = 32
NUM_WORKERS = 2

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, pad_token=PAD, unk_token="<unk>", mask_token=MASK)

class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=MAX_LEN):
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.ignore_index = -100
        #self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):

        # pandas DataFrame 행렬 데이터 불러오기
        turn = self._data.iloc[idx]

        # pandas DataFrmae idx 행의 Q,A 열 불러오기
        # 텍스트 데이터의 특수문자 제거 전처리
        q = turn['Q']
        q = re.sub(r"([?.!,])", r" ", q)
        
        a = turn['A']
        a = re.sub(r"([?.!,])", r" ", a)

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        # <usr> 안녕 </s>
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        # <sys> 반가워 </s>
        a_len = len(a_toked)

        if q_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        # labels = [self.mask,] * q_len + a_toked[1:]
        # ['<unused0>', '<unused0>', '<unused0>', '<unused0>', '반가워', '</s>']

        #labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        labels_ids = (
            [self.ignore_index] * q_len +         # 질문(Q) 전체를 -100으로 마스킹
            [self.ignore_index] * 1 +             # 답변의 시작 토큰인 <sys>도 -100으로 마스킹
            self.tokenizer.convert_tokens_to_ids(a_toked[1:]) # 실제 답변 내용
        )

        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)

        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]


        attention_mask = [1] * (q_len + a_len) + [0] * (self.max_len - q_len - a_len)
        
        # token_ids = [2, 2133, 14124, 324234, 213123, 112, 3, 3, 3, 3, 3...]
        # np.array(attention_mask) = [1 1 1 1 1 1 0 0 0 0 0 0 0...]
        # labels_ids = [9, 9, 9, 9, 213123, 112, 3, 3, 3....]

        # labels는 모델이 예측한 값과 비교하여 손실을 계산할 때 사용
        # attention_mask는 모델이 어떤 토큰을 봐야하고, 어떤 토큰을 무시해야하는 지 알려주는 역할
        # 질문이든 답변이든 모두 토큰 1 을 할당하여 모델이 답변을 생성할 때 질문의 맥락 정보를 참고하기 때문
        return (token_ids, np.array(attention_mask), labels_ids)

    # DataLoader가 DataSet 기준으로 batch 단위로 List를 모아 LongTensor로 변환
    def collate_batch(self, batch):
        data = [item[0] for item in batch]
        attention_mask = [item[1] for item in batch]
        labels = [item[2] for item in batch]

        return torch.LongTensor(data), torch.LongTensor(attention_mask), torch.LongTensor(labels)

# GPU 자원 사용량 조절을 위한 max Len = 40 (모델이 학습 시 일정한 길이의 데이터로 정의하면 빠르게 학습 가능)
train_set = ChatbotDataset(Chatbot_Data, max_len=MAX_LEN)

# GPU 자원 사용 효율을 위한 batch_size 설정, num_workers = 벙렬로 처리할 프로세스 개수, 0 = 메인 프로세스에서 모두 처리
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, collate_fn=train_set.collate_batch,)

# print("start")

# for batch_idx, sample in enumerate(train_dataloader):
#     token_ids, mask, label = sample
#     print("token_ids ===> ", token_ids)
#     print("mask ===> ", mask)
#     print("label ===> ", label)
# print("end")