from dataSet import koGPT2_TOKENIZER, train_dataloader
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from tqdm.auto import tqdm # 학습 진행 상황을 보여줌 (선택)
import time
import datetime



# # # # # # # # # # # # # #

# conda env (LLM Test), python=3.12

# # # # # # # # # # # # # # 
if __name__ == '__main__':
    start = time.time()


    # --- 1. 장치 설정 및 모델 로드 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device ==> ", device)

    # KoGPT2 모델
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    model.to(device)

    # <usr>, <sys> 커스텀 토큰을 추가했으므로 모델의 임베딩 레이어 크기를 토크나이저의 최종 크기에 맞게 조정
    model.resize_token_embeddings(koGPT2_TOKENIZER.vocab_size)

    EPOCHS = 80
    LEARNING_RATE = 5e-5

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


    for epoch in range(EPOCHS):
        total_loss = 0

        print(f"======== Epoch {epoch+1} =========")
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, (token_ids, attention_mask, labels) in enumerate(loop):

            # 데이터 바인딩
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 모델 포워드 
            outputs = model(
                input_ids = token_ids,
                attention_mask = attention_mask,
                labels = labels
            )

            # 손실 계산
            loss = outputs.loss
            total_loss += loss.item()

            # 역전파
            loss.backward()

            # 가중치 업데이트
            optimizer.step()

            # 진행 상황 업데이트
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        print(f"\n[Epoch {epoch+1}/{EPOCHS}] 완료. 평균 Loss: {avg_loss:.4f}")

    end = time.time()
    sec = (end - start)
    result = str(datetime.timedelta(seconds=sec)).split(".")
    print("소요 시간 ", result[0])
    print("학습 완료")

    model.save_pretrained("./kogpt2_chatbot_model")
    koGPT2_TOKENIZER.save_pretrained("./kogpt2_chatbot_model")