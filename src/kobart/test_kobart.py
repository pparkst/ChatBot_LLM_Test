import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

def generate_answer(input_text, trained_model, current_tokenizer, max_len=MAX_LEN, num_beams=5):
    device = trained_model.device

    input_ids = current_tokenizer.encode(
        input_text + EOS,
        return_tensors='pt',
        max_length=max_len,
        truncation=True
    ).to(device)

    output_ids = trained_model.generate(
        input_ids,
        max_length=max_len,
        num_beams=num_beams,
        no_repeat_ngram_size=2,
        early_stopping=True,
        eos_token_id=current_tokenizer.eos_token_id
    )

    answer = current_tokenizer.decode(output_ids.squeeze().tolist(), skip_special_tokens=True)
    return answer.strip()


final_model = AutoModelForSeq2SeqLM.from_pretrained("./final_kobart_chatbot_model")


dialogue_history = []

def get_response_with_history(new_question):
    history_string = " ".join([f"{q} {a}" for q, a in dialogue_history])
    
    full_input_text = f"{history_string}\n{new_question}"

    new_answer = generate_answer(full_input_text, final_model, kobart_tokenizer)

    print("모델 입력 전체 텍스트 " + (full_input_text.replace('\n', ' [EOL] ')))

    dialogue_history.append((new_question, new_answer))

    return new_answer

print(get_response_with_history("상의"))
print(get_response_with_history("종류가 뭐있어요"))
print(get_response_with_history("제가 뭘 물어봤죠"))

