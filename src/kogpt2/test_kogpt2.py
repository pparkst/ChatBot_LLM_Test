import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./kogpt2_chatbot_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"사용 장치: {device}")

try:
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    # 모델 장치로 이동
    model.to(device)

    # 추론 모드 설정
    model.eval()

    print("모델 및 토크나이저 로드 완료")

except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit()


def generate_response(prompt: str, max_length = 100, num_return_sequences = 1):
    print(f"\n[입력] : {prompt}")

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output_sequences = model.generate(
            input_ids = input_ids,
            max_length = input_ids.shape[-1] + max_length, # 생성 최대 토큰 수
            do_sample = True, # 샘플링 활성화
            temperature = 0.1, # 창의성 조절 (낮을수록 보수적)
            repetition_penalty = 1.2, # 반복 패널티
            pad_token_id = tokenizer.pad_token_id,
            #eos_token_id = tokenizer.eos.token_id,
            num_return_sequences = num_return_sequences,
        )

        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens = True)

        if generated_text.startswith(prompt):
            response = generated_text[len(prompt):].strip()
        else:
            response = generated_text.strip()

        print(f"[응답] : {response}")
        return response

generate_response("반품 문의 좀 하고싶어요")
generate_response("택배 상자가 파손되었어요")
generate_response("아니 개자식아")
generate_response("장난하세요?")
generate_response("죽여버린다")
generate_response("열받네 진짜")
generate_response("배송비 얼마에요")
generate_response("배송 언제 출발해요?")
generate_response("닥쳐")