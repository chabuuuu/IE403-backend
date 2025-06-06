import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

# Tắt bớt log của transformers để đỡ rối
logging.getLogger("transformers").setLevel(logging.ERROR)
load_dotenv() 

# --- PHẦN 1: KHỞI TẠO TOÀN CỤC ---
print("Bắt đầu quá trình khởi tạo cho production...")

# 1.1. Cấu hình Model và Thiết bị (giữ nguyên)
HF_TOKEN = os.environ.get("HF_TOKEN") 
MODEL_NAME_OR_PATH = "./model/Vistral-7B-Chat"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_dependencies():
    """Tải model, tokenizer và pipeline. Tác vụ này rất nặng."""
    print(f"Đang tải model từ đường dẫn cục bộ: '{MODEL_NAME_OR_PATH}'...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Sử dụng try-except để bắt lỗi nếu model không tồn tại hoặc token sai
    try:
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME_OR_PATH,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG: Không thể tải model từ đường dẫn cục bộ. Chi tiết: {e}")
        raise
    
    model.eval()

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        device_map="auto"
    )
    print("Model và pipeline đã sẵn sàng.")
    return pipe

# <<< THAY ĐỔI Ở ĐÂY: Hàm load_data cũ được thay bằng hàm mới đơn giản hơn
def load_processed_map(file_path: str) -> dict:
    """Tải trực tiếp file map đã được xử lý sẵn từ file JSON."""
    print(f"Đang tải file map đã xử lý từ: '{file_path}'...")
    if not os.path.exists(file_path):
        # Lỗi nghiêm trọng nếu file map không tồn tại, dừng chương trình
        raise FileNotFoundError(
            f"Không tìm thấy file map tại '{file_path}'. "
            "Bạn đã chạy script để generate file này chưa?"
        )
        
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset_map = json.load(f)
    
    print(f"Đã tải thành công map với {len(dataset_map)} mục.")
    return dataset_map

# 1.2. Thực hiện tải
# <<< THAY ĐỔI Ở ĐÂY: Cập nhật đường dẫn để trỏ đến file dataset_map.json
# Sử dụng đường dẫn tuyệt đối cho Colab
PROCESSED_MAP_PATH = './dataset/output_decoded_test.json'

# <<< THAY ĐỔI Ở ĐÂY: Gọi hàm mới với đường dẫn mới
DATASET_MAP = load_processed_map(PROCESSED_MAP_PATH)
INFERENCE_PIPELINE = load_model_and_dependencies()

# 1.3. Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="VlogQA API",
    description="API để hỏi đáp dựa trên ngữ cảnh từ các vlog với Vistral-7B.",
    version="1.2.0" # Tăng phiên bản để đánh dấu sự thay đổi
)

# --- PHẦN 2: ĐỊNH NGHĨA API (giữ nguyên không đổi) ---

class QueryRequest(BaseModel):
    title: str
    question: str

class AnswerResponse(BaseModel):
    answer: str

def generate_answer(context: str, question: str) -> str:
    """Hàm sinh câu trả lời, sử dụng pipeline đã được nạp sẵn."""
    PROMPT_TEMPLATE = "<s>[INST] {system_prompt} \nNgữ cảnh: {context_chunk} \nCâu hỏi: {question} [/INST]"
    SYSTEM_PROMPT = "Bạn là một trợ lý AI hữu ích. Hãy đọc kỹ ngữ cảnh được cung cấp và trả lời câu hỏi một cách ngắn gọn, chính xác, và chỉ dựa vào thông tin trong ngữ cảnh đó."

    prompt = PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context_chunk=context,
        question=question
    )
    
    result = INFERENCE_PIPELINE(prompt)
    answer = result[0]['generated_text'][len(prompt):].strip()
    return answer.splitlines()[0].strip()

@app.post("/query", response_model=AnswerResponse)
async def handle_query(request: QueryRequest):
    """
    Nhận 'title' và 'question', trả về câu trả lời từ model.
    """
    print(f"Nhận được yêu cầu cho title: '{request.title}'")
    
    article = DATASET_MAP.get(request.title)
    if not article:
        raise HTTPException(
            status_code=404, 
            detail=f"Title '{request.title}' not found."
        )
    
    context = article['paragraphs'][0]['context']
    answer = generate_answer(context, request.question)
    return AnswerResponse(answer=answer)

@app.get("/")
def read_root():
    return {"message": "Welcome to the VlogQA API (v1.2.0). Please use /docs to test."}