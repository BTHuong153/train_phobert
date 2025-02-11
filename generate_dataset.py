from flask import Flask, request, jsonify, send_from_directory
import torch
import numpy as np
from transformers import PreTrainedTokenizerFast, AutoModelForTokenClassification
import os
from datetime import datetime, timedelta
import random
import json

app = Flask(__name__, static_folder="demo", static_url_path="")

# Đường dẫn tới model đã fine-tuned (đã lưu từ quá trình huấn luyện)
MODEL_PATH = "phobert_leave_ner_finetuned"

# Tải tokenizer và model
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# Các thành phần sinh dữ liệu (cho dataset)
# ---------------------------
intro_phrases = [
    "Tôi xin nghỉ",
    "Tôi xin phép nghỉ",
    "Tôi xin phép được nghỉ",
    "Cho tôi xin phép nghỉ",
    "Cho tôi xin phép được nghỉ",
    "Xin phép nghỉ",
    "Cho phép tôi nghỉ",
    "Cho phép tôi xin nghỉ",
    "Tôi đề nghị nghỉ",
    "Tôi cần nghỉ",
    "Tôi cần được nghỉ",
    "Tôi mong được nghỉ",
    "Tôi mong muốn được nghỉ"
]

session_phrases = ["sáng", "trưa", "chiều", "tối"]

date_templates_numeric = [
    "{day}",
    "ngày {day}",
    "vào ngày {day}",
    "trong ngày {day}",
    "ngày {day}/{month}",
    "vào ngày {day}/{month}",
    "trong ngày {day}/{month}",
    "{day}/{month}",
    "ngày {day}/{month}/{year}",
    "vào ngày {day}/{month}/{year}",
    "trong ngày {day}/{month}/{year}",
    "{day}/{month}/{year}",
    "hôm {day}",
    "vào hôm {day}",
    "trong hôm {day}",
    "hôm {day}/{month}",
    "vào hôm {day}/{month}",
    "trong hôm {day}/{month}",
    "hôm {day}/{month}/{year}",
    "vào hôm {day}/{month}/{year}",
    "trong hôm {day}/{month}/{year}"
]

date_expressions = [
    "hôm nay",
    "ngày mai",
    "ngày kia",
    "ngày mốt",
    "ngày kìa",
    "nay",
    "mai",
    "kia",
    "mốt",
    "kìa"
]

reason_templates = [
    "do {reason}",
    "vì {reason}",
    "bởi {reason}",
    "tại {reason}",
    "{reason}",
    ""
]

reason_options = [
    "việc gia đình",
    "lý do sức khỏe",
    "việc cá nhân",
    "việc quan trọng",
    "bị ốm",
    "không khoẻ",
    "bị sốt"
]

connectors = ["và", "cùng với", "với"]

# Danh sách ngữ cảnh bổ sung (context) để thêm vào đầu hoặc cuối câu
context_phrases_begin = [
    "Trong bối cảnh công ty mở rộng quy mô,",
    "Theo thông báo của ban giám đốc,",
    "Trong tình hình hiện nay,",
    "Theo quyết định của phòng nhân sự,"
]

context_phrases_end = [
    "theo hướng dẫn của ban giám đốc.",
    "vì các vấn đề nội bộ.",
    "để đảm bảo hoạt động liên tục.",
    "theo quyết định của ban lãnh đạo."
]

def add_context(sentence):
    """
    Với một xác suất, thêm ngữ cảnh vào đầu hoặc cuối câu.
    """
    r = random.random()
    if r < 0.33:
        # Thêm ngữ cảnh ở đầu câu
        context = random.choice(context_phrases_begin)
        return context + " " + sentence
    elif r < 0.66:
        # Thêm ngữ cảnh ở cuối câu
        context = random.choice(context_phrases_end)
        return sentence + " " + context
    else:
        return sentence

# ---------------------------
# Hàm xử lý ngày
# ---------------------------
def process_date(date_str):
    """
    Chuyển đổi chuỗi biểu thị ngày thành định dạng dd/mm/YYYY.
    Nếu chuỗi là biểu thức tương đối (ví dụ "hôm nay", "ngày mai", "ngày kia",...),
    tính ngày dựa trên ngày hiện tại.
    Nếu là định dạng số (ví dụ "6/2" hoặc "ngày 6/2"), chuyển sang định dạng dd/mm/YYYY
    với năm hiện tại (hoặc năm sau nếu tháng nhỏ hơn tháng hiện tại).
    """
    relative_expressions = {
        "hôm nay": 0,
        "nay": 0,
        "ngày mai": 1,
        "mai": 1,
        "ngày kia": 2,
        "kia": 2,
        "ngày môt": 2,
        "môt": 2,
        "mốt": 2,
        "ngày kìa": 3,
        "kìa": 3
    }
    date_str_lower = date_str.lower().strip()
    if date_str_lower in relative_expressions:
        delta = relative_expressions[date_str_lower]
        new_date = datetime.now() + timedelta(days=delta)
        return new_date.strftime("%d/%m/%Y")
    else:
        try:
            parts = date_str.split("/")
            if len(parts) == 2:
                day, month = parts
                day = int(day)
                month = int(month)
                year = datetime.now().year
                if month < datetime.now().month:
                    year += 1
                return f"{day:02d}/{month:02d}/{year}"
            elif len(parts) == 3:
                day, month, year = parts
                return f"{int(day):02d}/{int(month):02d}/{year}"
            else:
                return date_str
        except Exception as e:
            return date_str

# ---------------------------
# Hàm sinh dữ liệu mẫu (Dataset Generation)
# ---------------------------
def generate_single_request():
    """
    Sinh một yêu cầu nghỉ gồm:
      - Buổi nghỉ (session)
      - Ngày nghỉ (date): sử dụng định dạng numeric hoặc từ chỉ ngày.
      - Lý do nghỉ (reason), có thể trống.
    Trả về tuple: (request_text, request_meta)
      request_meta là dict với keys: "session", "date", "reason", "date_type"
    """
    session = random.choice(session_phrases)
    use_numeric_date = random.random() < 0.5
    if use_numeric_date:
        date_str = random.choice(date_templates_numeric).format(
            day=random.randint(1,28),
            month=random.randint(1,12),
            year=2025
        )
        date_type = "numeric"
    else:
        date_str = random.choice(date_expressions)
        date_type = "relative"
    
    reason_template = random.choice(reason_templates)
    if reason_template.strip():
        reason = random.choice(reason_options)
        reason_str = reason_template.format(reason=reason)
    else:
        reason_str = ""
    
    template_choice = random.choice([1, 2, 3, 4])
    if template_choice == 1:
        req_text = f"{session} {date_str}"
    elif template_choice == 2:
        req_text = f"{date_str} {session}"
    elif template_choice == 3:
        req_text = f"{session} {date_str} {reason_str}" if reason_str else f"{session} {date_str}"
    elif template_choice == 4:
        req_text = f"{date_str} {session} {reason_str}" if reason_str else f"{date_str} {session}"
    
    req_meta = {"session": session, "date": date_str, "reason": reason_str, "date_type": date_type}
    return req_text, req_meta

def generate_complex_sample():
    """
    Sinh mẫu văn bản xin nghỉ phức tạp với nhiều kiểu cấu trúc:
      - Kiểu "discrete": nhiều yêu cầu nghỉ rời rạc, ví dụ:
            "sáng hôm nay, trưa ngày mai, và chiều ngày kia do bị sốt"
      - Kiểu "continuous": biểu thức liên tục, ví dụ:
            "từ sáng mai đến hết chiều kia do việc gia đình"
    Trả về tuple: (sentence, meta)
      meta: {"requests": [req_meta1, req_meta2, ...]}
    """
    complex_type = random.choice(["discrete", "continuous"])
    
    if complex_type == "discrete":
        n = random.randint(2, 3)  # 2 hoặc 3 yêu cầu
        requests_text = []
        requests_meta = []
        for _ in range(n):
            req_text, req_meta = generate_single_request()
            requests_text.append(req_text)
            requests_meta.append(req_meta)
        if n == 2:
            sentence = " ".join([requests_text[0], "và", requests_text[1]])
        else:
            sentence = ", ".join(requests_text[:-1]) + " và " + requests_text[-1]
        meta = {"requests": requests_meta}
        return sentence, meta
    else:
        # continuous type: "từ {request_start} đến hết {request_end}"
        req_text_start, req_meta_start = generate_single_request()
        req_text_end, req_meta_end = generate_single_request()
        sentence = f"từ {req_text_start} đến hết {req_text_end}"
        meta = {"requests": [req_meta_start, req_meta_end]}
        return sentence, meta

def generate_sample_wrapper():
    """
    Với một xác suất, sử dụng mẫu phức tạp; ngược lại sử dụng mẫu đơn giản.
    Sau đó, thêm ngữ cảnh vào đầu hoặc cuối câu (với một xác suất nhất định).
    Trả về tuple: (sentence, meta)  
    meta: {"requests": [req_meta, ...]}
    """
    if random.random() < 0.4:
        sentence, meta = generate_complex_sample()
    else:
        req_text, req_meta = generate_single_request()
        sentence = req_text
        meta = {"requests": [req_meta]}
    
    # Thêm ngữ cảnh vào đầu hoặc cuối câu với xác suất 50%
    if random.random() < 0.5:
        sentence = add_context(sentence)
    return sentence, meta

def add_context(sentence):
    """
    Thêm ngữ cảnh vào đầu hoặc cuối câu.
    """
    r = random.random()
    if r < 0.33:
        context = random.choice([
            "Trong bối cảnh công ty mở rộng quy mô,",
            "Theo thông báo của ban giám đốc,",
            "Trong tình hình hiện nay,",
            "Theo quyết định của phòng nhân sự,"
        ])
        return context + " " + sentence
    elif r < 0.66:
        context = random.choice([
            "theo hướng dẫn của ban giám đốc.",
            "vì các vấn đề nội bộ.",
            "để đảm bảo hoạt động liên tục.",
            "theo quyết định của ban lãnh đạo."
        ])
        return sentence + " " + context
    else:
        return sentence

def generate_token_labels_from_meta(sentence, meta):
    """
    Dựa vào meta (với key "requests") để gán nhãn cho từng token.
    Gán nhãn:
      - Các token của mỗi request:
          + Session: toàn bộ token gán "B-SESSION"
          + Date: nếu date_type == "numeric": chỉ gán nhãn cho token chứa chữ số 
                     (token đầu tiên "B-DATE", sau đó "I-DATE");
                    nếu date_type == "relative": toàn bộ token gán, token đầu tiên "B-DATE", sau đó "I-DATE"
          + Reason: nếu có, token đầu tiên "B-REASON", sau đó "I-REASON"
      - Các từ nối (như "và", ",", "từ", "đến hết", ...) được gán "O"
    """
    tokens = sentence.split()
    constructed_tokens = []
    constructed_labels = []
    
    for idx, req in enumerate(meta["requests"]):
        if idx > 0:
            # Giả sử rằng từ nối đã có trong câu (không thêm token nối mới)
            pass
        # Session
        session_tokens = req["session"].split() if req["session"] else []
        for token in session_tokens:
            constructed_tokens.append(token)
            constructed_labels.append("B-SESSION")
        # Date
        date_tokens = req["date"].split() if req["date"] else []
        if req.get("date_type", "relative") == "numeric":
            for i, token in enumerate(date_tokens):
                if any(ch.isdigit() for ch in token):
                    if i == 0:
                        constructed_tokens.append(token)
                        constructed_labels.append("B-DATE")
                    else:
                        constructed_tokens.append(token)
                        constructed_labels.append("I-DATE")
                else:
                    constructed_tokens.append(token)
                    constructed_labels.append("O")
        else:
            for i, token in enumerate(date_tokens):
                if i == 0:
                    constructed_tokens.append(token)
                    constructed_labels.append("B-DATE")
                else:
                    constructed_tokens.append(token)
                    constructed_labels.append("I-DATE")
        # Reason
        if req["reason"]:
            reason_tokens = req["reason"].split()
            for i, token in enumerate(reason_tokens):
                if i == 0:
                    constructed_tokens.append(token)
                    constructed_labels.append("B-REASON")
                else:
                    constructed_tokens.append(token)
                    constructed_labels.append("I-REASON")
    
    original_tokens = sentence.split()
    if len(constructed_tokens) < len(original_tokens):
        constructed_labels.extend(["O"] * (len(original_tokens) - len(constructed_tokens)))
    elif len(constructed_tokens) > len(original_tokens):
        constructed_labels = constructed_labels[:len(original_tokens)]
    
    return constructed_labels

# ---------------------------
# Sinh dataset với 50,000 mẫu duy nhất có cấu trúc khác biệt và thêm ngữ cảnh
# ---------------------------
unique_samples = {}
attempts = 0
target_samples = 50000

while len(unique_samples) < target_samples:
    sentence, meta = generate_sample_wrapper()
    tokens = sentence.split()
    ner_tags = generate_token_labels_from_meta(sentence, meta)
    if len(tokens) != len(ner_tags):
        continue
    if sentence not in unique_samples:
        unique_samples[sentence] = ner_tags
    attempts += 1
    if attempts % 1000 == 0:
        print("Attempts:", attempts, "Unique samples:", len(unique_samples))

dataset_list = [{"text": text, "ner_tags": tags} for text, tags in unique_samples.items()]
print(f"Generated {len(dataset_list)} unique samples after {attempts} attempts.")

with open("data/train.json", "w", encoding="utf-8") as f:
    json.dump(dataset_list, f, ensure_ascii=False, indent=2)

val_samples = random.sample(dataset_list, 200)
with open("data/validation.json", "w", encoding="utf-8") as f:
    json.dump(val_samples, f, ensure_ascii=False, indent=2)

print("Dataset generated successfully!")