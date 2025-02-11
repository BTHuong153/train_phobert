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
total_params = sum(p.numel() for p in model.parameters())
print("Tổng số tham số:", total_params)
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
# API để trích xuất thông tin xin nghỉ
# ---------------------------
def extract_leave_info(text):
    """
    Xử lý văn bản xin nghỉ phép để trích xuất thông tin:
      - "Nhân viên xin nghỉ": giá trị cố định "Test"
      - "Thời gian nghỉ": mảng các chuỗi, mỗi chuỗi kết hợp thông tin buổi và ngày.
      - "Lý do": mảng các chuỗi, mỗi chuỗi là lý do tương ứng với từng yêu cầu (nếu có, nếu không thì "Không có").
      
    Quá trình:
      1. Tokenize văn bản và dự đoán nhãn.
      2. Tách các yêu cầu nghỉ dựa trên từ nối "và" (hoặc dấu phẩy).
      3. Với mỗi yêu cầu, trích xuất các token có nhãn SESSION, DATE và REASON.
         - Nếu có cả SESSION và DATE, kết hợp thành "SESSION processed_date".
         - Nếu thiếu SESSION hoặc DATE, sử dụng phần có sẵn; nếu cả hai đều thiếu, gán "Không xác định".
         - Tương tự, nếu không có REASON, gán "Không có".
    """
    # Tokenize văn bản và dự đoán nhãn
    words = text.split()
    tokenized_inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
    word_ids = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).word_ids(batch_index=0)
    
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    predictions = torch.argmax(outputs.logits, dim=2).cpu().numpy()[0]
    
    # Xử lý id2label
    id2label = model.config.id2label
    if isinstance(next(iter(id2label.keys())), str):
        id2label = {int(k): v for k, v in id2label.items()}
    
    predicted_labels = []
    prev_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:
            predicted_labels.append(id2label.get(predictions[idx], "O"))
            prev_word_idx = word_idx
    word_tag_pairs = list(zip(words, predicted_labels))
    
    # Tách các yêu cầu nghỉ dựa trên từ nối "và" hoặc dấu phẩy
    segments = []
    current_segment = []
    for word, label in word_tag_pairs:
        if word.lower() in ["và", ","] and label == "O":
            if current_segment:
                segments.append(current_segment)
                current_segment = []
        else:
            current_segment.append((word, label))
    if current_segment:
        segments.append(current_segment)
    
    leave_times = []
    leave_reasons = []
    for segment in segments:
        session_tokens = []
        date_tokens = []
        reason_tokens = []
        for word, label in segment:
            if label in ["B-SESSION", "I-SESSION"]:
                session_tokens.append(word)
            elif label in ["B-DATE", "I-DATE"]:
                date_tokens.append(word)
            elif label in ["B-REASON", "I-REASON"]:
                reason_tokens.append(word)
        
        # Xử lý "Thời gian nghỉ"
        if session_tokens or date_tokens:
            session_str = " ".join(session_tokens) if session_tokens else ""
            date_str = " ".join(date_tokens) if date_tokens else ""
            processed_date = process_date(date_str) if date_str else ""
            if session_str and processed_date:
                leave_times.append(f"{session_str} {processed_date}")
            elif session_str:
                leave_times.append(session_str)
            elif processed_date:
                leave_times.append(processed_date)
            else:
                leave_times.append("Không xác định")
        else:
            leave_times.append("Không xác định")
        
        # Xử lý "Lý do"
        if reason_tokens:
            reason_str = " ".join(reason_tokens)
            leave_reasons.append(reason_str)
    
    result = {
        "employee": "Test",
        "leave_times": leave_times,
        "leave_reasons": leave_reasons
    }
    return result

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/extract_leave", methods=["POST"])
def extract_leave():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = extract_leave_info(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)