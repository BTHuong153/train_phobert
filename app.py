from flask import Flask, request, jsonify, send_from_directory
import torch
import numpy as np
from transformers import PreTrainedTokenizerFast, AutoModelForTokenClassification
import os
from datetime import datetime, timedelta

app = Flask(__name__, static_folder="demo", static_url_path="")

# Đường dẫn tới model đã fine-tuned (đã lưu từ quá trình huấn luyện)
MODEL_PATH = "phobert_leave_ner_finetuned"

# Tải tokenizer và model
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def process_date(date_str):
    """
    Chuyển đổi chuỗi biểu thị ngày thành định dạng YYYY-MM-DD.
    Nếu chuỗi là một biểu thức tương đối như "hôm nay", "ngày mai", "ngày kia",...
    thì tính ngày dựa trên ngày hiện tại.
    Nếu là định dạng số (ví dụ "6/2"), chuyển sang định dạng với năm hiện tại (hoặc năm tiếp theo nếu cần).
    """
    relative_expressions = {
        "hôm nay": 0,
        "nay": 0,
        "ngày mai": 1,
        "mai": 1,
        "ngày kia": 2,
        "kia": 2,
        "ngày mốt": 2,
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
        # Giả sử định dạng là "day/month"
        try:
            parts = date_str.split("/")
            if len(parts) == 2:
                day, month = parts
                day = int(day)
                month = int(month)
                year = datetime.now().year
                # Nếu tháng nhỏ hơn tháng hiện tại, có thể là năm sau
                if month < datetime.now().month:
                    year += 1
                return f"-{day:02d}/{month:02d}/{year}"
            else:
                return date_str
        except Exception as e:
            return date_str

def extract_leave_info(text):
    # Chuyển văn bản thành danh sách các từ dựa trên khoảng trắng
    words = text.split()
    
    # Tokenize với is_split_into_words=True để đảm bảo mapping giữa các token và từ gốc
    tokenized_inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
    
    # Lấy mapping giữa token và từ gốc
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
    
    # Lấy nhãn từ model.config.id2label; nếu key là str, chuyển đổi về int key
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
    
    # Nhóm các token theo nhãn
    current_date_tokens = []
    current_session_tokens = []
    current_reason_tokens = []
    for word, tag in word_tag_pairs:
        if tag in ["B-DATE", "I-DATE"]:
            current_date_tokens.append(word)
        elif tag in ["B-SESSION", "I-SESSION"]:
            current_session_tokens.append(word.lower())
        elif tag in ["B-REASON", "I-REASON"]:
            current_reason_tokens.append(word)
    
    # Xử lý ngày: ghép các token thành chuỗi, chuyển đổi sang định dạng YYYY-MM-DD
    date_extracted = " ".join(current_date_tokens).strip() if current_date_tokens else ""
    processed_date = process_date(date_extracted) if date_extracted else ""
    
    # Ghép các token cho session và reason
    session_extracted = " ".join(current_session_tokens).strip() if current_session_tokens else ""
    reason_extracted = " ".join(current_reason_tokens).strip() if current_reason_tokens else ""
    
    # Trả về kết quả với các khóa tiếng Việt
    result = {
        "employee_id": "Test",  # Trong thực tế, lấy từ thông tin người dùng
        "session": session_extracted if session_extracted else "Không xác định",
        "date": processed_date if processed_date else "Không xác định",
        "reason": reason_extracted if reason_extracted else "Không có"
    }
    return result

# Route để phục vụ giao diện HTML demo
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# Endpoint API để trích xuất thông tin xin nghỉ
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