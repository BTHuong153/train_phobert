import random
import json
import os

# Tạo thư mục data nếu chưa tồn tại
if not os.path.exists("data"):
    os.makedirs("data")

# Các thành phần cấu trúc câu

# Các cụm mở đầu khác nhau
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

# Các cụm chỉ buổi nghỉ
session_phrases = ["sáng", "trưa", "chiều", "tối"]

# Các mẫu định dạng ngày cụ thể (numeric)
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

# Các cụm từ chỉ ngày không cụ thể
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

# Các mẫu lý do (có thể có hoặc không)
reason_templates = [
    "do {reason}",
    "vì {reason}",
    "bởi {reason}",
    "tại {reason}",
    "{reason}",
    ""  # không có lý do
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

def generate_numeric_date():
    """Sinh một chuỗi ngày theo định dạng số với một trong các template."""
    date_template = random.choice(date_templates_numeric)
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = 2025  # cố định năm
    return date_template.format(day=day, month=month, year=year)

def generate_single_request():
    """
    Sinh một yêu cầu nghỉ (leave request) gồm:
      - Buổi nghỉ (session)
      - Ngày nghỉ (date): sử dụng định dạng numeric hoặc từ chỉ ngày.
      - Lý do nghỉ (reason), có thể trống.
    Trả về tuple: (request_text, request_meta) 
      request_meta là dict với keys: "session", "date", "reason"
    """
    session = random.choice(session_phrases)
    use_numeric_date = random.random() < 0.5  # 50% chance
    if use_numeric_date:
        date_str = generate_numeric_date()
    else:
        date_str = random.choice(date_expressions)
    
    reason_template = random.choice(reason_templates)
    if reason_template.strip():
        reason = random.choice(reason_options)
        reason_str = reason_template.format(reason=reason)
    else:
        reason_str = ""
    
    if reason_str:
        req_text = f"{session} {date_str} {reason_str}"
    else:
        req_text = f"{session} {date_str}"
    req_meta = {"session": session, "date": date_str, "reason": reason_str}
    return req_text, req_meta

def generate_sample():
    """
    Sinh một câu mẫu cho việc nghỉ nhiều buổi.
    Cấu trúc: {intro} {request1} [và {request2}]
    Trả về tuple: (sentence, meta)
      meta là dict: {"intro": intro, "requests": [req_meta1, req_meta2 (nếu có)]}
    """
    intro = random.choice(intro_phrases)
    # Quyết định số yêu cầu nghỉ: 1 hoặc 2
    num_requests = 2 if random.random() < 0.5 else 1
    
    requests_text = []
    requests_meta = []
    for _ in range(num_requests):
        req_text, req_meta = generate_single_request()
        requests_text.append(req_text)
        requests_meta.append(req_meta)
    
    if num_requests == 1:
        sentence = f"{intro} {requests_text[0]}"
    else:
        sentence = f"{intro} " + " và ".join(requests_text)
    
    sentence = " ".join(sentence.split())
    meta = {"intro": intro, "requests": requests_meta}
    return sentence, meta

def generate_token_labels_from_meta(sentence, meta):
    """
    Dựa vào meta (với key "intro" và "requests") để gán nhãn cho từng token.
    Gán nhãn:
      - Các token của phần intro: "O"
      - Các token của mỗi request:
          + Buổi nghỉ: gán "B-SESSION" cho tất cả các token của phần này.
          + Ngày nghỉ: token đầu tiên "B-DATE", các token sau "I-DATE".
          + Lý do: token đầu tiên "B-REASON", các token sau "I-REASON".
      - Các từ nối như "và" được gán "O".
    """
    tokens = sentence.split()
    constructed_tokens = []
    constructed_labels = []
    
    # Phần intro
    intro_tokens = meta["intro"].split()
    constructed_tokens.extend(intro_tokens)
    constructed_labels.extend(["O"] * len(intro_tokens))
    
    # Xử lý các request
    for idx, req in enumerate(meta["requests"]):
        # Nếu có nhiều request, thêm từ nối "và" giữa các request
        if idx > 0:
            constructed_tokens.append("và")
            constructed_labels.append("O")
        # Xử lý phần session
        session_tokens = req["session"].split() if req["session"] else []
        for token in session_tokens:
            constructed_tokens.append(token)
            constructed_labels.append("B-SESSION")
        # Xử lý phần date
        date_tokens = req["date"].split() if req["date"] else []
        for i, token in enumerate(date_tokens):
            if i == 0:
                constructed_tokens.append(token)
                constructed_labels.append("B-DATE")
            else:
                constructed_tokens.append(token)
                constructed_labels.append("I-DATE")
        # Xử lý phần reason
        if req["reason"]:
            reason_tokens = req["reason"].split()
            for i, token in enumerate(reason_tokens):
                if i == 0:
                    constructed_tokens.append(token)
                    constructed_labels.append("B-REASON")
                else:
                    constructed_tokens.append(token)
                    constructed_labels.append("I-REASON")
    
    # Nếu số token xây dựng không khớp với sentence ban đầu, điều chỉnh.
    if len(constructed_tokens) < len(tokens):
        constructed_labels.extend(["O"] * (len(tokens) - len(constructed_tokens)))
    elif len(constructed_tokens) > len(tokens):
        constructed_labels = constructed_labels[:len(tokens)]
    
    return constructed_labels

# Tạo tập dataset với 1000 mẫu khác nhau
unique_samples = {}
while len(unique_samples) < 1000:
    sentence, meta = generate_sample()
    tokens = sentence.split()
    ner_tags = generate_token_labels_from_meta(sentence, meta)
    if len(tokens) != len(ner_tags):
        continue
    if sentence not in unique_samples:
        unique_samples[sentence] = ner_tags

dataset_list = [{"text": text, "ner_tags": tags} for text, tags in unique_samples.items()]
print(f"Generated {len(dataset_list)} unique samples.")

# Lưu dataset vào file train.json
with open("data/train.json", "w", encoding="utf-8") as f:
    json.dump(dataset_list, f, ensure_ascii=False, indent=2)

# Tạo tập validation với 200 mẫu ngẫu nhiên từ dataset_list
val_samples = random.sample(dataset_list, 200)
with open("data/validation.json", "w", encoding="utf-8") as f:
    json.dump(val_samples, f, ensure_ascii=False, indent=2)

print("Dataset generated successfully!")