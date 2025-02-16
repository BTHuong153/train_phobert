import random
import json
import os

# Tạo thư mục data nếu chưa tồn tại
if not os.path.exists("data"):
    os.makedirs("data")

# -------------------------------
# Các thành phần cấu trúc câu
# -------------------------------

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
    "Tôi mong muốn được nghỉ",
]

# Các cụm chỉ buổi nghỉ
session_phrases = ["sáng", "trưa", "chiều", "tối"]

# Các mẫu định dạng ngày cụ thể (numeric) – dùng để tạo chữ ký cấu trúc
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
    "trong hôm {day}/{month}/{year}",
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
    "kìa",
]

# Các mẫu lý do (có thể có hoặc không)
reason_templates = [
    "do {reason}",
    "vì {reason}",
    "bởi {reason}",
    "tại {reason}",
    "{reason}",
    "",  # không có lý do
]

reason_options = [
    "việc gia đình",
    "lý do sức khỏe",
    "việc cá nhân",
    "việc quan trọng",
    "bị ốm",
    "không khoẻ",
    "bị sốt",
]

# -------------------------------
# Các hàm sinh dữ liệu
# -------------------------------


def generate_numeric_date():
    """
    Sinh một chuỗi ngày theo định dạng số dựa vào một template ngẫu nhiên.
    Trả về:
      - date_str: chuỗi ngày đã điền số
      - numeric_template: template được sử dụng (dùng cho cấu trúc)
    """
    numeric_template = random.choice(date_templates_numeric)
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = 2025  # cố định năm
    date_str = numeric_template.format(day=day, month=month, year=year)
    return date_str, numeric_template


def generate_single_request():
    """
    Sinh một yêu cầu nghỉ (leave request) gồm:
      - Buổi nghỉ (session)
      - Ngày nghỉ (date): sử dụng định dạng numeric hoặc từ chỉ ngày.
      - Lý do nghỉ (reason), có thể trống.
    Trả về tuple: (req_text, req_meta) với req_meta có thêm key "structure" dùng để xác định mẫu cấu trúc.
    """
    # Sinh buổi nghỉ
    session = random.choice(session_phrases)

    # Sinh ngày nghỉ: chọn giữa numeric hoặc expression (50% cơ hội mỗi kiểu)
    if random.random() < 0.5:
        date_str, numeric_template = generate_numeric_date()
        # Cấu trúc của date là kiểu numeric và lưu lại template
        date_structure = ("numeric", numeric_template)
    else:
        date_expr = random.choice(date_expressions)
        date_str = date_expr
        date_structure = ("expression", date_expr)

    # Sinh lý do nghỉ: chọn mẫu reason ngẫu nhiên
    reason_template = random.choice(reason_templates)
    if reason_template.strip():
        reason = random.choice(reason_options)
        reason_str = reason_template.format(reason=reason)
        reason_structure = ("reason", reason_template)
    else:
        reason_str = ""
        reason_structure = ("none",)

    # Xây dựng chuỗi yêu cầu
    if reason_str:
        req_text = f"{session} {date_str} {reason_str}"
    else:
        req_text = f"{session} {date_str}"

    # Lưu thông tin meta, bao gồm cả cấu trúc của yêu cầu
    req_meta = {
        "session": session,
        "date": date_str,
        "reason": reason_str,
        "structure": (session, date_structure, reason_structure),
    }
    return req_text, req_meta


def generate_sample():
    """
    Sinh một câu mẫu cho việc nghỉ nhiều buổi.
    Cấu trúc: {intro} {request1} [và {request2}]
    Trả về tuple: (sentence, meta)
      - meta là dict gồm: {"intro": intro, "requests": [req_meta1, req_meta2 (nếu có)]}
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
    Dựa vào meta để gán nhãn cho từng token trong câu.

    Các phần được gán nhãn như sau:
      - Phần intro: toàn bộ token được gán "O".
      - Phần yêu cầu nghỉ:
          + Buổi nghỉ (session): tất cả token được gán "B-SESSION".
          + Ngày nghỉ (date):
                * Các token thuộc danh sách từ bổ trợ (như "vào", "trong", "ngày", "hôm") được gán "O".
                * Các token còn lại (chỉ ngày cụ thể) được gán "B-DATE" (token đầu tiên) và "I-DATE" (các token sau).
          + Lý do (reason): nếu có, token đầu tiên được gán "B-REASON", các token sau "I-REASON".
      - Từ nối như "và" được gán "O".
    """
    tokens = sentence.split()
    constructed_tokens = []
    constructed_labels = []

    # Xử lý phần intro
    intro_tokens = meta["intro"].split()
    constructed_tokens.extend(intro_tokens)
    constructed_labels.extend(["O"] * len(intro_tokens))

    # Xử lý các yêu cầu nghỉ
    for idx, req in enumerate(meta["requests"]):
        # Nếu có nhiều yêu cầu, thêm từ nối "và"
        if idx > 0:
            constructed_tokens.append("và")
            constructed_labels.append("O")

        # Phần session
        session_tokens = req["session"].split() if req["session"] else []
        for token in session_tokens:
            constructed_tokens.append(token)
            constructed_labels.append("B-SESSION")

        # Phần date
        # Chúng ta bỏ gán nhãn cho các token bổ trợ như "vào", "trong", "ngày", "hôm"
        auxiliary_date_words = {"vào", "trong", "ngày", "hôm"}
        date_tokens = req["date"].split() if req["date"] else []
        first_specific_date = True  # đánh dấu token đầu tiên không phải bổ trợ
        for token in date_tokens:
            if token.lower() in auxiliary_date_words:
                constructed_tokens.append(token)
                constructed_labels.append("O")
            else:
                if first_specific_date:
                    constructed_tokens.append(token)
                    constructed_labels.append("B-DATE")
                    first_specific_date = False
                else:
                    constructed_tokens.append(token)
                    constructed_labels.append("I-DATE")

        # Phần reason (nếu có)
        if req["reason"]:
            reason_tokens = req["reason"].split()
            for i, token in enumerate(reason_tokens):
                if i == 0:
                    constructed_tokens.append(token)
                    constructed_labels.append("B-REASON")
                else:
                    constructed_tokens.append(token)
                    constructed_labels.append("I-REASON")

    # Nếu số token xây dựng không khớp với câu ban đầu, điều chỉnh.
    if len(constructed_tokens) < len(tokens):
        constructed_labels.extend(["O"] * (len(tokens) - len(constructed_tokens)))
    elif len(constructed_tokens) > len(tokens):
        constructed_labels = constructed_labels[: len(tokens)]

    return constructed_labels


# -------------------------------
# Sinh tập dataset với 10.000 mẫu có cấu trúc không trùng lặp
# -------------------------------

# Sử dụng tập để lưu chữ ký cấu trúc của từng mẫu
unique_structure_signatures = set()
unique_samples = []
validation_samples = []

while len(unique_samples) < 10000:
    sentence, meta = generate_sample()
    # Tạo chữ ký cấu trúc cho mẫu hiện tại: gồm phần intro, số yêu cầu và cấu trúc của từng request
    structure_signature = (
        meta["intro"],
        len(meta["requests"]),
        tuple(req["structure"] for req in meta["requests"]),
    )

    # Nếu cấu trúc này đã có, bỏ qua mẫu này
    if structure_signature in unique_structure_signatures:
        continue

    tokens = sentence.split()
    ner_tags = generate_token_labels_from_meta(sentence, meta)
    # Nếu số token không khớp với số nhãn, bỏ qua (trường hợp hiếm)
    if len(tokens) != len(ner_tags):
        continue

    unique_structure_signatures.add(structure_signature)
    unique_samples.append({"text": sentence, "ner_tags": ner_tags})

print(f"Generated {len(unique_samples)} unique samples with unique structures.")
# Lưu dataset huấn luyện vào file train.json
with open("data/train.json", "w", encoding="utf-8") as f:
    json.dump(unique_samples, f, ensure_ascii=False, indent=2)

# # Load train dataset để đảm bảo validation không trùng lặp hoàn toàn
# with open("data/train.json", "r", encoding="utf-8") as f:
#     train_data = json.load(f)
# # Lưu trữ tất cả chữ ký cấu trúc có trong train
# train_structure_signatures = set()
# for sample in train_data:
#     text = sample["text"]
#     train_structure_signatures.add(text)  # Sử dụng nội dung câu làm chữ ký đơn giản

# validation_samples = []
# while len(validation_samples) < 2000:
#     sentence, meta = generate_sample()
#     # Dùng nội dung câu làm chữ ký kiểm tra
#     structure_signature = sentence

#     # Đảm bảo mẫu này không trùng hoàn toàn với train
#     if structure_signature in train_structure_signatures:
#         continue

#     tokens = sentence.split()
#     ner_tags = generate_token_labels_from_meta(sentence, meta)

#     # Nếu số token không khớp với số nhãn, bỏ qua (trường hợp hiếm)
#     if len(tokens) != len(ner_tags):
#         continue

#     train_structure_signatures.add(structure_signature)  # Đánh dấu đã lấy
#     validation_samples.append({"text": sentence, "ner_tags": ner_tags})

# print(f"Generated {len(validation_samples)} unique validation samples.")

# # Lưu tập validation
# with open("data/validation.json", "w", encoding="utf-8") as f:
#     json.dump(validation_samples, f, ensure_ascii=False, indent=2)

print("Validation dataset generated successfully!")
