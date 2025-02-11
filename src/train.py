Em Vi QC, [2/11/2025 3:42 PM]
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import os

# 1. Tải dataset
data_files = {
    "train": "data/train.json",
    "validation": "data/validation.json"
}
dataset = load_dataset("json", data_files=data_files)

# 2. Định nghĩa nhãn và mapping
label_list = ["O", "B-DATE", "I-DATE", "B-SESSION", "I-SESSION", "B-REASON", "I-REASON"]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in enumerate(label_list)}
num_labels = len(label_list)

# 3. Tải fast tokenizer (ép buộc dùng PreTrainedTokenizerFast)
tokenizer = PreTrainedTokenizerFast.from_pretrained("vinai/phobert-base")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 4. Tải mô hình cho token classification
model = AutoModelForTokenClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
# Cập nhật kích thước embedding theo tokenizer mới (nếu có token mới được thêm vào)
model.resize_token_embeddings(len(tokenizer))

# 5. Cấu hình thiết bị và log thông tin
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda)  # Check CUDA version PyTorch was compiled with
print(torch.backends.cudnn.enabled)  # Should be True if CUDA is working
print(f"\n=== Using device: {device} ===\n")
print(f"Number of GPUs available: {torch.cuda.device_count()}")

if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    print(f"Cached:    {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

model.to(device)
print(f"Model device: {next(model.parameters()).device}\n")  # Verify model device

# 6. Hàm tokenize và gán nhãn
def tokenize_and_align_labels(examples):
    texts = [text.split() for text in examples["text"]]
    tokenized_inputs = tokenizer(
        texts,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )
    all_labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[word_labels[word_idx]])
            else:
                label_ids.append(label2id[word_labels[word_idx]])
            previous_word_idx = word_idx
        all_labels.append(label_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

encoded_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# 7. Cấu hình huấn luyện (tối ưu cho GPU)
training_args = TrainingArguments(
    output_dir="phobert_leave_ner_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # Effective batch size = 32 nếu GPU không đủ bộ nhớ cho batch 32
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="phobert_leave_ner_logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,      # Sử dụng mixed precision training để giảm bộ nhớ GPU và tăng tốc độ
    report_to=[]    # Tắt báo cáo qua W&B nếu không cần thiết
)

# 8. Tải metric seqeval để đánh giá NER
metric = evaluate.load("seqeval")

Em Vi QC, [2/11/2025 3:42 PM]
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 9. Tạo Trainer, đảm bảo mô hình được chuyển sang GPU
trainer = Trainer(
    model=model,  # Model đã được chuyển sang device trước đó
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)

# 10. Huấn luyện và lưu mô hình, tokenizer
if name == "main":
    trainer.train()
    print("\n=== Training completed ===")
    print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated(0)/1024**2:.2f} MB")
    model.save_pretrained("phobert_leave_ner_finetuned")
    tokenizer.save_pretrained("phobert_leave_ner_finetuned")