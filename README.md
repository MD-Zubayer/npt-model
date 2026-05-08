# NPT Model: Neural Pre-Trained Language Model

একটি সম্পূর্ণ LLM ট্রেনিং পাইপলাইন OpenAI, Gemini এবং Llama এর মতো আধুনিক architecture এবং প্র্যাকটিসেস অনুসরণ করে তৈরি।

## 🎯 প্রজেক্ট ওভারভিউ

এই প্রজেক্ট একটি end-to-end Large Language Model (LLM) ডেভেলপমেন্ট ফ্রেমওয়ার্ক প্রদান করে যা ডাটা সংগ্রহ থেকে মডেল ডিপ্লয়মেন্ট পর্যন্ত সবকিছু কভার করে।

### মূল বৈশিষ্ট্য:
- ✅ **সম্পূর্ণ ডাটা পাইপলাইন**: ওয়েব স্ক্র্যাপিং, PDF এক্সট্র্যাকশন, API ইনজেশন
- ✅ **ডাটা ক্লিনিং**: নর্মালাইজেশন, নয়েজ রিমুভাল, ডিডুপ্লিকেশন
- ✅ **কোয়ালিটি ফিল্টারিং**: টক্সিসিটি চেক, হিউরিস্টিক স্কোরিং, ক্লাসিফিকেশন
- ✅ **আধুনিক Transformer আর্কিটেকচার**: RoPE (Rotary Position Embeddings), Flash Attention
- ✅ **ডিস্ট্রিবিউটেড ট্রেনিং**: DeepSpeed এবং FSDP সাপোর্ট
- ✅ **ইনফারেন্স অপটিমাইজেশন**: কোয়ান্টাইজেশন, ONNX এক্সপোর্ট
- ✅ **মনিটরিং এবং লগিং**: WandB ইন্টিগ্রেশন

## 📁 প্রজেক্ট স্ট্রাকচার

```
npt-model/
├── data/                          # ডাটা স্টোরেজ
│   ├── raw/                      # কাঁচা ডাটা
│   │   ├── web_scraped/
│   │   ├── books/
│   │   ├── docs/
│   │   └── conversations/
│   ├── cleaned/                  # পরিষ্কৃত ডাটা
│   ├── deduplicated/             # ডিডুপ্লিকেটেড ডাটা
│   ├── filtered/                 # ফিল্টারড ডাটা
│   ├── tokenized/                # টোকেনাইজড ডাটা
│   └── datasets_manifest/        # ডাটাসেট ম্যানিফেস্ট
│
├── data_pipeline/                 # ডাটা প্রসেসিং
│   ├── collectors/               # ডাটা সংগ্রহ
│   ├── cleaning/                 # ডাটা ক্লিনিং
│   ├── deduplication/            # ডিডুপ্লিকেশন
│   ├── quality_filter/           # কোয়ালিটি ফিল্টারিং
│   ├── tokenizer_builder/        # টোকেনাইজার ট্রেনিং
│   └── validation/               # ডাটা ভ্যালিডেশন
│
├── model/                         # মডেল আর্কিটেকচার
│   ├── architecture/             # মডেল কোর
│   │   ├── config.py             # কনফিগারেশন
│   │   ├── transformer.py        # Transformer
│   │   ├── attention.py          # Multi-head Attention
│   │   ├── embeddings.py         # এমবেডিং লেয়ার
│   │   └── rotary_embeddings.py  # RoPE এমবেডিং
│   ├── tokenizer/                # টোকেনাইজার
│   ├── checkpoints/              # মডেল চেকপয়েন্ট
│   ├── pretrained/               # প্রি-ট্রেনড মডেল
│   └── exports/                  # মডেল এক্সপোর্ট
│       ├── onnx/
│       └── tflite/
│
├── training/                      # ট্রেনিং মডিউল
│   ├── pretraining/              # প্রি-ট্রেনিং
│   ├── finetuning/               # ফাইন-টিউনিং
│   ├── distributed/              # ডিস্ট্রিবিউটেড ট্রেনিং কনফিগ
│   ├── monitoring/               # ট্রেনিং মনিটরিং
│   └── configs/                  # ট্রেনিং কনফিগ
│
├── evaluation/                    # মূল্যায়ন
│   ├── benchmarks/               # বেঞ্চমার্ক ডাটাসেট
│   ├── metrics/                  # মেট্রিক ক্যালকুলেশন
│   ├── eval_runner.py            # ইভ্যালুয়েশন স্ক্রিপ্ট
│   └── human_eval/               # মানবিক মূল্যায়ন
│
├── inference/                     # ইনফারেন্স
│   ├── server.py                 # API সার্ভার
│   ├── generation.py             # টেক্সট জেনারেশন
│   ├── sampling.py               # স্যাম্পলিং কৌশল
│   ├── quantization.py           # মডেল কোয়ান্টাইজেশন
│   └── optimization/             # ইনফারেন্স অপটিমাইজেশন
│
├── alignment/                     # মডেল এলাইনমেন্ট
│   ├── reward_model/             # রিওয়ার্ড মডেল
│   ├── preference_data/          # পছন্দের ডাটা
│   └── rlhf/                     # RLHF ট্রেনিং
│
├── experiments/                   # এক্সপেরিমেন্ট লগ
│   ├── logs/                     # ট্রেনিং লগ
│   ├── wandb/                    # WandB আর্টিফ্যাক্ট
│   └── ablations/                # এবলেশন স্টাডি
│
├── infra/                         # ইনফ্রাস্ট্রাকচার
│   ├── docker/                   # Docker কনফিগ
│   ├── cluster/                  # ক্লাস্টার সেটআপ
│   └── deployment/               # ডিপ্লয়মেন্ট স্ক্রিপ্ট
│
├── scripts/                       # ইউটিলিটি স্ক্রিপ্ট
│   ├── setup_env.sh              # পরিবেশ সেটআপ
│   └── run_pipeline.py           # পাইপলাইন অর্কেস্ট্রেশন
│
├── requirements.txt               # ডিপেন্ডেন্সি
├── README.md                      # এই ফাইল
└── setup.py                       # প্যাকেজ সেটআপ
```

## 🚀 দ্রুত শুরু

### ১. পরিবেশ সেটআপ

```bash
# Python ভার্চুয়াল এনভায়রনমেন্ট তৈরি করুন
python -m venv venv
source venv/bin/activate  # Linux/Mac
# অথবা
venv\Scripts\activate  # Windows

# ডিপেন্ডেন্সি ইনস্টল করুন
pip install -r requirements.txt
```

### ২. ডাটা প্রসেসিং

```python
from data_pipeline.collectors.web_crawler import WebCrawler
from data_pipeline.cleaning.normalize import TextNormalizer
from data_pipeline.quality_filter.heuristic_score import HeuristicScorer

# ডাটা সংগ্রহ
crawler = WebCrawler()
documents = crawler.crawl(["https://example.com"])

# ডাটা ক্লিনিং
normalizer = TextNormalizer()
cleaned_docs = normalizer.normalize_batch(documents)

# কোয়ালিটি ফিল্টারিং
scorer = HeuristicScorer()
quality_docs = scorer.filter_by_quality(cleaned_docs, threshold=0.5)
```

### ৩. মডেল ট্রেনিং

```python
from model.architecture.config import ModelConfig, PresetConfigs
from model.architecture.transformer import NPTModel
from training.pretraining.train import main
import argparse

# মডেল কনফিগ বেছে নিন
config = PresetConfigs.base()  # বা small(), large(), xlarge()

# ট্রেনিং শুরু করুন
python training/pretraining/train.py \\
    --model-config model_config.json \\
    --output-dir ./outputs \\
    --num-epochs 3
```

### ৪. ইনফারেন্স

```python
from inference.generation import InferenceEngine

# ইনফারেন্স ইঞ্জিন লোড করুন
engine = InferenceEngine(
    model_path="./outputs/best_model.pt",
    device='cuda'
)

# টেক্সট জেনারেট করুন
prompts = ["আমার নাম", "পৃথিবীর"]
outputs = engine.generate(prompts, max_length=128)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output}\n")
```

## 🛠️ প্রধান উপাদান

### ডাটা পাইপলাইন

#### সংগ্রাহক (Collectors)
- **WebCrawler**: ওয়েবসাইট থেকে স্ক্র্যাপ করুন
- **PDFExtractor**: PDF ফাইল প্রসেস করুন
- **APIIngestor**: API এন্ডপয়েন্ট থেকে ডাটা আহরণ করুন

#### পরিষ্কারক (Cleaning)
- **TextNormalizer**: ইউনিকোড এবং হোয়াইটস্পেস নর্মালাইজেশন
- **NoiseRemover**: HTML, URLs এবং ইমেল সরান
- **LanguageFilter**: নির্দিষ্ট ভাষায় ফিল্টার করুন

#### কোয়ালিটি ফিল্টারিং
- **ToxicityFilter**: ক্ষতিকর কন্টেন্ট সনাক্ত করুন
- **HeuristicScorer**: দৈর্ঘ্য, ভাষাতত্ত্ব এবং পাঠযোগ্যতা স্কোর করুন
- **ClassifierFilter**: বিষয় বেসড ফিল্টারিং

### মডেল আর্কিটেকচার

```
┌─────────────────────────────────────┐
│    Input Embeddings                  │
│ (Token + Position + Segment)         │
└──────────────┬──────────────────────┘
               │
        ┌──────▼─────────┐
        │ Transformer    │
        │ Encoder Layer  │
        │ ×12 layers     │
        │ ┌────────────┐ │
        │ │ Attention  │ │
        │ │(Flash Attn)│ │
        │ └────────────┘ │
        │ ┌────────────┐ │
        │ │ Feed-Forward
        │ └────────────┘ │
        └──────┬─────────┘
               │
        ┌──────▼──────────┐
        │ Output Layer     │
        │ (MLM, NSP, etc)  │
        └──────────────────┘
```

### প্রধান ফিচার:
- **Rotary Position Embeddings (RoPE)**: Llama 3 এর মতো আধুনিক
- **Flash Attention**: দ্রুত এবং মেমোরি দক্ষ মনোযোগ
- **Gradient Checkpointing**: বড় মডেলের জন্য মেমোরি সাশ্রয়
- **bfloat16**: মিশ্র নির্ভুলতা ট্রেনিং

## 📊 ট্রেনিং

### ট্রেনিং কনফিগারেশন

```python
from model.architecture.config import TrainingConfig

config = TrainingConfig(
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=1e-4,
    warmup_steps=10000,
    use_distributed=True,  # মাল্টি-GPU
    distributed_backend='deepspeed'  # বা 'fsdp'
)
```

### বিতরণকৃত ট্রেনিং

```bash
# DeepSpeed এর সাথে ট্রেন করুন
deepspeed training/pretraining/train.py \\
    --deepspeed training/distributed/deepspeed_config.json \\
    --num-epochs 3

# বা FSDP এর সাথে
torchrun --nproc_per_node=8 training/pretraining/train.py \\
    --use-distributed \\
    --distributed-backend fsdp
```

## 📈 মূল্যায়ন

```python
from evaluation.eval_runner import BenchmarkRunner, Metrics

# মেট্রিক্স গণনা করুন
metrics = Metrics()
perplexity = metrics.perplexity(losses)
accuracy = metrics.accuracy(predictions, references)
f1 = metrics.f1_score(predictions, references)
bleu = metrics.bleu_score(predictions, references)

print(f"Perplexity: {perplexity:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"BLEU Score: {bleu:.4f}")
```

## 🚀 ডিপ্লয়মেন্ট

### মডেল এক্সপোর্ট

```python
import torch
from model.architecture.transformer import NPTModel
from model.architecture.config import ModelConfig

model = NPTModel(ModelConfig())

# ONNX এ এক্সপোর্ট করুন
torch.onnx.export(
    model,
    (torch.randn(1, 128, dtype=torch.long),),
    "model.onnx",
    input_names=['input_ids'],
    output_names=['output'],
    dynamic_axes={'input_ids': {1: 'seq_length'}}
)

# TensorFlow Lite এ (TensorFlow মডেল থেকে)
# converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
# tflite_model = converter.convert()
```

### API সার্ভার

```bash
# FastAPI সার্ভার চালান
python inference/server.py --model-path ./outputs/best_model.pt
```

## 📝 কনফিগারেশন

### মডেল প্রিসেটস

```python
from model.architecture.config import PresetConfigs

# বিভিন্ন সাইজের মডেল
tiny = PresetConfigs.tiny()      # টেস্টিং / device
small = PresetConfigs.small()    # 125M প্যারামিটার
base = PresetConfigs.base()      # 365M প্যারামিটার (default)
large = PresetConfigs.large()    # 770M প্যারামিটার
xlarge = PresetConfigs.xlarge()  # 3.5B প্যারামিটার
```

## 🔧 কাস্টম কনফিগারেশন

```json
{
  "vocab_size": 50000,
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "intermediate_size": 3072,
  "hidden_dropout_prob": 0.1,
  "attention_dropout_prob": 0.1,
  "max_position_embeddings": 2048,
  "embedding_type": "rotary",
  "use_flash_attention": true,
  "learning_rate": 1e-4,
  "warmup_steps": 10000,
  "num_train_epochs": 3,
  "per_device_train_batch_size": 32
}
```

## 📦 ডিপেন্ডেন্সি

- **PyTorch 2.1+**: ডিপ লার্নিং ফ্রেমওয়ার্ক
- **Transformers 4.36+**: প্রি-ট্রেনড মডেল এবং আর্কিটেকচার
- **NumPy/SciPy**: গাণিতিক অপারেশন
- **DeepSpeed/FSDP**: বিতরণকৃত ট্রেনিং
- **WandB**: এক্সপেরিমেন্ট ট্র্যাকিং

## 📚 আরও তথ্য

### রেফারেন্স

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [DeepSpeed](https://www.deepspeed.ai/)
- [FSDP](https://pytorch.org/docs/stable/fsdp.html)

## 🤝 অবদান

অবদান স্বাগত! লাইসেন্সের অধীনে এই প্রজেক্টে অবদান রাখুন।

## 📄 লাইসেন্স

এই প্রজেক্টটি MIT লাইসেন্সের অধীনে লাইসেন্সপ্রাপ্ত।

## 📞 সাপোর্ট

প্রশ্ন বা সমস্যার জন্য, একটি ইস্যু খুলুন বা আমাদের ডিসকাশন ফোরামে যোগ দিন।

---

**হ্যাপি ট্রেনিং! 🚀**
