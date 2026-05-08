# Quick Start Guide

## 5 মিনিটে শুরু করুন

### ধাপ ১: পরিবেশ সেটআপ

```bash
# প্রজেক্ট ডিরেক্টরিতে যান
cd npt-model

# সেটআপ স্ক্রিপ্ট চালান
bash scripts/setup_env.sh

# ভার্চুয়াল এনভায়রনমেন্ট অ্যাক্টিভেট করুন
source venv/bin/activate  # Linux/Mac
# অথবা
venv\Scripts\activate  # Windows
```

### ধাপ ২: সাধারণ ট্রেনিং

```bash
# কনফিগ ফাইল কাস্টমাইজ করুন (ঐচ্ছিক)
# nano pipeline_config.json

# পাইপলাইন চালান
python scripts/run_pipeline.py --stage training

# অথবা সরাসরি ট্রেনিং স্ক্রিপ্ট চালান
python training/pretraining/train.py \\
    --output-dir ./outputs \\
    --num-epochs 3
```

### ধাপ ৩: ইনফারেন্স

```python
# inference_test.py
from inference.generation import InferenceEngine

engine = InferenceEngine("./outputs/best_model.pt")
results = engine.generate(["আমার নাম"])
print(results)
```

## ডেটা প্রসেসিং পাইপলাইন

### ডেটা সংগ্রহ

```python
# ওয়েব থেকে সংগ্রহ করুন
from data_pipeline.collectors.web_crawler import WebCrawler

crawler = WebCrawler(max_workers=5)
documents = asyncio.run(crawler.crawl(["https://example.com"]))

# PDF থেকে এক্সট্র্যাক্ট করুন
from data_pipeline.collectors.pdf_extractor import PDFExtractor

extractor = PDFExtractor()
documents = extractor.extract_from_directory("./pdfs/")

# API থেকে ডেটা নিন
from data_pipeline.collectors.api_ingestor import APIIngestor

ingestor = APIIngestor()
endpoints = [{'url': 'https://api.example.com/data'}]
documents = asyncio.run(ingestor.fetch_multiple_endpoints(endpoints))
```

### ডেটা ক্লিনিং

```python
from data_pipeline.cleaning.normalize import TextNormalizer
from data_pipeline.cleaning.remove_noise import NoiseRemover
from data_pipeline.cleaning.language_filter import LanguageFilter

# নর্মালাইজ করুন
normalizer = TextNormalizer()
cleaned = normalizer.normalize_batch(documents)

# নয়েজ অপসারণ করুন
remover = NoiseRemover()
denoised = remover.remove_noise_batch(cleaned)

# ভাষা ফিল্টার করুন (শুধুমাত্র ইংরেজি রাখুন)
lang_filter = LanguageFilter({'en'})
filtered = lang_filter.filter_documents(denoised)
```

### কোয়ালিটি ফিল্টারিং

```python
from data_pipeline.quality_filter.toxicity_filter import ToxicityFilter
from data_pipeline.quality_filter.heuristic_score import HeuristicScorer

# টক্সিসিটি চেক করুন
toxicity_filter = ToxicityFilter()
safe_docs = toxicity_filter.filter_documents(filtered)

# কোয়ালিটি স্কোর করুন
scorer = HeuristicScorer()
quality_docs = scorer.filter_by_quality(safe_docs, threshold=0.5)
```

### ডিডুপ্লিকেশন

```python
from data_pipeline.deduplication.semantic_dedup import SemanticDeduplicator

deduplicator = SemanticDeduplicator(similarity_threshold=0.95)
unique_docs = deduplicator.deduplicate(quality_docs)
```

## মডেল কনফিগারেশন

### প্রি-সেট মডেল

```python
from model.architecture.config import PresetConfigs

# বিভিন্ন সাইজের মডেল
tiny = PresetConfigs.tiny()      # টেস্টিং
small = PresetConfigs.small()    # 125M প্যারামিটার
base = PresetConfigs.base()      # 365M প্যারামিটার (ডিফল্ট)
large = PresetConfigs.large()    # 770M প্যারামিটার
xlarge = PresetConfigs.xlarge()  # 3.5B প্যারামিটার
```

### কাস্টম কনফিগারেশন

```python
from model.architecture.config import ModelConfig

config = ModelConfig(
    vocab_size=50000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    use_flash_attention=True,
    embedding_type="rotary"
)

# সেভ করুন
config.save("my_model_config.json")

# লোড করুন
loaded_config = ModelConfig.load("my_model_config.json")
```

## ট্রেনিং অপশন

### একক GPU ট্রেনিং

```bash
python training/pretraining/train.py \\
    --output-dir ./outputs \\
    --per-device-train-batch-size 32 \\
    --num-epochs 3
```

### মাল্টি-GPU ট্রেনিং (DeepSpeed)

```bash
deepspeed training/pretraining/train.py \\
    --deepspeed training/distributed/deepspeed_config.json \\
    --num-epochs 3
```

### মাল্টি-GPU ট্রেনিং (FSDP)

```bash
torchrun --nproc_per_node=8 training/pretraining/train.py \\
    --use-distributed \\
    --distributed-backend fsdp \\
    --num-epochs 3
```

## মদেল ম্যানেজমেন্ট

### মডেল এক্সপোর্ট

```python
from inference.quantization import ModelQuantizer
from model.architecture.transformer import NPTModel
from model.architecture.config import ModelConfig

model = NPTModel(ModelConfig())
model.load_state_dict(torch.load("./outputs/best_model.pt"))

# ONNX এ এক্সপোর্ট করুন
ModelQuantizer.export_to_onnx(model, "model.onnx", input_shape=(1, 128))

# INT8 কোয়ান্টাইজ করুন
quantized_model = ModelQuantizer.quantize_int8(model)

# মডেল সাইজ দেখুন
size = ModelQuantizer.get_model_size(model)
print(f"Model size: {size:.2f} MB")
```

## ইনফারেন্স সার্ভার চালান

```bash
# FastAPI সার্ভার শুরু করুন
python inference/server.py

# ডিফল্ট: http://localhost:8000

# API ডকুমেন্টেশন: http://localhost:8000/docs
```

### API উদাহরণ

```bash
# টেক্সট জেনারেশন
curl -X POST "http://localhost:8000/generate" \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "আমার নাম",
    "max_length": 128,
    "temperature": 0.7
  }'

# এমবেডিং
curl -X POST "http://localhost:8000/embed" \\
  -H "Content-Type: application/json" \\
  -d '{
    "texts": ["আমার নাম", "পৃথিবীর"]
  }'

# হেলথ চেক
curl "http://localhost:8000/health"
```

## উন্নত বিকল্প

### গ্র্যাডিয়েন্ট চেকপয়েন্টিং

```python
# বড় মডেলের জন্য মেমোরি সাশ্রয়
config.gradient_checkpointing = True
```

### bfloat16 ট্রেনিং

```python
# দ্রুত এবং দক্ষ
config.use_bfloat16 = True
```

### লার্নিং রেট শিডিউলার

```python
from training.pretraining.scheduler import WarmupCosineScheduler

scheduler = WarmupCosineScheduler(
    optimizer,
    num_warmup_steps=10000,
    num_training_steps=100000
)
```

## টিপস এবং ট্রিকস

1. **মেমোরি সমস্যা?** 
   - `gradient_accumulation_steps` বাড়ান
   - `per_device_train_batch_size` কমান
   - `gradient_checkpointing` সক্ষম করুন

2. **ট্রেনিং ধীর?**
   - `use_flash_attention` চালু করুন
   - মাল্টি-GPU ট্রেনিং ব্যবহার করুন
   - সংক্ষিপ্ত সিকোয়েন্স দিয়ে শুরু করুন

3. **NaN সমস্যা?**
   - লার্নিং রেট কমিয়ে দেখুন
   - গ্র্যাডিয়েন্ট ক্লিপিং সক্ষম করুন
   - ডেটা নির্মাণ পরীক্ষা করুন

## সাহায্য এবং সম্পদ

- 📖 [README.md](README.md) - সম্পূর্ণ ডকুমেন্টেশন
- 🐛 Issues - সমস্যা রিপোর্ট করুন
- 💬 Discussions - প্রশ্ন জিজ্ঞাসা করুন

---

**হ্যাপি ট্রেনিং! 🚀**
