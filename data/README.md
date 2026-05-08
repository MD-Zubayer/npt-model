# Data Card (NPT Model)

## Source Overview
- `data/raw/web_scraped/`: web crawl content
- `data/raw/books/`: বই/লং-ফর্ম টেক্সট
- `data/raw/docs/`: টেকনিক্যাল ডকুমেন্টেশন
- `data/raw/conversations/`: ডায়ালগ/কনভারসেশন ডেটা

## Processing Stages
- `data/raw/` -> সংগ্রহ করা কাঁচা ডেটা
- `data/cleaned/` -> normalize + noise removal + language filtering
- `data/filtered/` -> toxicity + heuristic quality filtering
- `data/deduplicated/` -> semantic deduplication
- `data/tokenized/` -> tokenizer-ready/tokenized outputs

## Language Ratio (Target)
- বাংলা: ~80%
- ইংরেজি: ~20%

## Notes
- এখানে কোনো sensitive/personal data intentionally রাখা হবে না।
- final training এ যাওয়ার আগে sample-based manual audit করা উচিত।
