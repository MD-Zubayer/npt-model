import os
from datasets import load_dataset


RAW_DIR = "data/raw"
OUT_PATH = os.path.join(RAW_DIR, "train_data.txt")

os.makedirs(RAW_DIR, exist_ok=True)


def append_lines(path: str, lines):
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            text = (line or "").strip()
            if text:
                f.write(text + "\n")


def download_bengali_data():
    # Fallback list: first working source will be used.
    candidates = [
        ("wikimedia/wikipedia", "20231101.bn", "train", "text"),
        ("oscar-corpus/oscar", "unshuffled_deduplicated_bn", "train", "text"),
    ]

    for name, config, split, text_key in candidates:
        try:
            print(f"Downloading Bangla data: {name} ({config}) ...")
            ds = load_dataset(name, config, split=split)
            append_lines(OUT_PATH, (item.get(text_key, "") for item in ds))
            print(f"Bangla data added from {name}.")
            return
        except Exception as e:
            print(f"Bangla source failed ({name}/{config}): {e}")

    raise RuntimeError("No Bangla dataset source could be downloaded.")


def download_english_data():
    print("Downloading English data: wikitext-2-raw-v1 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    append_lines(OUT_PATH, (item.get("text", "") for item in ds))
    print("English data added.")


if __name__ == "__main__":
    open(OUT_PATH, "w", encoding="utf-8").close()
    try:
        download_bengali_data()
    except Exception as e:
        print(f"Bangla download skipped: {e}")
    try:
        download_english_data()
    except Exception as e:
        print(f"English download skipped: {e}")
    print(f"\nDone. Dataset ready: {OUT_PATH}")
