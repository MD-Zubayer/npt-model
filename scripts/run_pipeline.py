"""
End-to-end orchestration script for NPT training pipeline.
Includes a local laptop-friendly execution path.
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self, config_path: str, output_dir: str = "./outputs"):
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self.documents: List[Dict] = []

    def _setup_logging(self):
        log_file = self.output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict]:
        docs: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        docs.append(item)
                except json.JSONDecodeError:
                    docs.append({"id": f"{path.stem}-{i}", "content": line, "source": str(path)})
        return docs

    @staticmethod
    def _read_text(path: Path) -> List[Dict]:
        docs: List[Dict] = []
        chunk: List[str] = []
        chunk_size = 80  # lines per document chunk
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    docs.append(
                        {"id": f"{path.stem}-{len(docs)}", "content": " ".join(chunk), "source": str(path)}
                    )
                    chunk = []
                if len(docs) >= 5000:
                    break
        if chunk:
            docs.append({"id": f"{path.stem}-{len(docs)}", "content": " ".join(chunk), "source": str(path)})
        return docs

    @staticmethod
    def _write_jsonl(path: Path, docs: List[Dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    def _ensure_docs(self):
        if self.documents:
            return
        raw_dir = Path(self.config.get("data_collection", {}).get("raw_data_dir", "./data/raw"))
        docs: List[Dict] = []
        if raw_dir.exists():
            for p in raw_dir.rglob("*.jsonl"):
                docs.extend(self._read_jsonl(p))
            for p in raw_dir.rglob("*.txt"):
                docs.extend(self._read_text(p))
        if not docs:
            docs = [
                {"id": "seed-1", "content": "This is a sample English training paragraph for local testing only."},
                {"id": "seed-2", "content": "এটি একটি বাংলা স্যাম্পল ডকুমেন্ট, লোকাল পাইপলাইন টেস্ট করার জন্য।"},
            ]
            logger.warning("No raw data found. Using built-in sample documents.")
        self.documents = docs
        logger.info("Loaded %d raw documents", len(self.documents))

    def run_data_collection(self):
        logger.info("STAGE 1: Data Collection")
        if self.config.get("data_collection", {}).get("enabled", False):
            self._ensure_docs()
            out = Path(self.config["data_collection"].get("raw_data_dir", "./data/raw")) / "collected.jsonl"
            self._write_jsonl(out, self.documents)
            logger.info("Saved collected docs: %s", out)

    def run_data_cleaning(self):
        logger.info("STAGE 2: Data Cleaning")
        if not self.config.get("data_cleaning", {}).get("enabled", False):
            return
        from data_pipeline.cleaning.language_filter import LanguageFilter
        from data_pipeline.cleaning.normalize import TextNormalizer
        from data_pipeline.cleaning.remove_noise import NoiseRemover

        self._ensure_docs()
        normalizer = TextNormalizer()
        remover = NoiseRemover()
        lang_cfg = self.config["data_cleaning"]
        lang_filter = LanguageFilter(set(lang_cfg.get("target_languages", ["en"])))

        docs = normalizer.normalize_batch(self.documents) if lang_cfg.get("normalize", True) else self.documents
        docs = remover.remove_noise_batch(docs) if lang_cfg.get("remove_noise", True) else docs
        docs = lang_filter.filter_documents(docs) if lang_cfg.get("language_filter", True) else docs
        self.documents = docs

        out = Path(lang_cfg.get("output_dir", "./data/cleaned")) / "cleaned.jsonl"
        self._write_jsonl(out, self.documents)
        logger.info("Cleaned documents: %d", len(self.documents))

    def run_filtering(self):
        logger.info("STAGE 3: Quality Filtering")
        if not self.config.get("quality_filter", {}).get("enabled", False):
            return
        from data_pipeline.quality_filter.heuristic_score import HeuristicScorer
        from data_pipeline.quality_filter.toxicity_filter import ToxicityFilter

        self._ensure_docs()
        filt_cfg = self.config["quality_filter"]
        toxicity = ToxicityFilter(toxicity_threshold=filt_cfg.get("toxicity_threshold", 0.5))
        scorer = HeuristicScorer()
        docs = toxicity.filter_documents(self.documents)
        docs = scorer.filter_by_quality(docs, threshold=filt_cfg.get("quality_threshold", 0.5))
        self.documents = docs
        out = Path(filt_cfg.get("output_dir", "./data/filtered")) / "filtered.jsonl"
        self._write_jsonl(out, self.documents)
        logger.info("Filtered documents: %d", len(self.documents))

    def run_deduplication(self):
        logger.info("STAGE 4: Deduplication")
        if not self.config.get("deduplication", {}).get("enabled", False):
            return
        from data_pipeline.deduplication.semantic_dedup import SemanticDeduplicator

        self._ensure_docs()
        cfg = self.config["deduplication"]
        dedup = SemanticDeduplicator(similarity_threshold=cfg.get("similarity_threshold", 0.95))
        self.documents = dedup.deduplicate(self.documents)
        out = Path(cfg.get("output_dir", "./data/deduplicated")) / "deduplicated.jsonl"
        self._write_jsonl(out, self.documents)
        logger.info("Deduplicated documents: %d", len(self.documents))

    def run_tokenization(self):
        logger.info("STAGE 5: Tokenization")
        if not self.config.get("tokenization", {}).get("enabled", False):
            return
        self._ensure_docs()
        tok_cfg = self.config["tokenization"]
        vocab_size = int(tok_cfg.get("vocab_size", 50000))
        freq = {}
        for doc in self.documents:
            for token in doc.get("content", "").split():
                freq[token] = freq.get(token, 0) + 1
        vocab = ["<pad>", "<unk>"] + [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[: max(0, vocab_size - 2)]]
        out_dir = Path(tok_cfg.get("output_dir", "./model/tokenizer"))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
        logger.info("Tokenizer vocab saved: %s", out_dir / "vocab.txt")

    def run_training(self):
        logger.info("STAGE 6: Model Training")
        if not self.config.get("training", {}).get("enabled", False):
            return
        cfg = self.config["training"]
        output_dir = cfg.get("output_dir", "./outputs")
        cmd = [
            "python",
            "training/pretraining/train.py",
            "--output-dir",
            output_dir,
            "--num-epochs",
            "1",
            "--dataset-size",
            "128",
            "--seq-length",
            "64",
            "--batch-size",
            "8",
        ]
        subprocess.run(cmd, check=True)
        logger.info("Training finished. Checkpoints in: %s", output_dir)

    def run_evaluation(self):
        logger.info("STAGE 7: Evaluation")
        if self.config.get("evaluation", {}).get("enabled", False):
            eval_dir = Path(self.config["evaluation"].get("output_dir", "./evaluation/results"))
            eval_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "timestamp": datetime.now().isoformat(),
                "num_documents_after_pipeline": len(self.documents),
                "status": "ok",
            }
            (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logger.info("Evaluation summary saved: %s", eval_dir / "summary.json")

    def run_full_pipeline(self):
        self.run_data_collection()
        self.run_data_cleaning()
        self.run_filtering()
        self.run_deduplication()
        self.run_tokenization()
        self.run_training()
        self.run_evaluation()
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")


def main():
    parser = argparse.ArgumentParser(description="NPT Model Training Pipeline Orchestrator")
    parser.add_argument("--config", type=str, default="pipeline_config.json")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "collection", "cleaning", "filtering", "deduplication", "tokenization", "training", "evaluation"],
        default="all",
    )
    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(args.config, args.output_dir)
    if args.stage == "all":
        orchestrator.run_full_pipeline()
    elif args.stage == "collection":
        orchestrator.run_data_collection()
    elif args.stage == "cleaning":
        orchestrator.run_data_cleaning()
    elif args.stage == "filtering":
        orchestrator.run_filtering()
    elif args.stage == "deduplication":
        orchestrator.run_deduplication()
    elif args.stage == "tokenization":
        orchestrator.run_tokenization()
    elif args.stage == "training":
        orchestrator.run_training()
    elif args.stage == "evaluation":
        orchestrator.run_evaluation()


if __name__ == "__main__":
    main()
