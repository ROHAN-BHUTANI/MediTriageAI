import json
import logging
from collections.abc import Iterator
from pathlib import Path
import pandas as pd

from meditriage.builder.adapters.base import BaseAdapter


logger = logging.getLogger("meditriage.builder.adapters.meddialog_en")


class MeddialogEnAdapter(BaseAdapter):
    """Adapter for MedDialog dataset.

    Supports three file formats:
    - Parquet (HF splits with utterances/description columns)
    - JSONL (line-delimited JSON with utterances schema)
    - JSON array (merged-MedDialog.json with instruction/input/output schema)

    The production file (wangrongsheng/MedDialog-1.1M) is a ~4GB JSON array
    containing ~2.7M records, each with keys: instruction, input, output.
    We use ijson streaming to avoid loading the entire file into memory.
    """

    @property
    def dataset_source(self) -> str:
        return "meddialog_en"

    @property
    def version(self) -> str:
        return "2.0"

    def _extract_text_from_record(self, record: dict) -> str:
        """Extract raw_text from a record supporting multiple schemas.

        Schema 1 (MedDialog-1.1M): instruction, input, output
        Schema 2 (HF parquet/JSONL): utterances, description
        """
        parts = []

        # Schema 1: instruction/input/output (merged-MedDialog.json)
        for key in ("instruction", "input", "output"):
            val = record.get(key)
            if val and isinstance(val, str) and val.strip():
                parts.append(val.strip())

        # Schema 2: utterances/description (HF parquet splits)
        if not parts:
            utterances = record.get("utterances", [])
            if isinstance(utterances, (list, tuple)):
                parts.extend(str(u) for u in utterances if u)
            elif isinstance(utterances, str) and utterances.strip():
                parts.append(utterances.strip())

            if not parts:
                desc = record.get("description")
                if desc and isinstance(desc, str) and desc.strip():
                    parts.append(desc.strip())

        return "\n".join(parts).strip()

    def ingest(
        self, dataset_path: str, chunk_size: int = 100000
    ) -> Iterator[pd.DataFrame]:
        raw_dir = Path(dataset_path)

        # Prefer large JSON files (merged-MedDialog.json) over small parquet splits
        json_files = sorted(raw_dir.rglob("*.json"), key=lambda f: f.stat().st_size, reverse=True)
        # Filter out metadata/config files
        json_files = [f for f in json_files if f.name not in (
            "SOURCE_URL.txt", "metadata.json", "dataset_info.json",
        ) and ".cache" not in str(f)]

        if json_files:
            for j_f in json_files:
                yield from self._ingest_json_file(j_f, chunk_size)
            return

        # Fallback: JSONL files
        jsonl_files = list(raw_dir.rglob("*.jsonl"))
        if jsonl_files:
            for j_f in jsonl_files:
                yield from self._ingest_jsonl_file(j_f, chunk_size)
            return

        # Fallback: Parquet files
        parquet_files = list(raw_dir.rglob("*.parquet"))
        if parquet_files:
            yield from self._ingest_parquet_files(parquet_files, chunk_size)

    def _ingest_json_file(
        self, json_path: Path, chunk_size: int
    ) -> Iterator[pd.DataFrame]:
        """Stream a JSON array file using ijson for memory efficiency."""
        try:
            import ijson
        except ImportError:
            # Fallback to full json.load if ijson is unavailable
            yield from self._ingest_json_file_fallback(json_path, chunk_size)
            return

        batch = []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                for item in ijson.items(f, "item"):
                    if not isinstance(item, dict):
                        continue
                    raw_text = self._extract_text_from_record(item)
                    if not raw_text:
                        continue

                    comb_text = raw_text.lower()
                    department = self._classify_department(comb_text)

                    batch.append({
                        "dataset_source": self.dataset_source,
                        "raw_text": raw_text,
                        "department": department,
                        "triage_level": None,
                        "language": "en",
                    })

                    if len(batch) >= chunk_size:
                        yield pd.DataFrame(batch)
                        batch = []
        except Exception as e:
            logger.error(
                "MedDialog JSON ingestion FAILED for %s: %s",
                json_path, e,
            )
            raise RuntimeError(
                f"MedDialog JSON ingestion failed for '{json_path}': {e}. "
                f"The file may be corrupted, truncated, or an unresolved LFS pointer."
            ) from e

        if batch:
            yield pd.DataFrame(batch)

    def _ingest_json_file_fallback(
        self, json_path: Path, chunk_size: int
    ) -> Iterator[pd.DataFrame]:
        """Fallback: load entire JSON file into memory (for environments without ijson)."""
        batch = []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                return

            for item in data:
                if not isinstance(item, dict):
                    continue
                raw_text = self._extract_text_from_record(item)
                if not raw_text:
                    continue

                comb_text = raw_text.lower()
                department = self._classify_department(comb_text)

                batch.append({
                    "dataset_source": self.dataset_source,
                    "raw_text": raw_text,
                    "department": department,
                    "triage_level": None,
                    "language": "en",
                })

                if len(batch) >= chunk_size:
                    yield pd.DataFrame(batch)
                    batch = []
        except Exception as e:
            logger.warning(
                "MedDialog JSON fallback ingestion failed for %s: %s",
                json_path, e,
            )

        if batch:
            yield pd.DataFrame(batch)

    def _ingest_jsonl_file(
        self, jsonl_path: Path, chunk_size: int
    ) -> Iterator[pd.DataFrame]:
        """Parse line-delimited JSON files."""
        batch = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            continue
                        raw_text = self._extract_text_from_record(record)
                        if not raw_text:
                            continue

                        comb_text = raw_text.lower()
                        department = self._classify_department(comb_text)

                        batch.append({
                            "dataset_source": self.dataset_source,
                            "raw_text": raw_text,
                            "department": department,
                            "triage_level": None,
                            "language": "en",
                        })

                        if len(batch) >= chunk_size:
                            yield pd.DataFrame(batch)
                            batch = []
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(
                "MedDialog JSONL ingestion failed for %s: %s",
                jsonl_path, e,
            )

        if batch:
            yield pd.DataFrame(batch)

    def _ingest_parquet_files(
        self, parquet_files: list, chunk_size: int
    ) -> Iterator[pd.DataFrame]:
        """Parse Parquet files (HF dataset splits)."""
        batch = []
        for pq_f in parquet_files:
            try:
                df = pd.read_parquet(pq_f)
                for _, row in df.iterrows():
                    record = row.to_dict()
                    raw_text = self._extract_text_from_record(record)
                    if not raw_text:
                        continue

                    comb_text = raw_text.lower()
                    department = self._classify_department(comb_text)

                    batch.append({
                        "dataset_source": self.dataset_source,
                        "raw_text": raw_text,
                        "department": department,
                        "triage_level": None,
                        "language": "en",
                    })

                    if len(batch) >= chunk_size:
                        yield pd.DataFrame(batch)
                        batch = []
            except Exception:
                pass

        if batch:
            yield pd.DataFrame(batch)

    def _classify_department(self, comb_text: str) -> str:
        if any(k in comb_text for k in ["pediatric", "neonatal", "infant", "child", "baby", "toddler"]):
            return "PEDS"
        elif any(k in comb_text for k in ["gynecolog", "obstetric", "pregnancy", "pregnant", "period", "menstrual", "vagina", "ovary", "uterus"]):
            return "OBGYN"
        elif any(k in comb_text for k in ["oncolog", "carcinoma", "leukemia", "lymphoma", "metastatic", "chemotherapy", "melanoma", "sarcoma", "tumor", "cancer"]):
            return "ONCOLOGY_HEME"
        elif any(k in comb_text for k in ["cardiolog", "pulmonolog", "cardiac", "heart", "hypertension", "chest pain", "asthma", "bronchitis", "pneumonia"]):
            return "CARDIO_PULM"
        elif any(k in comb_text for k in ["neurolog", "neurosurg", "stroke", "seizure", "epilepsy", "brain", "cerebral", "aneurysm", "headache"]):
            return "NEURO"
        elif any(k in comb_text for k in ["orthoped", "fracture", "sprain", "joint", "knee", "bone", "spine", "back pain", "arthritis"]):
            return "ORTHO"
        elif any(k in comb_text for k in ["gastroenterolog", "gastrointestinal", "colon", "liver", "hepatic", "pancreatic", "bowel", "gastric", "ulcer"]):
            return "GI"
        elif any(k in comb_text for k in ["urolog", "nephrolog", "kidney", "renal", "dialysis", "bladder", "prostate"]):
            return "RENAL_URO"
        elif any(k in comb_text for k in ["dermatolog", "ophthalmolog", "otolaryngolog", "eye", "ear", "throat", "skin", "rash", "acne"]):
            return "ENT_OPHTHALMO"
        elif any(k in comb_text for k in ["psychiatr", "psycholog", "anxiety", "depression", "panic", "bipolar", "mental health"]):
            return "PSYCH"
        elif any(k in comb_text for k in ["surgeon", "surgery", "operation", "post-operative"]):
            return "SURGERY"
        return "GEN_MED"
