#!/usr/bin/env python3
# translate_i18n_json_nllb_multi_merge.py
#
# i18n JSON merge-translate with NLLB:
# - Reference: fr.json
# - Targets: en.json, de.json, it.json, es.json (configurable)
#
# Behavior:
# - Preserves existing non-empty translations by default.
# - Adds missing keys from fr.json into target files.
# - Fills translations for keys that exist but are missing/empty.
# - OPTIONAL: --treat-same-as-missing => if target string == french string, it will be treated as missing and retranslated.
#
# Usage:
#   pip install transformers torch sentencepiece accelerate
#   python translate_i18n_json_nllb_multi_merge.py --in ./i18n --out ./i18n --device auto --dtype fp16 --treat-same-as-missing
#
# Notes:
# - If you want to avoid accelerate: run with --device cuda and remove device_map="auto" in model load.

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
REF_NAME = "fr.json"

SUFFIX_TO_NLLB = {
    "fr": "fra_Latn",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "es": "spa_Latn",
}

# ---- Heuristics ----
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
ONLY_PLACEHOLDERS_RE = re.compile(r"^\s*(?:\{+[^}]+\}+|\%[sd]|\:[A-Za-z_]\w*|\$\{[^}]+\})\s*$")
ICU_MARKERS_RE = re.compile(r"\b(select|plural)\b|,\s*plural,|,\s*select,", re.IGNORECASE)

PLACEHOLDER_PATTERNS = [
    re.compile(r"\{\{[^}]+\}\}"),         # Angular {{var}}
    re.compile(r"\{[^}]+\}"),             # {var}
    re.compile(r"\$\{[^}]+\}"),           # ${var}
    re.compile(r"%\d*\$?[sd]"),           # %s, %d, %1$s
    re.compile(r"\:[A-Za-z_]\w*"),        # :name
]

def is_missing_translation(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return len(v.strip()) == 0
    return False

def should_translate_value(s: str) -> bool:
    if not s or not s.strip():
        return False
    if URL_RE.search(s):
        return False
    if ONLY_PLACEHOLDERS_RE.match(s):
        return False
    if ICU_MARKERS_RE.search(s):
        return False
    return True

class PlaceholderVault:
    def __init__(self) -> None:
        self._items: Dict[str, str] = {}
        self._i = 0

    def put(self, value: str) -> str:
        key = f"⟦PH{self._i}⟧"
        self._i += 1
        self._items[key] = value
        return key

    def protect(self, s: str) -> str:
        out = s
        for pat in PLACEHOLDER_PATTERNS:
            while True:
                m = pat.search(out)
                if not m:
                    break
                out = out[:m.start()] + self.put(m.group(0)) + out[m.end():]
        return out

    def restore(self, s: str) -> str:
        out = s
        for k, v in self._items.items():
            out = out.replace(k, v)
        return out

class NllbTranslator:
    def __init__(self, model_name: str, device: str, dtype: str, batch_size: int, max_new_tokens: int, num_beams: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=(torch.float16 if dtype == "fp16" else None),
            device_map=("auto" if device == "auto" else None),
        )
        if device in ("cpu", "cuda"):
            self.model.to(torch.device(device))

        self.model.eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

    @torch.inference_mode()
    def translate_batch(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = src_lang

        forced_bos = None
        if hasattr(self.tokenizer, "get_lang_id"):
            try:
                forced_bos = self.tokenizer.get_lang_id(tgt_lang)
            except Exception:
                forced_bos = None
        if forced_bos is None:
            forced_bos = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        if forced_bos is None or forced_bos == self.tokenizer.unk_token_id:
            raise ValueError(f"Unable to resolve target language token id for '{tgt_lang}'")

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        gen = self.model.generate(
            **enc,
            forced_bos_token_id=forced_bos,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
        )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

# ---- Merge + translate ----

Job = Tuple[Tuple[Any, ...], str, PlaceholderVault]  # (path, protected_src_text, vault)

def set_at_path(root: Any, path: Tuple[Any, ...], value: Any) -> None:
    cur = root
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value

def collect_jobs_and_merge_structure(
    src: Any,
    tgt: Any,
    path: Tuple[Any, ...],
    jobs: List[Job],
    *,
    treat_same_as_missing: bool,
) -> Any:
    """
    Returns merged target structure while collecting translation jobs for leaf strings
    where target is missing/empty (and optionally where target == french string).
    Existing non-empty target strings are preserved unless treat_same_as_missing=True and target==src.
    """
    if isinstance(src, dict):
        if not isinstance(tgt, dict):
            tgt = {}
        for k, src_v in src.items():
            tgt_v = tgt.get(k)
            tgt[k] = collect_jobs_and_merge_structure(
                src_v, tgt_v, path + (k,), jobs, treat_same_as_missing=treat_same_as_missing
            )
        return tgt

    if isinstance(src, list):
        if not isinstance(tgt, list):
            tgt = []
        if len(tgt) < len(src):
            tgt.extend([None] * (len(src) - len(tgt)))
        for i, src_v in enumerate(src):
            tgt[i] = collect_jobs_and_merge_structure(
                src_v, tgt[i], path + (i,), jobs, treat_same_as_missing=treat_same_as_missing
            )
        return tgt

    if isinstance(src, str):
        # Preserve existing translation unless it's missing OR (optional) equals french
        if isinstance(tgt, str) and not is_missing_translation(tgt):
            if not (treat_same_as_missing and tgt == src):
                return tgt
            # else: fallthrough -> will retranslate

        if should_translate_value(src):
            v = PlaceholderVault()
            jobs.append((path, v.protect(src), v))
            return ""  # will be filled
        else:
            return src

    # Non-string leaf: keep existing if any, else copy source
    return tgt if tgt is not None else src

def apply_translations(
    merged_tgt: Any,
    jobs: List[Job],
    tr: NllbTranslator,
    src_lang: str,
    tgt_lang: str,
) -> None:
    if not jobs:
        return

    texts = [j[1] for j in jobs]
    vaults = [j[2] for j in jobs]
    paths = [j[0] for j in jobs]

    outs: List[str] = []
    bs = tr.batch_size
    for i in range(0, len(texts), bs):
        outs.extend(tr.translate_batch(texts[i:i+bs], src_lang=src_lang, tgt_lang=tgt_lang))

    for path, vault, out in zip(paths, vaults, outs):
        set_at_path(merged_tgt, path, vault.restore(out))

# ---- IO helpers ----

def find_reference_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.name.lower() == REF_NAME else []
    return [p for p in root.rglob(REF_NAME) if p.is_file()]

def compute_output_path(in_root: Path, out_root: Path, ref_file: Path, target_suffix: str) -> Path:
    out_name = f"{target_suffix}.json"
    if in_root.is_file():
        if out_root.suffix.lower() == ".json":
            return out_root
        return out_root / out_name
    rel_dir = ref_file.parent.relative_to(in_root)
    return out_root / rel_dir / out_name

def read_json_if_exists(p: Path) -> Optional[Any]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="strict"))
    except Exception:
        return None

def main() -> int:
    ap = argparse.ArgumentParser(description="Merge-translate i18n JSON fr.json -> {lang}.json (preserve existing translations).")
    ap.add_argument("--in", dest="in_path", required=True, help="Input directory (or single fr.json file)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output directory")
    ap.add_argument("--targets", default="en,de,it,es", help="Comma-separated targets (default: en,de,it,es)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model (default: {DEFAULT_MODEL})")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device placement")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16"], help="Model dtype (fp16 recommended on GPU)")
    ap.add_argument("--batch", type=int, default=16, help="Batch size (strings)")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens per string")
    ap.add_argument("--beams", type=int, default=1, help="Beam search (1=faster)")
    ap.add_argument("--treat-same-as-missing", action="store_true",
                    help="If a target value equals the French value, treat it as missing and retranslate.")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    args = ap.parse_args()

    in_root = Path(args.in_path).resolve()
    out_root = Path(args.out_path).resolve()

    if not in_root.exists():
        print(f"Input path not found: {in_root}", file=sys.stderr)
        return 2

    out_root.mkdir(parents=True, exist_ok=True)

    targets = [t.strip().lower() for t in args.targets.split(",") if t.strip()]
    if not targets:
        print("No targets provided.", file=sys.stderr)
        return 2

    for t in targets:
        if t not in SUFFIX_TO_NLLB:
            print(f"Missing NLLB mapping for target '{t}'", file=sys.stderr)
            return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but torch.cuda.is_available() is False.", file=sys.stderr)
        return 2

    tr = NllbTranslator(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.beams,
    )

    refs = find_reference_files(in_root)
    if not refs:
        print(f"No reference files found named '{REF_NAME}'", file=sys.stderr)
        return 1

    src_lang = SUFFIX_TO_NLLB["fr"]
    t0 = time.time()

    total_written = 0
    total_filled = 0

    for ref in refs:
        try:
            src_data = json.loads(ref.read_text(encoding="utf-8", errors="strict"))
        except json.JSONDecodeError as e:
            print(f"JSON parse error in {ref}: {e}", file=sys.stderr)
            continue

        for t in targets:
            tgt_lang = SUFFIX_TO_NLLB[t]
            out_fp = compute_output_path(in_root, out_root, ref, t)
            out_fp.parent.mkdir(parents=True, exist_ok=True)

            existing_tgt = read_json_if_exists(out_fp)

            jobs: List[Job] = []
            merged_tgt = collect_jobs_and_merge_structure(
                src_data,
                existing_tgt,
                tuple(),
                jobs,
                treat_same_as_missing=args.treat_same_as_missing,
            )

            apply_translations(merged_tgt, jobs, tr, src_lang=src_lang, tgt_lang=tgt_lang)

            total_filled += len(jobs)

            if args.dry_run:
                if args.verbose:
                    print(f"DRY: {ref} -> {out_fp} (fill {len(jobs)} values)")
                continue

            out_fp.write_text(json.dumps(merged_tgt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            total_written += 1
            if args.verbose:
                print(f"OK: {out_fp} (filled {len(jobs)} values)")

    dt = time.time() - t0
    print(f"Done. Wrote {total_written} file(s). Filled {total_filled} missing/empty/same-as-FR value(s) in {dt:.1f}s.")
    print("Note: ICU/plural/select strings are skipped by default. Adjust should_translate_value() if needed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
