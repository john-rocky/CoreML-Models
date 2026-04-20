"""Export Llama 3.2 1B tokenizer vocab+merges into bundleable JSON / text files.

Writes:
  Llama3Vocab.json   — {"token": id} (as decoded strings, GPT-2 byte-level form)
  Llama3Merges.txt   — "<a> <b>" per line, ranked in order

Run once and drop both into sample_apps/NitroEDemo/NitroEDemo/.
"""

import argparse
import json
import os

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--out_dir",
                    default=os.path.join(os.path.dirname(__file__), "..",
                                         "sample_apps", "NitroEDemo", "NitroEDemo"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.repo)
    # `tokenizer.json` is authoritative for Llama 3 (`fast` tokenizer)
    json_path = tok.backend_tokenizer.to_str()
    cfg = json.loads(json_path)
    model = cfg["model"]
    assert model["type"] == "BPE", f"unexpected tokenizer type: {model['type']}"

    vocab = model["vocab"]  # dict str -> int
    merges = model["merges"]  # list of "a b" or ["a", "b"]
    if merges and isinstance(merges[0], list):
        merges = [" ".join(m) for m in merges]

    os.makedirs(args.out_dir, exist_ok=True)
    vocab_path = os.path.join(args.out_dir, "Llama3Vocab.json")
    merges_path = os.path.join(args.out_dir, "Llama3Merges.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("\n".join(merges))

    print(f"[wrote] {vocab_path}  ({len(vocab)} tokens)")
    print(f"[wrote] {merges_path}  ({len(merges)} merges)")


if __name__ == "__main__":
    main()
