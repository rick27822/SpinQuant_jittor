# coding=utf-8
# Jittor port of utils/data_utils.py
#
# Provides the WikiText-2 calibration / evaluation loaders used by the rotation
# optimization (training) and PTQ (evaluation) phases. Tokenization is done with
# HuggingFace `transformers`; we request numpy tensors so no torch dependency is
# introduced for the data itself. The returned arrays are converted to
# jt.Var inside the training / evaluation loops.
import os
import random

import numpy as np
import transformers

try:
    import datasets
except ImportError:  # datasets is only needed for WikiText-2
    datasets = None

_WT2_DIR = os.environ.get(
    "WIKITEXT2_DIR",
    "",
)


def _load_wt2():
    """Load WikiText-2 (raw): local parquet first (offline), HF hub as fallback."""
    if _WT2_DIR:
        for split in ("train", "test"):
            path = os.path.join(_WT2_DIR, f"{split}-00000-of-00001.parquet")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"WIKITEXT2_DIR is set but file is missing: {path}")
        return datasets.load_dataset(
            "parquet",
            data_files={
                "train": os.path.join(_WT2_DIR, "train-00000-of-00001.parquet"),
                "test": os.path.join(_WT2_DIR, "test-00000-of-00001.parquet"),
            },
        )
    return datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")


def get_wikitext2(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None,
                  eval_mode=False):
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    if eval_mode:
        if datasets is None:
            raise ImportError("`datasets` is required for WikiText-2 loading.")
        testdata = _load_wt2()["test"]
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="np")
        return testenc

    if datasets is None:
        raise ImportError("`datasets` is required for WikiText-2 loading.")
    traindata = _load_wt2()["train"]
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="np")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc["input_ids"].shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc["input_ids"][:, i:j].astype("int64")
        tar = inp.copy()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


class CustomJsonDataset:
    """Minimal numpy-based port of the original torch IterableDataset.

    Builds (input_ids, labels) blocks of `block_size` from a list of raw text
    dicts. Returns a plain list so it can be iterated directly (or wrapped by a
    simple loader) without depending on torch data primitives.
    """

    def __init__(self, dataset, tokenizer, block_size: int = 1024):
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized = [self.tokenize_function(d) for d in dataset]
        grouped = self.group_texts(tokenized)
        self.input_ids = grouped["input_ids"]
        self.labels = grouped["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        concatenated = {}
        for d in examples:
            for key in d.keys():
                concatenated.setdefault(key, []).extend(d[key])
        total_length = len(concatenated["input_ids"])
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i: i + self.block_size]
                for i in range(0, total_length, self.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
