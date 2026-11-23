"""Reviewer 1, comment 4: frozen transformer article embeddings for the encoder sensitivity.

Run with a PyTorch environment, for example
    PYTHONPATH=.embed_libs <torch-python> scripts/34_extract_embeddings.py
Encoders are frozen (no fine-tuning, no labels), and were pretrained before the
evaluation periods to avoid look-ahead: WangchanBERTa (Thai; pretraining corpora
collected before 2021) for the Thai pilot (test 2024-2026) and FinBERT (ProsusAI;
BERT-base further trained on Reuters TRC2 2008-2010 and Financial PhraseBank) for
the U.S. replication (test 2018-2020). Each article (headline + description, the
model_text of the label builder) is embedded by attention-masked mean pooling of the
last hidden layer; bag embeddings are formed later, inside 35_encoder_sensitivity.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
JOBS = {
    'thai': {'labels': ROOT / 'provenance/historical/p1_pilot_labels', 'model': 'airesearch/wangchanberta-base-att-spm-uncased',
             'max_length': 256, 'thai_preprocess': True},
    'us': {'labels': ROOT / 'results/fnspid_prepared/p1_labels', 'model': 'ProsusAI/finbert',
           'max_length': 256, 'thai_preprocess': False},
}
OUT = ROOT / 'results' / 'transformer_embeddings'


def preprocess(text, thai):
    text = ' '.join(str(text).split())
    return text.lower().replace(' ', '<_>') if thai else text


@torch.no_grad()
def embed(texts, model_name, max_length, device, batch=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    vectors = []
    for lo in range(0, len(texts), batch):
        enc = tokenizer(texts[lo:lo + batch], padding=True, truncation=True, max_length=max_length,
                        return_tensors='pt').to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
            hidden = model(**enc).last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        vectors.append(pooled.float().cpu().numpy())
    return np.vstack(vectors).astype(np.float32)


def main():
    if (OUT / 'COMPLETE.json').exists():
        raise RuntimeError(f'Refusing to overwrite completed run: {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    record = {'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
              'torch': torch.__version__, 'transformers': transformers.__version__, 'jobs': {}}
    for market, job in JOBS.items():
        began = time.time()
        articles = pd.read_csv(job['labels'] / 'article_to_stock_day_pilot.csv', low_memory=False)
        articles['model_text'] = articles['model_text'].fillna('').astype(str).str.strip()
        articles = articles[articles.model_text.ne('')].drop_duplicates('article_id')
        texts = [preprocess(t, job['thai_preprocess']) for t in articles.model_text]
        vectors = embed(texts, job['model'], job['max_length'], device)
        np.save(OUT / f'{market}_embeddings.npy', vectors)
        articles[['article_id']].to_csv(OUT / f'{market}_article_ids.csv', index=False)
        record['jobs'][market] = {'model': job['model'], 'articles': len(articles), 'dimension': int(vectors.shape[1]),
                                  'max_length': job['max_length'], 'seconds': round(time.time() - began, 1)}
        print(market, record['jobs'][market], flush=True)
    (OUT / 'COMPLETE.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
