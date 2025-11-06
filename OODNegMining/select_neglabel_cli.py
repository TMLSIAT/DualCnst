#!/usr/bin/env python
"""
OOD Negative Label Selection Script - Command Line Interface

Usage:
    python select_neglabel_cli.py --dataset ImageNet
    python select_neglabel_cli.py --dataset bird200 --prompt_idx_pos 85 --neg_topk 0.15
    python select_neglabel_cli.py --dataset CXR --load_dump_neg
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse
import clip
import numpy as np
import torch
from transformers import CLIPTokenizer
from classs_name import CLASS_NAME, prompt_templates
import torchvision.transforms as transforms
from transformers import CLIPModel
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def select_label(train_dataset="ImageNet", prompt_idx_pos=85, prompt_idx_neg=85, load_dump_neg=False,
                 emb_batchsize=1000, pos_topk=None, neg_topk=0.145, pencentile=0.95):
    """
    Main function for OOD negative label selection

    Args:
        train_dataset: Dataset name (ImageNet, ImageNet10, bird200, food101, etc.)
        prompt_idx_pos: Positive sample prompt template index
        prompt_idx_neg: Negative sample prompt template index
        load_dump_neg: Whether to load cached negative sample embeddings
        emb_batchsize: Batch size for embedding computation
        pos_topk: Select top-k negative samples most similar to positive samples
        neg_topk: Proportion of negative samples to select
        pencentile: Percentile threshold for similarity filtering
    """

    print(f"{'='*80}")
    print(f"OOD Negative Label Selection")
    print(f"{'='*80}")
    print(f"Dataset: {train_dataset}")
    print(f"Prompt Index (Pos/Neg): {prompt_idx_pos}/{prompt_idx_neg}")
    print(f"Negative Selection Ratio: {neg_topk}")
    print(f"{'='*80}\n")

    print("[1/6] Loading CLIP model...")
    clip_model, _ = clip.load('ViT-B/16', 0, jit=False)
    dataset_to_subdir = {
        'ImageNet': 'ImageNet', 'ImageNet10': 'ImageNet', 'ImageNet20': 'ImageNet',
        'ImageNet100': 'ImageNet', 'ImageNet_R': 'ImageNet', 'ImageNet_A': 'ImageNet',
        'bird200': 'CUB', 'food101': 'Food', 'car196': 'Stanford-Cars',
        'pet37': 'Oxford-Pet', 'CXR': 'CXR'
    }
    dataset_to_filename = {
        'ImageNet': ('ImageNet.txt', 'ImageNet_neg.txt'),
        'ImageNet10': ('ImageNet10.txt', 'ImageNet10_neg.txt'),
        'ImageNet20': ('ImageNet20.txt', 'ImageNet20_neg.txt'),
        'ImageNet100': ('ImageNet100.txt', 'ImageNet100_neg.txt'),
        'ImageNet_R': ('ImageNet_R.txt', 'ImageNet_R_neg.txt'),
        'ImageNet_A': ('ImageNet_A.txt', 'ImageNet_A_neg.txt'),
        'bird200': ('positive_samples_bird200.txt', 'low_similarity_neg_samples_bird200.txt'),
        'food101': ('positive_samples_food101.txt', 'low_similarity_neg_samples_food101.txt'),
        'car196': ('positive_samples_car196.txt', 'low_similarity_neg_samples_car196.txt'),
        'pet37': ('positive_samples_pet37.txt', 'low_similarity_neg_samples_pet37.txt'),
        'CXR': ('CXR.txt', 'CXR_neg.txt'),
    }

    subdir = dataset_to_subdir.get(train_dataset, train_dataset)
    pos_filename, neg_filename = dataset_to_filename.get(train_dataset,
                                                          (f'{train_dataset}.txt', f'{train_dataset}_neg.txt'))

    output_pos_path = PROJECT_ROOT / 'OODNegMining' / subdir / 'positive' / pos_filename
    output_neg_path = PROJECT_ROOT / 'OODNegMining' / subdir / 'negative' / neg_filename

    output_pos_path.parent.mkdir(parents=True, exist_ok=True)
    output_neg_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[2/6] Getting class names...")
    class_name = CLASS_NAME[train_dataset]
    print(f"  - Number of classes: {len(class_name)}")

    print(f"[3/6] Generating positive prompts...")
    prompts = [prompt_templates[prompt_idx_pos].format(c) for c in class_name]
    text_inputs_pos = torch.cat([clip.tokenize(f"{c}") for c in prompts]).to(0)

    os.makedirs(os.path.dirname(output_pos_path), exist_ok=True)
    with open(output_pos_path, 'w') as f:
        for line in prompts:
            f.write(line + '\n')
    print(f"  ✓ Positive samples saved: {output_pos_path}")
    print(f"  - Count: {len(prompts)}")

    with torch.no_grad():
        text_features_pos = clip_model.encode_text(text_inputs_pos).to(torch.float32)
        text_features_pos /= text_features_pos.norm(dim=-1, keepdim=True)

    print(f"[4/6] Processing negative samples...")
    wordnet_database = PROJECT_ROOT / 'OODNegMining' / 'txtfiles'
    neg_dump_path = PROJECT_ROOT / 'OODNegMining' / 'neg_dump.pth'

    if not load_dump_neg or not neg_dump_path.exists():
        print("  - Loading negative samples from WordNet database...")
        txtfiles = os.listdir(wordnet_database)

        words_noun, words_adj, dedup = [], [], {}
        prompt_idx_neg = -1 if prompt_idx_neg is None else prompt_idx_neg
        prompt_template = {'adj': 'This is a {} photo', 'noun': prompt_templates[prompt_idx_neg]}

        for file in txtfiles:
            filetype = file.split('.')[0]
            if filetype not in prompt_template:
                continue
            with open(wordnet_database / file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line_content = line.strip()
                    if line_content in dedup:
                        continue
                    dedup[line_content] = None

                    if filetype == 'noun':
                        words_noun.append(prompt_template[filetype].format(line_content))
                    elif filetype == 'adj':
                        words_adj.append(prompt_template[filetype].format(line_content))

        print(f"  - Noun count: {len(words_noun)}")
        print(f"  - Adjective count: {len(words_adj)}")

        print("  - Generating negative embeddings...")
        text_inputs_neg_noun = torch.cat([clip.tokenize(f"{c}") for c in words_noun]).to(0)
        text_inputs_neg_adj = torch.cat([clip.tokenize(f"{c}") for c in words_adj]).to(0)
        text_inputs_neg = torch.cat([text_inputs_neg_noun, text_inputs_neg_adj], dim=0)
        noun_length, adj_length = len(text_inputs_neg_noun), len(text_inputs_neg_adj)

        with torch.no_grad():
            text_features_neg = []
            for i in range(0, len(text_inputs_neg), emb_batchsize):
                x = clip_model.encode_text(text_inputs_neg[i: i + emb_batchsize])
                text_features_neg.append(x)
            text_features_neg = torch.cat(text_features_neg, dim=0)
            text_features_neg /= text_features_neg.norm(dim=-1, keepdim=True)

    print(f"[5/6] Filtering low-similarity negative samples...")
    with torch.no_grad():
        text_features_neg = text_features_neg.to(torch.float32)

        if pos_topk is not None:
            pos_mask = torch.zeros(len(text_features_neg), dtype=torch.bool, device=0)
            for i in range(text_features_pos.shape[0]):
                sim = text_features_pos[i].unsqueeze(0) @ text_features_neg.T
                _, ind = torch.topk(sim.squeeze(0), k=pos_topk)
                pos_mask[ind] = 1
            text_features_pos = torch.cat([text_features_pos, text_features_neg[pos_mask]])

        neg_sim = []
        for i in range(0, noun_length + adj_length, emb_batchsize):
            tmp = text_features_neg[i: i + emb_batchsize] @ text_features_pos.T
            tmp = tmp.to(torch.float32)
            sim = torch.quantile(tmp, q=pencentile, dim=-1)
            neg_sim.append(sim)
        neg_sim = torch.cat(neg_sim, dim=0)
        neg_sim_noun, neg_sim_adj = neg_sim[:noun_length], neg_sim[noun_length:]

        neg_sim_noun_with_idx = [(sim.item(), idx) for idx, sim in enumerate(neg_sim_noun)]
        neg_sim_adj_with_idx = [(sim.item(), idx) for idx, sim in enumerate(neg_sim_adj)]

        neg_sim_noun_with_idx_sorted = sorted(neg_sim_noun_with_idx, key=lambda x: x[0], reverse=True)[
                                       :int(len(neg_sim_noun) * neg_topk)]
        neg_sim_adj_with_idx_sorted = sorted(neg_sim_adj_with_idx, key=lambda x: x[0])[
                                      :int(len(neg_sim_adj) * neg_topk)]

        low_sim_noun_texts = [words_noun[idx] for _, idx in neg_sim_noun_with_idx_sorted]
        low_sim_adj_texts = [words_adj[idx] for _, idx in neg_sim_adj_with_idx_sorted]
        low_sim_texts = low_sim_noun_texts + low_sim_adj_texts

        print(f"  - Selected nouns: {len(low_sim_noun_texts)}")
        print(f"  - Selected adjectives: {len(low_sim_adj_texts)}")
        print(f"  - Total: {len(low_sim_texts)}")

    print(f"[6/6] Saving negative samples...")
    os.makedirs(os.path.dirname(output_neg_path), exist_ok=True)
    with open(output_neg_path, 'w') as f:
        for line in low_sim_texts:
            f.write(line + '\n')
    print(f"  ✓ Negative samples saved: {output_neg_path}")

    print(f"\n{'='*80}")
    print(f"✓ OOD Negative Label Selection Complete!")
    print(f"{'='*80}")
    print(f"Positive: {output_pos_path} ({len(prompts)} items)")
    print(f"Negative: {output_neg_path} ({len(low_sim_texts)} items)")
    print(f"{'='*80}\n")

    return low_sim_texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OOD negative label selection script')
    parser.add_argument('--dataset', type=str, default='ImageNet',
                        choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100',
                                'bird200', 'food101', 'car196', 'pet37', 'CXR'],
                        help='Dataset name')
    parser.add_argument('--prompt_idx_pos', type=int, default=85,
                        help='Positive sample prompt template index')
    parser.add_argument('--prompt_idx_neg', type=int, default=85,
                        help='Negative sample prompt template index')
    parser.add_argument('--load_dump_neg', action='store_true',
                        help='Load cached negative sample embeddings')
    parser.add_argument('--emb_batchsize', type=int, default=1000,
                        help='Batch size for embedding computation')
    parser.add_argument('--pos_topk', type=int, default=None,
                        help='Select top-k negatives most similar to positives')
    parser.add_argument('--neg_topk', type=float, default=0.145,
                        help='Proportion of negative samples to select')
    parser.add_argument('--pencentile', type=float, default=0.95,
                        help='Percentile threshold for similarity filtering')

    args = parser.parse_args()

    select_label(
        train_dataset=args.dataset,
        prompt_idx_pos=args.prompt_idx_pos,
        prompt_idx_neg=args.prompt_idx_neg,
        load_dump_neg=args.load_dump_neg,
        emb_batchsize=args.emb_batchsize,
        pos_topk=args.pos_topk,
        neg_topk=args.neg_topk,
        pencentile=args.pencentile
    )
