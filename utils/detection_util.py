import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import entropy
import torchvision
import clip
import sklearn.metrics as sk
from transformers import CLIPTokenizer
from torchvision import datasets
import torch.nn.functional as F
import torchvision
from utils.read_sd_image import load_data_by_batches
from PIL import Image

def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    """Set OOD loader for ImageNet-scale datasets"""
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Places'), transform=preprocess)
    elif out_dataset == 'placesbg':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'placesbg'), transform=preprocess)
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                                      transform=preprocess)
    elif out_dataset == 'ImageNet10':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet10', 'train'), transform=preprocess)
    elif out_dataset == 'ImageNet20':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet20', 'val'), transform=preprocess)
    elif out_dataset == 'ImageNet_A':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet_A'), transform=preprocess)
    elif out_dataset == 'ImageNet_R':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet_R'), transform=preprocess)
    elif out_dataset == 'ImageNet_S':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet_S'), transform=preprocess)
    elif out_dataset == 'ImageNet_V2':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet_V2'), transform=preprocess)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                                shuffle=False, num_workers=4)
    return testloaderOut


def print_measures(log, auroc, aupr, fpr, method_name='Ours', recall_level=0.95):
    if log == None:
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
        print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
        print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
    else:
        log.debug('\t\t\t\t' + method_name)
        log.debug('  FPR{:d} AUROC AUPR'.format(int(100 * recall_level)))
        log.debug('& {:.2f} & {:.2f} & {:.2f}'.format(100 * fpr, 100 * auroc, 100 * aupr))


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """High-precision cumsum with validation"""
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    """Calculate FPR at specified recall level"""
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    y_true = (y_true == pos_label)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    cutoff = np.argmin(np.abs(recall - recall_level))

    thresholds_score = y_score[cutoff]

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))




def get_measures(_pos, _neg, recall_level=0.95):
    """Calculate AUROC, AUPR, and FPR metrics"""
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def input_preprocessing(args, net, images, text_features=None, classifier=None):
    """Input preprocessing (unused)"""
    criterion = torch.nn.CrossEntropyLoss()
    if args.model == 'vit-Linear':
        image_features = net(pixel_values=images.float()).last_hidden_state
        image_features = image_features[:, 0, :]
    elif args.model == 'CLIP-Linear':
        image_features = net.encode_image(images).float()
    if classifier:
        outputs = classifier(image_features) / args.T
    else:
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        outputs = image_features @ text_features.T / args.T
    pseudo_labels = torch.argmax(outputs.detach(), dim=1)
    loss = criterion(outputs, pseudo_labels)
    loss.backward()

    sign_grad = torch.ge(images.grad.data, 0)
    sign_grad = (sign_grad.float() - 0.5) * 2

    std = (0.26862954, 0.26130258, 0.27577711)
    for i in range(3):
        sign_grad[:, i] = sign_grad[:, i] / std[i]

    processed_inputs = images.data - args.noiseMagnitude * sign_grad
    return processed_inputs


def get_mean_prec(args, net, train_loader):
    """Compute class-wise means and precision matrix for Mahalanobis score"""
    classwise_mean = torch.empty(args.n_cls, args.feat_dim, device=args.gpu)
    all_features = []
    from collections import defaultdict
    classwise_idx = defaultdict(list)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.cuda()
            if args.model == 'CLIP':
                features = net.get_image_features(pixel_values=images).float()
            if args.normalize:
                features /= features.norm(dim=-1, keepdim=True)
            for label in labels:
                classwise_idx[label.item()].append(idx)
            all_features.append(features.cpu())
    all_features = torch.cat(all_features)
    for cls in range(args.n_cls):
        classwise_mean[cls] = torch.mean(all_features[classwise_idx[cls]].float(), dim=0)
        if args.normalize:
            classwise_mean[cls] /= classwise_mean[cls].norm(dim=-1, keepdim=True)
    cov = torch.cov(all_features.T.double())
    precision = torch.linalg.inv(cov).float()
    print(f'cond number: {torch.linalg.cond(precision)}')
    torch.save(classwise_mean, os.path.join(args.template_dir,
                                            f'{args.model}_classwise_mean_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'))
    torch.save(precision, os.path.join(args.template_dir,
                                       f'{args.model}_precision_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'))
    return classwise_mean, precision


def get_Mahalanobis_score(args, net, test_loader, classwise_mean, precision, in_dist=True):
    """Calculate Mahalanobis distance scores"""
    Mahalanobis_score_all = []
    total_len = len(test_loader.dataset)
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            if (batch_idx >= total_len // args.batch_size) and in_dist is False:
                break
            images, labels = images.cuda(), labels.cuda()
            if args.model == 'CLIP':
                features = net.get_image_features(pixel_values=images).float()
            if args.normalize:
                features /= features.norm(dim=-1, keepdim=True)
            for i in range(args.n_cls):
                class_mean = classwise_mean[i]
                zero_f = features - class_mean
                Mahalanobis_dist = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                if i == 0:
                    Mahalanobis_score = Mahalanobis_dist.view(-1, 1)
                else:
                    Mahalanobis_score = torch.cat((Mahalanobis_score, Mahalanobis_dist.view(-1, 1)), 1)
            Mahalanobis_score, _ = torch.max(Mahalanobis_score, dim=1)
            Mahalanobis_score_all.extend(-Mahalanobis_score.cpu().numpy())

    return np.asarray(Mahalanobis_score_all, dtype=np.float32)


def grouping(args, pos, neg, num, ngroup=100, random_permute=False):
    """Compute grouped softmax scores for positive and negative samples"""
    group = ngroup
    drop = neg.shape[1] % ngroup

    if drop > 0:
        neg = neg[:, :-drop]
    if random_permute:
        SEED = 0
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        idx = torch.randperm(neg.shape[1], device="cuda:{}".format(0))
        neg = neg.T
        negs = neg[idx].T.reshape(pos.shape[0], group, -1).contiguous()
    else:
        negs = neg.reshape(pos.shape[0], group, -1).contiguous()

    scores = []

    for i in range(group):
        full_sim = torch.cat([pos, negs[:, i, :]], dim=-1) / args.T
        full_sim = full_sim.softmax(dim=-1)
        pos_score = full_sim[:, :pos.shape[1]].sum(dim=-1)
        scores.append(pos_score.unsqueeze(-1))

    scores = torch.cat(scores, dim=-1)
    if num is not None:
        scores = scores[:, 0:num - 1]
    score = scores.mean(dim=-1)
    return score


def _create_intermediate_feature_extractor(net, selected_layers):
    """Create hook-based feature extractor for intermediate CLIP layers"""
    extractor = {
        'features': {layer: [] for layer in selected_layers},
        'hooks': [],
        'pool': nn.AdaptiveAvgPool1d(1)
    }

    def create_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            feature = output.clone().detach()
            feature = feature.permute(1, 2, 0)
            pooled_feature = extractor['pool'](feature).squeeze(-1)
            extractor['features'][layer_idx].append(pooled_feature)
        return hook_fn

    for idx, resblock in enumerate(net.visual.transformer.resblocks, 1):
        if idx in selected_layers:
            handle = resblock.register_forward_hook(create_hook_fn(idx))
            extractor['hooks'].append(handle)

    return extractor


def _remove_hooks(extractor):
    """Remove all registered hooks"""
    for handle in extractor['hooks']:
        handle.remove()
    extractor['hooks'] = []


def _load_or_compute_sd_features(args, net, text_list, preprocess, selected_layers):
    """Load or compute cached features for SD-generated images"""
    os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.intermediate_cache_path), exist_ok=True)

    if os.path.exists(args.cache_path) and os.path.exists(args.intermediate_cache_path):
        image_features_sd_all = torch.tensor(np.load(args.cache_path)).cuda()
        sd_intermediate_features = torch.load(args.intermediate_cache_path)
        return image_features_sd_all, sd_intermediate_features

    image_features_sd_all = []
    sd_intermediate_features = {layer: [] for layer in selected_layers}

    loader1, _, _ = load_data_by_batches(text_list, args.data_dir, preprocess, batch_size=512)

    extractor = _create_intermediate_feature_extractor(net, selected_layers)

    for sd_images in loader1:
        sd_images = sd_images.cuda()

        extractor['features'] = {layer: [] for layer in selected_layers}

        image_features_sd = net.encode_image(sd_images).to(torch.float32)
        image_features_sd /= image_features_sd.norm(dim=-1, keepdim=True)
        image_features_sd_all.append(image_features_sd)

        for idx in selected_layers:
            sd_intermediate_features[idx].extend(extractor['features'][idx])

        torch.cuda.empty_cache()

    _remove_hooks(extractor)

    for idx in selected_layers:
        if len(sd_intermediate_features[idx]) > 0:
            sd_intermediate_features[idx] = torch.cat(sd_intermediate_features[idx], dim=0)
        else:
            raise ValueError(f"No features collected for layer {idx}.")

    image_features_sd_all = torch.cat(image_features_sd_all, dim=0)

    torch.save(sd_intermediate_features, args.intermediate_cache_path)
    np.save(args.cache_path, image_features_sd_all.cpu().numpy())

    return image_features_sd_all, sd_intermediate_features


def _compute_layer_similarity(intermediate_features, sd_intermediate_features, selected_layers, weight):
    """Compute weighted cosine similarity for intermediate layer features"""
    first_layer = selected_layers[0]
    if len(intermediate_features[first_layer]) > 0:
        sample_feat = torch.cat(intermediate_features[first_layer], dim=0)
        target_shape = (sample_feat.shape[0], sd_intermediate_features[first_layer].shape[0])
    else:
        raise ValueError("No intermediate features available")

    final_similarity = torch.zeros(target_shape, device=sample_feat.device)

    for layer in selected_layers:
        if len(intermediate_features[layer]) > 0:
            layer_feat = torch.cat(intermediate_features[layer], dim=0)
            layer_feat = F.normalize(layer_feat, p=2, dim=1)
            sd_layer_feat = F.normalize(sd_intermediate_features[layer], p=2, dim=1)

            cosine_similarity = layer_feat @ sd_layer_feat.T
            final_similarity += weight * cosine_similarity

    total_weight = weight * len(selected_layers)
    return final_similarity, total_weight


def get_ood_scores_clip(args, net, loader, preprocess):
    """Compute OOD detection scores using various methods"""
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    start, end = map(int, args.selected_layers.split(','))
    selected_layers = list(range(start, end))
    _score = []
    tqdm_object = tqdm(loader, total=len(loader))
    with open(args.pos_path, 'r') as f:
        text_pos = f.readlines()

    with open(args.neg_path, 'r') as f:
        text_neg = f.readlines()
    text_com = text_pos + text_neg

    with torch.no_grad():
        extractor = _create_intermediate_feature_extractor(net, selected_layers)

        for images, labels in tqdm_object:

            images = images.cuda()
            labels = labels.long().cuda()

            extractor['features'] = {layer: [] for layer in selected_layers}

            image_features = net.encode_image(images).to(torch.float32)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            intermediate_features = extractor['features']
            if args.model == 'CLIP':
                text_inputs_id = torch.cat([clip.tokenize(f"{c}") for c in text_pos]).to(args.gpu)
                text_inputs_ood = torch.cat([clip.tokenize(f"{c}") for c in text_neg]).to(args.gpu)
                text_features_id = net.encode_text(text_inputs_id).to(torch.float32)
                text_features_ood = net.encode_text(text_inputs_ood).to(torch.float32)

                text_features_id /= text_features_id.norm(dim=-1, keepdim=True)
                text_features_ood /= text_features_ood.norm(dim=-1, keepdim=True)
                output_id = (image_features @ text_features_id.T)
                output_ood = (image_features @ text_features_ood.T)

            if args.score == 'max-logit':
                smax = to_np(output_id)
            else:
                smax = to_np(F.softmax(output_id / args.T, dim=1))

            if args.score == 'energy':
                _score.append(to_np((args.T * torch.logsumexp(output_id / args.T, dim=1))))
            elif args.score == 'entropy':
                _score.append(entropy(smax, axis=1))
            elif args.score == 'var':
                _score.append(-np.var(smax, axis=1))
            elif args.score in ['MCM', 'max-logit']:
                _score.append(np.max(smax, axis=1))
            elif args.score == 'NegLabel':
                num = None
                pos_sim = output_id
                neg_sim = output_ood
                group_score = grouping(args, pos_sim, neg_sim, num, ngroup=100, random_permute=False)
                _score.append(to_np(group_score))
            elif args.score == 'DualCnst':
                image_features_sd_all, sd_intermediate_features = _load_or_compute_sd_features(
                    args, net, text_com, preprocess, selected_layers
                )

                image_output_all = image_features @ image_features_sd_all.T

                final_similarity, total_weight = _compute_layer_similarity(
                    intermediate_features, sd_intermediate_features, selected_layers, weight=args.layer_weight
                )

                image_output_all = (1 - total_weight) * image_output_all + final_similarity

                num = None
                img_pos_sim, img_neg_sim = torch.split(
                    image_output_all,
                    [output_id.shape[1], image_output_all.shape[1] - output_id.shape[1]],
                    dim=1
                )

                pos_sim = args.alpha * img_pos_sim + (1 - args.alpha) * output_id
                neg_sim = args.alpha * img_neg_sim + (1 - args.alpha) * output_ood

                group_score = grouping(args, pos_sim, neg_sim, num, ngroup=100, random_permute=False)
                _score.append(to_np(group_score))

        _remove_hooks(extractor)

    return concat(_score)[:len(loader.dataset)].copy()



def get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list):
    """Evaluate and print OOD detection performance metrics"""
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(in_score, out_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}')
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)
    print_measures(log, auroc, aupr, fpr, args.score)


class TextDataset(torch.utils.data.Dataset):
    """Text dataset wrapper for batch processing"""

    def __init__(self, texts, labels):
        self.labels = labels
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        X = self.texts[index]
        y = self.labels[index]
        return X, y
