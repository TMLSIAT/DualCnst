import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import json
import argparse
import numpy as np
import torch
from scipy import stats
from utils.args_pool import *
import clip
from utils.common import setup_seed, get_num_cls, get_test_labels
from utils.detection_util import get_Mahalanobis_score, get_mean_prec, print_measures, get_and_print_results, get_ood_scores_clip
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import  set_model_clip, set_train_loader, set_val_loader, set_ood_loader_ImageNet
#from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel

def process_args():
    parser = argparse.ArgumentParser(description='DualCnst evaluation runner for CLIP-based OOD scores',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # setting for each run
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100',
                                  'pet37', 'food101', 'car196', 'bird200', 'ImageNet_R', 'CXR'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood",
                        type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=5, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type = int,
                        help='the GPU indice to use')
    parser.add_argument('-b', '--batch-size', default=1024, type=int,
                        help='mini-batch size')
    parser.add_argument('--T', type=int, default=0.01,
                        help='temperature parameter')
    parser.add_argument('--alpha', type=int, default=0.2,
                        help='alpha parameter')
    parser.add_argument('--model', default='CLIP', type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14','RN50','RN101','RN50x4','RN50x16'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='MCM', type=str, choices=['DualCnst','NegLabel',
        'MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha'], help='score options')
    # for Mahalanobis score
    parser.add_argument('--feat_dim', type=int, default=512, help='feat dim； 512 for ViT-B and 768 for ViT-L')
    parser.add_argument('--normalize', type = bool, default = False, help='whether use normalized features for Maha score')
    parser.add_argument('--generate', type = bool, default = True, help='whether to generate class-wise means or read from files for Maha score')
    parser.add_argument('--template_dir', type = str, default = 'img_templates', help='the loc of stored classwise mean and precision matrix')
    parser.add_argument('--subset', default = False, type =bool, help = "whether uses a subset of samples in the training set")
    parser.add_argument('--max_count', default = 250, type =int, help = "how many samples are used to estimate classwise mean and precision matrix")
    # Parameters for get_ood_scores_clip
    parser.add_argument('--neglabel_dir', default='OODNegMining', type=str, help='Root directory for positive/negative label files')
    parser.add_argument('--pos_path', default=None, type=str, help='Override: explicit path to positive samples (auto-generated if not provided)')
    parser.add_argument('--neg_path', default=None, type=str, help='Override: explicit path to negative samples (auto-generated if not provided)')
    parser.add_argument('--data_dir', default='//data0/fayi/generation_image_xl', type=str, help='Root directory for generated images')
    parser.add_argument('--selected_layers', default='7,10', type=str, help='Comma-separated list of selected layer indices (e.g., 1,2,3 or 9,10,11)')
    parser.add_argument('--layer_weight', default=0.15, type=float, help='Weight for intermediate layer features in Dual and MCM_Dual scores')
    parser.add_argument('--cache_dir', default='similarity_caches', type=str, help='Root directory for cache files')
    parser.add_argument('--cache_path', default=None, type=str, help='Override: explicit path for main cache file (auto-generated if not provided)')
    parser.add_argument('--intermediate_cache_path', default=None, type=str, help='Override: explicit path for intermediate cache file (auto-generated if not provided)')
    parser.add_argument('--save_similarities', default=True, type=bool)
    args = parser.parse_args()

    args.n_cls = get_num_cls(args)
    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok=True)

    # Auto-generate label and cache paths if not explicitly provided
    if args.pos_path is None or args.neg_path is None:
        dataset_to_neglabel_mapping = {
            'ImageNet': 'ImageNet',
            'ImageNet10': 'ImageNet',
            'ImageNet20': 'ImageNet',
            'ImageNet100': 'ImageNet',
            'ImageNet_R': 'ImageNet',
            'ImageNet_A': 'ImageNet',
            'ImageNet_V2': 'ImageNet',
            'ImageNet_S': 'ImageNet',
            'bird200': 'CUB',
            'food101': 'Food',
            'car196': 'Stanford-Cars',
            'pet37': 'Oxford-Pet',
            'CXR': 'CXR',
        }

        dataset_to_filename_mapping = {
            'ImageNet': ('ImageNet.txt', 'ImageNet_neg.txt'),
            'ImageNet10': ('ImageNet10.txt', 'ImageNet10_neg.txt'),
            'ImageNet20': ('ImageNet20.txt', 'ImageNet20_neg.txt'),
            'ImageNet100': ('ImageNet100.txt', 'ImageNet100_neg.txt'),
            'ImageNet_R': ('ImageNet_R.txt', 'ImageNet_R_neg.txt'),
            'ImageNet_A': ('ImageNet_A.txt', 'ImageNet_A_neg.txt'),
            'ImageNet_V2': ('ImageNet_V2.txt', 'ImageNet_V2_neg.txt'),
            'ImageNet_S': ('ImageNet_S.txt', 'ImageNet_S_neg.txt'),
            'bird200': ('positive_samples_bird200.txt', 'low_similarity_neg_samples_bird200.txt'),
            'food101': ('positive_samples_food101.txt', 'low_similarity_neg_samples_food101.txt'),
            'car196': ('positive_samples_car196.txt', 'low_similarity_neg_samples_car196.txt'),
            'pet37': ('positive_samples_pet37.txt', 'low_similarity_neg_samples_pet37.txt'),
            'CXR': ('positive_samples_CXR.txt', 'low_similarity_neg_samples_CXR.txt'),
        }

        neglabel_subdir = dataset_to_neglabel_mapping.get(args.in_dataset, args.in_dataset)
        pos_filename, neg_filename = dataset_to_filename_mapping.get(
            args.in_dataset,
            (f'{args.in_dataset}.txt', f'{args.in_dataset}_neg.txt')
        )

        if args.pos_path is None:
            args.pos_path = os.path.join(args.neglabel_dir, neglabel_subdir, 'positive', pos_filename)

        if args.neg_path is None:
            args.neg_path = os.path.join(args.neglabel_dir, neglabel_subdir, 'negative', neg_filename)

    if args.cache_path is None or args.intermediate_cache_path is None:
        model_suffix = args.CLIP_ckpt.replace('/', '-')

        if args.score == 'DualCnst':
            layers_str = args.selected_layers.replace(',', '_')
            cache_subdir = f"{args.in_dataset}/{model_suffix}/layers_{layers_str}"
        else:
            cache_subdir = f"{args.in_dataset}/{model_suffix}"

        full_cache_dir = os.path.join(args.cache_dir, cache_subdir)

        if args.cache_path is None:
            args.cache_path = os.path.join(full_cache_dir, f"{args.in_dataset}_final_features.npy")

        if args.intermediate_cache_path is None:
            args.intermediate_cache_path = os.path.join(full_cache_dir, f"{args.in_dataset}_intermediate_features.pt")

    return args
def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)

    if args.score in ['DualCnst', 'NegLabel']:
        print(f"\n{'='*80}")
        print(f"Path Configuration for {args.in_dataset}:")
        print(f"  Positive labels: {args.pos_path}")
        print(f"  Negative labels: {args.neg_path}")
        if args.score == 'DualCnst':
            print(f"  Main cache: {args.cache_path}")
            print(f"  Intermediate cache: {args.intermediate_cache_path}")
        print(f"{'='*80}\n")

    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)
    ckpt= ckpt_mapping.get(args.CLIP_ckpt)
    # model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    # model.from_pretrained()
    net, preprocess = clip.load(ckpt, args.gpu, jit=False)
    net.eval()

    if args.in_dataset in ['ImageNet10']:
        out_datasets = ['iNaturalist']
    elif args.in_dataset in ['ImageNet20']:
        out_datasets = ['ImageNet10']
    elif args.in_dataset in ['ImageNet', 'ImageNet100', 'bird200', 'car196', 'food101', 'pet37']:
         out_datasets = ['iNaturalist', 'SUN', 'places365', 'dtd']
    else:
        out_datasets = dataset_mappings.get(args.in_dataset, [])

    test_loader = set_val_loader(args, preprocess)
    test_labels = get_test_labels(args, test_loader)
    #generation(test_labels,img_dir="F:\paper\causal\DG\A\generation_image",num_images=10)
    #np.savetxt('test_labels10.txt', test_labels, fmt='%s', delimiter=',')
    if args.score == 'maha':
        os.makedirs(args.template_dir, exist_ok = True)
        train_loader = set_train_loader(args, preprocess, subset = args.subset)
        if args.generate:
            classwise_mean, precision = get_mean_prec(args, net, train_loader)
        classwise_mean = torch.load(os.path.join(args.template_dir, f'{args.model}_classwise_mean_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'), map_location= 'cpu').cuda()
        precision = torch.load(os.path.join(args.template_dir,  f'{args.model}_precision_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'), map_location= 'cpu').cuda()
        in_score = get_Mahalanobis_score(args, net, test_loader, classwise_mean, precision, in_dist = True)
    else:
        in_score  = get_ood_scores_clip(args, net, test_loader, preprocess)
    #np.savetxt(f'OODNegMining/exp_pos_sim.txt', in_score, delimiter=',')

    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, root=os.path.join(args.root_dir, 'ImageNet_OOD_dataset'))
        if args.score == 'maha':
            out_score = get_Mahalanobis_score(args, net, ood_loader, classwise_mean, precision, in_dist = False)
        else:
            out_score = get_ood_scores_clip(args, net, ood_loader, preprocess)
            pass
        #np.savetxt(f'OODNegMining/exp_neg_sim.txt', out_score, delimiter=',')

        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        plot_distribution(args, in_score, out_score, out_dataset)
        get_and_print_results(args, log, in_score, out_score,
                              auroc_list, aupr_list, fpr_list)
    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                   np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)
if __name__ == '__main__':
    main()
