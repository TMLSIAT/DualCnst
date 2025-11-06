import sys
import os
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from transformers import CLIPModel
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from dataloaders import StanfordCars, Food101, OxfordIIITPet, Cub2011


def set_model_clip(args):
    ckpt_mapping = {"ViT-B/16":"/data0/fayi/clip_vit_b_16",
                    "ViT-B/32":"openai/clip-vit-base-patch32",
                    "ViT-L/14":"openai/clip-vit-large-patch14"}
    args.ckpt = ckpt_mapping[args.CLIP_ckpt]
    model =  CLIPModel.from_pretrained(args.ckpt)
    if args.model == 'CLIP-Linear':
        model.load_state_dict(torch.load(args.finetune_ckpt, map_location=torch.device(args.gpu)))
    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    return model, val_preprocess

def set_train_loader(args, preprocess=None, batch_size=None, shuffle=False, subset=False):
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
        preprocess = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if batch_size is None:
        batch_size = args.batch_size
        shuffle = True
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'train')
        dataset = datasets.ImageFolder(path, transform=preprocess)
        if subset:
            from collections import defaultdict
            classwise_count = defaultdict(int)
            indices = []
            for i, label in enumerate(dataset.targets):
                if classwise_count[label] < args.max_count:
                    indices.append(i)
                    classwise_count[label] += 1
            dataset = torch.utils.data.Subset(dataset, indices)
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset in ["ImageNet_and_R", "ImageNet_and_A", "ImageNet_and_V2", "ImageNet_and_S"]:
        imagenet_path = os.path.join(root, 'ImageNet', 'train')
        imagenet_dataset = datasets.ImageFolder(imagenet_path, transform=preprocess)

        second_dataset_name = args.in_dataset.split('_and_')[1]
        second_dataset_path = os.path.join(root, f'ImageNet_{second_dataset_name}')
        second_dataset = datasets.ImageFolder(second_dataset_path, transform=preprocess)

        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([imagenet_dataset, second_dataset])

        if subset:
            import random
            indices = random.sample(range(len(combined_dataset)), min(len(combined_dataset), args.max_count * args.n_cls))
            combined_dataset = torch.utils.data.Subset(combined_dataset, indices)

        train_loader = torch.utils.data.DataLoader(combined_dataset,
                                                  batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "UCM":
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder("/data0/fayi/datasets/RSdata/UCMerced_LandUse", transform=preprocess),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100"]:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'train'), transform=preprocess),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "car196":
        train_loader = torch.utils.data.DataLoader(
            StanfordCars(root, split="train", download=False, transform=preprocess),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "food101":
        train_loader = torch.utils.data.DataLoader(Food101(root, split="train", download=False, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "pet37":
        train_loader = torch.utils.data.DataLoader(
            OxfordIIITPet(root, split="trainval", download=False, transform=preprocess),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "bird200":
        train_loader = torch.utils.data.DataLoader(Cub2011(root, train=True, transform=preprocess),
                                                   batch_size=batch_size, shuffle=shuffle, **kwargs)
    return train_loader


def set_val_loader(args, preprocess=None):
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset in ["ImageNet", "ImageNet_and_ARSV"]:
        path = os.path.join(root, 'ImageNet', 'val')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset in ["ImageNet_and_R","ImageNet_Robust"]:
        imagenet_path = os.path.join(root, 'ImageNet', 'val')
        imagenet_val_dataset = datasets.ImageFolder(imagenet_path, transform=preprocess)

        imagenet_r_path = os.path.join(root, 'ImageNet_R', 'val')
        imagenet_r_dataset = datasets.ImageFolder(imagenet_r_path, transform=preprocess)

        from torch.utils.data import ConcatDataset
        combined_val_dataset = ConcatDataset([imagenet_val_dataset, imagenet_r_dataset])

        val_loader = torch.utils.data.DataLoader(combined_val_dataset,
                                               batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset in ["ImageNet_and_A", "ImageNet_and_V2", "ImageNet_and_S"]:
        imagenet_path = os.path.join(root, 'ImageNet', 'val')
        imagenet_val_dataset = datasets.ImageFolder(imagenet_path, transform=preprocess)

        second_dataset_name = args.in_dataset.split('_and_')[1]
        second_dataset_path = os.path.join(root, f'ImageNet_{second_dataset_name}')

        if os.path.exists(os.path.join(second_dataset_path, 'val')):
            second_dataset_path = os.path.join(second_dataset_path, 'val')

        second_val_dataset = datasets.ImageFolder(second_dataset_path, transform=preprocess)

        from torch.utils.data import ConcatDataset
        combined_val_dataset = ConcatDataset([imagenet_val_dataset, second_val_dataset])

        val_loader = torch.utils.data.DataLoader(combined_val_dataset,
                                               batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset in ["ImageNet_A","ImageNet_R","ImageNet_S","ImageNet_V2"]:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100"]:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "UCM":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder("/data0/fayi/datasets/RSdata/UCMerced_LandUse", transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "car196":
        val_loader = torch.utils.data.DataLoader(StanfordCars(root, split="test", download=False, transform=preprocess),
                                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "food101":
        val_loader = torch.utils.data.DataLoader(Food101(root, split="test", download=False, transform=preprocess),
                                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "pet37":
        val_loader = torch.utils.data.DataLoader(
            OxfordIIITPet(root, split="test", download=False, transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "bird200":
        val_loader = torch.utils.data.DataLoader(Cub2011(root, train=False, transform=preprocess),
                                                 batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset =="cifar10":
        val_loader =torch.utils.data.DataLoader(datasets.CIFAR10(root=os.path.join(root, "cifar10"), train=False, download=False, transform=preprocess),
                                                batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset =="cifar100":
        val_loader =torch.utils.data.DataLoader(datasets.CIFAR100(root=os.path.join(root, "cifar100"), train=False, download=False, transform=preprocess),
                                                batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "CXR":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(
                root, 'organized_images'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)

    return val_loader


def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':  # filtered places
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Places'), transform=preprocess)
    elif out_dataset == 'placesbg':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'placesbg'), transform=preprocess)
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                                      transform=preprocess)
    elif out_dataset == 'ssb_hard': 
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'images_largescale', 'ssb_hard'),
                                                      transform=preprocess)
    elif out_dataset == 'AID': 
        testsetout = torchvision.datasets.ImageFolder("/data0/fayi/datasets/RSdata/AID",
                                                      transform=preprocess)
    elif out_dataset == 'ninco': 
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'images_largescale', 'ninco'),
                                                      transform=preprocess)
    elif out_dataset == 'ImageNet_A':
        testsetout = torchvision.datasets.ImageFolder("/data0/fayi/datasets/ImageNet_A/val",
                                                      transform=preprocess)
    elif out_dataset == 'ImageNet_R':
        testsetout = torchvision.datasets.ImageFolder("/data0/fayi/datasets/ImageNet_R/val",
                                                      transform=preprocess)
    elif out_dataset == 'ImageNet_S':
        testsetout = torchvision.datasets.ImageFolder("/data0/fayi/datasets/ImageNet_S/val",
                                                      transform=preprocess)
    elif out_dataset == 'ImageNet_V2':
        testsetout = torchvision.datasets.ImageFolder("/data0/fayi/datasets/ImageNet_V2/val",
                                                      transform=preprocess)
    elif out_dataset == 'ImageNet10':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet10', 'val'), transform=preprocess)
    elif out_dataset == 'ImageNet20':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet20', 'val'), transform=preprocess)
    elif out_dataset == 'media':
            testsetout = torchvision.datasets.ImageFolder(root="/data0/fayi/data/images_test", transform=preprocess)
    elif out_dataset =="cifar10":
        testsetout = datasets.CIFAR10(
            root=os.path.join(args.root_dir, "cifar10"), 
            train=False, 
            download=False, 
            transform=preprocess
        )
    elif out_dataset =="cifar100":
        testsetout = datasets.CIFAR100(
            root=os.path.join(args.root_dir, "cifar100"), 
            train=False, 
            download=False, 
            transform=preprocess
        )
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                                shuffle=False, num_workers=4)
    
    return testloaderOut

