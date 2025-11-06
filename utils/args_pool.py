dataset_mappings = {
    # far ood
    'bird200': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'car196': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'food101': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'pet37': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'ImageNet_sketch': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'cifar10': ['cifar100'],
    'cifar100': ['cifar10'],
    # near ood
    'ImageNet10': ['ImageNet20'],
    'ImageNet20': ['ImageNet10'],
    # fine-grained ood
    'cub100_ID': ['cub100_OOD'],
    'car98_ID': ['car98_OOD'],
    'food50_ID': ['food50_OOD'],
    'pet18_ID': ['pet18_OOD'],

    'ImageNet_A': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'ImageNet_and_R': ['ssb_hard', 'ninco', 'iNaturalist', 'dtd'],
    'ImageNet_Robust': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'ImageNet_V2': ['iNaturalist', 'SUN', 'places365', 'dtd'],
    'ImageNet_and_ARSV': ['ImageNet_A', 'ImageNet_R', 'ImageNet_S', 'ImageNet_V2'],
    'UCM': ['AID'],#
    'CXR': ["media"]
}
ckpt_mapping = {
    "RN50": "/data0/fayi/CLIP_model/RN50.pt",
    "RN101": "/data0/fayi/CLIP_model/RN101.pt",
    "RN50x4": "/data0/fayi/CLIP_model/RN50x4.pt",
    "RN50x16": "/data0/fayi/CLIP_model/RN50x16.pt",
    "ViT-B/16": "/data0/fayi/CLIP_model/ViT-B-16.pt",
    "ViT-B/32": "/data0/fayi/CLIP_model/ViT-B-32.pt",
    "ViT-L/14": "/data0/fayi/CLIP_model/ViT-L-14.pt",
}