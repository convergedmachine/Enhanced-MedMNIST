import torch
from torchvision.transforms.functional import InterpolationMode

def get_module(use_v2):
    # Dynamically imports appropriate torchvision.transforms version
    if use_v2:
        import torchvision.transforms.v2
        return torchvision.transforms.v2
    else:
        import torchvision.transforms
        return torchvision.transforms


class ClassificationPresetTrain:
    """
    Transformation pipeline for training images, supporting both PIL images and tensors.
    """
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.3162, 0.3162, 0.3162),
        std=(0.3213, 0.3213, 0.3213),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
        use_v2=False,
    ):
        T = get_module(use_v2)

        transforms = []
        backend = backend.lower()
        
        # Add conditional preprocessing based on backend
        if backend == "tensor":
            # No conversion for tensors
            pass
        elif backend == "pil":
            # Ensure input is converted to tensor
            transforms.append(T.PILToTensor())
        else:
            raise ValueError(f"backend must be 'tensor' or 'pil', but got {backend}")

        # Add main augmentations
        transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transforms.append(T.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                transforms.append(T.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                transforms.append(T.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(T.AutoAugment(policy=aa_policy, interpolation=interpolation))

        # Convert tensor to float type and normalize
        transforms.extend([
            T.ConvertImageDtype(torch.float) if not use_v2 else T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
        ])
        
        # Add random erasing
        if random_erase_prob > 0:
            transforms.append(T.RandomErasing(p=random_erase_prob))

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    """
    Transformation pipeline for evaluation images, supporting both PIL images and tensors.
    """
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.3162, 0.3162, 0.3162),
        std=(0.3213, 0.3213, 0.3213),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil",
        use_v2=False,
    ):
        T = get_module(use_v2)

        transforms = []
        backend = backend.lower()

        if backend == "tensor":
            # No conversion for tensors
            pass
        elif backend == "pil":
            # Ensure input is converted to tensor
            transforms.append(T.PILToTensor())
        else:
            raise ValueError(f"backend must be 'tensor' or 'pil', but got {backend}")

        # Add resizing and cropping transformations
        transforms += [
            T.Resize(resize_size, interpolation=interpolation, antialias=True),
            T.CenterCrop(crop_size),
        ]

        # Convert tensor to float type and normalize
        transforms += [
            T.ConvertImageDtype(torch.float) if not use_v2 else T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
        ]

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
