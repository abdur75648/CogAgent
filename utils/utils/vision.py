import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

class BlipImageEvalProcessor:
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__()
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

from functools import partial

def blip2_image_processor_func_with_inputs(image_processor, image):
    return {'image': image_processor(image).unsqueeze(0), 'input_ids': torch.zeros(1, 1, dtype=torch.long), 'position_ids': None, 'attention_mask': torch.ones(1, 1, dtype=torch.long)}

def get_image_processor(image_size):
    return partial(blip2_image_processor_func_with_inputs, BlipImageEvalProcessor(image_size))

class GroundingImageProcessor:
    def __init__(self, image_size=512, mean=None, std=None):
        super().__init__()
        self.image_size = (image_size, image_size)
        if mean is None:
            mean = (123.675, 116.28, 103.53)
        if std is None:
            std = (58.395, 57.12, 57.375)
        self.normalize = transforms.Normalize(mean, std)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])
        
    def process(self, image):
        def get_size_with_aspect_ratio(image_size, size):
            w, h = image_size
            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
            return (oh, ow)

        def get_size(image_size, size):
            if isinstance(size, (list, tuple)):
                return size[::-1]
            else:
                return get_size_with_aspect_ratio(image_size, size)
        
        # COnvert PIL image to no array and check shape
        import numpy as np
        # print(f'\n\nImage shape:', np.array(image).shape) # Image shape: (1211, 2557, 3)
        size = get_size(image.size, self.image_size)
        # print(f'\n\nSize:', size) # Size: (512, 512)
        rescaled_image = F.resize(image, size)
        # print(f'\n\nRescaled Image shape:', np.array(rescaled_image).shape) # Rescaled Image shape: (512, 512, 3)
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
        normalized_image = self.transform(rescaled_image)
        # print(f'\n\nNormalized Image shape:', normalized_image.shape) # Normalized Image shape: torch.Size([3, 512, 512])
        return normalized_image, ratios

def get_grounding_image_processor(image_size):
    return GroundingImageProcessor(image_size)