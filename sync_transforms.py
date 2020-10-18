import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomScale(object):
    def __init__(self, base_size, crop_size, resize_scale_range):
        self.base_size = base_size
        self.crop_size = crop_size
        self.resize_scale_range = resize_scale_range

    def __call__(self, img, mask):
        w, h = img.size
        # print("img.size:", img.size)
        short_size = random.randint(int(self.base_size * self.resize_scale_range[0]),
                                    int(self.base_size * self.resize_scale_range[1]))
        # print("short_size:", short_size)
        #         if h > w:
        #             ow = short_size
        #             oh = int(1.0 * h * ow / w)
        #         else:
        #             oh = short_size
        #             ow = int(1.0 * w * oh / h)
        ow, oh = short_size, short_size
        # print("ow, oh = ", ow, oh)
        img, mask = img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        img = np.array(img)
        mask = np.array(mask)
        num_crop = 0
        while num_crop < 5:
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            endx = x + self.crop_size
            endy = y + self.crop_size
            patch = img[y:endy, x:endx]
            if (patch == 0).all():
                continue
            else:
                break
        img = img[y:endy, x:endx]
        mask = mask[y:endy, x:endx]
        img, mask = Image.fromarray(img), Image.fromarray(mask)
        return img, mask


class RandomFlip(object):
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, img, mask):
        if random.random() < self.flip_ratio:
            img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img, mask = img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask


class RandomGaussianBlur(object):
    def __init__(self, prop):
        self.prop = prop
    def __call__(self, img, mask, prop):
        if random.random() < self.prop:
            img = img.filter(ImageFilter.GaussianBlur)(radius=random.random())
        return img, mask

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)
