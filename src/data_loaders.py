import os
import gc
import glob
import math
import pickle

from tqdm import tqdm
import hydra
import cv2
import numpy as np
from shapely.geometry import Polygon
import torch
from torch.utils.data import Dataset, DataLoader
import imgaug
import imgaug as ia
import imgaug.augmenters as iaa
import pyclipper

import db_transforms
from utils import dict_to_device, minmax_scaler_img


class BaseDatasetIter(Dataset):
    def __init__(self,
                 train_dir,
                 train_gt_dir,
                 ignore_tags,
                 is_training=True,
                 image_size=640,
                 min_text_size=8,
                 shrink_ratio=0.4,
                 thresh_min=0.3,
                 thresh_max=0.7,
                 augment=None,
                 mean=[103.939, 116.779, 123.68],
                 debug=False,
                 train_otf=False,
                 **kwargs):

        self.train_dir = train_dir
        self.train_gt_dir = train_gt_dir
        self.ignore_tags = ignore_tags

        self.is_training = is_training
        self.image_size = image_size
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.augment = augment
        if self.augment is None:
            # self.augment = self._get_default_augment()
            self.augment = self.get_custom_augment()
            # print(self.augment)
            # self.crop_augment, self.augment = self._get_default_augment()

        self.mean = mean
        self.debug = debug
        self.train_otf = train_otf

        # load metadata
        self.image_paths, self.gt_paths = self.load_metadata(
            train_dir, train_gt_dir)

        if not self.train_otf:
            # load annotation
            self.all_anns = self.load_all_anns(self.gt_paths)
            assert len(self.image_paths) == len(self.all_anns)
        else:
            assert len(self.image_paths) == len(self.wordBB)

    def _get_default_augment(self):
        augment_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-10, 10)),
            iaa.Resize((0.5, 3.0))
        ])
        return augment_seq

    def get_custom_augment(self):

        augment_seq = iaa.Sequential(
            [
                iaa.Fliplr(p=0.1),
                iaa.Flipud(p=0.1),
                iaa.Grayscale(alpha=0.5),
                iaa.GaussianBlur(sigma=(0, 1.0)),
                iaa.LinearContrast((0.5, 1.5)),
                iaa.ChannelShuffle(0.3, channels=[0, 1, 2]),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.25),
                iaa.Multiply((0.7, 1.3), per_channel=0.25),
                iaa.JpegCompression(compression=(0, 25)),
                iaa.Invert(p=0.1),
                iaa.Affine(
                    scale={
                        "x": (0.95, 1.05),
                        "y": (0.95, 1.05)
                    },
                    translate_percent={
                        "x": (-0.1, 0.1),
                        "y": (-0.1, 0.1)
                    },
                    shear=(-10, 10),
                    order=[0, 1],
                    cval=(0, 255),
                    rotate=(-5, 5),
                    #         rotate = (-90, 90),
                    #         mode = ia.ALL,
                    mode="constant"),
                iaa.PerspectiveTransform(scale=(0.0, 0.1)),
                iaa.SomeOf((0.75), [
                    iaa.Dropout(p=(0.0, 0.05)),
                    iaa.CoarseDropout(
                        p=(0.005, 0.05), per_channel=True, size_percent=0.02),
                    iaa.OneOf([
                        iaa.Rot90(k=1),
                        iaa.Rot90(k=2),
                        iaa.Rot90(k=3),
                    ]),
                    iaa.Superpixels(p_replace=(0, 0.5), n_segments=(20, 50)),
                    iaa.MotionBlur(k=(3, 5)),
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 0.5)),
                        iaa.AverageBlur(k=(2, 5)),
                        iaa.MedianBlur(k=(3, 9))
                    ]),
                    iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)),
                    iaa.Emboss(alpha=(0, 0.5), strength=(0, 2.0)),
                    iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.7)),
                        iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0, 1.0))
                    ]),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.2), per_channel=0.5),
                        iaa.CoarseDropout((0.01, 0.2),
                                          size_percent=(0.02, 0.05),
                                          per_channel=0.2)
                    ]),
                    iaa.Invert(0.05, per_channel=True),
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
                    iaa.PiecewiseAffine(scale=(0.01, 0.05))
                ],
                           random_order=True),
                iaa.OneOf([
                    iaa.PadToFixedSize(width=i, height=i)
                    for i in range(200, 1800, 200)
                ]),
                iaa.OneOf([
                    iaa.CropToFixedSize(width=i, height=i)
                    for i in range(200, 1800, 200)
                ]),
                # iaa.CropToFixedSize(width=800, height=800),
                # iaa.Resize((0.3, 3.0))
            ],
            random_order=False)

        return augment_seq
    
    def remove_small_poly(self, anns):
        new_anns = []
        hs = []
        for ann in anns:
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])
            polygon = Polygon(poly)

            thresh = 5.0
            if height < thresh or width < thresh or not polygon.buffer(
                    0).is_valid or polygon.area < 10:
                continue

            hs.append(height)
            new_anns.append(ann)

        hs = np.mean(hs)
        return new_anns, hs

    def remove_lowh_poly(self, anns, hmean):
        new_anns = []
        for ann in anns:
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            if height >= hmean / 3:
                new_anns.append(ann)
        return new_anns
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        # image_path = self.image_paths[index]
        # anns = self.all_anns[index]

        image_path = self.image_paths[index]
        if self.train_otf:
            anns = self.wordBB[index]
            # customize for synthtext dataset
            anns = self.load_ann(anns)
        else:
            anns = self.all_anns[index]

        if self.debug:
            print(image_path)
            print(len(anns))

        img = cv2.imread(image_path)[:, :, ::-1]
        if self.is_training and self.augment is not None:
            augment_seq = self.augment.to_deterministic()
            img, anns = db_transforms.transform(augment_seq, img, anns)
            img, anns = db_transforms.crop(img, anns)

        img, anns = db_transforms.resize(self.image_size, img, anns)
        anns, hmean = self.remove_small_poly(anns)
        anns = self.remove_lowh_poly(anns, hmean)
        
        # anns = [ann for ann in anns if Polygon(ann['poly']).buffer(0).is_valid]
        gt = np.zeros((self.image_size, self.image_size),
                      dtype=np.float32)  # batch_gts
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        thresh_map = np.zeros((self.image_size, self.image_size),
                              dtype=np.float32)  # batch_thresh_maps
        # batch_thresh_masks
        thresh_mask = np.zeros((self.image_size, self.image_size),
                               dtype=np.float32)

        if self.debug:
            print(type(anns), len(anns))

        ignore_tags = []
        for ann in anns:
            # i.e shape = (4, 2) / (6, 2) / ...
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])
            polygon = Polygon(poly)

            # generate gt and mask
            if polygon.area < 1 or \
                    min(height, width) < self.min_text_size or \
                    ann['text'] in self.ignore_tags:
                ignore_tags.append(True)
                cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            else:
                # 6th equation
                distance = polygon.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                subject = [tuple(_l) for _l in ann['poly']]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)

                if len(shrinked) == 0:
                    ignore_tags.append(True)
                    cv2.fillPoly(mask,
                                 poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    if shrinked.shape[0] > 2 and \
                            Polygon(shrinked).buffer(0).is_valid:
                        ignore_tags.append(False)
                        cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                    else:
                        ignore_tags.append(True)
                        cv2.fillPoly(mask,
                                     poly.astype(np.int32)[np.newaxis, :, :],
                                     0)
                        continue

            # generate thresh map and thresh mask
            db_transforms.draw_thresh_map(ann['poly'],
                                          thresh_map,
                                          thresh_mask,
                                          shrink_ratio=self.shrink_ratio)

        thresh_map = thresh_map * \
            (self.thresh_max - self.thresh_min) + self.thresh_min

        img = img.astype(np.float32)
        img[..., 0] -= self.mean[0]
        img[..., 1] -= self.mean[1]
        img[..., 2] -= self.mean[2]

        img = np.transpose(img, (2, 0, 1))

        data_return = {
            "image_path": image_path,
            "img": img,
            "prob_map": gt,
            "supervision_mask": mask,
            "thresh_map": thresh_map,
            "text_area_map": thresh_mask,
        }
        # for batch_size = 1
        if not self.is_training:
            data_return["anns"] = [ann['poly'] for ann in anns]
            data_return["ignore_tags"] = ignore_tags

        # return image_path, img, gt, mask, thresh_map, thresh_mask
        return data_return


class TotalTextDatasetIter(BaseDatasetIter):
    def __init__(self, train_dir, train_gt_dir, ignore_tags, **kwargs):
        super().__init__(train_dir, train_gt_dir, ignore_tags, **kwargs)

    def load_metadata(self, img_dir, gt_dir):
        img_fps = sorted(glob.glob(os.path.join(img_dir, "*")))
        gt_fps = []
        for img_fp in img_fps:
            img_id = img_fp.split("/")[-1].replace("img", "").split(".")[0]
            gt_fn = "gt_img{}.txt".format(img_id)
            gt_fp = os.path.join(gt_dir, gt_fn)
            assert os.path.exists(img_fp)
            gt_fps.append(gt_fp)
        assert len(img_fps) == len(gt_fps)

        return img_fps, gt_fps

    def load_all_anns(self, gt_paths):
        res = []
        for gt in gt_paths:
            lines = []
            reader = open(gt, 'r').readlines()
            for line in reader:
                item = {}
                parts = line.strip().split(',')
                label = parts[-1]
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                num_points = math.floor((len(line) - 1) / 2) * 2
                poly = np.array(list(map(float, line[:num_points]))).reshape(
                    (-1, 2)).tolist()
                if len(poly) < 3:
                    continue
                item['poly'] = poly
                item['text'] = label
                # {'poly': [[184.0, 293.0], [222.0, 269.0], [273.0, 270.0], [269.0, 296.0], [230.0, 297.0], [202.0, 317.0]], 'text': 'alpaca'}
                lines.append(item)
            res.append(lines)
        return res


class CTW1500DatasetIter(BaseDatasetIter):
    def __init__(self, train_dir, train_gt_dir, ignore_tags, **kwargs):

        super().__init__(train_dir, train_gt_dir, ignore_tags, **kwargs)

    def load_metadata(self, img_dir, gt_dir):
        img_fps = sorted(glob.glob(os.path.join(img_dir, "*")))
        gt_fps = []
        for img_fp in img_fps:
            img_id = img_fp.split("/")[-1][:-4]
            gt_fn = "{}.txt".format(img_id)
            gt_fp = os.path.join(gt_dir, gt_fn)
            assert os.path.exists(img_fp)
            gt_fps.append(gt_fp)
        assert len(img_fps) == len(gt_fps)

        return img_fps, gt_fps

    def load_all_anns(self, gt_fps):
        """
        Reference: https://github.com/whai362/PSENet/blob/master/dataset/ctw1500_loader.py
        """
        res = []
        for gt_fp in gt_fps:
            lines = []
            with open(gt_fp, 'r') as f:
                for line in f:
                    item = {}
                    gt = line.strip().strip('\ufeff').strip('\xef\xbb\xbf')
                    gt = list(map(int, gt.split(',')))

                    x1 = np.int(gt[0])
                    y1 = np.int(gt[1])
                    bbox = [np.int(gt[i]) for i in range(4, 32)]
                    bbox = np.asarray(bbox) + ([x1, y1] * 14)
                    bbox = bbox.reshape(-1, 2).tolist()
                    item['poly'] = bbox
                    item['text'] = 'True'
                    lines.append(item)
            res.append(lines)
        return res


class ICDAR2015DatasetIter(BaseDatasetIter):
    def __init__(self, train_dir, train_gt_dir, ignore_tags, **kwargs):
        super().__init__(train_dir, train_gt_dir, ignore_tags, **kwargs)

    def load_metadata(self, img_dir, gt_dir):
        img_fps = glob.glob(os.path.join(img_dir, "*"))
        gt_fps = []
        for img_fp in img_fps:
            img_id = img_fp.split("/")[-1].split(".")[0]
            gt_fn = "gt_{}.txt".format(img_id)
            gt_fp = os.path.join(gt_dir, gt_fn)
            assert os.path.exists(img_fp)
            gt_fps.append(gt_fp)
        assert len(img_fps) == len(gt_fps)

        return img_fps, gt_fps

    def load_all_anns(self, gt_fps):
        res = []
        for gt_fp in gt_fps:
            lines = []
            with open(gt_fp, 'r') as f:
                for line in f:
                    item = {}
                    gt = line.strip().strip('\ufeff').strip(
                        '\xef\xbb\xbf').split(",")
                    label = ",".join(gt[8:])
                    if label == "###":
                        continue
                    poly = list(map(int, gt[:8]))
                    poly = np.asarray(poly).reshape(-1, 2).tolist()
                    item['poly'] = poly
                    item['text'] = label
                    lines.append(item)
            res.append(lines)
        return res


class MSRATD500DatasetIter(BaseDatasetIter):
    def __init__(self, train_dir, train_gt_dir, ignore_tags, **kwargs):
        super().__init__(train_dir, train_gt_dir, ignore_tags, **kwargs)

    def transform_four_points(self, points, center_point, theta):
        """Reference: https://stackoverflow.com/questions/622140
        """
        theta = -theta
        new_coords = []
        x_center, y_center = center_point

        for point in points:
            x, y = point
            x_new = x_center + (x - x_center) * np.cos(theta) + \
                (y - y_center) * np.sin(theta)
            y_new = y_center - (x - x_center) * np.sin(theta) + \
                (y - y_center) * np.cos(theta)
            x_new = int(x_new)
            y_new = int(y_new)
            new_coords.append((x_new, y_new))
        return new_coords

    def load_metadata(self, img_dir, gt_dir=None):
        # ignore gt_dir
        img_fps = sorted(glob.glob(os.path.join(img_dir, "*.JPG")))
        gt_fps = sorted(glob.glob(os.path.join(img_dir, "*.gt")))
        assert len(img_fps) == len(gt_fps)

        return img_fps, gt_fps

    def load_all_anns(self, gt_fps):
        res = []
        for gt_fp in gt_fps:
            lines = []
            with open(gt_fp, 'r') as f:
                for line in f:
                    item = {}
                    line = list(map(float, line.strip().split()))
                    index, dif, x_min, y_min, w, h, theta = line
                    if int(dif) == 1:  # difficult label
                        continue

                    c1 = (x_min, y_min)
                    c2 = (x_min + w, y_min)
                    c3 = (x_min + w, y_min + h)
                    c4 = (x_min, y_min + h)
                    center = (x_min + w / 2, y_min + h / 2)
                    rot_box = self.transform_four_points([c1, c2, c3, c4],
                                                         center, theta)
                    rot_box = np.array(rot_box).tolist()

                    item['poly'] = rot_box
                    item['text'] = 'True'
                    lines.append(item)
            res.append(lines)
        return res


class SynthTextDatasetIter(BaseDatasetIter):
    def __init__(self, train_dir, train_gt_dir=None, ignore_tags=[], **kwargs):

        self.dump_dir = kwargs.get("dump_dir")
        self.no_imgs = kwargs.get("no_imgs")
        self.train_otf = kwargs.get("train_otf")

        self.train_dir = train_dir
        with open(os.path.join(self.dump_dir, "train_imnames.pkl"),
                  "rb") as f:
            self.imnames = pickle.load(f)[:self.no_imgs]

        with open(os.path.join(self.dump_dir, "train_wordBB.pkl"),
                  "rb") as f:
            self.wordBB = pickle.load(f)[:self.no_imgs]

        # shuffle data
        print("Shuffling data...")
        indices = np.arange(len(self.imnames))
        self.imnames = np.array(self.imnames)
        print(type(self.wordBB))
        np.random.shuffle(indices)
        self.imnames = self.imnames[indices]
        self.wordBB = self.wordBB[indices]

        gc.collect()
        print("Number of images: {}".format(len(self.imnames)))
        print("Training on the fly: {}".format(self.train_otf))
        super().__init__(train_dir, train_gt_dir, ignore_tags, **kwargs)

    def load_metadata(self, img_dir, gt_dir=None):
        # ignore gt_dir
        img_fps = [os.path.join(self.train_dir, fn[0]) for fn in self.imnames]
        assert os.path.exists(img_fps[0])

        return img_fps, True

    def load_ann(self, anns):
        lines = []
        if len(anns.shape) == 2:
            anns = np.expand_dims(anns, axis=-1)

        anns = anns.transpose(2, 1, 0)
        for ann in anns:
            item = {}
            item["poly"] = ann.tolist()
            item["text"] = ""
            lines.append(item)
        return lines

    def load_all_anns(self, gt_fps):
        res = []
        for img_obj in tqdm(self.wordBB):
            lines = []
            if len(img_obj.shape) == 2:
                img_obj = np.expand_dims(img_obj, axis=-1)
            img_gts = img_obj.transpose(2, 1, 0)
            for img_gt in img_gts:
                item = {}
                item["poly"] = img_gt.tolist()
                item["text"] = ''
                lines.append(item)
            res.append(lines)

        return res


class CustomDatasetIter(BaseDatasetIter):
    def __init__(self, train_dir, train_gt_dir=None, ignore_tags=[], **kwargs):

        self.no_imgs = kwargs.get("no_imgs")
        self.train_otf = kwargs.get("train_otf")
        self.mode = kwargs.get("mode")
        self.train_gt_dir = train_gt_dir

        self.train_dir = train_dir  # img folder
        print("Loading {} dataset...".format(self.mode))
        imnames_fp = os.path.join(self.train_gt_dir,
                                  "{}_imnames.pkl".format(self.mode))
        wordBB_fp = os.path.join(self.train_gt_dir,
                                 "{}_wordBB.pkl".format(self.mode))
        assert os.path.exists(imnames_fp)
        assert os.path.exists(wordBB_fp)

        if self.mode == "test":
            self.no_imgs = 50
        with open(imnames_fp, "rb") as f:
            self.imnames = pickle.load(f)[:self.no_imgs]
        with open(wordBB_fp, "rb") as f:
            self.wordBB = pickle.load(f)[:self.no_imgs]

        # self.crop_augment, self.augment = self._get_default_augment()

        gc.collect()
        print("Number of images: {}".format(len(self.imnames)))
        print("Training on the fly: {}".format(self.train_otf))
        super().__init__(train_dir, train_gt_dir, ignore_tags, **kwargs)

    def _get_default_augment(self):

        if self.mode == "train":
            augment_seq = iaa.Sequential(
                [
                    iaa.Fliplr(p=0.05),
                    iaa.Flipud(p=0.05),
                    iaa.Grayscale(alpha=0.5),
                    iaa.GaussianBlur(sigma=(0, 1.0)),
                    iaa.LinearContrast((0.5, 1.5)),
                    iaa.ChannelShuffle(0.3, channels=[0, 1, 2]),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.25),
                    iaa.Multiply((0.7, 1.3), per_channel=0.25),
                    iaa.JpegCompression(compression=(0, 25)),
                    iaa.Invert(p=0.1),
                    iaa.Affine(
                        scale={
                            "x": (0.95, 1.05),
                            "y": (0.95, 1.05)
                        },
                        translate_percent={
                            "x": (-0.1, 0.1),
                            "y": (-0.1, 0.1)
                        },
                        shear=(-10, 10),
                        order=[0, 1],
                        cval=(0, 255),
                        rotate=(-5, 5),
                        #         rotate = (-90, 90),
                        #         mode = ia.ALL,
                        mode="constant"),
                    iaa.PerspectiveTransform(scale=(0.0, 0.1)),
                    iaa.SomeOf((0.75), [
                        iaa.Dropout(p=(0.0, 0.05)),
                        iaa.CoarseDropout(p=(0.005, 0.05),
                                          per_channel=True,
                                          size_percent=0.02),
                        iaa.OneOf([
                            iaa.Rot90(k=1),
                            iaa.Rot90(k=2),
                            iaa.Rot90(k=3),
                        ]),
                        iaa.Superpixels(p_replace=(0, 0.5),
                                        n_segments=(20, 50)),
                        iaa.MotionBlur(k=(3, 5)),
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 0.5)),
                            iaa.AverageBlur(k=(2, 5)),
                            iaa.MedianBlur(k=(3, 9))
                        ]),
                        iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)),
                        iaa.Emboss(alpha=(0, 0.5), strength=(0, 2.0)),
                        iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0, 0.7)),
                            iaa.DirectedEdgeDetect(alpha=(0, 0.7),
                                                   direction=(0, 1.0))
                        ]),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.2), per_channel=0.5),
                            iaa.CoarseDropout((0.01, 0.2),
                                              size_percent=(0.02, 0.05),
                                              per_channel=0.2)
                        ]),
                        iaa.Invert(0.05, per_channel=True),
                        iaa.Add((-10, 10), per_channel=0.5),
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                        iaa.ElasticTransformation(alpha=(0.5, 3.5),
                                                  sigma=0.25),
                        iaa.PiecewiseAffine(scale=(0.01, 0.05))
                    ],
                               random_order=True),
                    iaa.OneOf([
                        iaa.PadToFixedSize(width=i, height=i)
                        for i in range(200, 1800, 200)
                    ]),
                    iaa.OneOf([
                        iaa.CropToFixedSize(width=i, height=i)
                        for i in range(200, 1800, 200)
                    ]),
                    # iaa.CropToFixedSize(width=800, height=800),
                    # iaa.Resize((0.3, 3.0))
                ],
                random_order=False)
        else:
            augment_seq = iaa.Sequential([
                iaa.CropToFixedSize(width=960, height=960),
                iaa.Resize((0.7, 1.3))
            ])

        # crop_augment_seq = iaa.Sequential([
        #     iaa.Fliplr(p=0.05),
        #     iaa.Flipud(p=0.05),
        #     iaa.Grayscale(alpha=0.5),
        #     iaa.GaussianBlur(sigma=(0, 1.0)),
        #     iaa.LinearContrast((0.5, 1.5)),
        #     iaa.ChannelShuffle(0.3, channels=[0, 1, 2]),
        #     iaa.AdditiveGaussianNoise(loc=0,
        #                               scale=(0.0, 0.05 * 255),
        #                               per_channel=0.25),
        #     iaa.Multiply((0.7, 1.3), per_channel=0.25),
        #     iaa.JpegCompression(compression=(0, 25)),
        #     iaa.Invert(p=0.1),
        #     iaa.Affine(scale={
        #         "x": (0.95, 1.05),
        #         "y": (0.95, 1.05)
        #     },
        #                translate_percent={
        #                    "x": (-0.1, 0.1),
        #                    "y": (-0.1, 0.1)
        #                },
        #                shear=(-10, 10),
        #                order=[0, 1],
        #                cval=(0, 255),
        #                rotate=(-5, 5),
        #                mode="constant"),
        #     iaa.PerspectiveTransform(scale=(0.0, 0.06)),
        #     iaa.OneOf(
        #         iaa.CropToFixedSize(width=i, height=i)
        #         for i in range(200, 1800, 200)),
        # ])

        # augment_seq = iaa.Sequential([
        #     iaa.SomeOf((0.9), [
        #         iaa.Dropout(p=(0.0, 0.05)),
        #         iaa.CoarseDropout(
        #             p=(0.005, 0.05), per_channel=True, size_percent=0.02),
        #         iaa.OneOf([
        #             iaa.Rot90(k=1),
        #             iaa.Rot90(k=2),
        #             iaa.Rot90(k=3),
        #         ]),
        #         iaa.Superpixels(p_replace=(0, 0.5), n_segments=(20, 50)),
        #         iaa.MotionBlur(k=(3, 9)),
        #         iaa.OneOf([
        #             iaa.GaussianBlur((0, 0.5)),
        #             iaa.AverageBlur(k=(2, 5)),
        #             iaa.MedianBlur(k=(3, 9))
        #         ]),
        #         iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)),
        #         iaa.Emboss(alpha=(0, 0.5), strength=(0, 2.0)),
        #         iaa.OneOf([
        #             iaa.EdgeDetect(alpha=(0, 0.7)),
        #             iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0, 1.0))
        #         ]),
        #         iaa.AdditiveGaussianNoise(
        #             loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        #         iaa.OneOf([
        #             iaa.Dropout((0.01, 0.2), per_channel=0.5),
        #             iaa.CoarseDropout((0.01, 0.2),
        #                               size_percent=(0.02, 0.05),
        #                               per_channel=0.2)
        #         ]),
        #         iaa.Invert(0.05, per_channel=True),
        #         iaa.Add((-10, 10), per_channel=0.5),
        #         iaa.Multiply((0.5, 1.5), per_channel=0.5),
        #         iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
        #         iaa.Grayscale(alpha=(0.0, 1.0)),
        #         iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
        #         iaa.PiecewiseAffine(scale=(0.01, 0.05))
        #     ],
        #                random_order=True),
        #     iaa.OneOf(
        #         iaa.PadToFixedSize(width=i, height=i)
        #         for i in range(200, 2600, 200)),
        # ],
        #                              random_order=False)

        # return crop_augment_seq, augment_seq
        return augment_seq

    def load_metadata(self, img_dir, gt_dir=None):
        # ignore gt_dir
        img_fps = []
        for fn in self.imnames:
            fn = fn.split("/")[-1]
            img_fp = os.path.join(self.train_dir, fn)
            img_fps.append(img_fp)

        assert os.path.exists(img_fps[0])
        gc.collect()

        return img_fps, True

    def load_ann(self, anns):
        lines = []
        if len(anns.shape) == 2:
            anns = np.expand_dims(anns, axis=-1)

        anns = anns.transpose(2, 1, 0)
        for ann in anns:
            item = {}
            item["poly"] = ann.tolist()
            item["text"] = ""
            lines.append(item)
        return lines

    def load_all_anns(self, gt_fps):
        """REMOVED"""
        res = []
        for img_obj in tqdm(self.wordBB):
            lines = []
            if len(img_obj.shape) == 2:
                img_obj = np.expand_dims(img_obj, axis=-1)
            img_gts = img_obj.transpose(2, 1, 0)
            for img_gt in img_gts:
                item = {}
                item["poly"] = img_gt.tolist()
                item["text"] = ''
                lines.append(item)
            res.append(lines)

        del self.wordBB
        gc.collect()
        return res

    def remove_small_poly(self, anns):
        new_anns = []
        hs = []
        for ann in anns:
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])
            polygon = Polygon(poly)

            thresh = 3.0
            if height < thresh or width < thresh or not polygon.buffer(
                    0).is_valid or polygon.area < 10:
                continue

            hs.append(height)
            new_anns.append(ann)

        hs = np.mean(hs)
        return new_anns, hs

    def remove_lowh_poly(self, anns, hmean):
        new_anns = []
        for ann in anns:
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            if height >= hmean / 4:
                new_anns.append(ann)
        return new_anns

    def crop_transform(self, aug, image, anns):
        image_shape = image.shape
        image = aug.augment_image(image)
        new_anns = []
        for ann in anns:
            keypoints = [imgaug.Keypoint(p[0], p[1]) for p in ann['poly']]
            keypoints = aug.augment_keypoints(
                [imgaug.KeypointsOnImage(keypoints,
                                         shape=image_shape)])[0].keypoints
            poly = [(min(max(0, p.x), image.shape[1] - 1),
                     min(max(0, p.y), image.shape[0] - 1)) for p in keypoints]
            new_ann = {'poly': poly, 'text': ann['text']}
            new_anns.append(new_ann)
        return image, new_anns

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        if self.train_otf:
            anns = self.wordBB[index]
            anns = self.load_ann(anns)
        else:
            anns = self.all_anns[index]

        if self.debug:
            print(image_path)
            print(len(anns))

        img = cv2.imread(image_path)[:, :, ::-1]
        # if self.is_training and self.augment is not None:
        if self.augment is not None:
            augment_seq = self.augment.to_deterministic()
            img, anns = db_transforms.transform(augment_seq, img, anns)
            img, anns = db_transforms.crop(img, anns)

        # if self.augment is not None:
        #     crop_augment_seq = self.crop_augment.to_deterministic()
        #     augment_seq = self.augment.to_deterministic()

        #     crop_img, crop_anns = self.crop_transform(crop_augment_seq, img,
        #                                               anns)
        #     img, anns = db_transforms.transform(augment_seq, crop_img,
        #                                         crop_anns)
        #     img, anns = db_transforms.crop(img, anns)

        img, anns = db_transforms.resize(self.image_size, img, anns)

        anns, hmean = self.remove_small_poly(anns)
        anns = self.remove_lowh_poly(anns, hmean)

        gt = np.zeros((self.image_size, self.image_size),
                      dtype=np.float32)  # batch_gts
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        thresh_map = np.zeros((self.image_size, self.image_size),
                              dtype=np.float32)  # batch_thresh_maps
        # batch_thresh_masks
        thresh_mask = np.zeros((self.image_size, self.image_size),
                               dtype=np.float32)

        if self.debug:
            print(type(anns), len(anns))

        ignore_tags = []
        for ann in anns:
            # i.e shape = (4, 2) / (6, 2) / ...
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])
            polygon = Polygon(poly)

            # generate gt and mask
            if polygon.area < 1 or \
                    min(height, width) < self.min_text_size or \
                    ann['text'] in self.ignore_tags:
                ignore_tags.append(True)
                cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            else:
                # 6th equation
                distance = polygon.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                subject = [tuple(_l) for _l in ann['poly']]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)

                if len(shrinked) == 0:
                    ignore_tags.append(True)
                    cv2.fillPoly(mask,
                                 poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    if shrinked.shape[0] > 2 and \
                            Polygon(shrinked).buffer(0).is_valid:
                        ignore_tags.append(False)
                        cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                    else:
                        ignore_tags.append(True)
                        cv2.fillPoly(mask,
                                     poly.astype(np.int32)[np.newaxis, :, :],
                                     0)
                        continue

            # generate thresh map and thresh mask
            db_transforms.draw_thresh_map(ann['poly'],
                                          thresh_map,
                                          thresh_mask,
                                          shrink_ratio=self.shrink_ratio)

        thresh_map = thresh_map * \
            (self.thresh_max - self.thresh_min) + self.thresh_min

        img = img.astype(np.float32)
        img[..., 0] -= self.mean[0]
        img[..., 1] -= self.mean[1]
        img[..., 2] -= self.mean[2]

        img = np.transpose(img, (2, 0, 1))

        data_return = {
            "image_path": image_path,
            "img": img,
            "prob_map": gt,
            "supervision_mask": mask,
            "thresh_map": thresh_map,
            "text_area_map": thresh_mask,
        }
        # for batch_size = 1
        if not self.is_training:
            data_return["anns"] = [ann['poly'] for ann in anns]
            data_return["ignore_tags"] = ignore_tags

        # return image_path, img, gt, mask, thresh_map, thresh_mask
        return data_return


@hydra.main(config_path="../config.yaml", strict=False)
def run(cfg):
    dataset_name = cfg.dataset.name
    ignore_tags = cfg.data[dataset_name].ignore_tags
    train_dir = cfg.data[dataset_name].train_dir
    train_gt_dir = cfg.data[dataset_name].train_gt_dir

    if dataset_name == 'totaltext':
        TextDatasetIter = TotalTextDatasetIter
    elif dataset_name == 'ctw1500':
        TextDatasetIter = CTW1500DatasetIter
    elif dataset_name == 'icdar2015':
        TextDatasetIter = ICDAR2015DatasetIter
    elif dataset_name == 'msra_td500':
        TextDatasetIter = MSRATD500DatasetIter
    else:
        raise NotImplementedError("Pls provide valid dataset name!")
    train_iter = TextDatasetIter(train_dir,
                                 train_gt_dir,
                                 ignore_tags,
                                 is_training=True,
                                 debug=False)
    train_loader = DataLoader(dataset=train_iter,
                              batch_size=1,
                              shuffle=True,
                              num_workers=1)
    samples = next(iter(train_loader))
    samples = dict_to_device(samples, device='cpu')
    for k, v in samples.items():
        if isinstance(v, torch.Tensor):
            print(samples[k].device)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(minmax_scaler_img(samples['img'][0].numpy().transpose(1, 2, 0)))
    plt.imshow(samples['prob_map'][0], cmap='jet', alpha=0.35)
    plt.imshow(samples['thresh_map'][0], cmap='jet', alpha=0.5)
    # plt.imshow(samples['text_area_map'][0], cmap='jet', alpha=0.5)
    # plt.imshow(samples['supervision_mask'][0], cmap='jet', alpha=0.5)
    plt.savefig(os.path.join(cfg.meta.root_dir, 'tmp/foo.jpg'),
                bbox_inches='tight')


if __name__ == '__main__':
    run()
