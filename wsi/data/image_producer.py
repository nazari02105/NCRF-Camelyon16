import os

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import copy

np.random.seed(0)

from torchvision import transforms  # noqa

from wsi.data.annotation import Annotation  # noqa


class GridImageDataset(Dataset):
    """
    Data producer that generate a square grid, e.g. 3x3, of patches and their
    corresponding labels from pre-sampled images.
    """
    def __init__(self, data_path, json_path, img_size, patch_size,
                 crop_size=224, normalize=True, augmentation=False):
        """
        Initialize the data producer.

        Arguments:
            data_path: string, path to pre-sampled images using patch_gen.py
            json_path: string, path to the annotations in json format
            img_size: int, size of pre-sampled images, e.g. 768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
        """
        self._data_path = data_path
        self._json_path = json_path
        self._img_size = img_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._augmentation = augmentation
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self._preprocess()

    def _preprocess(self):
        if self._img_size % self._patch_size != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(self._img_size, self._patch_size))

        self._patch_per_side = self._img_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side

        self._pids = list(map(lambda x: x.strip('.json').lower(),
                              os.listdir(self._json_path)))

        self._annotations = {}
        for pid in self._pids:
            pid_json_path = os.path.join(self._json_path, pid.capitalize() + '.json')
            anno = Annotation()
            anno.from_json(pid_json_path)
            self._annotations[pid] = anno

        self._coords = []
        f = open(os.path.join(self._data_path, 'list.txt'))
        for line in f:
            pid, x_center, y_center = line.strip('\n').split(',')[0:3]
            x_center, y_center = int(x_center), int(y_center)
            self._coords.append((pid, x_center, y_center))
        f.close()

        self._num_image = len(self._coords)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        pid, x_center, y_center = self._coords[idx]

        x_top_left = int(x_center - self._img_size / 2)
        y_top_left = int(y_center - self._img_size / 2)

        # the grid of labels for each patch
        label_grid = np.zeros((self._patch_per_side, self._patch_per_side),
                              dtype=np.float32)
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # (x, y) is the center of each patch
                x = x_top_left + int((x_idx + 0.5) * self._patch_size)
                y = y_top_left + int((y_idx + 0.5) * self._patch_size)

                if self._annotations[pid].inside_polygons((x, y), True):
                    label = 1
                else:
                    label = 0

                # extracted images from WSI is transposed with respect to
                # the original WSI (x, y)
                label_grid[y_idx, x_idx] = label

        img = Image.open(os.path.join(self._data_path, '{}.png'.format(idx)))
        img = img.convert("L")

        # color jitter
        img = self._color_jitter(img)

        if self._augmentation:
            if idx % 3 == 0:
                # use left_right flip
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                label_grid = np.fliplr(label_grid)
            elif idx % 3 == 1:
                # use rotate
                num_rotate = 1
                img = img.rotate(90 * num_rotate)
                label_grid = np.rot90(label_grid, num_rotate)

        # PIL image:   H x W x C
        # torch image: C X H X W
        img = np.array(img, dtype=np.uint8)

        img = cv2.equalizeHist(img)

        # flatten the square grid
        img_flat = np.zeros(
            (self._grid_size, 1, self._crop_size, self._crop_size),
            dtype=np.float32)
        label_flat = np.zeros(self._grid_size, dtype=np.float32)

        idx = 0
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # center crop each patch
                x_start = int(
                    (x_idx + 0.5) * self._patch_size - self._crop_size / 2)
                x_end = x_start + self._crop_size
                y_start = int(
                    (y_idx + 0.5) * self._patch_size - self._crop_size / 2)
                y_end = y_start + self._crop_size
                img_flat[idx] = img[x_start:x_end, y_start:y_end]
                label_flat[idx] = label_grid[x_idx, y_idx]

                idx += 1

        if self._augmentation and idx % 3 == 2:
            first_list = [0, 1, 2, 3]
            second_list = [8, 7, 6, 5]
            for i in range(len(first_list)):
                size = img_flat[0].shape[1]
                first_number = first_list[i]
                second_number = second_list[i]
                temp = copy.deepcopy(img_flat[first_number])
                img_flat[first_number][:, int(1 / 8 * size):int(7 / 8 * size), int(1 / 8 * size):int(7 / 8 * size)] = \
                img_flat[second_number][:, int(1 / 8 * size):int(7 / 8 * size), int(1 / 8 * size):int(7 / 8 * size)]
                img_flat[second_number][:, int(1 / 8 * size):int(7 / 8 * size), int(1 / 8 * size):int(7 / 8 * size)] = \
                    temp[:, int(1 / 8 * size):int(7 / 8 * size), int(1 / 8 * size):int(7 / 8 * size)]
                label_flat[first_number], label_flat[second_number] = label_flat[second_number], label_flat[first_number]

            # creating mask
            size = img_flat[0].shape[1]
            mask = np.zeros((size, size), dtype="uint8")
            start_point = (int(1 / 8 * size), int(1 / 8 * size))
            end_point = (int(7 / 8 * size), int(7 / 8 * size))
            color = (255, 255, 255)
            thickness = 3
            mask = cv2.rectangle(mask, start_point, end_point, color, thickness)
            # adding in painting
            total_list = first_list + second_list
            for i in total_list:
                this_image = copy.deepcopy(img_flat[i])
                this_image = np.squeeze(this_image)
                this_image = this_image.astype(np.uint8)
                res_telea = cv2.inpaint(src=this_image, inpaintMask=mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
                # res_ns = cv2.inpaint(src=this_image, inpaintMask=mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
                res_telea = res_telea.reshape(1, res_telea.shape[0], res_telea.shape[1])
                img_flat[i][:, :, :] = res_telea[:, :, :]

        if self._normalize:
            img_flat = (img_flat - 128.0)/128.0

        return (img_flat, label_flat)
