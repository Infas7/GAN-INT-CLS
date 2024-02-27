import io
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
import h5py


class Text2ImageDataset(Dataset):

    def __init__(self, coresetFile, fullFile, transform=None, split=0, precompiled_keys=None):
        self.coresetFile = coresetFile
        self.fullFile = fullFile
        self.transform = transform
        self.dataset = None
        self.dataset_keys = None
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        self.h5py2int = lambda x: int(np.array(x))
        self.precompiled_keys = precompiled_keys

        self.dataset = h5py.File(self.fullFile, mode='r')
        self.orig_dataset = h5py.File(self.coresetFile, mode='r')

        self.full_keys = [str(k)
                          for k in self.dataset[self.split].keys()]

        if self.precompiled_keys is None:
            self.dataset_keys = [str(k)
                                 for k in self.orig_dataset[self.split].keys()]
        else:
            self.dataset_keys = self.precompiled_keys

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, idx):
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]

        right_image = bytes(np.array(example['img']))
        right_embed = np.array(example['embeddings'], dtype=float)
        # wrong_image = bytes(np.array(self.find_wrong_image(example['class'])))
        exter_embed, exter_image = self.find_exter_embed()
        # inter_embed = self.find_inter_embed()

        right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))
        exter_image = Image.open(io.BytesIO(exter_image)).resize((64, 64))
        # wrong_image = Image.open(io.BytesIO(wrong_image)).resize((64, 64))

        right_image = self.validate_image(right_image)
        exter_image = self.validate_image(exter_image)
        # wrong_image = self.validate_image(wrong_image)

        try:
            txt = np.array(example['txt']).astype(str)
        except:
            txt = 'This is a bird'

        jst_rand = 0.5

        sample = {
            'right_images': torch.FloatTensor(right_image),
            'right_embed': torch.FloatTensor(right_embed),
            # 'wrong_images': torch.FloatTensor(wrong_image),
            'new_embed': torch.FloatTensor(exter_embed),
            'new_images': torch.FloatTensor(exter_image),
            # # + torch.FloatTensor(right_embed))/2,
            # 'inter_embed': jst_rand * torch.FloatTensor(inter_embed) + (1 - jst_rand) * torch.FloatTensor(right_embed),
            'key': example_name,

            'txt': str(txt)
        }

        sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        sample['new_images'] = sample['new_images'].sub_(127.5).div_(127.5)
        # sample['wrong_images'] = sample['wrong_images'].sub_(127.5).div_(127.5)

        return sample

    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_wrong_image(category)

    def find_exter_embed(self):
        idx = np.random.randint(len(self.full_keys))
        example_name = self.full_keys[idx]
        example = self.dataset[self.split][example_name]
        return np.array(example['embeddings'], dtype=float), bytes(np.array(example['img']))

    def find_inter_embed(self):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        return np.array(example['embeddings'], dtype=float)

    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)
