import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from skimage.io import imread
import glob
from augmentations import build_train_augmentations, build_test_augmentations
from torch.utils.data import DataLoader

class UnetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        #self.img_path = glob.glob(image_dir + '*.tif')
        self.images = glob.glob(image_dir + '*.jpg')
        self.masks = glob.glob(mask_dir + '*.png')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = img_path.replace("/images/", "/masks/")
        mask_path = mask_path.replace(".jpg", ".png")
        image = imread(img_path).astype(np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        if((len(np.unique(mask)) > 2) or ((mask.min() != 0) and (mask.max() != 255))):
            print ("ERRO")
            
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        #print(image.size())    
        #print(mask.shape)

        return image, mask
    

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, 
                train_transform, val_transform, num_workers=4, pin_memory=True,):
    
    train_ds = UnetDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform,)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,pin_memory=pin_memory,shuffle=True,)

    val_ds = UnetDataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform,)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False,)

    return train_loader, val_loader


def create_loader(dataset, aug_parameters, batch_size, n_workers, pin_memory, in_channels = 2):
    train_transform = build_train_augmentations(in_channels)

    val_transforms = build_test_augmentations(in_channels)
    
    train_loader, val_loader = get_loaders(
        dataset["train_images"],
        dataset["train_masks"],
        dataset["val_images"],
        dataset["val_masks"],
        batch_size,
        train_transform,
        val_transforms,
        n_workers,
        pin_memory,
    )
    
    return train_loader, val_loader