import albumentations as A
from albumentations.pytorch import transforms as A_torch
from albumentations.augmentations.functional import _maybe_process_in_chunks, preserve_shape
import torch
from scipy.ndimage import gaussian_filter

from skimage import restoration

def rolling_ball(array):
    background = restoration.rolling_ball(array)
    array_restaured = array-background
    return array_restaured

def build_train_augmentations(in_channels = 2):
    """augmentations utilizadas pelo campeão do XView3. No caso do ScaleRotate, foi utilizada reflexão ao invés do padding com NaN""" 
    transforms = A.Compose(
		[
			#UnclippedRandomBrightnessContrast(brightness_limit=(-1,1), contrast_limit=0.1, image_in_log_space=False, p=0.25),
			#UnclippedGaussNoise(image_in_log_space=False, var_limit=(0.0001, 0.005), mean=0, per_channel=True, p=0.5),
			A.HorizontalFlip(p=0.2),
			A.VerticalFlip(p=0.2),
			#ElasticTransform(alpha=(10,100), p=0.1),
			#A.ShiftScaleRotate(scale_limit=0, rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
			#RandomGridShuffle(p=0.2),
			#A.MedianBlur(blur_limit=5, p=0.05),
			#A.GaussianBlur(blur_limit=(3,5),p=0.05),
            A.Normalize(
                mean=in_channels*[0.0],
                std=in_channels*[1.0],
                max_pixel_value=255.0,
            ),
            A.Resize(height = 512, width = 512),
			A_torch.ToTensorV2()
		]
    )
    return transforms

def build_test_augmentations(in_channels = 2):
    transforms = A.Compose(
		[
            A.Normalize(
                mean=in_channels*[0.0],
                std=in_channels*[1.0],
                max_pixel_value=255.0,
            ),
            A.Resize(height = 512, width = 512),
			A_torch.ToTensorV2()
		]
    )
    return transforms

def apply_TTA(image):
    flip_image = torch.Tensor(image.cpu().numpy()[:,:,:,::-1].copy())
    filter_image = torch.Tensor(gaussian_filter(image.cpu().numpy(), sigma=1).copy())
    #filter_image = torch.Tensor(rolling_ball(image.cpu().numpy()).copy())
    return flip_image, filter_image

def revert_TTA(flip_mask, filter_mask):
    flip_output = torch.Tensor(flip_mask.cpu().numpy()[:,:,:,::-1].copy())
    filter_output = torch.Tensor(filter_mask.cpu().numpy().copy())
    
    return flip_output, filter_output

def get_final_bin_mask(orig_bin_mask, h_bin_mask, v_bin_mask):
    sum_bin_mask = orig_bin_mask + h_bin_mask + v_bin_mask
    final_bin_mask = sum_bin_mask >= 2
    return final_bin_mask