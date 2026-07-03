import torch
import numpy as np
import cv2
from PIL import Image

class MaskProcessor:
    """蒙版处理工具集"""

    @staticmethod
    def tensor_to_mask(mask_tensor):
        if isinstance(mask_tensor, torch.Tensor):
            if mask_tensor.dim() == 4:
                mask_tensor = mask_tensor.squeeze(0)
            if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.squeeze(0)
            mask = mask_tensor.cpu().numpy()
        else:
            mask = mask_tensor

        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask /= 255.0
        return np.clip(mask, 0, 1)

    @staticmethod
    def mask_to_tensor(mask_np, device="cpu"):
        mask_np = np.clip(mask_np, 0, 1)
        tensor = torch.from_numpy(mask_np).float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device)

    @staticmethod
    def feather_mask(mask, radius=10):
        mask_uint8 = (mask * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(mask_uint8, (radius*2+1, radius*2+1), 0)
        return blurred.astype(np.float32) / 255.0

    @staticmethod
    def expand_mask(mask, pixels=20):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels*2+1, pixels*2+1))
        expanded = cv2.dilate((mask*255).astype(np.uint8), kernel)
        return expanded.astype(np.float32) / 255.0

    @staticmethod
    def shrink_mask(mask, pixels=20):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels*2+1, pixels*2+1))
        shrunk = cv2.erode((mask*255).astype(np.uint8), kernel)
        return shrunk.astype(np.float32) / 255.0

    @staticmethod
    def smooth_mask(mask, iterations=3):
        result = mask.copy()
        for _ in range(iterations):
            result = cv2.medianBlur((result*255).astype(np.uint8), 5)
            result = result.astype(np.float32) / 255.0
        return result

    @staticmethod
    def invert_mask(mask):
        return 1.0 - mask

    @staticmethod
    def combine_masks(mask1, mask2, operation="union"):
        h1, w1 = mask1.shape[:2]
        h2, w2 = mask2.shape[:2]
        if h1 != h2 or w1 != w2:
            mask2 = cv2.resize(mask2, (w1, h1), interpolation=cv2.INTER_LINEAR)

        if operation == "union":
            return np.maximum(mask1, mask2)
        elif operation == "intersection":
            return np.minimum(mask1, mask2)
        elif operation == "subtract":
            return np.clip(mask1 - mask2, 0, 1)
        elif operation == "multiply":
            return mask1 * mask2

        return mask1

    @staticmethod
    def create_radial_mask(height, width, center=None, radius=None, feather=0.2):
        if center is None:
            center = (width // 2, height // 2)
        if radius is None:
            radius = min(width, height) // 2

        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)

        mask = np.clip(1 - (dist / radius), 0, 1)

        if feather > 0:
            feather_pixels = int(radius * feather)
            mask = cv2.GaussianBlur((mask*255).astype(np.uint8),
                                     (feather_pixels*2+1, feather_pixels*2+1), 0)
            mask = mask.astype(np.float32) / 255.0

        return mask

    @staticmethod
    def create_gradient_mask(height, width, direction="top", feather=0.3):
        mask = np.zeros((height, width), dtype=np.float32)

        if direction == "top":
            mask = np.linspace(0, 1, height).reshape(-1, 1).repeat(width, axis=1)
        elif direction == "bottom":
            mask = np.linspace(1, 0, height).reshape(-1, 1).repeat(width, axis=1)
        elif direction == "left":
            mask = np.linspace(0, 1, width).reshape(1, -1).repeat(height, axis=0)
        elif direction == "right":
            mask = np.linspace(1, 0, width).reshape(1, -1).repeat(height, axis=0)

        if feather > 0:
            feather_pixels = int(max(height, width) * feather)
            mask = cv2.GaussianBlur((mask*255).astype(np.uint8),
                                     (feather_pixels*2+1, feather_pixels*2+1), 0)
            mask = mask.astype(np.float32) / 255.0

        return mask

    @staticmethod
    def extract_region_mask(image, color_range, tolerance=30):
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.dim() == 3 and image.shape[0] in [1, 3, 4]:
                image = image.permute(1, 2, 0)
            img = image.cpu().numpy()
        else:
            img = image

        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0

        hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV)

        lower = np.array([max(0, color_range[0][0] - tolerance),
                         max(0, color_range[1][0] - tolerance),
                         max(0, color_range[2][0] - tolerance)])
        upper = np.array([min(179, color_range[0][1] + tolerance),
                         min(255, color_range[1][1] + tolerance),
                         min(255, color_range[2][1] + tolerance)])

        mask = cv2.inRange(hsv, lower, upper).astype(np.float32) / 255.0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask
