import torch
import numpy as np
import cv2
from PIL import Image
import comfy.utils
from .color_match_core import ColorMatchEngine, SubjectDetector, ColorClone
from .mask_utils import MaskProcessor


class ColorMatchNode:
    """颜色匹配节点 - 基础追色功能"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "method": (["mean_shift", "dominant", "histogram", "lab_transfer", "reinhard", "mkl", "idt", "adaptive"], 
                          {"default": "mean_shift"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "preserve_luminance": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("matched_image", "applied_mask")
    FUNCTION = "match_color"
    CATEGORY = "🎨 ColorMatch Pro"

    def match_color(self, image, reference, method, strength, preserve_luminance, mask=None):
        engine = ColorMatchEngine(method=method, strength=strength)

        results = []
        masks = []

        for i in range(image.shape[0]):
            img = image[i]
            ref = reference[min(i, reference.shape[0]-1)]

            # 处理蒙版 - 修复维度问题
            m = None
            if mask is not None:
                if mask.dim() == 4:  # B,1,H,W -> 取第一个
                    m = mask[i] if i < mask.shape[0] else mask[0]
                elif mask.dim() == 3:
                    if mask.shape[0] == image.shape[0]:  # B,H,W
                        m = mask[i]
                    elif mask.shape[0] == 1:  # 1,H,W
                        m = mask[0]
                    elif mask.shape[-1] == 1:  # H,W,1
                        m = mask.squeeze(-1)
                    else:
                        m = mask
                elif mask.dim() == 2:  # H,W
                    m = mask

            result = engine.match(img, ref, m, preserve_luminance)
            results.append(torch.from_numpy(result).float())

            # 记录实际使用的蒙版
            if m is not None:
                m_np = MaskProcessor.tensor_to_mask(m)
                if m_np.shape[:2] != (img.shape[0], img.shape[1]):
                    m_np = cv2.resize(m_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                masks.append(torch.from_numpy(m_np).float())
            else:
                masks.append(torch.ones(img.shape[0], img.shape[1]))

        result_tensor = torch.stack(results)
        mask_tensor = torch.stack(masks).unsqueeze(-1)

        return (result_tensor, mask_tensor)


class SubjectAutoMaskNode:
    """自动主体检测蒙版节点"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["auto", "saliency", "face", "color_cluster", "edge"], 
                          {"default": "auto"}),
                "expand_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "detect_subject"
    CATEGORY = "🎨 ColorMatch Pro"

    def detect_subject(self, image, method, expand_ratio, invert):
        detector = SubjectDetector(method=method)

        masks = []
        previews = []

        for i in range(image.shape[0]):
            img = image[i]
            mask = detector.detect(img, expand_ratio)

            if invert:
                mask = 1.0 - mask

            mask_3ch = np.stack([mask]*3, axis=-1)
            preview = img.cpu().numpy() * 0.7 + mask_3ch * 0.3

            masks.append(torch.from_numpy(mask).float())
            previews.append(torch.from_numpy(preview).float())

        mask_tensor = torch.stack(masks).unsqueeze(-1)
        preview_tensor = torch.stack(previews)

        return (mask_tensor, preview_tensor)


class ColorCloneNode:
    """高级颜色克隆节点 - 模仿原图颜色风格"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "mode": (["color_adapter", "product", "smart", "rgb_match", "color_only", "full", "tone", "mood", "palette"], {"default": "color_adapter"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "mask": ("MASK",),
                "reference_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cloned_image",)
    FUNCTION = "clone_color"
    CATEGORY = "🎨 ColorMatch Pro"

    def clone_color(self, image, reference, mode, strength, mask=None, reference_mask=None):
        cloner = ColorClone()

        results = []

        for i in range(image.shape[0]):
            img = image[i]
            ref = reference[min(i, reference.shape[0]-1)]

            # 处理原图蒙版
            m = None
            if mask is not None:
                if mask.dim() == 4:
                    m = mask[i] if i < mask.shape[0] else mask[0]
                elif mask.dim() == 3:
                    if mask.shape[0] == image.shape[0]:
                        m = mask[i]
                    elif mask.shape[0] == 1:
                        m = mask[0]
                    elif mask.shape[-1] == 1:
                        m = mask.squeeze(-1)
                    else:
                        m = mask
                elif mask.dim() == 2:
                    m = mask

                if m is not None:
                    m = MaskProcessor.tensor_to_mask(m)

            # 处理参考图蒙版
            ref_m = None
            if reference_mask is not None:
                if reference_mask.dim() == 4:
                    ref_m = reference_mask[min(i, reference_mask.shape[0]-1)]
                elif reference_mask.dim() == 3:
                    if reference_mask.shape[0] == reference.shape[0]:
                        ref_m = reference_mask[min(i, reference_mask.shape[0]-1)]
                    elif reference_mask.shape[0] == 1:
                        ref_m = reference_mask[0]
                    elif reference_mask.shape[-1] == 1:
                        ref_m = reference_mask.squeeze(-1)
                    else:
                        ref_m = reference_mask
                elif reference_mask.dim() == 2:
                    ref_m = reference_mask

                if ref_m is not None:
                    ref_m = MaskProcessor.tensor_to_mask(ref_m)

            result = cloner.clone(img, ref, mode, m, ref_mask=ref_m, color_strength=strength)
            results.append(torch.from_numpy(result).float())

        return (torch.stack(results),)


class RegionalColorMatchNode:
    """区域调色节点 - 支持多区域分别调色"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "region_mask": ("MASK",),
                "reference": ("IMAGE",),
                "method": (["mean_shift", "histogram", "lab_transfer", "reinhard", "mkl", "adaptive"], 
                          {"default": "mean_shift"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "background_reference": ("IMAGE",),
                "bg_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("result", "processed_mask")
    FUNCTION = "regional_match"
    CATEGORY = "🎨 ColorMatch Pro"

    def regional_match(self, image, region_mask, reference, method, strength, feather,
                      background_reference=None, bg_strength=0.0):
        engine = ColorMatchEngine(method=method, strength=strength)
        processor = MaskProcessor()

        results = []
        processed_masks = []

        for i in range(image.shape[0]):
            img = image[i]
            ref = reference[min(i, reference.shape[0]-1)]

            # 处理蒙版
            mask = region_mask[min(i, region_mask.shape[0]-1)] if region_mask.dim() > 2 else region_mask
            mask_np = processor.tensor_to_mask(mask)

            # 确保蒙版尺寸匹配
            if mask_np.shape[:2] != (img.shape[0], img.shape[1]):
                mask_np = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

            # 可选羽化
            if feather > 0:
                mask_np = processor.feather_mask(mask_np, feather)

            result = engine.match(img, ref, mask_np, preserve_luminance=False)

            # 背景处理（可选）
            if background_reference is not None and bg_strength > 0:
                bg_ref = background_reference[min(i, background_reference.shape[0]-1)]
                bg_mask = 1.0 - mask_np
                bg_result = engine.match(img, bg_ref, bg_mask, preserve_luminance=False)
                result = result * mask_np[:,:,np.newaxis] + bg_result * bg_mask[:,:,np.newaxis]

            results.append(torch.from_numpy(result).float())
            processed_masks.append(torch.from_numpy(mask_np).float())

        result_tensor = torch.stack(results)
        mask_tensor = torch.stack(processed_masks).unsqueeze(-1)

        return (result_tensor, mask_tensor)


class MaskToolNode:
    """蒙版工具节点 - 蒙版处理操作"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "operation": (["feather", "expand", "shrink", "smooth", "invert", 
                              "threshold", "blur"], {"default": "feather"}),
                "amount": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.5}),
            },
            "optional": {
                "mask_b": ("MASK",),
                "combine_op": (["union", "intersection", "subtract", "multiply"], 
                              {"default": "union"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("processed_mask",)
    FUNCTION = "process_mask"
    CATEGORY = "🎨 ColorMatch Pro"

    def process_mask(self, mask, operation, amount, mask_b=None, combine_op="union"):
        processor = MaskProcessor()

        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.dim() == 4:
            mask = mask.squeeze(0).squeeze(-1)

        mask_np = processor.tensor_to_mask(mask)

        if mask_b is not None:
            if mask_b.dim() == 3 and mask_b.shape[-1] == 1:
                mask_b = mask_b.squeeze(-1)
            mask_b_np = processor.tensor_to_mask(mask_b)
            mask_np = processor.combine_masks(mask_np, mask_b_np, combine_op)

        if operation == "feather":
            result = processor.feather_mask(mask_np, int(amount))
        elif operation == "expand":
            result = processor.expand_mask(mask_np, int(amount))
        elif operation == "shrink":
            result = processor.shrink_mask(mask_np, int(amount))
        elif operation == "smooth":
            result = processor.smooth_mask(mask_np, int(amount))
        elif operation == "invert":
            result = processor.invert_mask(mask_np)
        elif operation == "threshold":
            result = (mask_np > amount/100.0).astype(np.float32)
        elif operation == "blur":
            result = cv2.GaussianBlur((mask_np*255).astype(np.uint8), 
                                     (int(amount)*2+1, int(amount)*2+1), 0).astype(np.float32)/255

        result_tensor = processor.mask_to_tensor(result)
        if result_tensor.dim() == 2:
            result_tensor = result_tensor.unsqueeze(0).unsqueeze(-1)
        elif result_tensor.dim() == 3:
            result_tensor = result_tensor.unsqueeze(-1)

        return (result_tensor,)


class ColorCorrectionNode:
    """色差修正节点 - 自动修正AI出图色差"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_color": (["auto", "neutral", "warm", "cool", "vintage", "cinematic"], 
                                {"default": "auto"}),
                "correction_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "white_balance": ("BOOLEAN", {"default": True}),
                "contrast_adjust": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_image",)
    FUNCTION = "correct_color"
    CATEGORY = "🎨 ColorMatch Pro"

    def correct_color(self, image, target_color, correction_strength, white_balance, 
                     contrast_adjust, mask=None):
        results = []

        for i in range(image.shape[0]):
            img = image[i].cpu().numpy()

            if white_balance:
                img = self._white_balance(img)

            if target_color != "auto":
                img = self._apply_tone(img, target_color, correction_strength)

            if contrast_adjust != 0:
                img = self._adjust_contrast(img, contrast_adjust)

            if mask is not None:
                m = mask[min(i, mask.shape[0]-1)] if mask.dim() > 2 else mask
                m = MaskProcessor.tensor_to_mask(m)
                m = cv2.resize(m, (img.shape[1], img.shape[0]))
                m_3ch = np.stack([m]*3, axis=-1)
                original = image[i].cpu().numpy()
                img = original * (1 - m_3ch) + img * m_3ch

            results.append(torch.from_numpy(np.clip(img, 0, 1)).float())

        return (torch.stack(results),)

    def _white_balance(self, img):
        result = img.copy()
        for c in range(3):
            mean_val = result[:,:,c].mean()
            result[:,:,c] = result[:,:,c] * (0.5 / (mean_val + 1e-7))
        return np.clip(result, 0, 1)

    def _apply_tone(self, img, tone, strength):
        tone_presets = {
            "neutral": np.array([1.0, 1.0, 1.0]),
            "warm": np.array([1.1, 1.0, 0.9]),
            "cool": np.array([0.9, 1.0, 1.1]),
            "vintage": np.array([1.2, 1.0, 0.8]),
            "cinematic": np.array([1.0, 1.05, 1.1]),
        }

        factor = tone_presets.get(tone, np.array([1.0, 1.0, 1.0]))
        factor = 1 + (factor - 1) * strength

        return np.clip(img * factor, 0, 1)

    def _adjust_contrast(self, img, amount):
        mean = img.mean()
        return np.clip((img - mean) * (1 + amount) + mean, 0, 1)


class BatchColorMatchNode:
    """批量颜色匹配 - 支持多参考图批量处理"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "references": ("IMAGE",),
                "method": (["mean_shift", "histogram", "lab_transfer", "reinhard", "mkl", "adaptive"], 
                          {"default": "mean_shift"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "masks": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matched_images",)
    FUNCTION = "batch_match"
    CATEGORY = "🎨 ColorMatch Pro"

    def batch_match(self, images, references, method, strength, masks=None):
        engine = ColorMatchEngine(method=method, strength=strength)

        results = []
        num_images = images.shape[0]
        num_refs = references.shape[0]

        for i in range(num_images):
            img = images[i]
            ref = references[i % num_refs]

            m = None
            if masks is not None:
                if masks.dim() == 4:
                    m = masks[i % masks.shape[0]]
                elif masks.dim() == 3:
                    if masks.shape[0] == num_images:
                        m = masks[i]
                    else:
                        m = masks
                m = MaskProcessor.tensor_to_mask(m)

            result = engine.match(img, ref, m, preserve_luminance=False)
            results.append(torch.from_numpy(result).float())

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "ColorMatchPro": ColorMatchNode,
    "SubjectAutoMask": SubjectAutoMaskNode,
    "ColorClonePro": ColorCloneNode,
    "RegionalColorMatch": RegionalColorMatchNode,
    "MaskToolPro": MaskToolNode,
    "ColorCorrectionPro": ColorCorrectionNode,
    "BatchColorMatch": BatchColorMatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorMatchPro": "🎨 Color Match (追色匹配)",
    "SubjectAutoMask": "🔍 Auto Subject Mask (自动主体蒙版)",
    "ColorClonePro": "🎭 Color Clone (颜色克隆)",
    "RegionalColorMatch": "🎯 Regional Color Match (区域调色)",
    "MaskToolPro": "🔧 Mask Tool (蒙版工具)",
    "ColorCorrectionPro": "⚡ Color Correction (色差修正)",
    "BatchColorMatch": "📦 Batch Color Match (批量追色)",
}
