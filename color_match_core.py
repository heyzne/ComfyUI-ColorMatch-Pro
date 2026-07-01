import torch
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from scipy import linalg
import warnings


class ColorMatchEngine:
    """高级颜色匹配引擎"""

    def __init__(self, method="mean_shift", strength=1.0):
        self.method = method
        self.strength = strength

    def match(self, source, target, mask=None, preserve_luminance=False):
        source_np = self._to_numpy(source)
        target_np = self._to_numpy(target)

        h, w = source_np.shape[:2]

        if target_np.shape[:2] != (h, w):
            target_np = cv2.resize(target_np, (w, h), interpolation=cv2.INTER_LANCZOS4)

        mask_np = None
        if mask is not None:
            mask_np = self._to_numpy_mask(mask)
            if mask_np.shape[:2] != (h, w):
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_np = np.clip(mask_np, 0, 1)

        result = self._match_global(source_np, target_np, mask_np, preserve_luminance)

        return np.clip(result, 0, 1).astype(np.float32)

    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
                tensor = tensor.permute(1, 2, 0)
            arr = tensor.cpu().numpy()
        elif isinstance(tensor, Image.Image):
            arr = np.array(tensor).astype(np.float32) / 255.0
        else:
            arr = tensor.astype(np.float32)
            if arr.max() > 1.0:
                arr /= 255.0

        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        return np.clip(arr, 0, 1)

    def _to_numpy_mask(self, mask):
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 4:
                mask = mask.squeeze(0)
            if mask.dim() == 3:
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                elif mask.shape[-1] == 1:
                    mask = mask.squeeze(-1)
                elif mask.shape[0] == 3 or mask.shape[0] == 4:
                    mask = mask[0]
            mask = mask.cpu().numpy()

        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask /= 255.0
        return np.clip(mask, 0, 1)

    def _match_global(self, source, target, mask, preserve_luminance):
        if self.method == "mean_shift":
            return self._mean_shift_match(source, target, mask, preserve_luminance)
        elif self.method == "dominant":
            return self._dominant_color_match(source, target, mask, preserve_luminance)
        elif self.method == "histogram":
            return self._histogram_match(source, target, mask, preserve_luminance)
        elif self.method == "lab_transfer":
            return self._lab_transfer(source, target, mask, preserve_luminance)
        elif self.method == "reinhard":
            return self._reinhard_transfer(source, target, mask, preserve_luminance)
        elif self.method == "mkl":
            return self._mkl_transfer(source, target, mask, preserve_luminance)
        elif self.method == "idt":
            return self._idt_transfer(source, target, mask, preserve_luminance)
        else:
            return self._adaptive_match(source, target, mask, preserve_luminance)

    def _mean_shift_match(self, source, target, mask, preserve_luminance):
        result = source.copy()

        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5
            if mask_bool.sum() > 10:
                for c in range(3):
                    s_mean = source[:,:,c][mask_bool].mean()
                    t_mean = target[:,:,c][mask_bool].mean()
                    shift = (t_mean - s_mean) * self.strength
                    result[:,:,c] = np.where(mask_bool, 
                                             np.clip(source[:,:,c] + shift, 0, 1),
                                             source[:,:,c])
            else:
                for c in range(3):
                    shift = (target[:,:,c].mean() - source[:,:,c].mean()) * self.strength
                    result[:,:,c] = np.clip(source[:,:,c] + shift, 0, 1)
        else:
            for c in range(3):
                shift = (target[:,:,c].mean() - source[:,:,c].mean()) * self.strength
                result[:,:,c] = np.clip(source[:,:,c] + shift, 0, 1)

        if preserve_luminance:
            result = self._preserve_luminance(source, result, mask)

        return result

    def _dominant_color_match(self, source, target, mask, preserve_luminance):
        result = source.copy()

        target_pixels = target.reshape(-1, 3)
        non_white_t = np.where(np.linalg.norm(target_pixels - 1.0, axis=1) > 0.12)[0]
        if len(non_white_t) > 50:
            target_pixels = target_pixels[non_white_t]

        source_pixels = source.reshape(-1, 3)
        non_white_s = np.where(np.linalg.norm(source_pixels - 1.0, axis=1) > 0.12)[0]
        if len(non_white_s) > 50:
            source_pixels = source_pixels[non_white_s]

        n_clusters = min(3, len(target_pixels) // 100)
        if n_clusters >= 2:
            kmeans_t = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_t.fit(target_pixels)
            target_palette = kmeans_t.cluster_centers_

            kmeans_s = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_s.fit(source_pixels)
            source_palette = kmeans_s.cluster_centers_

            t_order = np.argsort(target_palette.sum(axis=1))
            s_order = np.argsort(source_palette.sum(axis=1))

            for i in range(n_clusters):
                src_color = source_palette[s_order[i]]
                tgt_color = target_palette[t_order[i]]
                shift = (tgt_color - src_color) * self.strength

                distances = np.linalg.norm(source.reshape(-1, 3) - src_color, axis=1)
                weights = np.exp(-distances * 3).reshape(source.shape[:2])

                for c in range(3):
                    result[:,:,c] = np.clip(result[:,:,c] + shift[c] * weights, 0, 1)
        else:
            return self._mean_shift_match(source, target, mask, preserve_luminance)

        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5
            for c in range(3):
                result[:,:,c] = np.where(mask_bool, result[:,:,c], source[:,:,c])

        if preserve_luminance:
            result = self._preserve_luminance(source, result, mask)

        return result

    def _histogram_match(self, source, target, mask, preserve_luminance):
        result = np.zeros_like(source)

        for c in range(3):
            if mask is not None and mask.max() > 0.01:
                mask_bool = mask > 0.5
                s_vals = source[:,:,c][mask_bool].flatten()
                t_vals = target[:,:,c][mask_bool].flatten()
            else:
                s_vals = source[:,:,c].flatten()
                t_vals = target[:,:,c].flatten()

            if len(s_vals) < 10 or len(t_vals) < 10:
                result[:,:,c] = source[:,:,c]
                continue

            s_counts, s_bins = np.histogram(s_vals, bins=256, range=(0,1))
            t_counts, t_bins = np.histogram(t_vals, bins=256, range=(0,1))

            s_cdf = np.cumsum(s_counts).astype(np.float64)
            t_cdf = np.cumsum(t_counts).astype(np.float64)

            s_cdf = s_cdf / (s_cdf[-1] + 1e-10)
            t_cdf = t_cdf / (t_cdf[-1] + 1e-10)

            lut = np.interp(s_cdf, t_cdf, np.linspace(0, 1, 256))

            idx = (source[:,:,c].flatten() * 255).astype(np.uint8)
            matched = lut[idx].reshape(source.shape[:2])

            if mask is not None and mask.max() > 0.01:
                mask_bool = mask > 0.5
                result[:,:,c] = np.where(mask_bool, matched, source[:,:,c])
            else:
                result[:,:,c] = matched

        if preserve_luminance:
            result = self._preserve_luminance(source, result, mask)

        return result

    def _lab_transfer(self, source, target, mask, preserve_luminance):
        source_lab = cv2.cvtColor((source*255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor((target*255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

        for i in range(3):
            if mask is not None and mask.max() > 0.01:
                mask_bool = mask > 0.5
                mean_s = source_lab[:,:,i][mask_bool].mean()
                std_s = source_lab[:,:,i][mask_bool].std()
                mean_t = target_lab[:,:,i][mask_bool].mean()
                std_t = target_lab[:,:,i][mask_bool].std()
            else:
                mean_s = source_lab[:,:,i].mean()
                std_s = source_lab[:,:,i].std()
                mean_t = target_lab[:,:,i].mean()
                std_t = target_lab[:,:,i].std()

            if preserve_luminance and i == 0:
                continue

            if std_s > 1e-5:
                source_lab[:,:,i] = (source_lab[:,:,i] - mean_s) * (std_t / std_s) + mean_t

        result = cv2.cvtColor(np.clip(source_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255

        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5
            for c in range(3):
                result[:,:,c] = np.where(mask_bool, result[:,:,c], source[:,:,c])

        return np.clip(result, 0, 1)

    def _reinhard_transfer(self, source, target, mask, preserve_luminance):
        source_log = np.log(source + 1e-6)
        target_log = np.log(target + 1e-6)

        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5
            source_mean = source_log[mask_bool].mean(axis=0)
            target_mean = target_log[mask_bool].mean(axis=0)
            source_std = source_log[mask_bool].std(axis=0)
            target_std = target_log[mask_bool].std(axis=0)
        else:
            source_mean = source_log.mean(axis=(0,1))
            target_mean = target_log.mean(axis=(0,1))
            source_std = source_log.std(axis=(0,1))
            target_std = target_log.std(axis=(0,1))

        result_log = (source_log - source_mean) * (target_std / (source_std + 1e-7)) + target_mean
        result = np.exp(result_log) - 1e-6

        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5
            for c in range(3):
                result[:,:,c] = np.where(mask_bool, result[:,:,c], source[:,:,c])

        if preserve_luminance:
            result = self._preserve_luminance(source, result, mask)

        return np.clip(result, 0, 1)

    def _mkl_transfer(self, source, target, mask, preserve_luminance):
        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5
            s = source[mask_bool].reshape(-1, 3).T
            t = target[mask_bool].reshape(-1, 3).T
        else:
            s = source.reshape(-1, 3).T
            t = target.reshape(-1, 3).T

        cov_s = np.cov(s)
        cov_t = np.cov(t)

        try:
            Ms = linalg.sqrtm(cov_s)
            Ms_inv = linalg.inv(Ms)
            M = np.dot(Ms_inv, linalg.sqrtm(np.dot(np.dot(Ms, cov_t), Ms)))

            mean_s = s.mean(axis=1, keepdims=True)
            mean_t = t.mean(axis=1, keepdims=True)
            result_vec = np.dot(M, s - mean_s) + mean_t

            if mask is not None and mask.max() > 0.01:
                result = source.copy()
                result[mask_bool] = result_vec.T
            else:
                result = result_vec.T.reshape(source.shape)
        except:
            return self._lab_transfer(source, target, mask, preserve_luminance)

        if preserve_luminance:
            result = self._preserve_luminance(source, result, mask)

        return np.clip(result, 0, 1)

    def _idt_transfer(self, source, target, mask, preserve_luminance):
        result = source.copy()

        for _ in range(3):
            for c in range(3):
                if mask is not None and mask.max() > 0.01:
                    mask_bool = mask > 0.5
                    s_sorted = np.sort(result[:,:,c][mask_bool].flatten())
                    t_sorted = np.sort(target[:,:,c][mask_bool].flatten())
                else:
                    s_sorted = np.sort(result[:,:,c].flatten())
                    t_sorted = np.sort(target[:,:,c].flatten())

                if len(s_sorted) < 10 or len(t_sorted) < 10:
                    continue

                idx = np.searchsorted(s_sorted, result[:,:,c].flatten())
                idx = np.clip(idx, 0, len(t_sorted)-1)
                matched = t_sorted[idx].reshape(source.shape[:2])

                if mask is not None and mask.max() > 0.01:
                    mask_bool = mask > 0.5
                    result[:,:,c] = np.where(mask_bool, matched, source[:,:,c])
                else:
                    result[:,:,c] = matched

        if preserve_luminance:
            result = self._preserve_luminance(source, result, mask)

        return np.clip(result, 0, 1)

    def _adaptive_match(self, source, target, mask, preserve_luminance):
        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5
            source_mean = source[mask_bool].mean()
            target_mean = target[mask_bool].mean()
        else:
            source_mean = source.mean()
            target_mean = target.mean()

        color_diff = abs(source_mean - target_mean)

        if color_diff < 0.05:
            return self._mean_shift_match(source, target, mask, preserve_luminance)
        elif color_diff < 0.15:
            return self._lab_transfer(source, target, mask, preserve_luminance)
        else:
            return self._histogram_match(source, target, mask, preserve_luminance)

    def _preserve_luminance(self, source, result, mask):
        source_gray = 0.299*source[:,:,0] + 0.587*source[:,:,1] + 0.114*source[:,:,2]
        result_gray = 0.299*result[:,:,0] + 0.587*result[:,:,1] + 0.114*result[:,:,2]

        ratio = np.ones_like(source_gray)
        valid = result_gray > 0.01
        ratio[valid] = source_gray[valid] / result_gray[valid]

        for c in range(3):
            result[:,:,c] = np.clip(result[:,:,c] * ratio, 0, 1)

        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5
            for c in range(3):
                result[:,:,c] = np.where(mask_bool, result[:,:,c], source[:,:,c])

        return result


class SubjectDetector:
    """自动主体检测器"""

    def __init__(self, method="saliency"):
        self.method = method

    def detect(self, image, expand_ratio=0.1):
        img = self._to_numpy(image)
        h, w = img.shape[:2]

        if self.method == "saliency":
            mask = self._saliency_detect(img)
        elif self.method == "face":
            mask = self._face_detect(img)
        elif self.method == "color_cluster":
            mask = self._color_cluster(img)
        elif self.method == "edge":
            mask = self._edge_based(img)
        else:
            mask = self._auto_detect(img)

        if expand_ratio > 0:
            kernel_size = int(min(h, w) * expand_ratio)
            if kernel_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask = cv2.dilate(mask, kernel)

        return mask

    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
                tensor = tensor.permute(1, 2, 0)
            arr = tensor.cpu().numpy()
        else:
            arr = tensor.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        return arr

    def _saliency_detect(self, img):
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency.computeSaliency((img*255).astype(np.uint8))

        if not success:
            gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            saliencyMap = cv2.Laplacian(gray, cv2.CV_64F)
            saliencyMap = np.abs(saliencyMap)
            saliencyMap = cv2.normalize(saliencyMap, None, 0, 255, cv2.NORM_MINMAX)

        _, mask = cv2.threshold((saliencyMap * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask.astype(np.float32) / 255.0

    def _face_detect(self, img):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        mask = np.zeros(img.shape[:2], dtype=np.float32)
        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            axes = (int(w*0.6), int(h*0.7))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)

        if mask.max() == 0:
            return self._saliency_detect(img)

        return mask

    def _color_cluster(self, img, n_clusters=5):
        pixels = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)

        counts = np.bincount(labels)
        bg_label = counts.argmax()

        mask = (labels != bg_label).reshape(img.shape[:2]).astype(np.float32)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _edge_based(self, img):
        gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray, dtype=np.float32)
        cv2.drawContours(mask, contours, -1, 1, -1)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))

        return mask

    def _auto_detect(self, img):
        sal_mask = self._saliency_detect(img)
        edge_mask = self._edge_based(img)

        combined = np.maximum(sal_mask, edge_mask * 0.5)

        combined_uint8 = (combined * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_uint8)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = areas.argmax() + 1
            mask = (labels == largest_idx).astype(np.float32)
        else:
            mask = combined

        return mask


class ColorClone:
    """
    高级颜色克隆器

    新增 color_adapter 模式：
    模仿 LayerColor ColorAdapter 思路，
    在LAB空间做AB通道（色彩）偏移，保留L（亮度），
    同时排除参考图中的白色背景干扰。
    """

    def __init__(self):
        self.engine = ColorMatchEngine()

    def clone(self, source, reference, mode="product", mask=None, 
              preserve_structure=True, color_strength=1.0, ref_mask=None):
        source_np = self._to_numpy(source)
        ref_np = self._to_numpy(reference)

        if source_np.shape[:2] != ref_np.shape[:2]:
            ref_np = cv2.resize(ref_np, (source_np.shape[1], source_np.shape[0]), 
                                interpolation=cv2.INTER_LANCZOS4)

        mask_np = None
        if mask is not None:
            mask_np = self._to_numpy_mask(mask)
            if mask_np.shape[:2] != source_np.shape[:2]:
                mask_np = cv2.resize(mask_np, (source_np.shape[1], source_np.shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            mask_np = np.clip(mask_np, 0, 1)

        ref_mask = None
        if ref_mask is not None:
            ref_mask = self._to_numpy_mask(ref_mask)
            if ref_mask.shape[:2] != ref_np.shape[:2]:
                ref_mask = cv2.resize(ref_mask, (ref_np.shape[1], ref_np.shape[0]), 
                                         interpolation=cv2.INTER_LINEAR)
            ref_mask = np.clip(ref_mask, 0, 1)

        if mode == "product":
            return self._clone_product(source_np, ref_np, mask_np, color_strength, ref_mask)
        elif mode == "color_adapter":
            return self._clone_color_adapter(source_np, ref_np, mask_np, color_strength, ref_mask)
        elif mode == "color_only":
            return self._clone_color_only(source_np, ref_np, mask_np, color_strength, ref_mask)
        elif mode == "full":
            return self._clone_full(source_np, ref_np, mask_np, color_strength, ref_mask)
        elif mode == "tone":
            return self._clone_tone(source_np, ref_np, mask_np, color_strength, ref_mask)
        elif mode == "mood":
            return self._clone_mood(source_np, ref_np, mask_np, color_strength, ref_mask)
        elif mode == "palette":
            return self._clone_palette(source_np, ref_np, mask_np, color_strength, ref_mask)
        elif mode == "rgb_match":
            return self._clone_rgb_match(source_np, ref_np, mask_np, color_strength, ref_mask)
        elif mode == "smart":
            return self._clone_smart(source_np, ref_np, mask_np, color_strength, ref_mask)

        return source_np

    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
                tensor = tensor.permute(1, 2, 0)
            arr = tensor.cpu().numpy()
        else:
            arr = tensor.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        return arr

    def _to_numpy_mask(self, mask):
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 4:
                mask = mask.squeeze(0)
            if mask.dim() == 3:
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                elif mask.shape[-1] == 1:
                    mask = mask.squeeze(-1)
                elif mask.shape[0] == 3 or mask.shape[0] == 4:
                    mask = mask[0]
            mask = mask.cpu().numpy()

        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask /= 255.0
        return np.clip(mask, 0, 1)

    def _extract_product_color(self, image, mask=None, white_thresh=0.88):
        """智能提取产品颜色，排除白色/近白背景"""
        pixels = image.reshape(-1, 3)

        if mask is not None and mask.max() > 0.01:
            mask_flat = mask.flatten() > 0.5
            pixels = pixels[mask_flat]

        # 排除白色/近白像素
        luminance = 0.299*pixels[:,0] + 0.587*pixels[:,1] + 0.114*pixels[:,2]
        rgb_diff = np.abs(pixels - pixels.mean(axis=1, keepdims=True)).max(axis=1)

        # 白色：亮度高且RGB差异小（接近灰/白）
        is_white = (luminance > white_thresh) & (rgb_diff < 0.08)
        non_white = ~is_white

        if non_white.sum() > 50:
            product_pixels = pixels[non_white]
        else:
            product_pixels = pixels

        return product_pixels

    def _clone_color_adapter(self, source, reference, mask, strength, ref_mask=None):
        """
        ColorAdapter风格克隆 - 改进版

        核心改进：不用AB均值偏移，而是做AB通道直方图匹配
        这样即使参考图和原图颜色接近，也能精确对齐颜色分布
        避免均值偏移导致的过度校正和发灰问题

        步骤：
        1. 转到LAB空间
        2. 提取参考图产品区域（排除白色背景）
        3. 对AB通道做蒙版区域内的直方图匹配
        4. L通道完全不动
        5. 可选：如果颜色差异很小，减少匹配强度
        """
        # 处理 ref_mask
        ref_mask_np = None
        if ref_mask is not None:
            ref_mask_np = self._to_numpy_mask(ref_mask)
            if ref_mask_np.shape[:2] != reference.shape[:2]:
                ref_mask_np = cv2.resize(ref_mask_np, (reference.shape[1], reference.shape[0]), 
                                         interpolation=cv2.INTER_LINEAR)
            ref_mask_np = np.clip(ref_mask_np, 0, 1)

        # 转到LAB空间
        source_lab = cv2.cvtColor((source * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

        result_lab = source_lab.copy()

        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5

            if mask_bool.sum() > 10:
                # 提取参考图产品区域（排除白色背景）
                ref_product = self._extract_product_color(reference, ref_mask_np, white_thresh=0.88)

                # 将参考图产品像素转到LAB
                ref_product_uint8 = (ref_product * 255).astype(np.uint8).reshape(1, -1, 3)
                ref_product_lab = cv2.cvtColor(ref_product_uint8, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)

                # 提取原图蒙版区域像素
                src_product = source[mask_bool]
                src_product_uint8 = (src_product * 255).astype(np.uint8).reshape(1, -1, 3)
                src_product_lab = cv2.cvtColor(src_product_uint8, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)

                # 对A和B通道分别做直方图匹配（不是均值偏移！）
                for i in [1, 2]:  # A通道和B通道
                    s_vals = src_product_lab[:,i]
                    r_vals = ref_product_lab[:,i]

                    if len(s_vals) < 10 or len(r_vals) < 10:
                        continue

                    # 直方图匹配
                    s_counts, _ = np.histogram(s_vals, bins=256, range=(0, 255))
                    r_counts, _ = np.histogram(r_vals, bins=256, range=(0, 255))

                    s_cdf = np.cumsum(s_counts).astype(np.float64)
                    r_cdf = np.cumsum(r_counts).astype(np.float64)
                    s_cdf = s_cdf / (s_cdf[-1] + 1e-10)
                    r_cdf = r_cdf / (r_cdf[-1] + 1e-10)

                    lut = np.interp(s_cdf, r_cdf, np.linspace(0, 255, 256))

                    # 应用查找表到整个图像的该通道
                    idx = source_lab[:,:,i].astype(np.uint8)
                    matched = lut[idx].reshape(source.shape[:2])

                    # 只替换蒙版区域，平滑混合
                    result_lab[:,:,i] = np.where(mask_bool, 
                                                 source_lab[:,:,i] * (1 - strength) + matched * strength,
                                                 source_lab[:,:,i])

                # L通道完全不动，保留原图所有光影
                # 但检测颜色差异，如果差异很小，进一步降低AB匹配强度
                ref_ab_mean = np.array([ref_product_lab[:,1].mean(), ref_product_lab[:,2].mean()])
                src_ab_mean = np.array([src_product_lab[:,1].mean(), src_product_lab[:,2].mean()])
                ab_diff = np.linalg.norm(ref_ab_mean - src_ab_mean)

                # 如果AB差异很小（<5），说明颜色已经很接近，降低strength避免过度校正
                if ab_diff < 5:
                    # 颜色已经很接近，只做轻微校正
                    for i in [1, 2]:
                        shift = (ref_product_lab[:,i].mean() - src_product_lab[:,i].mean()) * strength * 0.3
                        result_lab[:,:,i] = np.where(mask_bool,
                                                     np.clip(source_lab[:,:,i] + shift, 0, 255),
                                                     source_lab[:,:,i])
        else:
            # 无蒙版，全图处理
            ref_product = self._extract_product_color(reference, ref_mask_np, white_thresh=0.88)
            ref_product_uint8 = (ref_product * 255).astype(np.uint8).reshape(1, -1, 3)
            ref_product_lab = cv2.cvtColor(ref_product_uint8, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)

            for i in [1, 2]:
                r_vals = ref_product_lab[:,i]
                s_vals = source_lab[:,:,i].flatten()

                s_counts, _ = np.histogram(s_vals, bins=256, range=(0, 255))
                r_counts, _ = np.histogram(r_vals, bins=256, range=(0, 255))

                s_cdf = np.cumsum(s_counts).astype(np.float64)
                r_cdf = np.cumsum(r_counts).astype(np.float64)
                s_cdf = s_cdf / (s_cdf[-1] + 1e-10)
                r_cdf = r_cdf / (r_cdf[-1] + 1e-10)

                lut = np.interp(s_cdf, r_cdf, np.linspace(0, 255, 256))

                idx = source_lab[:,:,i].astype(np.uint8)
                matched = lut[idx].reshape(source.shape[:2])

                result_lab[:,:,i] = source_lab[:,:,i] * (1 - strength) + matched * strength

        # 转回RGB
        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

        return np.clip(result, 0, 1)

    def _clone_product(self, source, reference, mask, strength, ref_mask=None):
        """产品专用克隆 - RGB直方图匹配 + 亮度保护"""
        result = source.copy()

        ref_product = self._extract_product_color(reference, ref_mask, white_thresh=0.88)

        if mask is None or mask.max() <= 0.01:
            for c in range(3):
                s_vals = source[:,:,c].flatten()
                r_vals = ref_product[:,c]

                s_counts, _ = np.histogram(s_vals, bins=256, range=(0,1))
                r_counts, _ = np.histogram(r_vals, bins=256, range=(0,1))

                s_cdf = np.cumsum(s_counts).astype(np.float64)
                r_cdf = np.cumsum(r_counts).astype(np.float64)
                s_cdf = s_cdf / (s_cdf[-1] + 1e-10)
                r_cdf = r_cdf / (r_cdf[-1] + 1e-10)

                lut = np.interp(s_cdf, r_cdf, np.linspace(0, 1, 256))

                idx = (source[:,:,c].flatten() * 255).astype(np.uint8)
                matched = lut[idx].reshape(source.shape[:2])

                result[:,:,c] = source[:,:,c] * (1 - strength) + matched * strength

            source_lum = 0.299*source[:,:,0] + 0.587*source[:,:,1] + 0.114*source[:,:,2]
            result_lum = 0.299*result[:,:,0] + 0.587*result[:,:,1] + 0.114*result[:,:,2]
            ratio = np.ones_like(source_lum)
            valid = result_lum > 0.01
            ratio[valid] = source_lum[valid] / result_lum[valid]
            for c in range(3):
                result[:,:,c] = np.clip(result[:,:,c] * ratio, 0, 1)

            return np.clip(result, 0, 1)

        mask_bool = mask > 0.5

        if mask_bool.sum() < 10:
            return source

        src_product = self._extract_product_color(source, mask, white_thresh=0.88)

        for c in range(3):
            s_vals = src_product[:,c]
            r_vals = ref_product[:,c]

            if len(s_vals) < 10 or len(r_vals) < 10:
                continue

            s_counts, _ = np.histogram(s_vals, bins=256, range=(0,1))
            r_counts, _ = np.histogram(r_vals, bins=256, range=(0,1))

            s_cdf = np.cumsum(s_counts).astype(np.float64)
            r_cdf = np.cumsum(r_counts).astype(np.float64)
            s_cdf = s_cdf / (s_cdf[-1] + 1e-10)
            r_cdf = r_cdf / (r_cdf[-1] + 1e-10)

            lut = np.interp(s_cdf, r_cdf, np.linspace(0, 1, 256))

            idx = (source[:,:,c].flatten() * 255).astype(np.uint8)
            matched = lut[idx].reshape(source.shape[:2])

            result[:,:,c] = np.where(mask_bool, 
                                     source[:,:,c] * (1 - strength) + matched * strength,
                                     source[:,:,c])

        source_lum = 0.299*source[:,:,0] + 0.587*source[:,:,1] + 0.114*source[:,:,2]
        result_lum = 0.299*result[:,:,0] + 0.587*result[:,:,1] + 0.114*result[:,:,2]

        ratio = np.ones_like(source_lum)
        valid = (result_lum > 0.01) & mask_bool
        ratio[valid] = source_lum[valid] / result_lum[valid]

        for c in range(3):
            adjusted = np.clip(result[:,:,c] * ratio, 0, 1)
            result[:,:,c] = np.where(mask_bool, adjusted, source[:,:,c])

        return np.clip(result, 0, 1)

    def _clone_smart(self, source, reference, mask, strength, ref_mask=None):
        """智能克隆 - K-Means平滑映射"""
        result = source.copy()

        if mask is None or mask.max() <= 0.01:
            return source

        mask_bool = mask > 0.5
        if mask_bool.sum() < 10:
            return source

        ref_product = self._extract_product_color(reference, ref_mask, white_thresh=0.88)
        src_product = self._extract_product_color(source, mask, white_thresh=0.88)

        if len(ref_product) < 50 or len(src_product) < 50:
            return self._clone_product(source, reference, mask, strength)

        n_clusters = min(5, len(ref_product) // 100, len(src_product) // 100)
        if n_clusters < 2:
            n_clusters = 2

        ref_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        ref_kmeans.fit(ref_product)
        ref_palette = ref_kmeans.cluster_centers_

        src_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        src_kmeans.fit(src_product)
        src_palette = src_kmeans.cluster_centers_

        ref_order = np.argsort(ref_palette.sum(axis=1))
        src_order = np.argsort(src_palette.sum(axis=1))

        src_flat = source.reshape(-1, 3)
        mask_flat = mask_bool.flatten()

        result_flat = result.reshape(-1, 3).copy()

        for i in range(len(src_flat)):
            if not mask_flat[i]:
                continue

            pixel = src_flat[i]

            distances = np.linalg.norm(src_palette - pixel, axis=1)
            src_idx = distances.argmin()

            ref_idx = ref_order[np.where(src_order == src_idx)[0][0]]
            target_color = ref_palette[ref_idx]

            shift = (target_color - src_palette[src_idx]) * strength

            weight = np.exp(-distances[src_idx] * 2)

            new_color = pixel + shift * weight
            result_flat[i] = np.clip(new_color, 0, 1)

        result = result_flat.reshape(source.shape)

        source_lum = 0.299*source[:,:,0] + 0.587*source[:,:,1] + 0.114*source[:,:,2]
        result_lum = 0.299*result[:,:,0] + 0.587*result[:,:,1] + 0.114*result[:,:,2]

        ratio = np.ones_like(source_lum)
        valid = (result_lum > 0.01) & mask_bool
        ratio[valid] = source_lum[valid] / result_lum[valid]

        for c in range(3):
            adjusted = np.clip(result[:,:,c] * ratio, 0, 1)
            result[:,:,c] = np.where(mask_bool, adjusted, source[:,:,c])

        return np.clip(result, 0, 1)

    def _clone_rgb_match(self, source, reference, mask, strength, ref_mask=None):
        """逐像素KNN匹配 + 亮度保护"""
        result = source.copy()

        if mask is None or mask.max() <= 0.01:
            return source

        mask_bool = mask > 0.5
        if mask_bool.sum() < 10:
            return source

        ref_product = self._extract_product_color(reference, ref_mask, white_thresh=0.88)

        if len(ref_product) < 10:
            return self._clone_product(source, reference, mask, strength)

        src_pixels = source[mask_bool]

        step = max(1, len(ref_product) // 5000)
        ref_sample = ref_product[::step]

        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        nn.fit(ref_sample)

        distances, indices = nn.kneighbors(src_pixels)
        matched_pixels = ref_sample[indices.flatten()]

        blended = src_pixels * (1 - strength) + matched_pixels * strength

        src_lum = 0.299*src_pixels[:,0] + 0.587*src_pixels[:,1] + 0.114*src_pixels[:,2]
        matched_lum = 0.299*blended[:,0] + 0.587*blended[:,1] + 0.114*blended[:,2]

        ratio = np.ones_like(src_lum)
        valid = matched_lum > 0.01
        ratio[valid] = src_lum[valid] / matched_lum[valid]

        for c in range(3):
            blended[:,c] = np.clip(blended[:,c] * ratio, 0, 1)

        result_flat = result.reshape(-1, 3)
        mask_flat = mask_bool.flatten()
        result_flat[mask_flat] = blended
        result = result_flat.reshape(source.shape)

        return np.clip(result, 0, 1)

    def _clone_color_only(self, source, reference, mask, strength, ref_mask=None):
        """只克隆颜色（色调+饱和度），完全不碰亮度"""
        source_hsv = cv2.cvtColor((source * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        ref_hsv = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

        result_hsv = source_hsv.copy()

        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5

            if mask_bool.sum() > 10:
                for i in [0, 1]:
                    s_mean = source_hsv[:,:,i][mask_bool].mean()
                    r_mean = ref_hsv[:,:,i][mask_bool].mean()

                    shift = (r_mean - s_mean) * strength

                    shifted = result_hsv[:,:,i] + shift
                    if i == 0:
                        shifted = np.mod(shifted, 180)
                    else:
                        shifted = np.clip(shifted, 0, 255)

                    result_hsv[:,:,i] = np.where(mask_bool, shifted, result_hsv[:,:,i])
        else:
            for i in [0, 1]:
                s_mean = source_hsv[:,:,i].mean()
                r_mean = ref_hsv[:,:,i].mean()
                shift = (r_mean - s_mean) * strength

                if i == 0:
                    result_hsv[:,:,i] = np.mod(result_hsv[:,:,i] + shift, 180)
                else:
                    result_hsv[:,:,i] = np.clip(result_hsv[:,:,i] + shift, 0, 255)

        result = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return np.clip(result, 0, 1)

    def _clone_full(self, source, reference, mask, strength, ref_mask=None):
        result = self.engine.match(source, reference, mask, preserve_luminance=False)
        return source * (1 - strength) + result * strength

    def _clone_tone(self, source, reference, mask, strength, ref_mask=None):
        source_hsv = cv2.cvtColor((source*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        ref_hsv = cv2.cvtColor((reference*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

        for i in [0, 1]:
            mean_s, std_s = source_hsv[:,:,i].mean(), source_hsv[:,:,i].std()
            mean_r, std_r = ref_hsv[:,:,i].mean(), ref_hsv[:,:,i].std()
            if std_s > 1e-5:
                source_hsv[:,:,i] = (source_hsv[:,:,i] - mean_s) * (std_r / std_s) + mean_r

        result = cv2.cvtColor(np.clip(source_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255

        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5
            for c in range(3):
                result[:,:,c] = np.where(mask_bool, result[:,:,c], source[:,:,c])

        return np.clip(source * (1 - strength) + result * strength, 0, 1)

    def _clone_mood(self, source, reference, mask, strength, ref_mask=None):
        source_lab = cv2.cvtColor((source*255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor((reference*255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

        mean_s, std_s = source_lab[:,:,0].mean(), source_lab[:,:,0].std()
        mean_r, std_r = ref_lab[:,:,0].mean(), ref_lab[:,:,0].std()
        if std_s > 1e-5:
            source_lab[:,:,0] = (source_lab[:,:,0] - mean_s) * (std_r / std_s) + mean_r

        for i in [1, 2]:
            mean_s, std_s = source_lab[:,:,i].mean(), source_lab[:,:,i].std()
            mean_r, std_r = ref_lab[:,:,i].mean(), ref_lab[:,:,i].std()
            if std_s > 1e-5:
                source_lab[:,:,i] = (source_lab[:,:,i] - mean_s) * (std_r / std_s) * 0.3 + mean_r * 0.3 + source_lab[:,:,i] * 0.7

        result = cv2.cvtColor(np.clip(source_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255

        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5
            for c in range(3):
                result[:,:,c] = np.where(mask_bool, result[:,:,c], source[:,:,c])

        return np.clip(source * (1 - strength) + result * strength, 0, 1)

    def _clone_palette(self, source, reference, mask, strength, ref_mask=None):
        ref_pixels = reference.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        ref_labels = kmeans.fit_predict(ref_pixels)
        ref_palette = kmeans.cluster_centers_

        src_pixels = source.reshape(-1, 3)
        src_kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        src_labels = src_kmeans.fit_predict(src_pixels)
        src_palette = src_kmeans.cluster_centers_

        mapping = {}
        for i, src_color in enumerate(src_palette):
            distances = np.linalg.norm(ref_palette - src_color, axis=1)
            mapping[i] = distances.argmin()

        result_pixels = src_pixels.copy()
        for i in range(len(src_pixels)):
            src_cluster = src_labels[i]
            target_cluster = mapping[src_cluster]
            result_pixels[i] = src_pixels[i] * 0.3 + ref_palette[target_cluster] * 0.7

        result = result_pixels.reshape(source.shape)

        if mask is not None and mask.max() > 0.01:
            mask_bool = mask > 0.5
            for c in range(3):
                result[:,:,c] = np.where(mask_bool, result[:,:,c], source[:,:,c])

        return np.clip(source * (1 - strength) + result * strength, 0, 1)
