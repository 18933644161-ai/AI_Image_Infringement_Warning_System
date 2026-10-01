import cv2
import numpy as np
from PIL import Image
import random
from typing import List, Tuple, Optional
import math

from config import Config


class ImageInterference:
    """图像干扰模块"""

    def __init__(self, config=None):
        self.config = config or Config()

    def apply_interference(self, image: np.ndarray, face_regions: List[List[int]],
                           method: str = "blur", strength: float = None) -> np.ndarray:
        """
        对图像应用干扰

        Args:
            image: 输入图像 (BGR格式)
            face_regions: 人脸区域列表 [[x1, y1, x2, y2], ...]
            method: 干扰方法
            strength: 干扰强度 (0-1)

        Returns:
            处理后的图像
        """
        if strength is None:
            strength = self.config.INTERFERENCE_STRENGTH

        if not face_regions:
            return image

        result = image.copy()

        for region in face_regions:
            x1, y1, x2, y2 = region

            # 确保区域在图像范围内
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # 提取人脸区域
            face_region = result[y1:y2, x1:x2]

            if method == "blur":
                processed_face = self._apply_blur(face_region, strength)
            elif method == "noise":
                processed_face = self._apply_noise(face_region, strength)
            elif method == "color_shift":
                processed_face = self._apply_color_shift(face_region, strength)
            elif method == "pixelate":
                processed_face = self._apply_pixelation(face_region, strength)
            elif method == "combination":
                processed_face = self._apply_combination(face_region, strength)
            elif method == "mask":
                processed_face = self._apply_mask(face_region, strength)
            else:
                # 默认使用模糊
                processed_face = self._apply_blur(face_region, strength)

            # 将处理后的区域放回原图
            result[y1:y2, x1:x2] = processed_face

        return result

    def _apply_blur(self, face_region: np.ndarray, strength: float) -> np.ndarray:
        """应用高斯模糊"""
        # 根据强度计算模糊核大小
        kernel_size = max(3, int(15 * strength))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        # 应用高斯模糊
        blurred = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)

        return blurred

    def _apply_noise(self, face_region: np.ndarray, strength: float) -> np.ndarray:
        """添加随机噪声"""
        # 生成噪声
        noise = np.random.normal(0, 30 * strength, face_region.shape)

        # 添加噪声
        noisy_face = face_region.astype(np.float32) + noise
        noisy_face = np.clip(noisy_face, 0, 255).astype(np.uint8)

        return noisy_face

    def _apply_color_shift(self, face_region: np.ndarray, strength: float) -> np.ndarray:
        """应用颜色偏移"""
        # 随机选择颜色偏移方式
        shift_type = random.choice(['hue', 'saturation', 'brightness', 'contrast'])

        if shift_type == 'hue':
            # 转换到HSV空间调整色相
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            hue_shift = int(30 * strength * random.uniform(-1, 1))
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif shift_type == 'saturation':
            # 调整饱和度
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            sat_shift = 1.0 + strength * random.uniform(-0.5, 0.5)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_shift, 0, 255)
            shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        elif shift_type == 'brightness':
            # 调整亮度
            value_shift = 1.0 + strength * random.uniform(-0.3, 0.3)
            shifted = np.clip(face_region.astype(np.float32) * value_shift, 0, 255).astype(np.uint8)

        else:  # contrast
            # 调整对比度
            contrast = 1.0 + strength * random.uniform(-0.4, 0.4)
            mean = np.mean(face_region)
            shifted = np.clip((face_region.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)

        return shifted

    def _apply_pixelation(self, face_region: np.ndarray, strength: float) -> np.ndarray:
        """应用像素化"""
        h, w = face_region.shape[:2]

        # 根据强度计算像素块大小
        block_size = max(2, int(20 * strength))

        # 缩小图像
        small_h, small_w = max(1, h // block_size), max(1, w // block_size)
        small = cv2.resize(face_region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

        # 放大回原尺寸
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return pixelated

    def _apply_combination(self, face_region: np.ndarray, strength: float) -> np.ndarray:
        """应用组合干扰"""
        processed = face_region.copy()

        # 随机选择2-3种方法
        methods = ['blur', 'noise', 'color_shift', 'pixelate']
        selected_methods = random.sample(methods, random.randint(2, 3))

        for method in selected_methods:
            if method == 'blur':
                processed = self._apply_blur(processed, strength * 0.7)
            elif method == 'noise':
                processed = self._apply_noise(processed, strength * 0.5)
            elif method == 'color_shift':
                processed = self._apply_color_shift(processed, strength * 0.8)
            elif method == 'pixelate':
                processed = self._apply_pixelation(processed, strength * 0.6)

        return processed

    def _apply_mask(self, face_region: np.ndarray, strength: float) -> np.ndarray:
        """应用遮罩（马赛克）"""
        h, w = face_region.shape[:2]

        # 创建马赛克效果
        block_size = max(4, int(15 * strength))

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)

                # 取方块内的平均颜色
                block = face_region[y:y_end, x:x_end]
                if block.size > 0:
                    avg_color = np.mean(block, axis=(0, 1))
                    face_region[y:y_end, x:x_end] = avg_color

        return face_region

    def generate_interference_preview(self, image: np.ndarray, face_regions: List[List[int]]) -> List[np.ndarray]:
        """生成所有干扰方法的预览"""
        previews = []
        methods = self.config.INTERFERENCE_METHODS

        for method in methods:
            processed = self.apply_interference(image.copy(), face_regions, method)
            previews.append((method, processed))

        return previews

    def calculate_interference_strength(self, similarity: float, threshold: float) -> float:
        """
        根据相似度计算干扰强度

        Args:
            similarity: 相似度分数 (0-1)
            threshold: 相似度阈值

        Returns:
            干扰强度 (0-1)
        """
        if similarity <= threshold:
            return 0.0

        # 相似度超过阈值时，根据超过的程度计算强度
        excess = (similarity - threshold) / (1.0 - threshold)

        # 使用非线性映射（指数函数）
        strength = min(1.0, 0.3 + 0.7 * (excess ** 0.5))

        return strength