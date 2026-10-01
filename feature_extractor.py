import cv2
import numpy as np
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
from typing import List, Dict, Optional, Union
import warnings

warnings.filterwarnings('ignore')

from config import Config


class FaceFeatureExtractor:
    """人脸特征提取器"""

    def __init__(self, model_name=None):
        self.config = Config()
        self.model_name = model_name or self.config.FACE_MODEL_NAME
        self.model_dir = str(self.config.MODEL_DIR / self.model_name)
        self.app = None
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型"""
        try:
            # 设置模型路径
            import os
            os.environ['INSIGHTFACE_MODELS_ROOT'] = str(self.config.MODEL_DIR)

            # 初始化FaceAnalysis
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

            # 准备模型
            self.app.prepare(
                ctx_id=0,
                det_thresh=self.config.FACE_DETECTION_THRESHOLD,
                det_size=(640, 640)
            )

            print(f"人脸识别模型加载成功: {self.model_name}")

        except Exception as e:
            print(f"模型初始化失败: {e}")
            raise

    def extract_from_image_file(self, image_path: Union[str, Path]) -> List[Dict]:
        """从图像文件提取人脸特征"""
        if isinstance(image_path, str):
            image_path = Path(image_path)

        if not image_path.exists():
            print(f"图像文件不存在: {image_path}")
            return []

        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"无法读取图像: {image_path}")
            return []

        return self.extract_from_array(img)

    def extract_from_array(self, image_array: np.ndarray) -> List[Dict]:
        """从numpy数组提取人脸特征"""
        if self.app is None:
            self._initialize_model()

        try:
            # 检测人脸
            faces = self.app.get(image_array)

            features = []
            for face in faces:
                # 获取边界框
                bbox = [int(coord) for coord in face.bbox]  # [x1, y1, x2, y2]

                # 获取特征向量
                embedding = face.normed_embedding

                # 获取关键点（如果可用）
                landmarks = []
                if hasattr(face, 'kps') and face.kps is not None:
                    landmarks = face.kps.tolist()

                features.append({
                    'bbox': bbox,
                    'embedding': embedding,
                    'landmarks': landmarks,
                    'confidence': float(face.det_score),
                    'face_area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                })

            return features

        except Exception as e:
            print(f"特征提取失败: {e}")
            return []

    def extract_from_pil_image(self, pil_image) -> List[Dict]:
        """从PIL图像提取人脸特征"""
        # 转换为numpy数组
        img_array = np.array(pil_image)

        # 转换颜色空间 (RGB -> BGR)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        return self.extract_from_array(img_array)

    def draw_faces_on_image(self, image_array: np.ndarray, features: List[Dict]) -> np.ndarray:
        """在图像上绘制人脸框和关键点"""
        result = image_array.copy()

        for feature in features:
            bbox = feature['bbox']
            landmarks = feature.get('landmarks', [])
            confidence = feature.get('confidence', 0)

            # 绘制边界框
            color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255) if confidence > 0.6 else (0, 0, 255)
            thickness = 2

            cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

            # 绘制置信度
            label = f"Face: {confidence:.2f}"
            cv2.putText(result, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 绘制关键点
            if landmarks:
                for (x, y) in landmarks:
                    cv2.circle(result, (int(x), int(y)), 2, (0, 255, 255), -1)

        return result

    def get_dominant_face(self, features: List[Dict]) -> Optional[Dict]:
        """获取主要人脸（最大或最清晰的人脸）"""
        if not features:
            return None

        # 按人脸面积排序
        sorted_faces = sorted(features, key=lambda x: x.get('face_area', 0), reverse=True)

        return sorted_faces[0] if sorted_faces else None