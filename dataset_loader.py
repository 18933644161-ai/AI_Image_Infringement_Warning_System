import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import csv
import os
from PIL import Image
import cv2
import re

from config import Config
from feature_extractor import FaceFeatureExtractor


class DatasetLoader:
    """VGG-Face2数据集加载器"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.identity_mapping_path = self.config.VGGFACE2_IDENTITY_FILE
        self.train_dir = self.config.VGGFACE2_TRAIN_DIR
        self.val_dir = self.config.VGGFACE2_VAL_DIR
        self.feature_extractor = FaceFeatureExtractor()
        self.identity_map = {}  # 缓存身份映射

    def _clean_identity_name(self, name: str) -> str:
        """清理身份名称 - 完全去除所有引号"""
        if not name:
            return name

        # 去除首尾空格
        name = name.strip()

        # 方法1: 使用循环去除所有层的引号
        while name.startswith('"') and name.endswith('"'):
            name = name[1:-1].strip()

        # 方法2: 如果还有 " " 包裹的情况
        if name.startswith(' "') and name.endswith('" '):
            name = name[2:-2].strip()
        elif name.startswith(' "') and name.endswith('"'):
            name = name[2:-1].strip()
        elif name.startswith('"') and name.endswith('" '):
            name = name[1:-2].strip()

        # 处理内部的双引号
        name = name.replace('""', '"')

        # 去除多余空格
        name = ' '.join(name.split())

        # 最后再检查一次首尾引号
        name = name.strip('"').strip()

        return name

    def _is_valid_identity_name(self, name: str) -> bool:
        """检查身份名称是否有效"""
        if not name or len(name) < 2:
            return False

        # 检查是否包含明显乱码字符
        invalid_patterns = ['鈥', '聽', '谩', '艒', '�']
        for pattern in invalid_patterns:
            if pattern in name:
                return False

        # 检查是否是有效的英文名称（至少包含字母）
        if not re.search(r'[A-Za-z]', name):
            return False

        return True

    def load_identity_mapping(self) -> Dict[str, str]:
        """加载VGG-Face2身份映射文件，返回 {ID: Name} 字典"""
        if not self.identity_mapping_path.exists():
            print(f"身份映射文件不存在: {self.identity_mapping_path}")
            return {}

        if self.identity_map:
            return self.identity_map

        print(f"加载VGG-Face2身份映射文件: {self.identity_mapping_path}")

        identity_map = {}

        try:
            # 使用gbk编码
            with open(self.identity_mapping_path, 'r', encoding='gbk', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    # 分割字段
                    parts = line.split(',', 3)  # 最多分割3次
                    if len(parts) < 2:
                        continue

                    identity_id = parts[0].strip()
                    name = parts[1].strip()

                    # 使用新的清理方法
                    name = self._clean_identity_name(name)

                    # 只保留有效的身份
                    if self._is_valid_identity_name(name):
                        identity_map[identity_id] = name

            print(f"成功加载 {len(identity_map)} 个有效的身份映射")

            # 显示前10个身份映射
            if identity_map:
                print("\n前10个有效的身份映射:")
                for i, (identity_id, name) in enumerate(list(identity_map.items())[:10]):
                    print(f"  {i + 1:2d}. {identity_id} -> {name}")

                # 保存到缓存
                self.identity_map = identity_map
            else:
                print("警告: 没有解析到任何有效的身份映射")

        except Exception as e:
            print(f"加载身份映射文件失败: {e}")
            import traceback
            traceback.print_exc()

        return identity_map

    def get_valid_identities(self, max_count: int = None) -> List[Dict]:
        """
        获取有效的身份列表（有对应图片目录且名称有效的身份）

        Returns:
            List of dicts with keys: ['id', 'name', 'image_count']
        """
        identity_map = self.load_identity_mapping()
        valid_identities = []

        # 按ID顺序排序
        all_ids = sorted(identity_map.keys())

        for identity_id in all_ids:
            # 获取身份名称
            name = identity_map[identity_id]

            # 检查是否有对应的图片目录
            train_dir = self.train_dir / identity_id
            has_images = False
            image_count = 0

            if train_dir.exists() and train_dir.is_dir():
                # 统计JPG图片数量
                jpg_files = list(train_dir.glob("*.jpg"))
                if jpg_files:
                    has_images = True
                    image_count = len(jpg_files)

            # 如果该身份有图片，添加到有效列表
            if has_images:
                valid_identities.append({
                    'id': identity_id,
                    'name': name,
                    'image_count': image_count
                })

                # 如果达到最大数量，停止
                if max_count and len(valid_identities) >= max_count:
                    break

        print(f"找到 {len(valid_identities)} 个有效的身份（有图片）")

        # 显示前几个
        if valid_identities:
            print("\n前10个有效身份:")
            for i, identity in enumerate(valid_identities[:10]):
                print(f"  {i + 1:2d}. {identity['name']} (ID: {identity['id']}, 图片: {identity['image_count']})")

        return valid_identities

    def get_identity_name(self, identity_id: str) -> str:
        """根据ID获取身份名称"""
        if not self.identity_map:
            self.load_identity_mapping()

        # 如果找不到映射，返回ID本身
        return self.identity_map.get(identity_id, identity_id)

    def get_identity_id_by_name(self, name: str) -> Optional[str]:
        """根据身份名称获取ID"""
        if not self.identity_map:
            self.load_identity_mapping()

        # 反向查找
        for identity_id, identity_name in self.identity_map.items():
            if identity_name == name:
                return identity_id

        return None

    def get_all_identities(self, max_count: int = None) -> List[str]:
        """获取所有身份名称，可选限制数量"""
        identity_map = self.load_identity_mapping()

        # 获取所有身份名称
        identities = list(identity_map.values())

        # 去重和排序
        identities = sorted(list(set(identities)))

        if max_count is not None and max_count > 0:
            identities = identities[:max_count]

        print(f"获取到 {len(identities)} 个身份")

        return identities

    def get_image_paths_for_identity(self, identity_name: str,
                                     use_train_set: bool = True,
                                     use_val_set: bool = False) -> List[Path]:
        """根据身份名称获取图片路径列表"""
        image_paths = []

        # 首先尝试通过身份名称查找对应的ID
        identity_id = self.get_identity_id_by_name(identity_name)
        if not identity_id:
            # 如果没有找到映射，假设身份名称就是ID
            identity_id = identity_name

        # 在训练集中查找
        if use_train_set and self.train_dir.exists():
            identity_train_dir = self.train_dir / identity_id
            if identity_train_dir.exists() and identity_train_dir.is_dir():
                # 查找所有支持的图片格式
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_paths.extend(list(identity_train_dir.glob(f"*{ext}")))
                    image_paths.extend(list(identity_train_dir.glob(f"*{ext.upper()}")))

        # 在验证集中查找
        if use_val_set and self.val_dir.exists():
            identity_val_dir = self.val_dir / identity_id
            if identity_val_dir.exists() and identity_val_dir.is_dir():
                # 查找所有支持的图片格式
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_paths.extend(list(identity_val_dir.glob(f"*{ext}")))
                    image_paths.extend(list(identity_val_dir.glob(f"*{ext.upper()}")))

        return image_paths

    def get_images_for_identity(self, identity_name: str,
                                max_images: int = None,
                                use_train_set: bool = True,
                                use_val_set: bool = False) -> List[Dict]:
        """获取指定身份的所有图片信息"""
        # 获取图片路径
        image_paths = self.get_image_paths_for_identity(
            identity_name,
            use_train_set=use_train_set,
            use_val_set=use_val_set
        )

        images = []
        for img_path in image_paths:
            # 确定图片来自哪个数据集
            dataset = "train" if self.train_dir in img_path.parents else "val"

            images.append({
                'image_name': img_path.name,
                'identity': identity_name,
                'identity_id': self.get_identity_id_by_name(identity_name) or identity_name,
                'image_path': str(img_path),
                'dataset': dataset
            })

        if max_images is not None and max_images > 0:
            images = images[:max_images]

        print(f"为身份 '{identity_name}' 找到 {len(images)} 张图片")

        return images

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载图片"""
        path = Path(image_path)
        if not path.exists():
            return None

        try:
            img = cv2.imread(str(path))
            if img is not None:
                return img
        except Exception as e:
            print(f"加载图片失败 {path}: {e}")

        return None

    def extract_features_for_identity(self, identity_info: Dict,
                                      max_images: int = 10) -> List[np.ndarray]:
        """
        为指定身份提取特征

        Args:
            identity_info: 身份信息字典，包含 'id', 'name', 'image_count'
            max_images: 最大提取图片数

        Returns:
            特征向量列表
        """
        identity_name = identity_info['name']
        embeddings = []

        # 获取该身份的图片
        images = self.get_images_for_identity(identity_name, max_images=max_images)

        print(f"为身份 '{identity_name}' 提取特征，找到 {len(images)} 张图片")

        for i, image_info in enumerate(images):
            try:
                # 加载图片
                img = self.load_image(image_info['image_path'])
                if img is None:
                    continue

                # 提取特征
                features = self.feature_extractor.extract_from_array(img)
                if features:
                    # 取主要人脸的特征
                    main_face = self.feature_extractor.get_dominant_face(features)
                    if main_face:
                        embeddings.append(main_face['embedding'])
                        print(f"  图片 {i + 1}/{len(images)}: 成功提取特征")
                    else:
                        print(f"  图片 {i + 1}/{len(images)}: 未检测到人脸")
                else:
                    print(f"  图片 {i + 1}/{len(images)}: 未检测到人脸")

            except Exception as e:
                print(f"  图片 {i + 1}/{len(images)}: 特征提取失败 - {e}")

        print(f"为身份 '{identity_name}' 提取了 {len(embeddings)} 个特征")
        return embeddings

    def test_vggface2_dataset(self):
        """测试VGG-Face2数据集"""
        print("=" * 60)
        print("VGG-Face2数据集测试")
        print("=" * 60)

        # 1. 测试身份映射加载
        print("\n1. 测试身份映射文件加载...")
        identity_map = self.load_identity_mapping()

        if not identity_map:
            print("✗ 无法加载身份映射文件")
            return False

        print(f"✓ 成功加载 {len(identity_map)} 个身份映射")

        # 2. 获取有效身份
        print("\n2. 获取有效身份列表...")
        valid_identities = self.get_valid_identities(max_count=10)

        if not valid_identities:
            print("✗ 没有找到有效身份")
            return False

        print(f"✓ 找到 {len(valid_identities)} 个有效身份")

        # 3. 测试图片加载
        print("\n3. 测试图片加载...")
        test_identity = valid_identities[0]
        images = self.get_images_for_identity(test_identity['name'], max_images=1)

        if images:
            test_image_path = images[0]['image_path']
            print(f"测试图片: {test_image_path}")

            img = self.load_image(test_image_path)
            if img is not None:
                print(f"✓ 图片加载成功，尺寸: {img.shape}")
                return True
            else:
                print("✗ 图片加载失败")

        return False


# 测试代码
if __name__ == "__main__":
    config = Config()
    loader = DatasetLoader(config)

    # 运行测试
    success = loader.test_vggface2_dataset()

    if success:
        print("\n" + "=" * 60)
        print("VGG-Face2数据集测试成功！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("VGG-Face2数据集测试失败，请检查配置和文件路径")
        print("=" * 60)