import os
from pathlib import Path


class Config:
    """配置文件"""

    # 路径配置
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"
    FEATURE_DB_DIR = BASE_DIR / "feature_db"

    # VGG-Face2数据集路径
    VGGFACE2_TRAIN_DIR = DATA_DIR / "train"  # 训练集目录
    VGGFACE2_VAL_DIR = DATA_DIR / "val"  # 验证集目录
    VGGFACE2_IDENTITY_FILE = DATA_DIR / "vggface2_identity_trans.csv"  # 身份映射文件

    # 模型配置
    FACE_MODEL_NAME = "antelopev2"

    # 特征提取配置
    FACE_DETECTION_THRESHOLD = 0.6
    SIMILARITY_THRESHOLD = 0.7  # 侵权风险阈值
    MEDIUM_RISK_THRESHOLD = 0.8  # 中等风险阈值
    HIGH_RISK_THRESHOLD = 0.9  # 高风险阈值
    EMBEDDING_SIZE = 512

    # 干扰机制配置
    INTERFERENCE_STRENGTH = 0.3  # 干扰强度
    INTERFERENCE_METHODS = ["blur", "noise", "color_shift", "pixelate", "combination"]

    # 数据库配置
    DATABASE_PATH = FEATURE_DB_DIR / "high_risk_features.db"

    # Web界面配置
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 8080

    # 图像处理配置
    MAX_IMAGE_SIZE = (1920, 1080)
    SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    # 数据库构建配置
    DATABASE_BUILD_CONFIG = {
        "max_identities": 100,  # 最大身份数量
        "max_images_per_identity": 10,  # 每个身份最大图片数
        "min_features_per_identity": 1,  # 每个身份最少特征数
        "default_max_identities": 100,  # 默认最大身份数
        "default_max_images": 10,  # 默认每个身份最大图片数
        "use_train_set": True,  # 使用训练集
        "use_val_set": False,  # 不使用验证集
        "dataset_mode": "vggface2"  # 数据集模式
    }

    # 系统信息
    SYSTEM_NAME = "AI Image Infringement Warning System"
    VERSION = "1.0.0"

    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.MODEL_DIR / cls.FACE_MODEL_NAME,
            cls.DATA_DIR,
            cls.VGGFACE2_TRAIN_DIR,
            cls.VGGFACE2_VAL_DIR,
            cls.OUTPUT_DIR,
            cls.FEATURE_DB_DIR,
            cls.OUTPUT_DIR / "protected",
            cls.OUTPUT_DIR / "reports",
            cls.OUTPUT_DIR / "logs"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {directory}")