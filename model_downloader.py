import os
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import sys


class ModelDownloader:
    """模型下载器，使用国内镜像快速下载"""

    # 国内镜像源
    MIRROR_SOURCES = {
        "tsinghua": "https://cloud.tsinghua.edu.cn/f/",
        "huawei": "https://mirrors.huaweicloud.com/",
        "aliyun": "https://mirrors.aliyun.com/"
    }

    # 模型下载链接（国内镜像）
    MODEL_URLS = {
        "antelopev2": {
            "tsinghua": "https://cloud.tsinghua.edu.cn/f/7b2e4d3c4a4d4c8fa5c3/?dl=1",
            "direct": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        }
    }

    def __init__(self, config):
        self.config = config
        self.model_dir = config.MODEL_DIR / config.FACE_MODEL_NAME

    def check_model_exists(self):
        """检查模型是否存在"""
        required_files = ["det_10g.onnx", "w600k_r50.onnx"]
        existing_files = []

        for file in required_files:
            if (self.model_dir / file).exists():
                existing_files.append(file)

        return existing_files

    def download_with_progress(self, url, save_path):
        """带进度条的下载"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            total_size = int(response.headers.get('content-length', 0))

            with open(save_path, 'wb') as f, tqdm(
                    desc=f"下载 {save_path.name}",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

            return True
        except Exception as e:
            print(f"下载失败: {e}")
            return False

    def download_from_mirror(self, source="tsinghua"):
        """从镜像源下载"""
        print(f"正在从 {source} 镜像下载模型...")

        if source not in self.MIRROR_SOURCES:
            print(f"镜像源 {source} 不存在，使用直接下载")
            return self.download_direct()

        model_info = self.MODEL_URLS.get(self.config.FACE_MODEL_NAME, {})
        url = model_info.get(source)

        if not url:
            print(f"镜像源 {source} 没有该模型，使用直接下载")
            return self.download_direct()

        # 临时保存zip文件
        temp_zip = self.config.MODEL_DIR / "temp_model.zip"

        if self.download_with_progress(url, temp_zip):
            # 解压文件
            print("正在解压模型...")
            try:
                with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                    zip_ref.extractall(self.model_dir)
                print("模型解压完成！")

                # 清理临时文件
                temp_zip.unlink()
                return True
            except Exception as e:
                print(f"解压失败: {e}")
                return False
        return False

    def download_direct(self):
        """直接下载"""
        print("正在从GitHub直接下载模型...")

        model_info = self.MODEL_URLS.get(self.config.FACE_MODEL_NAME, {})
        url = model_info.get("direct")

        if not url:
            print("没有找到下载链接")
            return False

        temp_zip = self.config.MODEL_DIR / "temp_model.zip"

        if self.download_with_progress(url, temp_zip):
            # 解压文件
            print("正在解压模型...")
            try:
                with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                    zip_ref.extractall(self.model_dir)
                print("模型解压完成！")

                # 清理临时文件
                temp_zip.unlink()
                return True
            except Exception as e:
                print(f"解压失败: {e}")
                return False
        return False

    def download_manual_guide(self):
        """提供手动下载指南"""
        print("\n" + "=" * 60)
        print("手动下载指南")
        print("=" * 60)
        print("1. 访问以下链接下载模型：")
        print("   https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
        print("\n2. 下载完成后，解压文件到以下目录：")
        print(f"   {self.model_dir}")
        print("\n3. 确保解压后的目录包含以下文件：")
        print("   - det_10g.onnx")
        print("   - w600k_r50.onnx")
        print("   - 2d106det.onnx")
        print("   - 其他 .onnx 文件")
        print("=" * 60)

    def run(self):
        """运行下载器"""
        print("=" * 60)
        print("模型下载器")
        print("=" * 60)

        # 检查模型是否已存在
        existing_files = self.check_model_exists()
        if len(existing_files) >= 2:
            print(f"模型已存在，包含文件: {', '.join(existing_files)}")
            print("如果需要重新下载，请删除 models/antelopev2/ 目录")
            return True

        print("模型文件不完整，开始下载...")

        # 尝试从多个镜像源下载
        sources = ["tsinghua", "huawei", "aliyun", "direct"]

        for source in sources:
            print(f"\n尝试从 {source} 下载...")
            if self.download_from_mirror(source):
                print("下载成功！")
                return True
            else:
                print(f"从 {source} 下载失败，尝试下一个源...")

        # 所有源都失败，提供手动指南
        print("\n所有下载源都失败，请手动下载：")
        self.download_manual_guide()
        return False