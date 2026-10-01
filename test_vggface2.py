#!/usr/bin/env python3
"""
测试VGG-Face2数据加载
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from dataset_loader import DatasetLoader


def main():
    """主函数"""
    print("=" * 60)
    print("VGG-Face2数据加载测试")
    print("=" * 60)

    # 初始化配置
    config = Config()
    config.create_directories()

    # 创建数据加载器
    loader = DatasetLoader(config)

    # 测试1: 解析CSV文件
    print("\n1. 测试CSV文件解析...")
    identity_map = loader.load_identity_mapping()

    if not identity_map:
        print("✗ 无法加载身份映射文件")
        return False

    print(f"✓ 成功加载 {len(identity_map)} 个身份映射")

    # 显示前10个映射
    print("\n前10个身份映射:")
    for i, (identity_id, name) in enumerate(list(identity_map.items())[:10]):
        print(f"  {i + 1:2d}. {identity_id} -> {name}")

    # 测试2: 获取身份列表
    print("\n" + "=" * 60)
    print("2. 获取身份列表测试...")

    # 获取所有身份
    all_identities = loader.get_all_identities()
    print(f"总共 {len(all_identities)} 个身份")

    # 获取前20个身份
    top_identities = loader.get_all_identities(max_count=20)
    print(f"前20个身份: {top_identities}")

    # 测试3: 获取某个身份的图片
    print("\n" + "=" * 60)
    print("3. 测试获取身份图片...")

    if top_identities:
        # 找一个名字简单的身份
        for test_identity in top_identities:
            # 检查是否有图片
            images = loader.get_images_for_identity(test_identity, max_images=1)
            if images:
                print(f"测试身份: {test_identity}")
                break

        if not images:
            test_identity = top_identities[0]
            print(f"测试身份: {test_identity} (可能没有图片)")

        # 获取身份ID
        identity_id = loader.get_identity_id_by_name(test_identity)
        print(f"身份ID: {identity_id}")

        # 获取图片
        images = loader.get_images_for_identity(test_identity, max_images=5)
        print(f"找到 {len(images)} 张图片")

        if images:
            print("前5张图片:")
            for i, img in enumerate(images[:5]):
                print(f"  {i + 1}. {img['image_name']}")
                print(f"     路径: {img['image_path']}")
                print(f"     身份ID: {img.get('identity_id', 'N/A')}")
                print(f"     数据集: {img.get('dataset', 'N/A')}")
        else:
            print("警告: 没有找到图片")
            print("请检查目录结构:")
            print(f"  训练集目录是否存在: {config.VGGFACE2_TRAIN_DIR.exists()}")
            if config.VGGFACE2_TRAIN_DIR.exists():
                identity_dir = config.VGGFACE2_TRAIN_DIR / identity_id
                print(f"  身份目录是否存在: {identity_dir.exists()} ({identity_dir})")

    # 测试4: 测试图片加载
    print("\n" + "=" * 60)
    print("4. 测试图片加载...")

    if top_identities and images:
        test_image_path = images[0]['image_path']
        print(f"测试图片: {test_image_path}")

        img = loader.load_image(test_image_path)
        if img is not None:
            print(f"✓ 图片加载成功，尺寸: {img.shape}")
        else:
            print("✗ 图片加载失败")
    else:
        print("没有找到测试图片")

    # 测试5: 检查目录结构
    print("\n" + "=" * 60)
    print("5. 检查目录结构...")

    print(f"数据目录: {config.DATA_DIR}")
    print(f"训练集目录: {config.VGGFACE2_TRAIN_DIR}")
    print(f"训练集是否存在: {config.VGGFACE2_TRAIN_DIR.exists()}")
    print(f"验证集目录: {config.VGGFACE2_VAL_DIR}")
    print(f"验证集是否存在: {config.VGGFACE2_VAL_DIR.exists()}")

    if config.VGGFACE2_TRAIN_DIR.exists():
        # 列出前几个目录
        dirs = [d for d in config.VGGFACE2_TRAIN_DIR.iterdir() if d.is_dir()]
        print(f"训练集有 {len(dirs)} 个身份目录")
        if dirs:
            print("前5个身份目录:")
            for i, d in enumerate(dirs[:5]):
                jpg_count = len(list(d.glob("*.jpg")))
                print(f"  {i + 1}. {d.name}: {jpg_count} 张JPG图片")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

    return True


if __name__ == "__main__":
    main()