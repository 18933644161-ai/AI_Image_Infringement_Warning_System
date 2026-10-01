#!/usr/bin/env python3
"""
AI图像侵权预警系统主入口
"""

import argparse
import sys
from pathlib import Path

from config import Config
from model_downloader import ModelDownloader
from image_processor import ImageProcessor
from dataset_loader import DatasetLoader
from feature_database import FeatureDatabase
from web_interface import WebInterface


def setup_system():
    """系统初始化"""
    print("=" * 60)
    print("AI图像侵权预警系统 - 初始化")
    print("=" * 60)

    # 创建配置
    config = Config()

    # 创建目录
    config.create_directories()

    return config


def check_models(config):
    """检查并下载模型"""
    print("\n检查模型文件...")

    downloader = ModelDownloader(config)
    existing_files = downloader.check_model_exists()

    if len(existing_files) >= 2:
        print(f"✓ 模型文件已存在: {', '.join(existing_files)}")
        return True
    else:
        print(f"✗ 模型文件不完整，需要下载")
        return downloader.run()


def build_feature_database(config, max_identities=None, max_images=None,
                           use_train_set=None, use_val_set=None):
    """构建特征数据库"""
    print("\n构建特征数据库...")

    processor = ImageProcessor(config)

    # 使用参数或配置默认值
    build_config = config.DATABASE_BUILD_CONFIG

    if max_identities is None:
        max_identities = build_config["default_max_identities"]
    if max_images is None:
        max_images = build_config["default_max_images"]
    if use_train_set is None:
        use_train_set = build_config["use_train_set"]
    if use_val_set is None:
        use_val_set = build_config["use_val_set"]

    print(f"构建配置:")
    print(f"  最大身份数: {max_identities}")
    print(f"  每个身份最大图片数: {max_images}")
    print(f"  使用训练集: {use_train_set}")
    print(f"  使用验证集: {use_val_set}")

    result = processor.build_feature_database_from_dataset(
        max_identities=max_identities,
        max_images_per_identity=max_images,
        use_train_set=use_train_set,
        use_val_set=use_val_set
    )

    return result


def test_dataset(config):
    """测试数据集加载"""
    print("\n测试VGG-Face2数据集加载...")

    loader = DatasetLoader(config)
    success = loader.test_vggface2_dataset()

    if success:
        print("\n✓ 数据集测试成功！")

        # 显示数据集统计
        stats = loader.get_identity_statistics()
        print(f"\n数据集统计:")
        print(f"  总身份数: {stats['total_identities']}")
        print(f"  训练集身份数: {stats.get('train_identities', 'N/A')}")
        print(f"  验证集身份数: {stats.get('val_identities', 'N/A')}")

        # 显示前10个身份
        identities = loader.get_all_identities(max_count=10)
        print(f"\n前10个身份:")
        for i, identity in enumerate(identities, 1):
            identity_id = loader.get_identity_id_by_name(identity)
            print(f"  {i:2d}. {identity} (ID: {identity_id})")

            # 获取图片数量
            image_paths = loader.get_image_paths_for_identity(identity)
            print(f"      图片数: {len(image_paths)}")
    else:
        print("\n✗ 数据集测试失败")

    return success


def process_single_image(config, image_path, threshold=None, method=None):
    """处理单张图片"""
    print(f"\n处理单张图片: {image_path}")

    processor = ImageProcessor(config)
    result = processor.process_single_image(
        image_path=image_path,
        threshold=threshold,
        interference_method=method,
        save_result=True
    )

    if result["success"]:
        print(f"✓ 处理成功")
        print(f"  检测到人脸: {result['face_detected']}")

        if result["face_detected"]:
            print(f"  人脸数量: {result['face_count']}")
            print(f"  最高相似度: {result['max_similarity']:.4f}")
            print(f"  风险等级: {result['risk_level']}")

            if result["similar_persons"]:
                print(f"  匹配人物: {result['similar_persons'][0]['name']}")

            if result["interference_applied"]:
                print(f"  已应用干扰: {result['interference_method']}")

            if result["output_path"]:
                print(f"  输出文件: {result['output_path']}")
    else:
        print(f"✗ 处理失败: {result.get('error', '未知错误')}")

    return result


def process_batch(config, input_dir, output_dir=None, threshold=None, method=None):
    """批量处理图片"""
    print(f"\n批量处理图片目录: {input_dir}")

    processor = ImageProcessor(config)
    result = processor.process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        threshold=threshold,
        interference_method=method
    )

    if result["success"]:
        stats = result["statistics"]
        print(f"✓ 批量处理完成")
        print(f"  总图片数: {stats['total']}")
        print(f"  成功处理: {stats['success']}")
        print(f"  检测到人脸: {stats['face_detected']}")
        print(f"  高风险图片: {stats['high_risk']}")
        print(f"  应用干扰: {stats['interference_applied']}")
        print(f"  输出目录: {result['output_dir']}")
        print(f"  报告文件: {result['report_path']}")
    else:
        print(f"✗ 批量处理失败: {result.get('error', '未知错误')}")

    return result


def show_database_info(config):
    """显示数据库信息"""
    print("\n数据库信息:")

    db = FeatureDatabase(config)
    stats = db.get_statistics()

    print(f"  人物数量: {stats['person_count']}")
    print(f"  特征数量: {stats['feature_count']}")
    print(f"  处理日志: {stats['log_count']}")
    print(f"  最后处理: {stats['last_processed'] or '无'}")

    # 显示人物列表
    persons = db.get_all_persons()
    if persons:
        print(f"\n人物列表 (共 {len(persons)} 人):")
        for i, person in enumerate(persons[:20], 1):
            print(f"  {i:2d}. {person['name']}: {person['feature_count']} 个特征")

        if len(persons) > 20:
            print(f"  ... 还有 {len(persons) - 20} 个人物")

    return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AI图像侵权预警系统 - VGG-Face2版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --setup                    # 系统初始化
  %(prog)s --test-dataset             # 测试VGG-Face2数据集
  %(prog)s --build-db                 # 构建特征数据库
  %(prog)s --process image.jpg        # 处理单张图片
  %(prog)s --batch input_dir          # 批量处理图片
  %(prog)s --web                      # 启动Web界面
  %(prog)s --info                     # 显示系统信息

数据库构建选项:
  --max-identities N       最大身份数量 (默认: 100)
  --max-images N          每个身份最大图片数 (默认: 10)
  --use-train             使用训练集 (默认: True)
  --use-val               使用验证集 (默认: False)
        """
    )

    # 主要模式
    parser.add_argument('--setup', action='store_true', help='系统初始化')
    parser.add_argument('--test-dataset', action='store_true', help='测试VGG-Face2数据集')
    parser.add_argument('--build-db', action='store_true', help='构建特征数据库')
    parser.add_argument('--process', type=str, metavar='IMAGE', help='处理单张图片')
    parser.add_argument('--batch', type=str, metavar='DIR', help='批量处理图片目录')
    parser.add_argument('--web', action='store_true', help='启动Web界面')
    parser.add_argument('--info', action='store_true', help='显示系统信息')

    # 数据库构建选项
    parser.add_argument('--max-identities', type=int, default=None,
                        help=f'最大身份数量 (默认: {Config.DATABASE_BUILD_CONFIG["default_max_identities"]})')
    parser.add_argument('--max-images', type=int, default=None,
                        help=f'每个身份最大图片数 (默认: {Config.DATABASE_BUILD_CONFIG["default_max_images"]})')
    parser.add_argument('--use-train', action='store_true', default=None,
                        help='使用训练集构建数据库')
    parser.add_argument('--use-val', action='store_true', default=None,
                        help='使用验证集构建数据库')

    # 处理选项
    parser.add_argument('--threshold', type=float, default=None,
                        help=f'相似度阈值 (默认: {Config.SIMILARITY_THRESHOLD})')
    parser.add_argument('--method', type=str, default='blur',
                        choices=Config.INTERFERENCE_METHODS,
                        help='干扰方法')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (仅用于批量处理)')

    args = parser.parse_args()

    # 如果没有参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # 初始化配置
    config = setup_system()

    # 执行不同模式
    if args.setup:
        # 检查并下载模型
        if check_models(config):
            print("\n✓ 系统初始化完成")
        else:
            print("\n✗ 系统初始化失败")
            sys.exit(1)

    elif args.test_dataset:
        # 测试数据集
        test_dataset(config)

    elif args.build_db:
        # 检查模型
        if not check_models(config):
            print("✗ 模型文件不完整，无法构建数据库")
            sys.exit(1)

        # 确定是否使用训练集和验证集
        use_train_set = args.use_train if args.use_train is not None else config.DATABASE_BUILD_CONFIG["use_train_set"]
        use_val_set = args.use_val if args.use_val is not None else config.DATABASE_BUILD_CONFIG["use_val_set"]

        # 构建特征数据库
        result = build_feature_database(
            config=config,
            max_identities=args.max_identities,
            max_images=args.max_images,
            use_train_set=use_train_set,
            use_val_set=use_val_set
        )

        print(f"\n✓ 数据库构建完成:")
        print(f"  添加了 {result['total_features']} 个特征")
        print(f"  来自 {result['total_identities']} 个身份")

    elif args.process:
        # 检查模型
        if not check_models(config):
            print("✗ 模型文件不完整，无法处理图片")
            sys.exit(1)

        # 检查图片文件
        image_path = Path(args.process)
        if not image_path.exists():
            print(f"✗ 图片文件不存在: {args.process}")
            sys.exit(1)

        # 处理单张图片
        process_single_image(config, str(image_path), args.threshold, args.method)

    elif args.batch:
        # 检查模型
        if not check_models(config):
            print("✗ 模型文件不完整，无法批量处理")
            sys.exit(1)

        # 检查输入目录
        input_dir = Path(args.batch)
        if not input_dir.exists():
            print(f"✗ 输入目录不存在: {args.batch}")
            sys.exit(1)

        # 批量处理
        process_batch(config, str(input_dir), args.output_dir, args.threshold, args.method)

    elif args.web:
        # 检查模型
        if not check_models(config):
            print("✗ 模型文件不完整，无法启动Web界面")
            sys.exit(1)

        # 启动Web界面
        print("\n启动Web界面...")
        interface = WebInterface(config)
        interface.launch()

    elif args.info:
        # 显示系统信息
        show_database_info(config)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()