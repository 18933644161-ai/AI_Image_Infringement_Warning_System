"""
图像处理器模块
处理上传的图片，进行人脸检测、特征提取、相似度比对和干扰处理
"""

from datetime import datetime
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import json
import sqlite3
from PIL import Image as PILImage
import hashlib

from config import Config
from feature_extractor import FaceFeatureExtractor
from feature_database import FeatureDatabase
from image_interference import ImageInterference


class ImageProcessor:
    """图片处理器 - 核心处理类"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.feature_extractor = FaceFeatureExtractor()
        self.feature_db = FeatureDatabase(config)
        self.interference = ImageInterference(config)

        # 创建输出目录
        self.output_dir = self.config.OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)

    def process_single_image(self, image_path: str,
                             threshold: float = None,
                             interference_method: str = None,
                             save_result: bool = True,
                             return_interference_comparison: bool = False) -> Dict[str, Any]:
        """
        处理单张图片，可返回干扰前后对比数据

        Args:
            image_path: 图片路径
            threshold: 相似度阈值
            interference_method: 干扰方法
            save_result: 是否保存结果
            return_interference_comparison: 是否返回干扰前后对比数据（用于测试）

        Returns:
            处理结果字典，包含干扰前后相似度对比
        """
        start_time = time.time()

        if threshold is None:
            threshold = self.config.SIMILARITY_THRESHOLD

        if interference_method is None:
            interference_method = "blur"

        # 验证图片路径
        if not Path(image_path).exists():
            return {
                "success": False,
                "error": f"图片不存在: {image_path}",
                "processing_time": 0
            }

        try:
            # 读取图片
            original_image = cv2.imread(image_path)
            if original_image is None:
                return {
                    "success": False,
                    "error": f"无法读取图片: {image_path}",
                    "processing_time": 0
                }

            # 提取人脸特征
            features = self.feature_extractor.extract_from_array(original_image)

            if not features:
                processing_time = time.time() - start_time

                # 记录处理日志
                self.feature_db.log_processing(
                    image_path=image_path,
                    person_id=None,
                    similarity_score=None,
                    interference_applied=False,
                    interference_method=None,
                    processing_time=processing_time
                )

                return {
                    "success": True,
                    "image_path": image_path,
                    "face_detected": False,
                    "face_count": 0,
                    "max_similarity": 0.0,
                    "before_interference_similarity": 0.0,  # 新增：干扰前相似度
                    "similar_persons": [],
                    "risk_level": "safe",
                    "interference_applied": False,
                    "processing_time": processing_time
                }

            # 获取主要人脸
            main_face = self.feature_extractor.get_dominant_face(features)
            if not main_face:
                main_face = features[0]

            # 与数据库比对（获取原始相似度）
            similar_persons = self.feature_db.find_similar(
                main_face['embedding'],
                threshold=threshold
            )

            # 记录干扰前的相似度
            before_interference_similarity = similar_persons[0][2] if similar_persons else 0.0

            # 确定风险等级
            max_similarity = before_interference_similarity
            risk_level = self._determine_risk_level(max_similarity, threshold)

            # 准备人脸区域
            face_regions = [feature['bbox'] for feature in features]

            # 应用干扰（如果需要）
            interference_applied = False
            processed_image = original_image.copy()
            after_interference_similarity = 0.0  # 干扰后相似度

            if similar_persons and max_similarity > threshold:
                # 计算干扰强度
                interference_strength = self.interference.calculate_interference_strength(
                    max_similarity, threshold
                )

                if interference_strength > 0:
                    # 应用干扰
                    processed_image = self.interference.apply_interference(
                        original_image.copy(),
                        face_regions,
                        interference_method,
                        interference_strength
                    )
                    interference_applied = True

                    # 如果需要测试干扰效果，计算干扰后的相似度
                    if return_interference_comparison:
                        after_interference_similarity = self._calculate_after_interference_similarity(
                            processed_image, face_regions, similar_persons[0][0] if similar_persons else None
                        )
                    else:
                        # 默认使用干扰前的相似度（因为干扰后可能仍然相似）
                        after_interference_similarity = max_similarity

            # 如果没有应用干扰，干扰前后相似度相同
            if not interference_applied:
                after_interference_similarity = before_interference_similarity

            processing_time = time.time() - start_time

            # 记录处理日志（新增干扰前后相似度字段）
            person_id = similar_persons[0][0] if similar_persons else None
            self.feature_db.log_processing_with_comparison(
                image_path=image_path,
                person_id=person_id,
                before_similarity=before_interference_similarity,
                after_similarity=after_interference_similarity,
                interference_applied=interference_applied,
                interference_method=interference_method if interference_applied else None,
                processing_time=processing_time
            )

            # 保存结果
            output_path = None
            if save_result:
                output_path = self._save_processed_image(
                    image_path,
                    processed_image,
                    original_image,
                    max_similarity,
                    after_interference_similarity,  # 新增参数
                    similar_persons,
                    interference_applied,
                    interference_method
                )

            # 构建结果
            result = {
                "success": True,
                "image_path": image_path,
                "output_path": str(output_path) if output_path else None,
                "face_detected": True,
                "face_count": len(features),
                "face_regions": face_regions,
                "max_similarity": float(max_similarity),  # 这是干扰前的最高相似度
                "before_interference_similarity": float(before_interference_similarity),  # 新增：干扰前相似度
                "after_interference_similarity": float(after_interference_similarity),  # 新增：干扰后相似度
                "similarity_reduction": float(before_interference_similarity - after_interference_similarity),  # 新增：相似度降低值
                "similar_persons": [
                    {
                        "id": person_id,
                        "name": person_name,
                        "similarity": float(similarity),
                        "weighted_similarity": float(weighted_sim)
                    }
                    for person_id, person_name, weighted_sim, similarity, _, _ in similar_persons
                ],
                "risk_level": risk_level,
                "interference_applied": interference_applied,
                "interference_method": interference_method if interference_applied else None,
                "interference_strength": float(interference_strength) if interference_applied else 0.0,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }

    def _calculate_after_interference_similarity(self, processed_image: np.ndarray,
                                                face_regions: List[List[int]],
                                                person_id: Optional[int]) -> float:
        """
        计算干扰后的相似度（对干扰后的图片重新提取特征并比对）

        Args:
            processed_image: 干扰后的图片
            face_regions: 人脸区域
            person_id: 原始匹配的人物ID

        Returns:
            干扰后的相似度分数
        """
        try:
            # 对干扰后的图片重新提取特征
            features_after = self.feature_extractor.extract_from_array(processed_image)

            if not features_after:
                return 0.0

            # 获取主要人脸
            main_face_after = self.feature_extractor.get_dominant_face(features_after)
            if not main_face_after:
                return 0.0

            # 重新与数据库比对
            if person_id:
                # 只与原始匹配的人物进行比对
                conn = sqlite3.connect(self.feature_db.db_path)
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT embedding FROM face_features 
                    WHERE person_id = ? 
                    ORDER BY quality_score DESC LIMIT 1
                ''', (person_id,))

                row = cursor.fetchone()
                conn.close()

                if row:
                    stored_embedding = np.frombuffer(row[0], dtype=np.float32)
                    after_similarity = self.feature_db._cosine_similarity(
                        main_face_after['embedding'], stored_embedding
                    )
                    return float(after_similarity)

            # 如果没有特定person_id，与整个数据库比对取最高
            similar_persons_after = self.feature_db.find_similar(
                main_face_after['embedding'],
                threshold=0.1  # 使用很低的阈值，确保能检测到
            )

            if similar_persons_after:
                return float(similar_persons_after[0][2])

            return 0.0

        except Exception as e:
            print(f"计算干扰后相似度失败: {e}")
            return 0.0

    def process_batch(self, input_dir: str, output_dir: str = None,
                      threshold: float = None, interference_method: str = None) -> Dict[str, Any]:
        """
        批量处理图片

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            threshold: 相似度阈值
            interference_method: 干扰方法

        Returns:
            批量处理结果
        """
        start_time = time.time()

        input_path = Path(input_dir)
        if not input_path.exists():
            return {
                "success": False,
                "error": f"输入目录不存在: {input_dir}"
            }

        # 设置输出目录
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.output_dir / "batch_results" / datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path.mkdir(parents=True, exist_ok=True)

        # 收集图片文件
        image_files = []
        for ext in self.config.SUPPORTED_IMAGE_EXTENSIONS:
            image_files.extend(list(input_path.glob(f"*{ext}")))
            image_files.extend(list(input_path.glob(f"*{ext.upper()}")))

        print(f"找到 {len(image_files)} 张图片")

        # 处理每张图片
        results = []
        statistics = {
            "total": len(image_files),
            "processed": 0,
            "success": 0,
            "failed": 0,
            "face_detected": 0,
            "high_risk": 0,
            "interference_applied": 0,
            "total_similarity_reduction": 0.0,  # 新增：总相似度降低值
            "avg_similarity_reduction": 0.0     # 新增：平均相似度降低值
        }

        for i, image_file in enumerate(image_files, 1):
            print(f"处理图片 {i}/{len(image_files)}: {image_file.name}")

            result = self.process_single_image(
                str(image_file),
                threshold=threshold,
                interference_method=interference_method,
                save_result=True,
                return_interference_comparison=True  # 批量处理时获取干扰前后对比
            )

            if result["success"]:
                statistics["success"] += 1
                if result.get("face_detected", False):
                    statistics["face_detected"] += 1
                if result.get("risk_level") == "high":
                    statistics["high_risk"] += 1
                if result.get("interference_applied", False):
                    statistics["interference_applied"] += 1
                    # 累加相似度降低值
                    reduction = result.get("similarity_reduction", 0.0)
                    statistics["total_similarity_reduction"] += reduction
            else:
                statistics["failed"] += 1

            statistics["processed"] += 1
            results.append(result)

        # 计算平均相似度降低值
        if statistics["interference_applied"] > 0:
            statistics["avg_similarity_reduction"] = (
                statistics["total_similarity_reduction"] / statistics["interference_applied"]
            )

        total_time = time.time() - start_time

        # 生成报告
        report = self._generate_batch_report(results, statistics, total_time)

        # 保存报告
        report_path = output_path / "batch_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "input_dir": str(input_dir),
            "output_dir": str(output_path),
            "report_path": str(report_path),
            "statistics": statistics,
            "total_time": total_time,
            "average_time": total_time / len(image_files) if image_files else 0,
            "results": results
        }

    def build_feature_database_from_dataset(self,
                                            max_identities: int = None,
                                            max_images_per_identity: int = None,
                                            use_train_set: bool = None,
                                            use_val_set: bool = None):
        """从VGG-Face2数据集构建特征数据库"""
        from dataset_loader import DatasetLoader

        # 使用配置或默认值
        config = self.config.DATABASE_BUILD_CONFIG
        if max_identities is None:
            max_identities = config["max_identities"]
        if max_images_per_identity is None:
            max_images_per_identity = config["max_images_per_identity"]
        if use_train_set is None:
            use_train_set = config["use_train_set"]
        if use_val_set is None:
            use_val_set = config["use_val_set"]

        print(f"开始从VGG-Face2数据集构建特征数据库...")
        print(f"配置: 最多 {max_identities} 个身份，每个最多 {max_images_per_identity} 张图片")
        print(f"使用训练集: {use_train_set}, 使用验证集: {use_val_set}")

        # 清空现有数据库
        self.feature_db.clear_database()

        # 加载数据集
        dataset_loader = DatasetLoader(self.config)

        # 获取有效的身份列表（跳过无效身份）
        print(f"\n获取有效的身份列表...")
        valid_identities = dataset_loader.get_valid_identities(max_count=max_identities)

        if not valid_identities:
            print("错误: 没有找到有效的身份")
            return {
                "success": False,
                "error": "没有找到有效的身份"
            }

        print(f"找到 {len(valid_identities)} 个有效身份用于构建特征库")

        total_features = 0
        skipped_identities = 0

        for identity_info in valid_identities:
            identity_name = identity_info['name']
            identity_id = identity_info['id']

            # 提取特征
            embeddings = dataset_loader.extract_features_for_identity(
                identity_info,
                max_images=max_images_per_identity
            )

            # 检查是否提取到足够的特征
            min_features = self.config.DATABASE_BUILD_CONFIG["min_features_per_identity"]
            if len(embeddings) < min_features:
                print(f"  跳过 {identity_name}: 只提取到 {len(embeddings)} 个特征（需要至少 {min_features} 个）")
                skipped_identities += 1
                continue

            # 添加人物到数据库
            person_id = self.feature_db.add_person(
                name=identity_name,
                category="celebrity",
                description=f"从VGG-Face2数据集添加的身份: {identity_name}"
            )

            # 添加特征
            for i, embedding in enumerate(embeddings):
                self.feature_db.add_feature(
                    person_id=person_id,
                    embedding=embedding,
                    source_image=f"{identity_name}_{i + 1}",
                    quality_score=1.0,
                    metadata={
                        "source_dataset": "vggface2",
                        "feature_index": i,
                        "extraction_time": datetime.now().isoformat()
                    }
                )
                total_features += 1

            print(f"  添加身份: {identity_name} ({len(embeddings)} 个特征)")

        print(f"\n特征数据库构建完成！")
        print(f"- 总共处理 {len(valid_identities)} 个身份")
        print(f"- 跳过了 {skipped_identities} 个身份（特征不足）")
        print(f"- 共添加 {total_features} 个特征")

        # 显示统计信息
        stats = self.feature_db.get_statistics()
        print(f"- 数据库统计: {stats['person_count']} 个人物，{stats['feature_count']} 个特征")

        return {
            "success": True,
            "total_identities": len(valid_identities),
            "skipped_identities": skipped_identities,
            "total_features": total_features,
            "database_stats": stats,
            "build_config": {
                "max_identities": max_identities,
                "max_images_per_identity": max_images_per_identity,
                "min_features_per_identity": self.config.DATABASE_BUILD_CONFIG["min_features_per_identity"],
                "use_train_set": use_train_set,
                "use_val_set": use_val_set
            }
        }

    def _determine_risk_level(self, similarity: float, threshold: float) -> str:
        """确定风险等级"""
        if similarity < threshold:
            return "safe"
        elif similarity < self.config.MEDIUM_RISK_THRESHOLD:
            return "low"
        elif similarity < self.config.HIGH_RISK_THRESHOLD:
            return "medium"
        else:
            return "high"

    def _save_processed_image(self, original_path: str, processed_image: np.ndarray,
                              original_image: np.ndarray, before_similarity: float,
                              after_similarity: float,  # 新增参数
                              similar_persons: List, interference_applied: bool,
                              interference_method: str) -> Path:
        """保存处理后的图片"""
        # 生成文件名
        original_name = Path(original_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if similar_persons:
            person_name = similar_persons[0][1].replace(" ", "_")
            risk_level = self._determine_risk_level(before_similarity, self.config.SIMILARITY_THRESHOLD)
            filename = f"{risk_level}_{person_name}_{original_name}_{timestamp}.jpg"
        else:
            filename = f"safe_{original_name}_{timestamp}.jpg"

        # 保存处理后的图片
        output_path = self.output_dir / "protected" / filename
        output_path.parent.mkdir(exist_ok=True)

        cv2.imwrite(str(output_path), processed_image)

        # 保存原始图片（如果不同）
        if not np.array_equal(processed_image, original_image):
            original_output = self.output_dir / "original" / f"original_{filename}"
            original_output.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(original_output), original_image)

        # 保存元数据（新增干扰前后相似度对比）
        metadata = {
            "original_path": original_path,
            "output_path": str(output_path),
            "before_interference_similarity": float(before_similarity),
            "after_interference_similarity": float(after_similarity),
            "similarity_reduction": float(before_similarity - after_similarity),
            "similar_persons": [
                {
                    "name": person_name,
                    "similarity": float(similarity),
                    "weighted_similarity": float(weighted_sim)
                }
                for _, person_name, weighted_sim, similarity, _, _ in similar_persons
            ],
            "interference_applied": interference_applied,
            "interference_method": interference_method if interference_applied else None,
            "timestamp": datetime.now().isoformat()
        }

        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return output_path

    def _generate_batch_report(self, results: List[Dict], statistics: Dict, total_time: float) -> Dict:
        """生成批量处理报告（新增相似度降低统计）"""
        # 高风险图片
        high_risk_images = []
        for result in results:
            if result.get("success", False) and result.get("risk_level") == "high":
                high_risk_images.append({
                    "image_path": result.get("image_path"),
                    "before_similarity": result.get("before_interference_similarity"),
                    "after_similarity": result.get("after_interference_similarity"),
                    "similarity_reduction": result.get("similarity_reduction"),
                    "similar_persons": result.get("similar_persons", []),
                    "interference_applied": result.get("interference_applied", False)
                })

        # 按相似度排序
        high_risk_images.sort(key=lambda x: x["before_similarity"], reverse=True)

        # 最常见的相似人物
        person_counts = {}
        for result in results:
            if result.get("success", False) and result.get("similar_persons"):
                for person in result["similar_persons"]:
                    person_name = person.get("name", "Unknown")
                    person_counts[person_name] = person_counts.get(person_name, 0) + 1

        common_persons = sorted(person_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "summary": {
                "total_images": statistics["total"],
                "successfully_processed": statistics["success"],
                "failed": statistics["failed"],
                "face_detected": statistics["face_detected"],
                "high_risk_detected": statistics["high_risk"],
                "interference_applied": statistics["interference_applied"],
                "total_similarity_reduction": statistics["total_similarity_reduction"],
                "average_similarity_reduction": statistics["avg_similarity_reduction"],
                "total_processing_time": total_time,
                "average_processing_time": total_time / statistics["total"] if statistics["total"] > 0 else 0,
                "generated_at": datetime.now().isoformat()
            },
            "high_risk_images": high_risk_images[:10],  # 前10个高风险图片
            "common_persons": [
                {"name": name, "count": count} for name, count in common_persons
            ],
            "processing_statistics": {
                "success_rate": statistics["success"] / statistics["total"] * 100 if statistics["total"] > 0 else 0,
                "face_detection_rate": statistics["face_detected"] / statistics["success"] * 100 if statistics["success"] > 0 else 0,
                "high_risk_rate": statistics["high_risk"] / statistics["face_detected"] * 100 if statistics["face_detected"] > 0 else 0,
                "interference_effectiveness": statistics["avg_similarity_reduction"] * 100  # 干扰效果指标
            }
        }


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("ImageProcessor 测试")
    print("=" * 60)

    config = Config()
    processor = ImageProcessor(config)

    print(f"✓ ImageProcessor 初始化成功")
    print(f"配置:")
    print(f"  特征提取器: {type(processor.feature_extractor).__name__}")
    print(f"  特征数据库: {type(processor.feature_db).__name__}")
    print(f"  干扰模块: {type(processor.interference).__name__}")
    print(f"  输出目录: {processor.output_dir}")

    print("\n✓ ImageProcessor 测试通过")