import sqlite3
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import hashlib
from datetime import datetime

from config import Config


class FeatureDatabase:
    """特征数据库管理器"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.db_path = self.config.DATABASE_PATH
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建高风险人物表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS high_risk_persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT,
                description TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建特征表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                source_image TEXT,
                quality_score REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES high_risk_persons (id) ON DELETE CASCADE
            )
        ''')

        # 创建处理日志表（包含干扰前后相似度字段）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT NOT NULL,
                image_path TEXT,
                person_id INTEGER,
                similarity_score REAL,
                before_similarity REAL,     -- 新增：干扰前相似度
                after_similarity REAL,      -- 新增：干扰后相似度
                interference_applied BOOLEAN,
                interference_method TEXT,
                processing_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES high_risk_persons (id)
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_person_name ON high_risk_persons(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_hash ON processing_logs(image_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON processing_logs(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_similarity ON processing_logs(similarity_score)')

        conn.commit()
        conn.close()

    def add_person(self, name: str, category: str = "celebrity",
                   description: str = "", metadata: Dict = None) -> int:
        """添加高风险人物"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"

        # 检查是否已存在
        cursor.execute('SELECT id FROM high_risk_persons WHERE name = ?', (name,))
        existing = cursor.fetchone()

        if existing:
            person_id = existing[0]
            # 更新现有记录
            cursor.execute('''
                UPDATE high_risk_persons 
                SET category = ?, description = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (category, description, metadata_json, person_id))
        else:
            # 插入新记录
            cursor.execute('''
                INSERT INTO high_risk_persons (name, category, description, metadata)
                VALUES (?, ?, ?, ?)
            ''', (name, category, description, metadata_json))
            person_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return person_id

    def add_feature(self, person_id: int, embedding: np.ndarray,
                    source_image: str = "", quality_score: float = 1.0,
                    metadata: Dict = None) -> int:
        """添加人脸特征"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 序列化特征向量
        embedding_blob = embedding.astype(np.float32).tobytes()
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"

        cursor.execute('''
            INSERT INTO face_features (person_id, embedding, source_image, quality_score, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (person_id, embedding_blob, source_image, quality_score, metadata_json))

        feature_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return feature_id

    def find_similar(self, query_embedding: np.ndarray,
                     threshold: float = None,
                     top_k: int = 5) -> List[Tuple]:
        """查找相似的高风险人物"""
        if threshold is None:
            threshold = self.config.SIMILARITY_THRESHOLD

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取所有特征
        cursor.execute('''
            SELECT ff.id, ff.person_id, hp.name, ff.embedding, ff.quality_score
            FROM face_features ff
            JOIN high_risk_persons hp ON ff.person_id = hp.id
        ''')

        results = []

        for row in cursor.fetchall():
            feature_id, person_id, person_name, embedding_blob, quality_score = row

            try:
                # 反序列化特征向量
                stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)

                # 计算相似度（考虑质量分数）
                similarity = self._cosine_similarity(query_embedding, stored_embedding)
                weighted_similarity = similarity * quality_score

                if weighted_similarity >= threshold:
                    results.append((person_id, person_name, weighted_similarity,
                                    similarity, quality_score, feature_id))
            except Exception as e:
                print(f"处理特征 {feature_id} 时出错: {e}")
                continue

        conn.close()

        # 按加权相似度排序
        results.sort(key=lambda x: x[2], reverse=True)

        # 按人物分组，取最高相似度
        person_results = {}
        for person_id, person_name, weighted_sim, sim, quality, feature_id in results:
            if person_id not in person_results or weighted_sim > person_results[person_id][2]:
                person_results[person_id] = (person_id, person_name, weighted_sim, sim, quality, feature_id)

        # 返回前top_k个结果
        top_results = list(person_results.values())[:top_k]

        return top_results

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0

        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_all_persons(self) -> List[Dict]:
        """获取所有高风险人物"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, name, category, description, metadata, 
                   created_at, updated_at
            FROM high_risk_persons
            ORDER BY name
        ''')

        persons = []
        for row in cursor.fetchall():
            person = {
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'description': row[3],
                'metadata': json.loads(row[4]) if row[4] else {},
                'created_at': row[5],
                'updated_at': row[6],
                'feature_count': 0
            }
            persons.append(person)

        conn.close()

        # 获取每个人物的特征数量
        for person in persons:
            person['feature_count'] = self.get_feature_count(person['id'])

        return persons

    def get_feature_count(self, person_id: int) -> int:
        """获取指定人物的特征数量"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM face_features WHERE person_id = ?', (person_id,))
        count = cursor.fetchone()[0]

        conn.close()
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 人物数量
        cursor.execute('SELECT COUNT(*) FROM high_risk_persons')
        person_count = cursor.fetchone()[0]

        # 特征数量
        cursor.execute('SELECT COUNT(*) FROM face_features')
        feature_count = cursor.fetchone()[0]

        # 日志数量
        cursor.execute('SELECT COUNT(*) FROM processing_logs')
        log_count = cursor.fetchone()[0]

        # 最近处理时间
        cursor.execute('SELECT MAX(timestamp) FROM processing_logs')
        last_processed = cursor.fetchone()[0]

        # 新增：干扰效果统计
        cursor.execute('''
            SELECT 
                COUNT(*) as interference_count,
                AVG(before_similarity) as avg_before,
                AVG(after_similarity) as avg_after,
                AVG(before_similarity - after_similarity) as avg_reduction
            FROM processing_logs 
            WHERE interference_applied = 1 
            AND before_similarity IS NOT NULL 
            AND after_similarity IS NOT NULL
        ''')
        interference_stats_row = cursor.fetchone()

        interference_stats = {
            'interference_count': interference_stats_row[0] if interference_stats_row else 0,
            'avg_before_similarity': float(interference_stats_row[1]) if interference_stats_row and
                                                                         interference_stats_row[1] else 0.0,
            'avg_after_similarity': float(interference_stats_row[2]) if interference_stats_row and
                                                                        interference_stats_row[2] else 0.0,
            'avg_reduction': float(interference_stats_row[3]) if interference_stats_row and interference_stats_row[
                3] else 0.0
        }

        conn.close()

        return {
            'person_count': person_count,
            'feature_count': feature_count,
            'log_count': log_count,
            'last_processed': last_processed,
            'interference_stats': interference_stats,  # 新增：干扰效果统计
            'database_size': self.db_path.stat().st_size if self.db_path.exists() else 0
        }

    def log_processing(self, image_path: str, person_id: Optional[int],
                       similarity_score: Optional[float], interference_applied: bool,
                       interference_method: Optional[str], processing_time: float):
        """记录处理日志（兼容旧版本）"""
        self.log_processing_with_comparison(
            image_path=image_path,
            person_id=person_id,
            before_similarity=similarity_score,  # 旧版本只记录一个相似度，用作before
            after_similarity=similarity_score,  # after也设为相同值
            interference_applied=interference_applied,
            interference_method=interference_method,
            processing_time=processing_time
        )

    def log_processing_with_comparison(self, image_path: str, person_id: Optional[int],
                                       before_similarity: Optional[float], after_similarity: Optional[float],
                                       interference_applied: bool, interference_method: Optional[str],
                                       processing_time: float):
        """记录处理日志（包含干扰前后相似度对比）"""
        # 计算图像哈希
        try:
            with open(image_path, 'rb') as f:
                image_hash = hashlib.md5(f.read()).hexdigest()
        except:
            image_hash = hashlib.md5(image_path.encode()).hexdigest()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 确定存储哪个相似度作为主要相似度分数
        # 如果有干扰，存储干扰后的相似度；如果没有，存储原始相似度
        main_similarity_score = after_similarity if interference_applied and after_similarity is not None else before_similarity

        cursor.execute('''
            INSERT INTO processing_logs 
            (image_hash, image_path, person_id, similarity_score, 
             before_similarity, after_similarity,
             interference_applied, interference_method, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (image_hash, image_path, person_id, main_similarity_score,
              before_similarity, after_similarity,
              interference_applied, interference_method, processing_time))

        conn.commit()
        conn.close()

    def get_interference_analysis(self, limit: int = 100) -> List[Dict]:
        """获取干扰效果分析数据（用于论文）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT 
                id, image_path, person_id, 
                before_similarity, after_similarity,
                (before_similarity - after_similarity) as reduction,
                interference_applied, interference_method,
                timestamp
            FROM processing_logs 
            WHERE interference_applied = 1 
            AND before_similarity IS NOT NULL 
            AND after_similarity IS NOT NULL
            ORDER BY reduction DESC
            LIMIT ?
        ''', (limit,))

        results = []
        for row in cursor.fetchall():
            log_id, image_path, person_id, before_sim, after_sim, reduction, applied, method, timestamp = row

            # 获取人物名称
            person_name = None
            if person_id:
                cursor2 = conn.cursor()
                cursor2.execute('SELECT name FROM high_risk_persons WHERE id = ?', (person_id,))
                person_row = cursor2.fetchone()
                if person_row:
                    person_name = person_row[0]

            results.append({
                'id': log_id,
                'image_path': image_path,
                'person_id': person_id,
                'person_name': person_name,
                'before_similarity': float(before_sim) if before_sim else 0.0,
                'after_similarity': float(after_sim) if after_sim else 0.0,
                'reduction': float(reduction) if reduction else 0.0,
                'reduction_percentage': float(
                    (before_sim - after_sim) / before_sim * 100) if before_sim and before_sim > 0 else 0.0,
                'interference_applied': bool(applied),
                'interference_method': method,
                'timestamp': timestamp,
                'risk_level_before': self._get_risk_level(before_sim),
                'risk_level_after': self._get_risk_level(after_sim)
            })

        conn.close()
        return results

    def _get_risk_level(self, similarity: float) -> str:
        """根据相似度获取风险等级"""
        config = Config()
        if similarity < config.SIMILARITY_THRESHOLD:
            return "safe"
        elif similarity < config.MEDIUM_RISK_THRESHOLD:
            return "low"
        elif similarity < config.HIGH_RISK_THRESHOLD:
            return "medium"
        else:
            return "high"

    def get_similarity_comparison_report(self) -> Dict[str, Any]:
        """生成相似度对比报告（论文用）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取所有干扰案例的统计
        cursor.execute('''
            SELECT 
                COUNT(*) as total_cases,
                AVG(before_similarity) as avg_before,
                AVG(after_similarity) as avg_after,
                AVG(before_similarity - after_similarity) as avg_reduction,
                MIN(before_similarity - after_similarity) as min_reduction,
                MAX(before_similarity - after_similarity) as max_reduction,
                interference_method,
                COUNT(*) as method_count
            FROM processing_logs 
            WHERE interference_applied = 1 
            AND before_similarity IS NOT NULL 
            AND after_similarity IS NOT NULL
            GROUP BY interference_method
        ''')

        method_stats = []
        for row in cursor.fetchall():
            method_stats.append({
                'method': row[6] if row[6] else 'unknown',
                'count': row[7],
                'avg_before': float(row[1]) if row[1] else 0.0,
                'avg_after': float(row[2]) if row[2] else 0.0,
                'avg_reduction': float(row[3]) if row[3] else 0.0,
                'min_reduction': float(row[4]) if row[4] else 0.0,
                'max_reduction': float(row[5]) if row[5] else 0.0
            })

        # 按干扰效果排序
        method_stats.sort(key=lambda x: x['avg_reduction'], reverse=True)

        # 获取风险等级转换统计
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN before_similarity >= ? AND after_similarity < ? THEN 1 ELSE 0 END) as high_to_safe,
                SUM(CASE WHEN before_similarity >= ? AND after_similarity >= ? AND after_similarity < ? THEN 1 ELSE 0 END) as high_to_low,
                SUM(CASE WHEN before_similarity >= ? AND after_similarity >= ? THEN 1 ELSE 0 END) as high_still_high
            FROM processing_logs 
            WHERE interference_applied = 1 
            AND before_similarity IS NOT NULL 
            AND after_similarity IS NOT NULL
        ''', (Config().SIMILARITY_THRESHOLD, Config().SIMILARITY_THRESHOLD,
              Config().SIMILARITY_THRESHOLD, Config().SIMILARITY_THRESHOLD, Config().MEDIUM_RISK_THRESHOLD,
              Config().HIGH_RISK_THRESHOLD, Config().HIGH_RISK_THRESHOLD))

        risk_conversion_row = cursor.fetchone()

        risk_conversion = {
            'total_cases': risk_conversion_row[0] if risk_conversion_row else 0,
            'high_to_safe': risk_conversion_row[1] if risk_conversion_row else 0,
            'high_to_safe_rate': (risk_conversion_row[1] / risk_conversion_row[0] * 100) if risk_conversion_row and
                                                                                            risk_conversion_row[
                                                                                                0] > 0 else 0.0,
            'high_to_low': risk_conversion_row[2] if risk_conversion_row else 0,
            'high_to_low_rate': (risk_conversion_row[2] / risk_conversion_row[0] * 100) if risk_conversion_row and
                                                                                           risk_conversion_row[
                                                                                               0] > 0 else 0.0,
            'high_still_high': risk_conversion_row[3] if risk_conversion_row else 0,
            'high_still_high_rate': (risk_conversion_row[3] / risk_conversion_row[0] * 100) if risk_conversion_row and
                                                                                               risk_conversion_row[
                                                                                                   0] > 0 else 0.0
        }

        conn.close()

        return {
            'interference_methods': method_stats,
            'risk_conversion': risk_conversion,
            'summary': {
                'most_effective_method': method_stats[0]['method'] if method_stats else 'none',
                'best_reduction': method_stats[0]['max_reduction'] if method_stats else 0.0,
                'avg_overall_reduction': sum(m['avg_reduction'] * m['count'] for m in method_stats) / sum(
                    m['count'] for m in method_stats) if method_stats else 0.0,
                'success_rate': risk_conversion['high_to_safe_rate'] + risk_conversion['high_to_low_rate']
            }
        }

    def clear_database(self):
        """清空数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM face_features')
        cursor.execute('DELETE FROM high_risk_persons')
        cursor.execute('DELETE FROM processing_logs')

        # 重置自增ID
        cursor.execute(
            'DELETE FROM sqlite_sequence WHERE name IN ("high_risk_persons", "face_features", "processing_logs")')

        conn.commit()
        conn.close()

        print("数据库已清空")