import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import tempfile
from datetime import datetime
from pathlib import Path
import cv2

from config import Config
from image_processor import ImageProcessor
from feature_database import FeatureDatabase
from feature_extractor import FaceFeatureExtractor


class WebInterface:
    """Web界面类"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.processor = ImageProcessor(config)
        self.feature_db = FeatureDatabase(config)
        self.feature_extractor = FaceFeatureExtractor()
        self.demo = None

    def process_image(self, image, threshold, interference_method, show_details):
        """处理上传的图像"""
        if image is None:
            return None, None, "请上传图像", None

        try:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image_path = tmp.name
                image.save(image_path)

            # 处理图像
            result = self.processor.process_single_image(
                image_path=image_path,
                threshold=threshold,
                interference_method=interference_method,
                save_result=True,
                return_interference_comparison=True  # 确保获取干扰前后对比
            )

            # 清理临时文件
            os.unlink(image_path)

            if not result["success"]:
                error_msg = result.get("error", "处理失败")
                return None, None, f"❌ {error_msg}", None

            # 读取处理后的图像
            processed_image = None
            if result.get("output_path"):
                processed_img = cv2.imread(result["output_path"])
                if processed_img is not None:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                    processed_image = Image.fromarray(processed_img)

            # 生成报告
            report = self._generate_report(result)

            # 生成可视化
            fig = self._create_visualization(result, threshold)

            # 显示详细信息
            details = None
            if show_details:
                details = self._generate_details(result)

            return processed_image, fig, report, details

        except Exception as e:
            return None, None, f"❌ 处理过程中发生错误: {str(e)}", None

    def _generate_report(self, result):
        """生成报告 - 包含干扰前后相似度对比"""
        report = f"## 📊 Analysis Report\n\n"

        if not result["face_detected"]:
            report += "✅ **No face detected**\n"
            report += "No face detected in the image. Infringement check is not required.\n"
        else:
            report += f"**Detected {result['face_count']} face(s)**\n\n"

            if result["similar_persons"]:
                # 获取干扰前后相似度
                before_sim = result.get("before_interference_similarity", result.get("max_similarity", 0.0))
                after_sim = result.get("after_interference_similarity", 0.0)
                reduction = result.get("similarity_reduction", 0.0)

                risk_level = result["risk_level"]

                report += f"**Before Interference Similarity**: {before_sim:.4f}\n"
                if result["interference_applied"] and after_sim is not None:
                    report += f"**After Interference Similarity**: {after_sim:.4f}\n"
                    report += f"**Similarity Reduction**: {reduction:.4f}\n"
                    if before_sim > 0:
                        report += f"**Reduction Percentage**: {(reduction / before_sim * 100):.1f}%\n"

                report += f"**Risk Level**: {self._get_risk_emoji(risk_level)} {risk_level.upper()}\n\n"

                report += "**Similar Person Matches**:\n"
                for person in result["similar_persons"][:3]:  # 显示前3个
                    report += f"- {person['name']}: {person['similarity']:.4f}\n"

                if result["interference_applied"]:
                    report += f"\n✅ **Interference Applied**\n"
                    report += f"Interference Method: {result['interference_method']}\n"
                    if "interference_strength" in result:
                        report += f"Interference Strength: {result.get('interference_strength', 0.0):.2f}\n"
                else:
                    report += "\n✅ **No Interference Required**\nSimilarity is within the safe range.\n"
            else:
                report += "✅ **No High-Risk Person Matched**\n"
                report += "The face does not match anyone in the high-risk feature database.\n"

        report += f"\n**Processing Time**: {result['processing_time']:.2f}秒\n"
        report += f"**Timestamp**: {result.get('timestamp', datetime.now().isoformat())}"

        return report

    def _get_risk_emoji(self, risk_level):
        """获取风险等级对应的表情符号"""
        emoji_map = {
            "safe": "✅",
            "low": "⚠️",
            "medium": "🚨",
            "high": "⛔"
        }
        return emoji_map.get(risk_level, "❓")

    def _create_visualization(self, result, threshold):
        """创建可视化图表（全英文版）"""
        fig, ax = plt.subplots(figsize=(12, 7))

        if result["face_detected"] and result["similar_persons"]:
            # 创建两个子图
            ax1 = plt.subplot(1, 2, 1)  # 相似度条形图
            ax2 = plt.subplot(1, 2, 2)  # 干扰前后对比图

            # 图1：相似度条形图
            persons = result["similar_persons"][:5]  # 显示前5个
            names = [p["name"] for p in persons]
            similarities = [p["similarity"] for p in persons]

            colors = []
            for sim in similarities:
                if sim < threshold:
                    colors.append('green')
                elif sim < self.config.MEDIUM_RISK_THRESHOLD:
                    colors.append('yellow')
                elif sim < self.config.HIGH_RISK_THRESHOLD:
                    colors.append('orange')
                else:
                    colors.append('red')

            bars = ax1.barh(names, similarities, color=colors)
            ax1.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold: {threshold}')
            ax1.set_xlabel('Similarity')
            ax1.set_title('Similarity Matching Result')
            ax1.set_xlim(0, 1)
            ax1.legend()

            # 在条形上添加数值
            for bar, sim in zip(bars, similarities):
                width = bar.get_width()
                ax1.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{sim:.3f}', ha='left', va='center')

            # 图2：干扰前后对比图（如果应用了干扰）
            if result["interference_applied"]:
                before_sim = result.get("before_interference_similarity", result.get("max_similarity", 0.0))
                after_sim = result.get("after_interference_similarity", 0.0)

                labels = ['Before', 'After']
                values = [before_sim, after_sim]
                colors = ['red' if before_sim >= threshold else 'green',
                          'green' if after_sim < threshold else 'orange']

                bars2 = ax2.bar(labels, values, color=colors)
                ax2.set_ylabel('Similarity')
                ax2.set_title('Interference Comparison')
                ax2.set_ylim(0, 1)
                ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold: {threshold}')
                ax2.legend()

                # 在条形上添加数值
                for bar, val in zip(bars2, values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                             f'{val:.3f}', ha='center', va='bottom')

                # 添加降低值标注
                reduction = before_sim - after_sim
                ax2.text(0.5, max(values) * 0.8, f'Reduction: {reduction:.3f}',
                         ha='center', va='center', fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
            else:
                # 如果没有应用干扰，显示提示
                ax2.text(0.5, 0.5, 'No Interference Applied\nSimilarity is Safe',
                         horizontalalignment='center', verticalalignment='center',
                         transform=ax2.transAxes, fontsize=12, color='gray')
                ax2.set_title('Interference Comparison')
                ax2.set_xticks([])
                ax2.set_yticks([])

        else:
            # 显示无结果的图
            ax.text(0.5, 0.5, 'No Matching Result',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=16, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Similarity Matching Result')

        plt.subplots_adjust(top=0.88, wspace=0.35)
        return fig

    def _generate_details(self, result):
        """生成详细信息 - 包含干扰前后对比数据"""
        if not result["face_detected"]:
            return "无详细信息"

        details = "## 📋 Detailed Information\n\n"

        details += "### Face Detection\n"
        details += f"- Number of Faces Detected: {result['face_count']}\n"

        if result.get("face_regions"):
            details += f"- Face Regions: {result['face_regions']}\n"

        details += "\n### Feature Matching\n"

        if result["similar_persons"]:
            details += f"- Matched {len(result['similar_persons'])} 个相似人物\n"

            for i, person in enumerate(result["similar_persons"][:5], 1):
                details += f"\n**{i}. {person['name']}**\n"
                details += f"  - Similarity: {person['similarity']:.4f}\n"
                details += f"  - Weighted Similarity: {person['weighted_similarity']:.4f}\n"
        else:
            details += "- No similar person matched\n"

        details += "\n### Similarity Comparison\n"
        before_sim = result.get("before_interference_similarity", result.get("max_similarity", 0.0))
        after_sim = result.get("after_interference_similarity", 0.0)
        reduction = result.get("similarity_reduction", 0.0)

        details += f"- Before Interference Similarity: {before_sim:.4f}\n"
        if result["interference_applied"] and after_sim is not None:
            details += f"- After Interference Similarity: {after_sim:.4f}\n"
            details += f"- Similarity Reduction: {reduction:.4f}\n"
            if before_sim > 0:
                details += f"- Reduction Percentage: {(reduction / before_sim * 100):.1f}%\n"

        details += "\n### Interference Processing\n"
        if result["interference_applied"]:
            details += f"- Interference Applied\n"
            details += f"- Interference Method: {result['interference_method']}\n"
            if "interference_strength" in result:
                details += f"- Interference Strength: {result.get('interference_strength', 0.0):.2f}\n"
            details += f"- Risk Level: {result['risk_level']}\n"
        else:
            details += "- Interference Not Applied\n"

        details += f"\n### Performance Statistics\n"
        details += f"- Processing Time: {result['processing_time']:.3f}秒\n"
        details += f"- Timestamp: {result.get('timestamp', 'N/A')}\n"

        if result.get("output_path"):
            details += f"- Output File: {result['output_path']}\n"

        return details

    def batch_process(self, files, threshold, interference_method):
        """批量处理文件"""
        if not files:
            return "请选择文件"

        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        input_dir = Path(temp_dir) / "input"
        input_dir.mkdir(exist_ok=True)

        # 保存上传的文件
        for file_info in files:
            with open(file_info.name, 'rb') as src, open(input_dir / Path(file_info.name).name, 'wb') as dst:
                dst.write(src.read())

        # 批量处理
        result = self.processor.process_batch(
            input_dir=str(input_dir),
            threshold=threshold,
            interference_method=interference_method
        )

        if not result["success"]:
            return f"批量处理失败: {result.get('error', '未知错误')}"

        # 生成报告
        stats = result["statistics"]
        report = f"## 📁 批量处理完成\n\n"
        report += f"**输入目录**: {result['input_dir']}\n"
        report += f"**输出目录**: {result['output_dir']}\n"
        report += f"**报告文件**: {result['report_path']}\n\n"

        report += f"**统计信息**:\n"
        report += f"- 总图片数: {stats['total']}\n"
        report += f"- 成功处理: {stats['success']}\n"
        report += f"- 处理失败: {stats['failed']}\n"
        report += f"- 检测到人脸: {stats['face_detected']}\n"
        report += f"- 高风险图片: {stats['high_risk']}\n"
        report += f"- 应用干扰: {stats['interference_applied']}\n"
        if stats['interference_applied'] > 0:
            report += f"- 总相似度降低: {stats['total_similarity_reduction']:.4f}\n"
            report += f"- 平均相似度降低: {stats['avg_similarity_reduction']:.4f}\n"
        report += f"- 总耗时: {result['total_time']:.2f}秒\n"
        report += f"- 平均耗时: {result['average_time']:.2f}秒/图片\n\n"

        # 高风险图片列表
        if stats['high_risk'] > 0:
            report += f"**高风险图片 ({stats['high_risk']} 张)**:\n"

            high_risk_count = 0
            for res in result["results"]:
                if res.get("success") and res.get("risk_level") == "high":
                    high_risk_count += 1
                    report += f"{high_risk_count}. {Path(res['image_path']).name}: "
                    report += f"干扰前 {res.get('before_interference_similarity', 0):.3f} → "
                    if res.get("interference_applied", False):
                        report += f"干扰后 {res.get('after_interference_similarity', 0):.3f} "
                        report += f"(降低 {res.get('similarity_reduction', 0):.3f})\n"
                    else:
                        report += "未应用干扰\n"

        return report

    def get_system_info(self):
        """获取系统信息"""
        stats = self.feature_db.get_statistics()

        info = {
            "system_name": self.config.SYSTEM_NAME,
            "version": self.config.VERSION,
            "database": {
                "total_persons": stats["person_count"],
                "total_features": stats["feature_count"],
                "last_processed": stats["last_processed"],
                "interference_stats": stats.get("interference_stats", {})
            },
            "thresholds": {
                "similarity_threshold": self.config.SIMILARITY_THRESHOLD,
                "medium_risk_threshold": self.config.MEDIUM_RISK_THRESHOLD,
                "high_risk_threshold": self.config.HIGH_RISK_THRESHOLD
            },
            "interference": {
                "methods": self.config.INTERFERENCE_METHODS,
                "default_strength": self.config.INTERFERENCE_STRENGTH
            }
        }

        return info

    def create_interface(self):
        """创建Web界面"""
        with gr.Blocks(title=self.config.SYSTEM_NAME, theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"# 🛡️ {self.config.SYSTEM_NAME}")
            gr.Markdown("AI-Generated Image Infringement Warning Method Based on High-Risk Face Feature Database")

            with gr.Row():
                with gr.Column(scale=1):
                    # 输入部分
                    image_input = gr.Image(label="📷 Upload Image", type="pil", height=300)

                    with gr.Accordion("⚙️ Parameters", open=True):
                        threshold = gr.Slider(
                            minimum=0.1, maximum=0.9,
                            value=self.config.SIMILARITY_THRESHOLD,
                            step=0.05, label="Similarity Threshold",
                            info="Exceeding this threshold triggers interference"
                        )

                        interference_method = gr.Dropdown(
                            choices=self.config.INTERFERENCE_METHODS,
                            value="blur",
                            label="Interference Method",
                            info="Method applied when infringement is detected"
                        )

                        show_details = gr.Checkbox(
                            label="Show Details", value=False,
                            info="Show detailed processing info and statistics"
                        )

                    process_btn = gr.Button("🚀 Start Analysis", variant="primary", size="lg")

                    # 批量处理
                    with gr.Accordion("📁 Batch Processing", open=False):
                        file_input = gr.File(
                            label="Select multiple image files",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        batch_threshold = gr.Slider(
                            minimum=0.1, maximum=0.9,
                            value=self.config.SIMILARITY_THRESHOLD,
                            step=0.05, label="Batch Threshold"
                        )
                        batch_method = gr.Dropdown(
                            choices=self.config.INTERFERENCE_METHODS,
                            value="blur",
                            label="Batch Interference Method"
                        )
                        batch_process_btn = gr.Button("🔄 Process Batch", variant="secondary")
                        batch_output = gr.Markdown(label="Batch Results")

                    # 系统信息
                    with gr.Accordion("📊 System Info", open=False):
                        system_info = self.get_system_info()
                        info_text = f"""
                        **System Name**: {system_info['system_name']}
                        **Version**: {system_info['version']}

                        **Database Statistics**:
                        - Total Persons: {system_info['database']['total_persons']}
                        - Total Features: {system_info['database']['total_features']}
                        - Last Processed: {system_info['database']['last_processed'] or 'None'}
                        - Interference Cases: {system_info['database']['interference_stats'].get('interference_count', 0)}

                        **Threshold Settings**:
                        - Similarity Threshold: {system_info['thresholds']['similarity_threshold']}
                        - Medium Risk Threshold: {system_info['thresholds']['medium_risk_threshold']}
                        - High Risk Threshold: {system_info['thresholds']['high_risk_threshold']}

                        **Interference Settings**:
                        - Available Methods: {', '.join(system_info['interference']['methods'])}
                        - Default Strength: {system_info['interference']['default_strength']}
                        """
                        gr.Markdown(info_text)

                with gr.Column(scale=1):
                    # 输出部分
                    image_output = gr.Image(label="🖼️ Processed Image", type="pil", height=300)
                    plot_output = gr.Plot(label="📈 Similarity Analysis")
                    report_output = gr.Markdown(label="📋 Analysis Report")

                    # 详细信息输出
                    details_output = gr.Markdown(label="📊 Details", visible=False)

            # 处理按钮事件
            process_btn.click(
                fn=self.process_image,
                inputs=[image_input, threshold, interference_method, show_details],
                outputs=[image_output, plot_output, report_output, details_output]
            ).then(
                fn=lambda show: gr.update(visible=show),
                inputs=[show_details],
                outputs=[details_output]
            )

            # 批量处理按钮事件
            batch_process_btn.click(
                fn=self.batch_process,
                inputs=[file_input, batch_threshold, batch_method],
                outputs=batch_output
            )

            # 使用说明
            with gr.Accordion("📖 Documentation", open=False):
                gr.Markdown("""
                ### System Features

                1. **Upload Image**: Click the upload button or drag and drop an image.
                2. **Set Parameters**:
                   - Similarity Threshold: Exceeding this value triggers interference.
                   - Interference Method: Select the method used when infringement is detected.
                   - Show Details: View detailed data and statistics.
                3. **Start Analysis**: Click the "Start Analysis" button.
                4. **View Results**:
                   - View the processed image.
                   - View the similarity analysis chart.
                   - Read the detailed report (including before/after interference comparison).

                ### Interference Mechanism

                The system automatically applies one of the following interferences when high-risk similarity is detected:
                - **Blur**: Gaussian blur on face regions.
                - **Noise**: Random noise on face regions.
                - **Color Shift**: Adjust facial color features.
                - **Pixelate**: Reduce the resolution of face regions.
                - **Combined**: Combine multiple interference methods.

                ### Output Information

                - **Before Interference Similarity**: Original match score.
                - **After Interference Similarity**: Match score after applying interference.
                - **Similarity Reduction**: Quantitative metric of interference effect.
                - **Matched High-Risk Person**: The person with the highest similarity.
                - **Risk Level**: Safe / Low / Medium / High.

                ### Technical Details

                The system uses face feature extraction and similarity comparison technology:
                1. Extract face features from the input image.
                2. Compare with the high-risk feature database.
                3. Calculate cosine similarity.
                4. Apply interference based on threshold.
                5. Record and display before/after comparison data.
                """)

        self.demo = demo
        return demo

    def launch(self, share=False, server_port=None, server_name=None):
        """启动Web界面"""
        if self.demo is None:
            self.create_interface()

        if server_port is None:
            server_port = self.config.WEB_PORT
        if server_name is None:
            server_name = self.config.WEB_HOST

        print("=" * 60)
        print(f"{self.config.SYSTEM_NAME} - Web界面")
        print("=" * 60)
        print(f"访问地址: http://{server_name}:{server_port}")
        print("按 Ctrl+C 停止服务")
        print("=" * 60)

        self.demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            quiet=False
        )


def start_web_interface(config=None):
    """启动Web界面"""
    interface = WebInterface(config)
    interface.create_interface()
    interface.launch()
if __name__ == '__main__':
        start_web_interface()