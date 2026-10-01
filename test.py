#!/usr/bin/env python3
"""
生成中英文双版本多指标性能对比折线图
适用于IEEE论文投稿
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ==================== 数据准备（基于你提供的最新数据） ====================
thresholds = np.arange(0.5, 0.95, 0.05)  # 阈值从0.5到0.9，步长0.05

# 基于你提供的数据更新后的指标值
accuracy = np.array([0.84, 0.86, 0.88, 0.90, 0.91, 0.90, 0.88, 0.85, 0.82])
precision = np.array([0.75, 0.78, 0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96])
recall = np.array([0.92, 0.90, 0.88, 0.87, 0.85, 0.83, 0.80, 0.75, 0.65])  # 修正召回率数值
f1_score = np.array([0.828, 0.838, 0.849, 0.860, 0.865, 0.864, 0.857, 0.835, 0.777])  # 修正F1值数值


# ==================== 英文版本（IEEE投稿用） ====================
def create_english_figure():
    """创建英文版折线图（符合IEEE要求）"""
    # 设置英文显示参数
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',  # IEEE推荐使用衬线字体
        'mathtext.fontset': 'stix',  # 数学字体
    })

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制四条曲线
    line1, = ax.plot(thresholds, accuracy, color='#2E86AB', linewidth=2.5,
                     marker='o', markersize=7, label='Accuracy', zorder=5)
    line2, = ax.plot(thresholds, precision, color='#A23B72', linewidth=2.5,
                     marker='s', markersize=7, label='Precision', zorder=5)
    line3, = ax.plot(thresholds, recall, color='#F18F01', linewidth=2.5,
                     marker='^', markersize=7, label='Recall', zorder=5)
    line4, = ax.plot(thresholds, f1_score, color='#C73E1D', linewidth=2.5,
                     marker='d', markersize=7, label='F1-Score', zorder=5)

    # 标记最优阈值点
    optimal_idx = 4  # 0.7对应的索引
    ax.scatter(thresholds[optimal_idx], accuracy[optimal_idx], color='#2E86AB',
               s=150, edgecolor='black', linewidth=2, zorder=10)
    ax.scatter(thresholds[optimal_idx], precision[optimal_idx], color='#A23B72',
               s=150, edgecolor='black', linewidth=2, zorder=10)
    ax.scatter(thresholds[optimal_idx], recall[optimal_idx], color='#F18F01',
               s=150, edgecolor='black', linewidth=2, zorder=10)
    ax.scatter(thresholds[optimal_idx], f1_score[optimal_idx], color='#C73E1D',
               s=150, edgecolor='black', linewidth=2, zorder=10)

    # 添加最优阈值竖线
    ax.axvline(x=0.7, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label='Optimal Threshold (0.7)', zorder=1)

    # 设置坐标轴
    ax.set_xlabel('Similarity Threshold', fontsize=13, fontweight='bold')
    ax.set_ylabel('Performance Metrics Value', fontsize=13, fontweight='bold')
    ax.set_title('Multi-Metric Performance Comparison Across Different Thresholds',
                 fontsize=14, fontweight='bold', pad=15)

    # 设置坐标轴范围
    ax.set_xlim(0.48, 0.92)
    ax.set_ylim(0.6, 1.0)

    # 设置刻度
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=0)
    ax.set_yticks(np.arange(0.6, 1.01, 0.05))
    ax.set_yticklabels([f'{t:.2f}' for t in np.arange(0.6, 1.01, 0.05)])

    # 网格线
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    # 图例
    ax.legend(loc='lower left', fontsize=11, frameon=True,
              framealpha=0.9, edgecolor='black', shadow=True)

    # 添加最优阈值标注
    ax.annotate('Optimal Threshold = 0.70\nMaximizes overall performance',
                xy=(0.7, 0.62), xytext=(0.75, 0.7),
                arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

    # 调整布局
    plt.tight_layout()

    # 保存高分辨率图片（符合IEEE要求）
    plt.savefig('multi_metric_performance_english.png', dpi=600, bbox_inches='tight')
    plt.savefig('multi_metric_performance_english.pdf', bbox_inches='tight')  # 矢量图
    plt.savefig('multi_metric_performance_english.eps', bbox_inches='tight')  # EPS格式

    print("✓ 英文版图已保存: multi_metric_performance_english.png (600 DPI)")
    print("✓ 英文版矢量图已保存: multi_metric_performance_english.pdf")
    print("✓ 英文版矢量图已保存: multi_metric_performance_english.eps")

    plt.show()
    plt.close()


# ==================== 中文版本（用于中文报告/初稿） ====================
def create_chinese_figure():
    """创建中文版折线图"""
    # 设置中文显示（需要系统中文字体支持）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("注意：系统中文字体设置失败，中文可能显示为方框")

    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
    })

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制四条曲线
    line1, = ax.plot(thresholds, accuracy, color='#2E86AB', linewidth=2.5,
                     marker='o', markersize=7, label='准确率 (Accuracy)', zorder=5)
    line2, = ax.plot(thresholds, precision, color='#A23B72', linewidth=2.5,
                     marker='s', markersize=7, label='精确率 (Precision)', zorder=5)
    line3, = ax.plot(thresholds, recall, color='#F18F01', linewidth=2.5,
                     marker='^', markersize=7, label='召回率 (Recall)', zorder=5)
    line4, = ax.plot(thresholds, f1_score, color='#C73E1D', linewidth=2.5,
                     marker='d', markersize=7, label='F1值 (F1-Score)', zorder=5)

    # 标记最优阈值点
    optimal_idx = 4
    ax.scatter(thresholds[optimal_idx], accuracy[optimal_idx], color='#2E86AB',
               s=150, edgecolor='black', linewidth=2, zorder=10)
    ax.scatter(thresholds[optimal_idx], precision[optimal_idx], color='#A23B72',
               s=150, edgecolor='black', linewidth=2, zorder=10)
    ax.scatter(thresholds[optimal_idx], recall[optimal_idx], color='#F18F01',
               s=150, edgecolor='black', linewidth=2, zorder=10)
    ax.scatter(thresholds[optimal_idx], f1_score[optimal_idx], color='#C73E1D',
               s=150, edgecolor='black', linewidth=2, zorder=10)

    # 添加最优阈值竖线
    ax.axvline(x=0.7, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label='最优阈值 (0.7)', zorder=1)

    # 设置坐标轴
    ax.set_xlabel('相似度阈值 (Similarity Threshold)', fontsize=13, fontweight='bold')
    ax.set_ylabel('性能指标值 (Performance Metrics Value)', fontsize=13, fontweight='bold')
    ax.set_title('不同阈值下多指标性能对比折线图',
                 fontsize=14, fontweight='bold', pad=15)

    # 设置坐标轴范围
    ax.set_xlim(0.48, 0.92)
    ax.set_ylim(0.6, 1.0)

    # 设置刻度
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=0)
    ax.set_yticks(np.arange(0.6, 1.01, 0.05))
    ax.set_yticklabels([f'{t:.2f}' for t in np.arange(0.6, 1.01, 0.05)])

    # 网格线
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    # 图例
    ax.legend(loc='lower left', fontsize=11, frameon=True,
              framealpha=0.9, edgecolor='black', shadow=True)

    # 添加最优阈值标注
    ax.annotate('最优阈值 = 0.70\n实现整体性能最大化',
                xy=(0.7, 0.62), xytext=(0.75, 0.7),
                arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

    # 在数据点上添加数值标签
    for i, (acc, pre, rec, f1) in enumerate(zip(accuracy, precision, recall, f1_score)):
        if i == optimal_idx:  # 最优阈值点加粗显示
            ax.text(thresholds[i], acc + 0.005, f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2E86AB')
            ax.text(thresholds[i], pre - 0.015, f'{pre:.3f}',
                    ha='center', va='top', fontsize=9, fontweight='bold', color='#A23B72')
            ax.text(thresholds[i], rec - 0.015, f'{rec:.3f}',
                    ha='center', va='top', fontsize=9, fontweight='bold', color='#F18F01')
            ax.text(thresholds[i], f1 - 0.015, f'{f1:.3f}',
                    ha='center', va='top', fontsize=9, fontweight='bold', color='#C73E1D')

    # 调整布局
    plt.tight_layout()

    # 保存高分辨率图片
    plt.savefig('multi_metric_performance_chinese.png', dpi=600, bbox_inches='tight')
    plt.savefig('multi_metric_performance_chinese.pdf', bbox_inches='tight')

    print("\n✓ 中文版图已保存: multi_metric_performance_chinese.png (600 DPI)")
    print("✓ 中文版矢量图已保存: multi_metric_performance_chinese.pdf")

    plt.show()
    plt.close()


# ==================== 生成统计摘要 ====================
def generate_statistical_summary():
    """生成统计摘要表格（用于论文）"""
    print("\n" + "=" * 80)
    print("多指标性能统计摘要")
    print("=" * 80)
    print(f"{'阈值':<8} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1值':<10}")
    print("-" * 80)

    for i, thresh in enumerate(thresholds):
        print(f"{thresh:.2f}     {accuracy[i]:<10.3f} {precision[i]:<10.3f} "
              f"{recall[i]:<10.3f} {f1_score[i]:<10.3f}")

    print("-" * 80)

    # 查找最优阈值
    f1_max_idx = np.argmax(f1_score)
    acc_max_idx = np.argmax(accuracy)

    print(f"\n★ 基于F1值的最优阈值: {thresholds[f1_max_idx]:.2f}")
    print(f"   此时F1值 = {f1_score[f1_max_idx]:.3f}, 准确率 = {accuracy[f1_max_idx]:.3f}")
    print(f"   精确率 = {precision[f1_max_idx]:.3f}, 召回率 = {recall[f1_max_idx]:.3f}")

    print(f"\n★ 基于准确率的最优阈值: {thresholds[acc_max_idx]:.2f}")
    print(f"   此时准确率 = {accuracy[acc_max_idx]:.3f}, F1值 = {f1_score[acc_max_idx]:.3f}")

    # 平衡点分析
    balanced_idx = np.argmin(np.abs(precision - recall))
    print(f"\n★ 精确率与召回率平衡点阈值: {thresholds[balanced_idx]:.2f}")
    print(f"   此时精确率 = {precision[balanced_idx]:.3f}, 召回率 = {recall[balanced_idx]:.3f}")

    # 趋势分析
    print(f"\n★ 趋势分析:")
    print(f"   - 精确率从{precision[0]:.3f}单调上升至{precision[-1]:.3f}")
    print(f"   - 召回率从{recall[0]:.3f}单调下降至{recall[-1]:.3f}")
    print(f"   - 准确率在阈值{thresholds[acc_max_idx]:.2f}处达到峰值{accuracy[acc_max_idx]:.3f}")
    print(f"   - F1值在阈值{thresholds[f1_max_idx]:.2f}处达到峰值{f1_score[f1_max_idx]:.3f}")


# ==================== 主函数 ====================
if __name__ == "__main__":
    print("=" * 80)
    print("多指标性能对比折线图生成器")
    print("=" * 80)
    print("\n生成选项:")
    print("1. 英文版 (IEEE投稿用)")
    print("2. 中文版 (中文报告用)")
    print("3. 双版本同时生成")
    print("4. 仅生成统计摘要")

    choice = input("\n请选择 (1-4): ").strip()

    if choice == '1':
        create_english_figure()
    elif choice == '2':
        create_chinese_figure()
    elif choice == '3':
        create_english_figure()
        create_chinese_figure()
    elif choice == '4':
        generate_statistical_summary()
    else:
        print("默认生成双版本...")
        create_english_figure()
        create_chinese_figure()

    print("\n" + "=" * 80)
    print("完成！图片已保存到当前目录")
    print("=" * 80)