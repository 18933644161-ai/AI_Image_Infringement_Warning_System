#!/usr/bin/env python3
"""
专门测试CSV文件解析
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config


def analyze_csv_file():
    """分析CSV文件结构"""
    config = Config()
    csv_path = config.VGGFACE2_IDENTITY_FILE

    print("=" * 60)
    print("CSV文件分析")
    print("=" * 60)
    print(f"文件路径: {csv_path}")
    print(f"文件存在: {csv_path.exists()}")

    if not csv_path.exists():
        print("文件不存在！")
        return

    # 读取文件
    with open(csv_path, 'rb') as f:
        raw_content = f.read()

    print(f"\n文件大小: {len(raw_content)} 字节")

    # 尝试不同的编码
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin-1', 'cp1252']

    for encoding in encodings:
        try:
            content = raw_content.decode(encoding)
            print(f"\n使用 {encoding} 编码成功解码")
            break
        except UnicodeDecodeError:
            continue
    else:
        print("无法用常见编码解码，尝试自动检测...")
        import chardet
        result = chardet.detect(raw_content)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"检测到编码: {encoding} (置信度: {confidence})")
        content = raw_content.decode(encoding, errors='ignore')

    # 分析行和列
    lines = content.strip().split('\n')
    print(f"\n总行数: {len(lines)}")

    print("\n前10行内容:")
    for i, line in enumerate(lines[:10]):
        print(f"行 {i + 1}: {repr(line)}")

    # 分析分隔符
    print("\n分析分隔符:")
    sample_line = lines[0] if lines else ""
    tab_count = sample_line.count('\t')
    comma_count = sample_line.count(',')
    semicolon_count = sample_line.count(';')

    print(f"制表符数量: {tab_count}")
    print(f"逗号数量: {comma_count}")
    print(f"分号数量: {semicolon_count}")

    if tab_count > 0:
        print("推测分隔符: 制表符 (\\t)")
        delimiter = '\t'
    elif semicolon_count > comma_count:
        print("推测分隔符: 分号 (;)")
        delimiter = ';'
    else:
        print("推测分隔符: 逗号 (,)")
        delimiter = ','

    # 解析示例行
    print("\n解析示例行:")
    for i, line in enumerate(lines[:5]):
        print(f"\n行 {i + 1}:")
        print(f"  原始: {repr(line)}")

        # 按推测的分隔符分割
        parts = line.split(delimiter)
        print(f"  分割为 {len(parts)} 部分:")
        for j, part in enumerate(parts):
            print(f"    列{j}: {repr(part)}")

    # 尝试解析整个文件
    print("\n" + "=" * 60)
    print("尝试解析整个文件...")

    identity_map = {}

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        # 按分隔符分割
        parts = line.split(delimiter)

        if len(parts) >= 2:
            identity_id = parts[0].strip()
            name = parts[1].strip()

            # 清理引号
            if name.startswith('"') and name.endswith('"'):
                name = name[1:-1].strip()

            identity_map[identity_id] = name

    print(f"成功解析 {len(identity_map)} 个身份映射")

    if identity_map:
        print("\n前10个身份映射:")
        for i, (identity_id, name) in enumerate(list(identity_map.items())[:10]):
            print(f"  {i + 1:2d}. {identity_id} -> {name}")

        # 检查是否有重复的ID
        ids = list(identity_map.keys())
        if len(ids) != len(set(ids)):
            print("\n警告: 有重复的ID!")

        # 保存解析结果到临时文件（用于验证）
        output_path = csv_path.parent / "parsed_identities.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for identity_id, name in identity_map.items():
                f.write(f"{identity_id}\t{name}\n")

        print(f"\n解析结果已保存到: {output_path}")

    return identity_map


def test_parse_with_pandas():
    """使用pandas尝试解析"""
    print("\n" + "=" * 60)
    print("使用pandas解析...")

    config = Config()
    csv_path = config.VGGFACE2_IDENTITY_FILE

    try:
        import pandas as pd

        # 尝试不同的参数
        for sep in ['\t', ',', ';']:
            try:
                df = pd.read_csv(csv_path, sep=sep, header=None, engine='python')
                print(f"使用分隔符 '{repr(sep)}' 成功读取，形状: {df.shape}")

                if df.shape[1] >= 2:
                    print(f"前5行数据:")
                    for i in range(min(5, len(df))):
                        print(f"  行{i}: {df.iloc[i].tolist()}")

                    # 提取身份映射
                    identity_map = {}
                    for _, row in df.iterrows():
                        if pd.notna(row[0]) and pd.notna(row[1]):
                            identity_id = str(row[0]).strip()
                            name = str(row[1]).strip()
                            if name.startswith('"') and name.endswith('"'):
                                name = name[1:-1].strip()
                            identity_map[identity_id] = name

                    print(f"\n提取到 {len(identity_map)} 个身份映射")
                    return identity_map

            except Exception as e:
                print(f"使用分隔符 '{repr(sep)}' 失败: {e}")
                continue

    except ImportError:
        print("pandas未安装，跳过pandas解析")

    return {}


if __name__ == "__main__":
    print("开始分析CSV文件...")

    # 方法1：手动解析
    identity_map1 = analyze_csv_file()

    # 方法2：使用pandas解析
    identity_map2 = test_parse_with_pandas()

    print("\n" + "=" * 60)
    print("解析完成")
    print("=" * 60)

    if identity_map1 or identity_map2:
        print(f"手动解析: {len(identity_map1)} 个身份")
        print(f"pandas解析: {len(identity_map2)} 个身份")

        if identity_map1:
            print("\n建议在dataset_loader.py中使用以下逻辑:")
            print("""
def load_identity_mapping(self):
    if not self.identity_mapping_path.exists():
        return {}

    identity_map = {}
    with open(self.identity_mapping_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 按制表符分割
            parts = line.split('\\t')
            if len(parts) >= 2:
                identity_id = parts[0].strip()
                name = parts[1].strip()

                # 清理引号
                if name.startswith('"') and name.endswith('"'):
                    name = name[1:-1].strip()

                identity_map[identity_id] = name

    return identity_map
            """)
    else:
        print("未能解析到任何身份映射")
        print("\n请检查文件格式，可能需要手动修复CSV文件")