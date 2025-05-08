import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 设置与文件夹创建 ---
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 定义图片输出文件夹
output_folder = "描述性统计分析文件夹"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"已创建文件夹: {output_folder}")

# --- 数据加载与预处理 ---
try:
    # 从用户上传的文件加载数据
    df = pd.read_csv("数据详情值.csv")
except FileNotFoundError:
    print("错误：未找到 '数据详情值.csv' 文件。请确保文件已上传并且名称正确。")
    exit()

# 删除完全为空的行 (通常是CSV末尾的空行)
df.dropna(how='all', inplace=True)
# 重置索引，以防原始数据中有间断的索引
df.reset_index(drop=True, inplace=True)


# 数据清洗函数：去除选项前缀 (如 "A.", "B.")
def clean_prefix(x):
    if isinstance(x, str) and len(x) > 2 and x[1] == '.':
        return x[2:]
    return x


# 根据最新的CSV文件更新列名
# 注意: '3.**过去30天**您使用回收箱的次数' 是从新CSV中直接复制的列名
usage_frequency_col_actual = '3.**过去30天**您使用回收箱的次数'

# 应用前缀清洗到相关列
categorical_cols_to_clean = [
    '1.您所在社区目前**是否已设置**“快递包装共享回收箱”？',
    '2.您**每月收到的快递数量**',
    usage_frequency_col_actual,  # 使用实际的列名
    '6.若将回收服务成本计入物业费，您是否接受每月物业费**相比当前增加不超过2元**？',
    '7.您的性别',
    '8.您的年龄',
    '9.您的最高学历',
    '10.您的职业'
]
for col in categorical_cols_to_clean:
    if col in df.columns:
        df[col] = df[col].apply(clean_prefix)
    # else:
    # print(f"数据清洗警告: 预定义分类列 '{col}' 在CSV中未找到。")

# Likert量表选项到数值的映射
likert_mapping = {
    "A.非常不同意": 1, "非常不同意": 1,
    "B.不同意": 2, "不同意": 2,
    "C.一般": 3, "一般": 3,
    "D.同意": 4, "同意": 4,
    "E.非常同意": 5, "非常同意": 5
}

# 获取所有Likert量表相关的列 (问题11到20)
# CSV中的Likert列名格式为 "11.感知有用性:显著减少生活垃圾"
likert_cols_pattern_prefixes = [f"{i}." for i in range(11, 21)]
likert_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in likert_cols_pattern_prefixes)]

for col in likert_cols:
    if col in df.columns:  # 确保列存在
        # 清洗Likert列中的选项前缀 (如果存在，例如 "A.非常不同意")
        # 假设Likert量表的数据本身已经是 "非常不同意", "不同意" 等，或者 "A.非常不同意"
        # 如果数据已经是 "1", "2" 等数字，则map会失败，需要调整
        # 首先尝试去除 "A." 前缀，然后映射
        df[col] = df[col].apply(lambda x: clean_prefix(x) if isinstance(x, str) else x).map(likert_mapping)

print("--- 数据加载与预处理完成 ---")
print(f"总样本数: {len(df)}")
if len(df) < 100:
    print(f"注意: 当前样本量为 {len(df)}，小于项目要求的100份有效样本。统计结果可能不具有广泛代表性。")
print("\n")

# --- 1. 用户基本特征分析 ---
print("--- 1. 用户基本特征分析 ---")

# 1.1 绘制受访者性别、年龄、学历分布饼图
demographic_features = {
    '7.您的性别': '性别分布',
    '8.您的年龄': '年龄分布',
    '9.您的最高学历': '学历分布'
}

for col, title in demographic_features.items():
    if col in df.columns and not df[col].dropna().empty:
        fig, ax = plt.subplots(figsize=(8, 8))
        counts = df[col].value_counts()
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        ax.set_title(title)
        ax.axis('equal')

        filename = f"{title.replace(' ', '_').replace('/', '_')}.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath)
        print(f"图表已保存到: {filepath}")

        plt.show()
        plt.close(fig)
        print(f"\n{title}:\n{counts}\n")
    else:
        print(f"警告: 找不到列 '{col}' 或列中无有效数据，无法生成 {title}")

# 1.2 制作不同人群回收箱使用频率交叉表
demographic_cols_for_crosstab = {
    '7.您的性别': '性别',
    '8.您的年龄': '年龄',
    '9.您的最高学历': '学历',
    '10.您的职业': '职业'
}

if usage_frequency_col_actual in df.columns:
    for col, name in demographic_cols_for_crosstab.items():
        if col in df.columns:
            if not df[col].dropna().empty and not df[usage_frequency_col_actual].dropna().empty:
                print(f"\n回收箱使用频率与{name}的交叉表:")
                crosstab_result = pd.crosstab(df[col], df[usage_frequency_col_actual])
                print(crosstab_result)
            else:
                print(f"警告: 列 '{col}' 或 '{usage_frequency_col_actual}' 中无有效数据，无法生成与 {name} 的交叉表")
            print("\n")
        else:
            print(f"警告: 找不到列 '{col}' 用于交叉表分析")
else:
    print(f"警告: 找不到列 '{usage_frequency_col_actual}' 用于交叉表分析。请检查CSV列名与代码中的定义是否完全一致。")

# --- 2. 回收行为现状分析 ---
print("--- 2. 回收行为现状分析 ---")

# 2.1 统计回收箱使用频率分布情况
if usage_frequency_col_actual in df.columns and not df[usage_frequency_col_actual].dropna().empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    usage_counts = df[usage_frequency_col_actual].value_counts().sort_index()
    usage_counts.plot(kind='bar', ax=ax, rot=45)
    ax.set_title('回收箱使用频率分布')
    ax.set_xlabel('使用次数 (过去30天)')
    ax.set_ylabel('人数')
    plt.tight_layout()

    filename = "回收箱使用频率分布.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)
    print(f"图表已保存到: {filepath}")

    plt.show()
    plt.close(fig)
    print(f"\n回收箱使用频率分布:\n{usage_counts}\n")
else:
    print(f"警告: 找不到列 '{usage_frequency_col_actual}' 或列中无有效数据，无法进行回收行为分析。")

# 2.2 分析投递包装类型占比
# 列名与CSV一致: '4.您投入的主要包装类型（可多选）:纸箱'
packaging_type_cols = [
    '4.您投入的主要包装类型（可多选）:纸箱',
    '4.您投入的主要包装类型（可多选）:塑料袋/快递袋',
    '4.您投入的主要包装类型（可多选）:填充物',
    '4.您投入的主要包装类型（可多选）:其他'
]
packaging_counts = {}
valid_packaging_data = False
for col in packaging_type_cols:
    if col in df.columns:
        type_name = col.split(':')[-1]
        count = df[col].count()
        if count > 0:
            valid_packaging_data = True
        packaging_counts[type_name] = count
    # else:
    # print(f"警告: 包装类型列 '{col}' 在CSV中未找到。")

if valid_packaging_data:
    packaging_series = pd.Series(packaging_counts).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    packaging_series.plot(kind='bar', ax=ax, rot=45)
    ax.set_title('投递的主要包装类型占比')
    ax.set_xlabel('包装类型')
    ax.set_ylabel('选择人次')
    plt.tight_layout()

    filename = "投递的主要包装类型占比.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)
    print(f"图表已保存到: {filepath}")

    plt.show()
    plt.close(fig)
    print(f"\n投递的主要包装类型人次:\n{packaging_series}\n")
else:
    print("警告: 未找到包装类型相关列或所有相关列均无数据。")

# 2.3 排序主要使用障碍因素
# 列名与CSV一致: '5.您在使用回收箱时遇到的**主要障碍**（可多选）:位置不便'
obstacle_cols = [
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:位置不便',
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:排队/拥堵',
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:不清楚可回收物',
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:无奖励',
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:卫生差',
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:担心信息泄露',
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:其他'
]
obstacle_counts = {}
valid_obstacle_data = False
for col in obstacle_cols:
    if col in df.columns:
        obstacle_name = col.split(':')[-1]
        count = df[col].count()
        if count > 0:
            valid_obstacle_data = True
        obstacle_counts[obstacle_name] = count
    # else:
    # print(f"警告: 障碍因素列 '{col}' 在CSV中未找到。")

if valid_obstacle_data:
    obstacle_series = pd.Series(obstacle_counts).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    obstacle_series.plot(kind='bar', ax=ax, rot=45)
    ax.set_title('主要使用障碍因素排序')
    ax.set_xlabel('障碍因素')
    ax.set_ylabel('选择人次')
    plt.tight_layout()

    filename = "主要使用障碍因素排序.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)
    print(f"图表已保存到: {filepath}")

    plt.show()
    plt.close(fig)
    print(f"\n主要使用障碍因素人次:\n{obstacle_series}\n")
else:
    print("警告: 未找到障碍因素相关列或所有相关列均无数据。")

# --- 3. 态度与心理因素分析 ---
print("--- 3. 态度与心理因素分析 ---")

# 3.1 计算各维度得分均值并制作雷达图
# 使用之前定义的 likert_cols，它已经是基于CSV列名动态生成的
psychological_dimensions = {
    '感知有用性': [col for col in likert_cols if col.startswith('11.感知有用性:')],
    '感知便利性': [col for col in likert_cols if col.startswith('12.感知便利性:')],
    '感知风险性': [col for col in likert_cols if col.startswith('13.感知风险性:')],
    '感知趣味性': [col for col in likert_cols if col.startswith('14.感知趣味性:')],
    '信任程度': [col for col in likert_cols if col.startswith('15.信任程度:')],
    '主观规范': [col for col in likert_cols if col.startswith('16.主观规范:')],
    '环境责任': [col for col in likert_cols if col.startswith('17.环境责任:')],
    '使用习惯': [col for col in likert_cols if col.startswith('18.使用习惯:')],
    '未来使用意向': [col for col in likert_cols if col.startswith('20.未来使用意向:')]
}

dimension_mean_scores = {}
print("\n各心理维度下各项指标的平均分:")
valid_dimension_data_for_radar = False
for dim_name, item_cols_for_dim in psychological_dimensions.items():
    actual_item_cols_in_df = [col for col in item_cols_for_dim if col in df.columns]

    if not actual_item_cols_in_df:
        print(f"警告: 维度 '{dim_name}' 没有在数据中找到对应的题目列。")
        continue

    # 确保这些列的数据类型是数值型，如果不是，则尝试转换，失败则跳过
    try:
        df[actual_item_cols_in_df] = df[actual_item_cols_in_df].apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        print(f"警告: 转换维度 '{dim_name}' 的列为数值型时出错: {e}。跳过此维度。")
        continue

    if df[actual_item_cols_in_df].isnull().all().all():
        print(f"警告: 维度 '{dim_name}' 的所有题目列均无有效数值数据。")
        continue

    item_means = df[actual_item_cols_in_df].mean()
    if not item_means.empty and not item_means.isnull().all():
        valid_dimension_data_for_radar = True
        print(f"\n维度: {dim_name}")
        for item_col_name, item_mean_val in item_means.items():
            indicator_name = item_col_name.split(':')[-1] if ':' in item_col_name else item_col_name
            print(f"  - {indicator_name}: {item_mean_val:.2f}")

        df[dim_name + '_mean'] = df[actual_item_cols_in_df].mean(axis=1)
        dimension_mean_scores[dim_name] = df[dim_name + '_mean'].mean()
    else:
        print(f"警告: 维度 '{dim_name}' 的题目列计算平均分后为空或全为NaN。")

if dimension_mean_scores and valid_dimension_data_for_radar:
    labels = list(dimension_mean_scores.keys())
    stats_raw = [dimension_mean_scores[label] for label in labels]

    valid_indices = [i for i, stat_val in enumerate(stats_raw) if not pd.isna(stat_val)]
    labels = [labels[i] for i in valid_indices]
    stats = [stats_raw[i] for i in valid_indices]

    if not labels:
        print("警告: 所有维度平均分均为NaN，无法生成雷达图。")
    else:
        print("\n各心理维度总平均分 (用于雷达图):")
        for label, stat_val in zip(labels, stats):
            print(f"- {label}: {stat_val:.2f}")

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        stats_closed = np.concatenate((stats, [stats[0]]))
        angles_closed = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.plot(angles_closed, stats_closed, linewidth=2, linestyle='solid')
        ax.fill(angles_closed, stats_closed, alpha=0.4)
        ax.set_thetagrids(np.degrees(angles), labels)
        ax.set_title('态度与心理因素雷达图 (各维度平均分)', size=20, y=1.1)
        ax.set_yticks(np.arange(1, 6, 1))
        ax.set_ylim(0, 5)

        filename = "态度与心理因素雷达图.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath)
        print(f"图表已保存到: {filepath}")

        plt.show()
        plt.close(fig)
else:
    print("警告: 未能计算维度平均分或数据不足，无法生成雷达图。")

# 3.2 排序不同激励方案的平均兴趣度
incentive_cols_q19 = [col for col in likert_cols if col.startswith('19.激励方案兴趣度:')]
incentive_mean_interest = {}
valid_incentive_data = False

if incentive_cols_q19:
    print("\n各激励方案的平均兴趣度:")
    for col in incentive_cols_q19:
        if col in df.columns:
            # 确保数据是数值型
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"警告: 转换激励方案列 '{col}' 为数值型时出错: {e}。跳过此列。")
                continue

            if not df[col].isnull().all():
                incentive_name = col.split(':')[-1] if ':' in col else col
                mean_val = df[col].mean()
                if not pd.isna(mean_val):
                    incentive_mean_interest[incentive_name] = mean_val
                    print(f"- {incentive_name}: {mean_val:.2f}")
                    valid_incentive_data = True
                else:
                    print(f"- {incentive_name}: 数据不足无法计算平均值 (全是NaN)")
            # else:
            # print(f"提示: 激励方案列 '{col}' 全是NaN值。")
        # else:
        # print(f"警告: 激励方案列 '{col}' 在CSV中未找到。")

    if valid_incentive_data and incentive_mean_interest:
        incentive_series = pd.Series(incentive_mean_interest).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        incentive_series.plot(kind='barh', ax=ax)
        ax.set_title('不同激励方案的平均兴趣度排序')
        ax.set_xlabel('平均兴趣度 (1-5分)')
        ax.set_ylabel('激励方案')
        ax.invert_yaxis()
        plt.tight_layout()

        filename = "不同激励方案的平均兴趣度排序.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath)
        print(f"图表已保存到: {filepath}")

        plt.show()
        plt.close(fig)
        print(f"\n不同激励方案的平均兴趣度 (已排序):\n{incentive_series}\n")
    else:
        print("警告: 无有效数据计算激励方案平均兴趣度，无法生成图表。")

else:
    print("警告: 未找到激励方案相关列 (以 '19.激励方案兴趣度:' 开头)。")

print("--- 描述性统计分析完成 ---")
print(f"所有图表已尝试保存到 '{output_folder}' 文件夹中。")
