import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# --- 数据加载与预处理 ---
try:
    # 从用户上传的文件加载数据
    df = pd.read_csv("数据详情值.csv")
except FileNotFoundError:
    print("错误：未找到 '数据详情值精简.csv' 文件。请确保文件已上传并且名称正确。")
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


# 应用前缀清洗到相关列
categorical_cols = [
    '1.您所在社区目前**是否已设置**“快递包装共享回收箱”？',
    '2.您**每月收到的快递数量**',
    '3.过去30天您使用回收箱的次数',
    '6.若将回收服务成本计入物业费，您是否接受每月物业费**相比当前增加不超过2元**？',
    '7.您的性别',
    '8.您的年龄',
    '9.您的最高学历',
    '10.您的职业'
]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_prefix)

# Likert量表选项到数值的映射
likert_mapping = {
    "A.非常不同意": 1, "非常不同意": 1,
    "B.不同意": 2, "不同意": 2,
    "C.一般": 3, "一般": 3,
    "D.同意": 4, "同意": 4,
    "E.非常同意": 5, "非常同意": 5
}

# 获取所有Likert量表相关的列 (问题11到20)
likert_cols = [col for col in df.columns if col.startswith(tuple([f"{i}." for i in range(11, 21)]))]

for col in likert_cols:
    df[col] = df[col].apply(clean_prefix).map(likert_mapping)

print("--- 数据加载与预处理完成 ---")
print(f"总样本数: {len(df)}")
if len(df) < 100:
    print("注意: 当前样本量小于100，统计结果可能不具有广泛代表性。")
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
    if col in df.columns:
        plt.figure(figsize=(8, 8))
        counts = df[col].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        plt.title(title)
        plt.axis('equal')  # 确保饼图是圆的
        plt.show()
        print(f"\n{title}:\n{counts}\n")
    else:
        print(f"警告: 找不到列 '{col}' 用于 {title}")

# 1.2 制作不同人群回收箱使用频率交叉表
usage_frequency_col = '3.过去30天您使用回收箱的次数'
demographic_cols_for_crosstab = {
    '7.您的性别': '性别',
    '8.您的年龄': '年龄',
    '9.您的最高学历': '学历',
    '10.您的职业': '职业'
}

if usage_frequency_col in df.columns:
    for col, name in demographic_cols_for_crosstab.items():
        if col in df.columns:
            print(f"\n回收箱使用频率与{name}的交叉表:")
            crosstab_result = pd.crosstab(df[col], df[usage_frequency_col])
            print(crosstab_result)
            print("\n")
        else:
            print(f"警告: 找不到列 '{col}' 用于交叉表分析")
else:
    print(f"警告: 找不到列 '{usage_frequency_col}' 用于交叉表分析")

# --- 2. 回收行为现状分析 ---
print("--- 2. 回收行为现状分析 ---")

# 2.1 统计回收箱使用频率分布情况
if usage_frequency_col in df.columns:
    plt.figure(figsize=(10, 6))
    usage_counts = df[usage_frequency_col].value_counts().sort_index()
    usage_counts.plot(kind='bar', rot=45)
    plt.title('回收箱使用频率分布')
    plt.xlabel('使用次数 (过去30天)')
    plt.ylabel('人数')
    plt.tight_layout()
    plt.show()
    print(f"\n回收箱使用频率分布:\n{usage_counts}\n")
else:
    print(f"警告: 找不到列 '{usage_frequency_col}' 用于回收行为分析")

# 2.2 分析投递包装类型占比
packaging_type_cols = [
    '4.您投入的主要包装类型（可多选）:纸箱',
    '4.您投入的主要包装类型（可多选）:塑料袋/快递袋',
    '4.您投入的主要包装类型（可多选）:填充物',
    '4.您投入的主要包装类型（可多选）:其他'
]
packaging_counts = {}
for col in packaging_type_cols:
    if col in df.columns:
        # 提取类型名称 (冒号后的部分)
        type_name = col.split(':')[-1]
        packaging_counts[type_name] = df[col].count()  # count() 会统计非NaN值的数量

if packaging_counts:
    packaging_series = pd.Series(packaging_counts).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    packaging_series.plot(kind='bar', rot=45)
    plt.title('投递的主要包装类型占比')
    plt.xlabel('包装类型')
    plt.ylabel('选择人次')
    plt.tight_layout()
    plt.show()
    print(f"\n投递的主要包装类型人次:\n{packaging_series}\n")
else:
    print("警告: 未找到包装类型相关列或无数据。")

# 2.3 排序主要使用障碍因素
obstacle_cols = [
    '5.您在使用回收箱时遇到的主要障碍（可多选）:位置不便',
    '5.您在使用回收箱时遇到的主要障碍（可多选）:排队/拥堵',
    '5.您在使用回收箱时遇到的主要障碍（可多选）:不清楚可回收物',
    '5.您在使用回收箱时遇到的主要障碍（可多选）:无奖励',
    '5.您在使用回收箱时遇到的主要障碍（可多选）:卫生差',
    '5.您在使用回收箱时遇到的主要障碍（可多选）:担心信息泄露',
    '5.您在使用回收箱时遇到的主要障碍（可多选）:其他'
]
obstacle_counts = {}
for col in obstacle_cols:
    if col in df.columns:
        obstacle_name = col.split(':')[-1]
        obstacle_counts[obstacle_name] = df[col].count()

if obstacle_counts:
    obstacle_series = pd.Series(obstacle_counts).sort_values(ascending=False)
    plt.figure(figsize=(12, 7))
    obstacle_series.plot(kind='bar', rot=45)
    plt.title('主要使用障碍因素排序')
    plt.xlabel('障碍因素')
    plt.ylabel('选择人次')
    plt.tight_layout()
    plt.show()
    print(f"\n主要使用障碍因素人次:\n{obstacle_series}\n")
else:
    print("警告: 未找到障碍因素相关列或无数据。")

# --- 3. 态度与心理因素分析 ---
print("--- 3. 态度与心理因素分析 ---")

# 3.1 计算各维度得分均值并制作雷达图
psychological_dimensions = {
    '感知有用性': [col for col in likert_cols if '11.感知有用性' in col],
    '感知便利性': [col for col in likert_cols if '12.感知便利性' in col],
    '感知风险性': [col for col in likert_cols if '13.感知风险性' in col],
    '感知趣味性': [col for col in likert_cols if '14.感知趣味性' in col],
    '信任程度': [col for col in likert_cols if '15.信任程度' in col],
    '主观规范': [col for col in likert_cols if '16.主观规范' in col],
    '环境责任': [col for col in likert_cols if '17.环境责任' in col],
    '使用习惯': [col for col in likert_cols if '18.使用习惯' in col],
    '未来使用意向': [col for col in likert_cols if '20.未来使用意向' in col]
}

dimension_mean_scores = {}
print("\n各心理维度下各项指标的平均分:")
for dim_name, item_cols in psychological_dimensions.items():
    if item_cols:  # 确保维度下有题目列
        # 计算每个题目列的平均分
        item_means = df[item_cols].mean()
        print(f"\n维度: {dim_name}")
        for item_col_name, item_mean_val in item_means.items():
            # 提取指标名称 (冒号后的部分)
            indicator_name = item_col_name.split(':')[-1]
            print(f"  - {indicator_name}: {item_mean_val:.2f}")

        # 计算该维度在所有受访者中的总平均分
        # 先计算每个受访者在该维度上的平均分
        df[dim_name + '_mean'] = df[item_cols].mean(axis=1)
        # 再计算所有受访者在该维度上的总平均分
        dimension_mean_scores[dim_name] = df[dim_name + '_mean'].mean()
    else:
        print(f"警告: 维度 '{dim_name}' 没有找到对应的题目列。")

if dimension_mean_scores:
    labels = list(dimension_mean_scores.keys())
    stats = list(dimension_mean_scores.values())

    print("\n各心理维度总平均分 (用于雷达图):")
    for label, stat in zip(labels, stats):
        print(f"- {label}: {stat:.2f}")

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))  # 使雷达图闭合
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.plot(angles, stats, linewidth=2, linestyle='solid')
    ax.fill(angles, stats, alpha=0.4)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title('态度与心理因素雷达图 (各维度平均分)', size=20, y=1.1)
    ax.set_yticks(np.arange(1, 6, 1))  # 假设评分是1-5
    ax.set_ylim(0, 5)  # 假设评分范围
    plt.show()
else:
    print("警告: 未能计算维度平均分，无法生成雷达图。")

# 3.2 排序不同激励方案的平均兴趣度
incentive_cols = [col for col in likert_cols if '19.激励方案兴趣度' in col]
incentive_mean_interest = {}

if incentive_cols:
    print("\n各激励方案的平均兴趣度:")
    for col in incentive_cols:
        # 提取激励方案名称 (冒号后的部分)
        incentive_name = col.split(':')[-1]
        mean_val = df[col].mean()
        incentive_mean_interest[incentive_name] = mean_val
        print(f"- {incentive_name}: {mean_val:.2f}")

    incentive_series = pd.Series(incentive_mean_interest).sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    incentive_series.plot(kind='barh')  # 使用水平条形图，标签更易读
    plt.title('不同激励方案的平均兴趣度排序')
    plt.xlabel('平均兴趣度 (1-5分)')
    plt.ylabel('激励方案')
    plt.gca().invert_yaxis()  # 使最高的在顶部
    plt.tight_layout()
    plt.show()
    print(f"\n不同激励方案的平均兴趣度 (已排序):\n{incentive_series}\n")
else:
    print("警告: 未找到激励方案相关列或无数据。")

print("--- 描述性统计分析完成 ---")

