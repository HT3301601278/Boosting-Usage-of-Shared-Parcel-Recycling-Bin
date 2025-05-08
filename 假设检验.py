import numpy as np
import pandas as pd
from scipy.stats import spearmanr, f_oneway, mannwhitneyu

# --- 全局设置 ---
# Matplotlib中文显示设置 (如果需要绘图，但此脚本主要关注统计输出)
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows系统中的黑体
# plt.rcParams['axes.unicode_minus'] = False # 解决负号'-'显示为方块的问题

# --- 数据加载与预处理 ---
print("--- 正在加载和预处理数据... ---")
try:
    df = pd.read_csv("数据详情值.csv")
except FileNotFoundError:
    print("错误：未找到 '数据详情值.csv' 文件。请确保文件与脚本在同一目录下，或提供正确路径。")
    exit()

# 删除完全为空的行
df.dropna(how='all', inplace=True)
df.reset_index(drop=True, inplace=True)


# 数据清洗函数：去除选项前缀 (如 "A.", "B.")
def clean_prefix(x):
    if isinstance(x, str) and len(x) > 2 and x[1] == '.' and x[0].isalpha():
        return x[2:]
    return x


# --- 列名定义 ---
usage_frequency_col_actual = '3.**过去30天**您使用回收箱的次数'  # 使用频率 (因变量 H2, H3)
# 感知有用性相关列 (自变量 H2)
perceived_usefulness_cols = [f'11.感知有用性:{item}' for item in
                             ['显著减少生活垃圾', '提升社区形象', '节省处理时间', '助力双碳目标']]
# 位置便利性相关列 (障碍问题中的“位置不便” 用于 H3)
obstacle_location_col = '5.您在使用回收箱时遇到的**主要障碍**（可多选）:位置不便'
# 未来使用意向相关列 (因变量 H1)
future_intention_cols = [f'20.未来使用意向:{item}' for item in
                         ['每月持续使用', '高频使用意愿', '无奖励坚持', '社交分享', '志愿推广']]
# 激励机制兴趣度相关列 (自变量 H1)
incentive_interest_cols_prefix = '19.激励方案兴趣度:'
# 从实际列名中提取所有激励方案的列
all_df_cols = df.columns.tolist()
incentive_cols_actual = [col for col in all_df_cols if col.startswith(incentive_interest_cols_prefix)]

# --- 应用前缀清洗 ---
cols_to_clean_prefix = [usage_frequency_col_actual] + \
                       [col for col in df.columns if col.startswith('11.感知有用性:')] + \
                       [col for col in df.columns if col.startswith('20.未来使用意向:')] + \
                       incentive_cols_actual + \
                       [obstacle_location_col if obstacle_location_col in df.columns else None]  # 清洗障碍列（如果存在）

# 清洗其他可能影响映射的分类列
other_categorical_cols_for_cleaning = [
    '7.您的性别', '8.您的年龄', '9.您的最高学历', '10.您的职业'
]
cols_to_clean_prefix.extend(other_categorical_cols_for_cleaning)
cols_to_clean_prefix = [col for col in cols_to_clean_prefix if col is not None and col in df.columns]  # 确保列存在

for col in cols_to_clean_prefix:
    df[col] = df[col].apply(clean_prefix)

# --- 数值编码与变量计算 ---
# 1. 使用频率编码 (H2, H3 的因变量)
usage_freq_mapping = {"0次": 0, "1-2次": 1, "3-5次": 2, "6-10次": 3, "10次以上": 4}
if usage_frequency_col_actual in df.columns:
    df['使用频率_数值'] = df[usage_frequency_col_actual].map(usage_freq_mapping)
else:
    print(f"关键列警告: 使用频率列 '{usage_frequency_col_actual}' 未在CSV中找到。H2和H3检验可能无法执行。")
    df['使用频率_数值'] = np.nan

# 2. Likert量表映射
likert_mapping = {"非常不同意": 1, "不同意": 2, "一般": 3, "同意": 4, "非常同意": 5}

# 3. 感知有用性平均分 (H2 的自变量)
actual_pu_cols = [col for col in perceived_usefulness_cols if col in df.columns]
if actual_pu_cols:
    for col in actual_pu_cols:
        df[col] = df[col].map(likert_mapping)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['感知有用性_平均分'] = df[actual_pu_cols].mean(axis=1)
else:
    print("警告: 感知有用性的部分或全部列未找到，无法计算平均分。H2检验可能受影响。")
    df['感知有用性_平均分'] = np.nan

# 4. 未来使用意向平均分 (H1 的因变量)
actual_fi_cols = [col for col in future_intention_cols if col in df.columns]
if actual_fi_cols:
    for col in actual_fi_cols:
        df[col] = df[col].map(likert_mapping)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['未来使用意向_平均分'] = df[actual_fi_cols].mean(axis=1)
else:
    print("警告: 未来使用意向的部分或全部列未找到，无法计算平均分。H1检验可能受影响。")
    df['未来使用意向_平均分'] = np.nan

# 5. 激励机制兴趣度评分 (H1 的自变量)
actual_incentive_rating_cols = {}  # 存储清理后的激励措施列名及其评分
for col in incentive_cols_actual:
    if col in df.columns:
        # col已经是清洗过前缀的，例如 '19.激励方案兴趣度:现金红包'
        # df[col]中的值也应该是清洗过前缀的，例如 '非常不同意'
        df[col] = df[col].map(likert_mapping)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        incentive_name = col.split(':')[-1]  # 获取激励名称，如 '现金红包'
        actual_incentive_rating_cols[incentive_name] = df[col]  # 存储的是Series对象

# 6. 位置便利性二元变量 (H3 的分组变量)
if obstacle_location_col in df.columns:
    # 如果 '位置不便' 列不是NaN (即被用户提及为障碍)，则便利性为0 (不便利)
    # 否则 (是NaN，即未提及为障碍)，则便利性为1 (相对便利)
    df['位置便利性_二元'] = df[obstacle_location_col].isna().astype(int)
else:
    print(f"警告: 障碍列 '{obstacle_location_col}' 未找到。H3检验可能无法执行。")
    df['位置便利性_二元'] = np.nan

print("--- 数据加载与预处理完成 ---")
print(f"总样本数: {len(df)}")
if len(df) < 100:
    print(f"注意: 当前样本量为 {len(df)}。统计检验结果的稳定性可能受影响。")
print("\n" + "=" * 60 + "\n")

# ==============================================================================
# --- 四、假设检验 ---
# ==============================================================================
print("--- 开始执行假设检验 ---")

# --- H1: 不同激励机制对使用意愿的影响存在显著差异 ---
# 方法: ANOVA - 比较用户对不同激励机制的“未来使用意向”均值。
# 操作：找出每个用户评分最高的激励机制（或评分达到某个阈值的机制），然后按此分组比较未来使用意向。
# 一个更直接的ANOVA方法：比较不同激励机制获得的平均“未来使用意向分数”。
# 这里我们采用ANOVA比较不同激励机制的平均“未来使用意向_平均分”。
# 我们需要将数据转换为长格式，或者为每个激励机制创建一个组。
# 更符合“不同激励机制对使用意愿的影响”的ANOVA：
# 将用户按其“最偏好的激励机制”分组，然后比较这些组的“未来使用意向_平均分”。

print("\n--- H1: 不同激励机制对使用意愿的影响存在显著差异 ---")
print("    原假设 (H0): 用户对不同激励机制的未来使用意向均值没有显著差异。")
print("    备择假设 (H1): 用户对至少两种激励机制的未来使用意向均值存在显著差异。")
print("    检验方法: 单因素方差分析 (ANOVA)")

# 准备ANOVA数据
# 对于每个用户，找出他们评分最高的激励机制。
# 如果有多个并列最高，则该用户可能对多个机制有同等强烈偏好。为简化，我们随机选一个或排除。
# 这里，我们为每个激励机制创建一个“高兴趣组”（例如评分4或5），然后比较这些组的未来使用意向。
# 这更像是多个t检验。
#
# 为了执行ANOVA，我们需要一个表示“激励机制类型”的分类变量。
# 我们将找出每个用户评分最高的那个激励机制作为其“最偏好激励”。

df_h1 = df.copy()
# 获取所有激励机制的评分列名
incentive_score_cols_for_h1 = [col for col in df.columns if col.startswith(incentive_interest_cols_prefix)]

if not incentive_score_cols_for_h1 or df['未来使用意向_平均分'].isnull().all():
    print("H1检验警告: 激励机制评分列或未来使用意向数据不足，无法执行ANOVA。")
else:
    # 找出每个用户评分最高的激励机制名称
    df_h1['最偏好激励'] = df_h1[incentive_score_cols_for_h1].idxmax(axis=1)
    # 清理列名，只保留激励名称
    df_h1['最偏好激励'] = df_h1['最偏好激励'].apply(lambda x: x.split(':')[-1] if isinstance(x, str) else None)

    # 移除未来使用意向或最偏好激励为空的行
    df_h1_anova = df_h1[['最偏好激励', '未来使用意向_平均分']].dropna()

    if df_h1_anova['最偏好激励'].nunique() > 1 and len(df_h1_anova) > 0:
        anova_groups = [group_data['未来使用意向_平均分'].values for name, group_data in
                        df_h1_anova.groupby('最偏好激励')]

        # 确保每个组至少有2个样本才能进行ANOVA
        valid_anova_groups = [g for g in anova_groups if len(g) >= 2]

        if len(valid_anova_groups) > 1:  # 需要至少两个有效的组
            f_statistic, p_value_h1 = f_oneway(*valid_anova_groups)
            print(f"  ANOVA F统计量: {f_statistic:.3f}")
            print(f"  P值: {p_value_h1:.3f}")
            if p_value_h1 < 0.05:
                print("  结论: P值 < 0.05，拒绝原假设。不同激励机制（按用户最偏好分组）对未来使用意向的均值存在显著差异。")
            else:
                print(
                    "  结论: P值 >= 0.05，不拒绝原假设。没有足够证据表明不同激励机制（按用户最偏好分组）对未来使用意向的均值存在显著差异。")
        else:
            print("H1检验警告: 有效分组（每组至少2个样本）数量不足两个，无法执行ANOVA。")
            print(f"  按最偏好激励分组后，各组样本数: {[len(g) for g in anova_groups]}")
    else:
        print("H1检验警告: '最偏好激励' 的类别不足2个或数据不足，无法执行ANOVA。")

# --- H2: 感知有用性与使用频率呈正相关 ---
print("\n--- H2: 感知有用性与使用频率呈正相关 ---")
print("    原假设 (H0): 感知有用性与使用频率之间不存在正相关关系 (rho <= 0)。")
print("    备择假设 (H1): 感知有用性与使用频率之间存在正相关关系 (rho > 0)。")
print("    检验方法: Spearman等级相关性检验 (单尾)")

df_h2 = df[['感知有用性_平均分', '使用频率_数值']].dropna()
if not df_h2.empty and len(df_h2) >= 2:
    corr_h2, p_value_h2_two_tailed = spearmanr(df_h2['感知有用性_平均分'], df_h2['使用频率_数值'])
    # 对于单尾检验 (H1: rho > 0)，如果corr > 0, p_one_tailed = p_two_tailed / 2; 否则 p_one_tailed = 1 - (p_two_tailed / 2)
    if corr_h2 > 0:
        p_value_h2_one_tailed = p_value_h2_two_tailed / 2
    else:
        p_value_h2_one_tailed = 1 - (p_value_h2_two_tailed / 2)

    print(f"  Spearman相关系数 (rho): {corr_h2:.3f}")
    print(f"  P值 (双尾): {p_value_h2_two_tailed:.3f}")
    print(f"  P值 (单尾, H1: rho > 0): {p_value_h2_one_tailed:.3f}")

    if p_value_h2_one_tailed < 0.05 and corr_h2 > 0:  # 确保相关性方向与假设一致
        print("  结论: P值 < 0.05 且 rho > 0，拒绝原假设。感知有用性与使用频率之间存在显著的正相关关系。")
    else:
        print(
            "  结论: P值 >= 0.05 或 rho <= 0，不拒绝原假设。没有足够证据表明感知有用性与使用频率之间存在显著的正相关关系。")
else:
    print("H2检验警告: 数据不足，无法计算Spearman相关性。")

# --- H3: 位置便利性是影响使用率的关键因素 ---
print("\n--- H3: 位置便利性是影响使用率的关键因素 ---")
print("    原假设 (H0): 位置便利的群体与位置不便利的群体在使用频率上没有显著差异。")
print("    备择假设 (H1): 位置便利的群体与位置不便利的群体在使用频率上存在显著差异。")
print("    检验方法: 独立样本t检验 (或Mann-Whitney U检验)")

if '位置便利性_二元' in df.columns and '使用频率_数值' in df.columns:
    df_h3 = df[['位置便利性_二元', '使用频率_数值']].dropna()
    if df_h3['位置便利性_二元'].nunique() == 2 and len(df_h3) > 0:  # 确保有两组数据
        group_convenient = df_h3[df_h3['位置便利性_二元'] == 1]['使用频率_数值']
        group_inconvenient = df_h3[df_h3['位置便利性_二元'] == 0]['使用频率_数值']

        if len(group_convenient) >= 2 and len(group_inconvenient) >= 2:  # t-test需要每组至少2个数据点
            # 优先使用Mann-Whitney U检验，因为它对数据分布的假设较少，更适合有序数据
            print("  (使用Mann-Whitney U检验，因使用频率为有序数据)")
            u_statistic, p_value_h3 = mannwhitneyu(group_convenient, group_inconvenient, alternative='two-sided')
            print(f"  Mann-Whitney U统计量: {u_statistic:.3f}")
            print(f"  P值: {p_value_h3:.3f}")

            # 打印均值以供参考
            print(f"  位置便利组平均使用频率: {group_convenient.mean():.2f} (N={len(group_convenient)})")
            print(f"  位置不便利组平均使用频率: {group_inconvenient.mean():.2f} (N={len(group_inconvenient)})")

            if p_value_h3 < 0.05:
                print("  结论: P值 < 0.05，拒绝原假设。位置便利性对使用频率有显著影响。")
            else:
                print("  结论: P值 >= 0.05，不拒绝原假设。没有足够证据表明位置便利性对使用频率有显著影响。")
        else:
            print("H3检验警告: 分组后某组样本量不足 (少于2个)，无法进行Mann-Whitney U检验。")
            print(f"  便利组样本数: {len(group_convenient)}, 不便利组样本数: {len(group_inconvenient)}")

    else:
        print("H3检验警告: '位置便利性_二元' 未能形成两组有效数据，或数据不足，无法进行检验。")
else:
    print("H3检验警告: '位置便利性_二元' 或 '使用频率_数值' 列准备失败，无法进行检验。")

print("\n" + "=" * 60)
print("--- 假设检验脚本执行完毕 ---")
print("请检查控制台输出的统计结果。P值小于0.05通常被认为在统计上是显著的。")
