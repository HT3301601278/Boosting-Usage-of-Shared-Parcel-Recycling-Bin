import numpy as np
import pandas as pd
from scipy import stats
pd.set_option('future.no_silent_downcasting', True)

print("--- 开始数据分析与假设检验 ---")

# --- 1. 数据加载 ---
try:
    df = pd.read_csv('数据详情值.csv')
    print(f"成功加载数据，共 {len(df)} 条记录。")
except FileNotFoundError:
    print("错误：未找到 '数据详情值.csv' 文件。请确保文件路径正确。")
    exit()

# --- 2. 数据预处理与编码 ---
print("\n--- 2.1 开始数据预处理与编码 ---")


# 2.1.1 李克特量表转换函数
def likert_to_numeric(series):
    mapping = {
        'A.非常不同意': 1, 'B.不同意': 2, 'C.一般': 3, 'D.同意': 4, 'E.非常同意': 5,
        'A.是': 1, 'B.否': 0, 'C.视服务质量而定': 2  # 假设第6题这样编码
    }
    # 仅替换存在于mapping中的值，其他保持不变（可能是NaN或其他）
    return series.map(mapping).fillna(series)  # 如果mapping中没有，则保留原值


# 2.1.2 使用频率转换 (Q3: '3.**过去30天**您使用回收箱的次数')
freq_mapping = {
    'A.0次': 0,
    'B.1-2次': 1.5,  # 使用区间中值
    'C.3-5次': 4,
    'D.6-10次': 8,
    'E.10次以上': 12  # 设定一个较高的代表值
}
df['使用频率_编码'] = df['3.**过去30天**您使用回收箱的次数'].replace(freq_mapping)
df['使用频率_编码'] = pd.to_numeric(df['使用频率_编码'], errors='coerce')

# 2.1.3 年龄、学历等人口学特征编码 (Q8, Q9)
age_mapping = {
    'A.15-20': 0, 'B.21-25': 1, 'C.26-30': 2, 'D.31-35': 3, 'E.36-45': 4, 'F.46及以上': 5
}
df['年龄_编码'] = df['8.您的年龄'].replace(age_mapping)
df['年龄_编码'] = pd.to_numeric(df['年龄_编码'], errors='coerce')

edu_mapping = {
    'A.初中及以下': 0, 'B.高中/中专': 1, 'C.大专': 2, 'D.本科': 3, 'E.硕士': 4, 'F.博士及以上': 5
}
df['最高学历_编码'] = df['9.您的最高学历'].replace(edu_mapping)
df['最高学历_编码'] = pd.to_numeric(df['最高学历_编码'], errors='coerce')

# 2.1.4 应用李克特转换到相关列 (Q11-Q20)
likert_cols_q11 = [
    '11.感知有用性:显著减少生活垃圾', '11.感知有用性:提升社区形象',
    '11.感知有用性:节省处理时间', '11.感知有用性:助力双碳目标'
]
likert_cols_q12 = [
    '12.感知便利性:投递步骤易懂', '12.感知便利性:回收箱离家距离近', '12.感知便利性:开放时间充足'
]
likert_cols_q13 = [
    '13.感知风险性:担心个人信息泄露', '13.感知风险性:担心箱体卫生导致虫鼠', '13.感知风险性:担心未来产生额外收费'
]
likert_cols_q14 = [
    '14.感知趣味性:投递过程让我心情愉快', '14.感知趣味性:完成投递让我有成就感'
]
likert_cols_q15 = [
    '15.信任程度:相信包装会被妥善处理', '15.信任程度:相信积分/奖励可正常兑换', '15.信任程度:相信平台的数据隐私保护'
]
likert_cols_q16 = [
    '16.主观规范:家人朋友赞成', '16.主观规范:社区干部建议', '16.主观规范:邻居使用影响', '16.主观规范:媒体宣传倡导'
]
likert_cols_q17 = [
    '17.环境责任:环保是个人责任', '17.环境责任:在意包装浪费', '17.环境责任:为后代保护环境', '17.环境责任:愿减少一次性用品'
]
likert_cols_q18 = [
    '18.使用习惯:已形成习惯性投递', '18.使用习惯:不回收会感到不自在', '18.使用习惯:回收行为对我来说很自然'
]
likert_cols_q19_interest = [
    '19.激励方案兴趣度:现金红包', '19.激励方案兴趣度:积分商城', '19.激励方案兴趣度:快递费减免',
    '19.激励方案兴趣度:绿色碳积分', '19.激励方案兴趣度:公益植树', '19.激励方案兴趣度:社区荣誉排行',
    '19.激励方案兴趣度:物业费折扣', '19.激励方案兴趣度:抽盲盒', '19.激励方案兴趣度:商家优惠券',
    '19.激励方案兴趣度:二手盒返现'
]
likert_cols_q20_intention = [
    '20.未来使用意向:每月持续使用', '20.未来使用意向:高频使用意愿', '20.未来使用意向:无奖励坚持',
    '20.未来使用意向:社交分享', '20.未来使用意向:志愿推广'
]

all_likert_cols = (likert_cols_q11 + likert_cols_q12 + likert_cols_q13 + likert_cols_q14 +
                   likert_cols_q15 + likert_cols_q16 + likert_cols_q17 + likert_cols_q18 +
                   likert_cols_q19_interest + likert_cols_q20_intention)

for col in all_likert_cols:
    if col in df.columns:
        df[col] = likert_to_numeric(df[col])
        df[col] = pd.to_numeric(df[col], errors='coerce')  # 确保是数值型，无法转换的变NaN
    else:
        print(f"警告: 列 '{col}' 在CSV文件中不存在，跳过处理。")


# 2.1.5 计算综合得分 (均值)
# 确保所有子列都存在且已转换为数值型才计算均值
def calculate_mean_score(df, cols_list, new_col_name):
    valid_cols = [col for col in cols_list if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if valid_cols:
        df[new_col_name] = df[valid_cols].mean(axis=1, skipna=True)  # skipna=True很重要
    else:
        df[new_col_name] = np.nan
        print(f"警告: 无法计算 '{new_col_name}' 因为其子列均不存在或非数值型。")


calculate_mean_score(df, likert_cols_q11, '感知有用性_均值')
calculate_mean_score(df, likert_cols_q12, '感知便利性_均值')
calculate_mean_score(df, likert_cols_q13, '感知风险性_均值')
calculate_mean_score(df, likert_cols_q17, '环境责任感_均值')
calculate_mean_score(df, likert_cols_q20_intention, '未来使用意向_均值')

# 2.1.6 处理多选题障碍 (Q5) - 创建0/1变量
obstacle_cols_prefix = '5.您在使用回收箱时遇到的**主要障碍**（可多选）:'
obstacle_types = ['位置不便', '排队/拥堵', '不清楚可回收物', '无奖励', '卫生差', '担心信息泄露', '其他']
for obs_type in obstacle_types:
    col_name = f"{obstacle_cols_prefix}{obs_type}"
    if col_name in df.columns:
        df[f'障碍_{obs_type}'] = df[col_name].notna().astype(int)  # 选中为1，未选中（NaN）为0
    else:
        df[f'障碍_{obs_type}'] = 0  # 如果原始列不存在，则该障碍为0
        print(f"警告: 障碍列 '{col_name}' 在CSV中不存在，已创建为全0列 '障碍_{obs_type}'。")

# 2.1.7 数据清洗：移除关键分析列为NaN的行
# '12.感知便利性:回收箱离家距离近' 是 likert_cols_q12 中的一列，在上面已转换为数值
key_vars_for_dropna = ['使用频率_编码', '未来使用意向_均值', '感知有用性_均值', '12.感知便利性:回收箱离家距离近']
# 确保这些列都存在
existing_key_vars = [var for var in key_vars_for_dropna if var in df.columns]

if len(existing_key_vars) < len(key_vars_for_dropna):
    print(f"警告: 部分关键变量用于dropna不存在: {set(key_vars_for_dropna) - set(existing_key_vars)}")

df_cleaned = df.dropna(subset=existing_key_vars)  # 仅对存在的关键变量进行dropna

print(f"数据预处理完成。原始数据量: {len(df)}, 清洗后用于分析的数据量: {len(df_cleaned)}")
if len(df_cleaned) == 0:
    print("错误：清洗后无有效数据，请检查数据和预处理步骤。后续分析将无法进行。")
    exit()
print("--- 数据预处理与编码完成 ---")

# --- 3. 核心假设检验 ---
print("\n--- 3. 核心假设检验 ---")
alpha = 0.05  # 显著性水平

# H1: 不同激励机制对使用意愿的影响存在显著差异
print("\n--- H1: 不同激励机制对使用意愿（兴趣度）的影响存在显著差异 ---")
# 提取各项激励方案的兴趣度得分, 确保列存在且是数值型
valid_interest_cols = [col for col in likert_cols_q19_interest if
                       col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col])]

if not valid_interest_cols:
    print("H1检验：无有效的激励方案兴趣度列用于分析。")
else:
    interest_scores_df = df_cleaned[valid_interest_cols].dropna()  # 确保没有NaN影响Friedman检验

    if len(interest_scores_df) < 2 or interest_scores_df.shape[1] < 2:
        print(
            f"H1检验：数据不足 (行: {len(interest_scores_df)}, 列: {interest_scores_df.shape[1]})，无法进行Friedman检验。")
    else:
        try:
            statistic, p_value = stats.friedmanchisquare(
                *(interest_scores_df[col] for col in interest_scores_df.columns))
            print(f"Friedman检验统计量: {statistic:.4f}")
            print(f"P值: {p_value:.4f}")

            if p_value < alpha:
                print(f"结论: P值 ({p_value:.4f}) < {alpha}，拒绝原假设。不同激励机制对用户的吸引力（兴趣度）存在显著差异。")
                mean_interests = interest_scores_df.mean().sort_values(ascending=False)
                print("\n各项激励方案平均兴趣度 (降序):")
                print(mean_interests)
            else:
                print(
                    f"结论: P值 ({p_value:.4f}) >= {alpha}，接受原假设。不同激励机制对用户的吸引力（兴趣度）未表现出显著差异。")
        except ValueError as e:
            print(f"H1检验错误: {e}. 可能由于数据问题（如所有值相同）。")

# H2: 感知有用性与使用频率呈正相关
print("\n--- H2: 感知有用性与使用频率呈正相关 ---")
if '感知有用性_均值' in df_cleaned.columns and '使用频率_编码' in df_cleaned.columns and \
        pd.api.types.is_numeric_dtype(df_cleaned['感知有用性_均值']) and pd.api.types.is_numeric_dtype(
    df_cleaned['使用频率_编码']):
    temp_df_h2 = df_cleaned[['感知有用性_均值', '使用频率_编码']].dropna()
    if len(temp_df_h2) > 1:
        correlation, p_value = stats.spearmanr(temp_df_h2['感知有用性_均值'], temp_df_h2['使用频率_编码'],
                                               nan_policy='omit')
        print(f"Spearman相关系数: {correlation:.4f}")
        print(f"P值 (双尾): {p_value:.4f}")

        # 对于单尾检验 (正相关)，如果correlation > 0 且 p_value/2 < alpha
        if correlation > 0 and (p_value / 2) < alpha:
            print(
                f"结论: P值/2 ({(p_value / 2):.4f}) < {alpha} 且相关系数为正，拒绝原假设。感知有用性与回收箱使用频率之间存在显著的正相关关系。")
        elif correlation <= 0 and (p_value / 2) < alpha:
            print(
                f"结论: P值/2 ({(p_value / 2):.4f}) < {alpha} 但相关系数为负或零。感知有用性与回收箱使用频率之间存在显著的负相关或无相关关系（与假设矛盾）。")
        else:
            print(
                f"结论: P值/2 ({(p_value / 2):.4f}) >= {alpha}，接受原假设。感知有用性与回收箱使用频率之间未发现显著的正相关关系。")
    else:
        print("H2检验：有效数据不足 (少于2条)，无法进行Spearman相关性分析。")
else:
    print("H2检验：必要的列（感知有用性_均值 或 使用频率_编码）不存在或非数值型。")

# H3: 位置便利性是影响使用率的关键因素
print("\n--- H3: 位置便利性是影响使用率的关键因素 ---")
# 使用 '障碍_位置不便' (0/1) 和 '使用频率_编码'
if '障碍_位置不便' in df_cleaned.columns and '使用频率_编码' in df_cleaned.columns and \
        pd.api.types.is_numeric_dtype(df_cleaned['障碍_位置不便']) and pd.api.types.is_numeric_dtype(
    df_cleaned['使用频率_编码']):
    group1 = df_cleaned[df_cleaned['障碍_位置不便'] == 1]['使用频率_编码'].dropna()  # 认为位置不便的组
    group2 = df_cleaned[df_cleaned['障碍_位置不便'] == 0]['使用频率_编码'].dropna()  # 不认为位置不便的组

    if len(group1) > 0 and len(group2) > 0:
        try:
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='less',
                                                    nan_policy='omit')  # H_a: group1 < group2
            print(f"Mann-Whitney U检验统计量: {statistic:.4f}")
            print(f"P值 (单尾, 'less'): {p_value:.4f}")
            print(f"认为位置不便组的平均使用频率编码: {group1.mean():.2f} (N={len(group1)})")
            print(f"不认为位置不便组的平均使用频率编码: {group2.mean():.2f} (N={len(group2)})")

            if p_value < alpha:
                print(
                    f"结论: P值 ({p_value:.4f}) < {alpha}，拒绝原假设。认为回收箱位置不便的用户，其使用频率显著低于不认为位置不便的用户。位置便利性是影响使用率的关键因素。")
            else:
                print(
                    f"结论: P值 ({p_value:.4f}) >= {alpha}，接受原假设。未发现认为位置不便与否对回收箱使用频率有显著影响，或者影响方向与假设不符。")
        except ValueError as e:
            print(f"H3检验错误: {e}. 可能由于数据问题（如所有值相同或样本量太小）。")

    else:
        print("H3检验：分组后一个或两个组数据不足，无法进行Mann-Whitney U检验。")
else:
    print("H3检验：必要的列（障碍_位置不便 或 使用频率_编码）不存在或非数值型。")

# --- 4. 额外提出的假设及其检验 ---
print("\n--- 4. 额外提出的假设及其检验 ---")

# H4: 环境责任感越强的用户，其回收箱使用频率越高
print("\n--- H4: 环境责任感越强的用户，其回收箱使用频率越高 ---")
if '环境责任感_均值' in df_cleaned.columns and '使用频率_编码' in df_cleaned.columns and \
        pd.api.types.is_numeric_dtype(df_cleaned['环境责任感_均值']) and pd.api.types.is_numeric_dtype(
    df_cleaned['使用频率_编码']):
    temp_df_h4 = df_cleaned[['环境责任感_均值', '使用频率_编码']].dropna()
    if len(temp_df_h4) > 1:
        correlation, p_value = stats.spearmanr(temp_df_h4['环境责任感_均值'], temp_df_h4['使用频率_编码'],
                                               nan_policy='omit')
        print(f"Spearman相关系数: {correlation:.4f}")
        print(f"P值 (双尾): {p_value:.4f}")

        if correlation > 0 and (p_value / 2) < alpha:
            print(
                f"结论: P值/2 ({(p_value / 2):.4f}) < {alpha} 且相关系数为正，拒绝原假设。环境责任感与回收箱使用频率之间存在显著的正相关关系。")
        elif correlation <= 0 and (p_value / 2) < alpha:
            print(
                f"结论: P值/2 ({(p_value / 2):.4f}) < {alpha} 但相关系数为负或零。环境责任感与回收箱使用频率之间存在显著的负相关或无相关关系（与假设矛盾）。")
        else:
            print(
                f"结论: P值/2 ({(p_value / 2):.4f}) >= {alpha}，接受原假设。环境责任感与回收箱使用频率之间未发现显著的正相关关系。")
    else:
        print("H4检验：有效数据不足 (少于2条)，无法进行Spearman相关性分析。")
else:
    print("H4检验：必要的列（环境责任感_均值 或 使用频率_编码）不存在或非数值型。")

# H5: 不同年龄段的用户在回收箱使用频率上存在显著差异
print("\n--- H5: 不同年龄段的用户在回收箱使用频率上存在显著差异 ---")
if '年龄_编码' in df_cleaned.columns and '使用频率_编码' in df_cleaned.columns and \
        pd.api.types.is_numeric_dtype(df_cleaned['年龄_编码']) and pd.api.types.is_numeric_dtype(
    df_cleaned['使用频率_编码']):
    temp_df_h5 = df_cleaned[['年龄_编码', '使用频率_编码']].dropna()

    if not temp_df_h5.empty and temp_df_h5['年龄_编码'].nunique() > 1:
        # 准备Kruskal-Wallis检验的数据
        samples = [temp_df_h5['使用频率_编码'][temp_df_h5['年龄_编码'] == age_code] for age_code in
                   sorted(temp_df_h5['年龄_编码'].unique())]
        samples = [s for s in samples if not s.empty and len(s) > 0]  # 过滤掉空或长度为0的样本组

        if len(samples) >= 2:  # Kruskal-Wallis至少需要2组
            try:
                statistic, p_value = stats.kruskal(*samples)
                print(f"Kruskal-Wallis H检验统计量: {statistic:.4f}")
                print(f"P值: {p_value:.4f}")

                if p_value < alpha:
                    print(
                        f"结论: P值 ({p_value:.4f}) < {alpha}，拒绝原假设。不同年龄段的用户在回收箱使用频率上存在显著差异。")
                    print("\n各年龄段平均使用频率编码:")
                    for age_code_val in sorted(temp_df_h5['年龄_编码'].unique()):
                        # 确保 age_code_val 不是 NaN
                        if pd.notna(age_code_val):
                            mean_freq = temp_df_h5['使用频率_编码'][temp_df_h5['年龄_编码'] == age_code_val].mean()
                            count = len(temp_df_h5[temp_df_h5['年龄_编码'] == age_code_val])
                            age_label_list = [k for k, v in age_mapping.items() if v == age_code_val]
                            age_label = age_label_list[0] if age_label_list else f"未知编码 {age_code_val}"
                            print(f"年龄段 {age_label} (编码 {age_code_val}): 平均频率编码 {mean_freq:.2f} (N={count})")
                else:
                    print(
                        f"结论: P值 ({p_value:.4f}) >= {alpha}，接受原假设。不同年龄段的用户在回收箱使用频率上未表现出显著差异。")
            except ValueError as e:
                print(f"H5检验错误: {e}. 可能由于数据问题（如所有组内值相同或样本量太小）。")
        else:
            print(f"H5检验：分组后有效的样本组少于2个 ({len(samples)}组)，无法进行Kruskal-Wallis检验。")
    else:
        print("H5检验：数据不足或年龄编码只有一个有效组别，无法进行Kruskal-Wallis检验。")
else:
    print("H5检验：必要的列（年龄_编码 或 使用频率_编码）不存在或非数值型。")

# H6: 认为“无奖励”是主要障碍的用户，其未来使用意向显著低于不认为“无奖励”是障碍的用户
print("\n--- H6: 认为“无奖励”是主要障碍的用户，其未来使用意向显著低于不认为“无奖励”是障碍的用户 ---")
if '障碍_无奖励' in df_cleaned.columns and '未来使用意向_均值' in df_cleaned.columns and \
        pd.api.types.is_numeric_dtype(df_cleaned['障碍_无奖励']) and pd.api.types.is_numeric_dtype(
    df_cleaned['未来使用意向_均值']):
    group1_h6 = df_cleaned[df_cleaned['障碍_无奖励'] == 1]['未来使用意向_均值'].dropna()
    group2_h6 = df_cleaned[df_cleaned['障碍_无奖励'] == 0]['未来使用意向_均值'].dropna()

    if len(group1_h6) > 0 and len(group2_h6) > 0:
        try:
            statistic, p_value = stats.mannwhitneyu(group1_h6, group2_h6, alternative='less', nan_policy='omit')
            print(f"Mann-Whitney U检验统计量: {statistic:.4f}")
            print(f"P值 (单尾, 'less'): {p_value:.4f}")
            print(f"认为无奖励是障碍组的平均未来使用意向: {group1_h6.mean():.2f} (N={len(group1_h6)})")
            print(f"不认为无奖励是障碍组的平均未来使用意向: {group2_h6.mean():.2f} (N={len(group2_h6)})")

            if p_value < alpha:
                print(
                    f"结论: P值 ({p_value:.4f}) < {alpha}，拒绝原假设。认为“无奖励”是主要障碍的用户，其未来使用意向显著低于不认为“无奖励”是障碍的用户。")
            else:
                print(
                    f"结论: P值 ({p_value:.4f}) >= {alpha}，接受原假设。未发现认为“无奖励”是障碍与否对未来使用意向有显著影响，或者影响方向与假设不符。")
        except ValueError as e:
            print(f"H6检验错误: {e}. 可能由于数据问题。")
    else:
        print("H6检验：分组后一个或两个组数据不足，无法进行Mann-Whitney U检验。")
else:
    print("H6检验：必要的列（障碍_无奖励 或 未来使用意向_均值）不存在或非数值型。")

print("\n--- 所有假设检验完成 ---")

print("""
--- 总结与说明 ---
1.  数据编码：为了进行统计分析，对分类和序数数据进行了数值编码。
    李克特量表编码为1-5分，使用频率编码为序数或区间中值。
    多选题障碍因素转换为0/1的二元变量。
2.  检验方法选择：
    - Friedman检验：比较多个相关样本（如不同激励方案的兴趣度）。
    - Mann-Whitney U检验：两个独立样本的比较（非参数t检验）。
    - Kruskal-Wallis H检验：三个或更多独立样本的比较（非参数ANOVA）。
    - Spearman等级相关系数：两个序数/连续变量之间的关系。
    这些非参数检验方法对数据分布的要求较低。
3.  P值解读：显著性水平alpha设为0.05。P值 < alpha 则拒绝原假设。
4.  单尾与双尾检验：根据假设方向选择，如 'alternative' 参数或调整P值。
5.  数据量：清洗和分组可能导致某些组样本量过小，影响检验效力。
    代码中加入了基本的数据量和类型检查。
""")