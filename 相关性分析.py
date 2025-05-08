import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # 用于热力图
from scipy.stats import spearmanr, pearsonr, ttest_ind, f_oneway, kruskal  # 导入统计检验函数

# --- 全局设置与文件夹创建 ---
# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 定义图片输出文件夹
output_folder_corr = "相关性分析图表文件夹"
if not os.path.exists(output_folder_corr):
    os.makedirs(output_folder_corr)
    print(f"已创建文件夹: {output_folder_corr}")

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


# --- 列名定义 (根据您提供的CSV文件) ---
# 核心行为变量
usage_frequency_col_actual = '3.**过去30天**您使用回收箱的次数'  # 使用频率

# 人口统计学列
gender_col = '7.您的性别'
age_col = '8.您的年龄'
education_col = '9.您的最高学历'
occupation_col = '10.您的职业'

# 心理因素和激励方案等Likert量表题目前缀
likert_q_prefixes = [f"{i}." for i in range(11, 21)]  # 问题11到20

# 障碍因素列 (多选)
obstacle_cols_raw = [
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:位置不便',
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:排队/拥堵',
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:不清楚可回收物',
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:无奖励',
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:卫生差',
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:担心信息泄露',
    '5.您在使用回收箱时遇到的**主要障碍**（可多选）:其他'
]

# --- 应用前缀清洗到分类列 ---
categorical_cols_to_clean_for_mapping = [
    usage_frequency_col_actual, gender_col, age_col, education_col, occupation_col
]
# 也包括问卷第一题和第六题，虽然它们不直接用于这里的相关性分析，但保持清洗一致性
other_categorical_cols = [
    '1.您所在社区目前**是否已设置**“快递包装共享回收箱”？',
    '2.您**每月收到的快递数量**',
    '6.若将回收服务成本计入物业费，您是否接受每月物业费**相比当前增加不超过2元**？'
]
all_cols_to_clean_prefix = categorical_cols_to_clean_for_mapping + other_categorical_cols

for col in all_cols_to_clean_prefix:
    if col in df.columns:
        df[col] = df[col].apply(clean_prefix)
    else:
        print(f"数据清洗提示: 预定义分类列 '{col}' 在CSV中未找到。")

# --- 为相关性分析准备数值编码 ---
# 1. 使用频率编码
usage_freq_mapping = {"0次": 0, "1-2次": 1, "3-5次": 2, "6-10次": 3, "10次以上": 4}
if usage_frequency_col_actual in df.columns:
    df['使用频率_数值'] = df[usage_frequency_col_actual].map(usage_freq_mapping)
else:
    print(f"关键列警告: 使用频率列 '{usage_frequency_col_actual}' 未在CSV中找到。部分分析将无法执行。")
    df['使用频率_数值'] = np.nan

# 2. 年龄编码
age_mapping = {"15-20": 0, "21-25": 1, "26-30": 2, "31-35": 3, "36-45": 4, "46及以上": 5}
if age_col in df.columns:
    df['年龄_数值'] = df[age_col].map(age_mapping)

# 3. 学历编码
education_mapping = {"初中及以下": 0, "高中/中专": 1, "大专": 2, "本科": 3, "硕士": 4, "博士": 5}
if education_col in df.columns:
    df['学历_数值'] = df[education_col].map(education_mapping)

# 4. 性别编码
if gender_col in df.columns:
    df['性别_数值'] = df[gender_col].map({'男': 0, '女': 1})

# --- Likert量表数据处理 ---
likert_mapping = {"非常不同意": 1, "不同意": 2, "一般": 3, "同意": 4, "非常同意": 5}
likert_cols_from_csv = [col for col in df.columns if any(col.startswith(prefix) for prefix in likert_q_prefixes)]

for col in likert_cols_from_csv:
    if col in df.columns:
        # 应用前缀清洗 (例如，如果数据是 "A.非常不同意")
        df[col] = df[col].apply(lambda x: clean_prefix(x) if isinstance(x, str) else x)
        # 应用映射
        df[col] = df[col].map(likert_mapping)
        # 转换为数值型，无法转换的变为NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- 计算心理因素各维度平均分 ---
psychological_dimensions_def = {
    '感知有用性': [col for col in likert_cols_from_csv if col.startswith('11.感知有用性:')],
    '感知便利性': [col for col in likert_cols_from_csv if col.startswith('12.感知便利性:')],
    '感知风险性': [col for col in likert_cols_from_csv if col.startswith('13.感知风险性:')],
    '感知趣味性': [col for col in likert_cols_from_csv if col.startswith('14.感知趣味性:')],
    '信任程度': [col for col in likert_cols_from_csv if col.startswith('15.信任程度:')],
    '主观规范': [col for col in likert_cols_from_csv if col.startswith('16.主观规范:')],
    '环境责任': [col for col in likert_cols_from_csv if col.startswith('17.环境责任:')],
    '使用习惯': [col for col in likert_cols_from_csv if col.startswith('18.使用习惯:')],
    '未来使用意向': [col for col in likert_cols_from_csv if col.startswith('20.未来使用意向:')]
}
dimension_mean_score_cols_generated = []  # 存储生成的维度平均分列名
for dim_name, item_cols in psychological_dimensions_def.items():
    actual_item_cols = [col for col in item_cols if col in df.columns]  # 只使用DataFrame中实际存在的列
    dim_mean_col_name = f"{dim_name}_平均分"
    dimension_mean_score_cols_generated.append(dim_mean_col_name)
    if actual_item_cols:  # 如果该维度下有实际的列
        df[dim_mean_col_name] = df[actual_item_cols].mean(axis=1)
    else:
        df[dim_mean_col_name] = np.nan  # 如果没有列，则填充NaN
        print(f"心理维度提示: 维度 '{dim_name}' 没有在CSV中找到对应的题目列。")

print("--- 数据加载与预处理完成 ---")
print(f"总样本数: {len(df)}")
if len(df) < 100:
    print(f"注意: 当前样本量为 {len(df)}，小于项目建议的100份有效样本。统计结果的普适性可能受限。")
print("\n" + "=" * 60 + "\n")

# ==============================================================================
# --- 二、相关性分析 ---
# ==============================================================================
print("--- 开始执行相关性分析 ---")

# --- 2.1 使用行为影响因素分析 ---
print("\n--- 2.1 使用行为影响因素分析 ---")

# 2.1.1 分析人口统计特征与使用频率的关系
print("\n--- 2.1.1 人口统计特征与使用频率的关系 ---")
target_usage_freq_numeric = '使用频率_数值'

if target_usage_freq_numeric not in df.columns or df[target_usage_freq_numeric].isnull().all():
    print(f"警告: '{target_usage_freq_numeric}' 列不存在或全为空，无法执行人口统计特征与使用频率的相关性分析。")
else:
    demographics_for_corr_map = {
        '性别_数值': '性别',
        '年龄_数值': '年龄',
        '学历_数值': '学历',
    }
    print("\n人口统计特征 (有序/二分) 与使用频率的Spearman相关性:")
    for num_col, name in demographics_for_corr_map.items():
        if num_col in df.columns and not df[num_col].isnull().all():
            temp_df = df[[num_col, target_usage_freq_numeric]].dropna()
            if not temp_df.empty and len(temp_df) >= 2:  # Spearman需要至少2个数据点
                corr, p_value = spearmanr(temp_df[num_col], temp_df[target_usage_freq_numeric])
                print(f"  {name} vs 使用频率: 相关系数={corr:.3f}, P值={p_value:.3f}{' *' if p_value < 0.05 else ''}")
            else:
                print(f"  {name} vs 使用频率: 数据不足无法计算相关性。")
        else:
            print(f"  人口统计特征列 '{num_col}' ({name}) 未找到或数据全为空。")

    # 职业（分类）与使用频率（有序）的关系 - Kruskal-Wallis H 检验
    if occupation_col in df.columns and not df[occupation_col].isnull().all():  # 使用原始职业列进行分组
        print("\n职业与使用频率的组间差异 (Kruskal-Wallis H 检验):")
        groups_for_kruskal = []
        # df[occupation_col] 此时已经是清洗过前缀的职业名称
        for job_name in df[occupation_col].dropna().unique():
            group_data = df[df[occupation_col] == job_name][target_usage_freq_numeric].dropna()
            if len(group_data) >= 1:  # Kruskal-Wallis对每组样本量要求不高，但至少要有数据
                groups_for_kruskal.append(group_data)

        if len(groups_for_kruskal) >= 2:  # 需要至少两个组进行比较
            try:
                stat, p_value = kruskal(*groups_for_kruskal)
                print(f"  Kruskal-Wallis H 检验: H统计量={stat:.3f}, P值={p_value:.3f}{' *' if p_value < 0.05 else ''}")
                if p_value < 0.05:
                    print("    结论: 不同职业群体在使用频率上存在显著差异。")
                else:
                    print("    结论: 不同职业群体在使用频率上无显著差异。")
            except ValueError as e:
                print(f"  Kruskal-Wallis检验出错: {e} (可能是由于样本量过小或数据分布问题)")
        else:
            print("  职业分组不足或数据不足，无法进行Kruskal-Wallis检验。")

# 2.1.2 计算心理因素与使用频率的相关系数
print("\n--- 2.1.2 心理因素与使用频率的相关系数 ---")
if target_usage_freq_numeric not in df.columns or df[target_usage_freq_numeric].isnull().all():
    print(f"警告: '{target_usage_freq_numeric}' 列不存在或全为空，无法执行心理因素与使用频率的相关性分析。")
else:
    print("\n心理因素各维度平均分与使用频率的Spearman相关性:")
    for dim_mean_col in dimension_mean_score_cols_generated:
        if dim_mean_col in df.columns and not df[dim_mean_col].isnull().all():
            temp_df = df[[dim_mean_col, target_usage_freq_numeric]].dropna()
            if not temp_df.empty and len(temp_df) >= 2:
                corr, p_value = spearmanr(temp_df[dim_mean_col], temp_df[target_usage_freq_numeric])
                dim_name_simple = dim_mean_col.replace('_平均分', '')
                print(
                    f"  {dim_name_simple} vs 使用频率: 相关系数={corr:.3f}, P值={p_value:.3f}{' *' if p_value < 0.05 else ''}")
            else:
                print(f"  {dim_mean_col.replace('_平均分', '')} vs 使用频率: 数据不足无法计算相关性。")
        else:
            print(f"  心理维度平均分列 '{dim_mean_col}' 未找到或数据全为空。")

# --- 2.2 障碍与态度关系分析 ---
print("\n\n--- 2.2 障碍与态度关系分析 ---")
# 2.2.1 分析主要障碍因素与使用意愿的关联
print("\n--- 2.2.1 主要障碍因素与未来使用意愿的关联 ---")
future_intention_mean_col = '未来使用意向_平均分'

if future_intention_mean_col not in df.columns or df[future_intention_mean_col].isnull().all():
    print(f"警告: '{future_intention_mean_col}' 列不存在或全为空，无法执行障碍与使用意愿的关联分析。")
else:
    print("\n主要障碍因素与未来使用意愿的点双列相关性 (Pearson r):")
    for obs_col_name_raw in obstacle_cols_raw:  # 使用原始障碍列名
        if obs_col_name_raw in df.columns:
            obstacle_name_simple = obs_col_name_raw.split(':')[-1]
            # 创建二元障碍列 (1=提及该障碍, 0=未提及)
            # 如果原始列中不是NaN，则表示提及 (因为CSV中未选中的是空白)
            df[f"{obstacle_name_simple}_提及"] = df[obs_col_name_raw].notna().astype(int)

            obstacle_binary_col = f"{obstacle_name_simple}_提及"
            temp_df = df[[obstacle_binary_col, future_intention_mean_col]].dropna()

            if not temp_df.empty and len(temp_df) >= 2 and temp_df[obstacle_binary_col].nunique() > 1:
                corr, p_value = pearsonr(temp_df[obstacle_binary_col], temp_df[future_intention_mean_col])
                print(
                    f"  障碍 '{obstacle_name_simple}' vs 未来使用意愿: 相关系数={corr:.3f}, P值={p_value:.3f}{' *' if p_value < 0.05 else ''}")
            else:
                print(f"  障碍 '{obstacle_name_simple}' vs 未来使用意愿: 数据不足或障碍提及情况单一，无法计算相关性。")
        else:
            print(f"  原始障碍列 '{obs_col_name_raw}' 在CSV中未找到。")

# 2.2.2 使用Python创建心理因素之间的相关矩阵
print("\n--- 2.2.2 心理因素之间的相关矩阵 ---")
# 选择所有维度平均分列进行相关性分析
psych_dims_for_matrix = [col for col in dimension_mean_score_cols_generated if
                         col in df.columns and not df[col].isnull().all()]

if psych_dims_for_matrix and len(psych_dims_for_matrix) > 1:
    psych_dims_df_for_matrix = df[psych_dims_for_matrix].dropna()  # dropna以避免NaN影响相关性计算
    if not psych_dims_df_for_matrix.empty and len(psych_dims_df_for_matrix) >= 2:
        correlation_matrix_psych = psych_dims_df_for_matrix.corr(method='spearman')
        print("\n心理因素各维度平均分之间的Spearman相关矩阵:")
        print(correlation_matrix_psych.round(3))

        plt.figure(figsize=(14, 12))  # 调整图像大小以容纳更多标签
        sns.heatmap(correlation_matrix_psych, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                    annot_kws={"size": 8}, cbar_kws={'label': 'Spearman Correlation'})  # 调整注释字体大小
        plt.title('心理因素各维度平均分之间的Spearman相关矩阵', fontsize=16)
        # 调整标签，去除"_平均分"
        cleaned_labels = [label.replace('_平均分', '') for label in correlation_matrix_psych.columns]
        plt.xticks(ticks=np.arange(len(cleaned_labels)) + 0.5, labels=cleaned_labels, rotation=45, ha='right',
                   fontsize=10)
        plt.yticks(ticks=np.arange(len(cleaned_labels)) + 0.5, labels=cleaned_labels, rotation=0, fontsize=10)
        plt.tight_layout()

        filename_heatmap = "心理因素相关矩阵_热力图.png"
        filepath_heatmap = os.path.join(output_folder_corr, filename_heatmap)
        try:
            plt.savefig(filepath_heatmap)
            print(f"\n热力图已保存到: {filepath_heatmap}")
        except Exception as e:
            print(f"\n保存热力图失败: {e}")
        plt.show()
        plt.close()
    else:
        print("心理因素各维度平均分数据不足 (dropna后为空或少于2行)，无法生成相关矩阵。")
else:
    print("有效的心理因素各维度平均分列不足两个，无法生成相关矩阵。")

# --- 2.3 激励偏好分析 ---
print("\n\n--- 2.3 激励偏好分析 ---")
# 激励方案的列 (原始Likert评分，值为1-5)
incentive_item_cols_all = [col for col in likert_cols_from_csv if col.startswith('19.激励方案兴趣度:')]

# 2.3.1 比较不同用户群体对各类激励的偏好差异
print("\n--- 2.3.1 不同用户群体对各类激励的偏好差异 (t-检验/ANOVA) ---")

demographic_cols_for_grouping = {
    '性别': gender_col,  # 原始分类列名 (已清洗前缀)
    '年龄段': age_col,  # 原始分类列名
    '学历层次': education_col,  # 原始分类列名
    '职业类型': occupation_col  # 原始分类列名
}

for incentive_col_name in incentive_item_cols_all:
    incentive_name_simple = incentive_col_name.split(':')[-1]
    if incentive_col_name not in df.columns or df[incentive_col_name].isnull().all():
        print(f"\n激励方案 '{incentive_name_simple}' 数据不足，跳过分析。")
        continue

    print(f"\n--- 对激励方案 '{incentive_name_simple}' 的偏好差异分析 ---")
    for group_category_name, demo_col_name_raw in demographic_cols_for_grouping.items():
        if demo_col_name_raw not in df.columns or df[demo_col_name_raw].isnull().all():
            print(f"  人口统计特征 '{group_category_name}' ({demo_col_name_raw}) 数据不足或列不存在，跳过。")
            continue

        print(f"  按 '{group_category_name}' 分析:")
        # df[demo_col_name_raw] 应该是已经清洗过前缀的分类值
        unique_groups_in_demo = df[demo_col_name_raw].dropna().unique()

        if len(unique_groups_in_demo) == 2:  # T-test
            group1_values = df[df[demo_col_name_raw] == unique_groups_in_demo[0]][incentive_col_name].dropna()
            group2_values = df[df[demo_col_name_raw] == unique_groups_in_demo[1]][incentive_col_name].dropna()
            if len(group1_values) >= 2 and len(group2_values) >= 2:  # t-test需要每组至少2个数据点
                stat, p_value = ttest_ind(group1_values, group2_values, equal_var=False)  # Welch's t-test
                print(
                    f"    {unique_groups_in_demo[0]} (均值={group1_values.mean():.2f},N={len(group1_values)}) vs {unique_groups_in_demo[1]} (均值={group2_values.mean():.2f},N={len(group2_values)}): T统计量={stat:.2f}, P值={p_value:.3f}{' *' if p_value < 0.05 else ''}")
            else:
                print(
                    f"    组别 '{unique_groups_in_demo[0]}' 或 '{unique_groups_in_demo[1]}' 数据不足 (少于2个有效值) 进行t-检验。")
        elif len(unique_groups_in_demo) > 2:  # ANOVA
            anova_groups_data = []
            valid_groups_for_anova = 0
            group_means_summary = []
            for group_value in unique_groups_in_demo:
                current_group_incentive_data = df[df[demo_col_name_raw] == group_value][incentive_col_name].dropna()
                if len(current_group_incentive_data) >= 2:  # ANOVA通常也建议每组至少2个
                    anova_groups_data.append(current_group_incentive_data)
                    group_means_summary.append(
                        f"{group_value}(均值={current_group_incentive_data.mean():.2f},N={len(current_group_incentive_data)})")
                    valid_groups_for_anova += 1

            if valid_groups_for_anova >= 2:  # 需要至少两个有效组进行比较
                try:
                    stat, p_value = f_oneway(*anova_groups_data)
                    print(
                        f"    ANOVA检验 (比较组: {', '.join(group_means_summary)}): F统计量={stat:.2f}, P值={p_value:.3f}{' *' if p_value < 0.05 else ''}")
                    if p_value < 0.05:
                        print("      结论: 不同组别在该激励方案偏好上存在显著差异。")
                except ValueError as e:
                    print(f"    ANOVA检验出错: {e} (可能是由于样本量过小、方差为0或数据问题)")
            else:
                print(f"    按 '{group_category_name}' 分组后，有效组别 (每组至少2个数据点) 不足2个，无法进行ANOVA检验。")
        else:  # 少于2个组
            print(f"    按 '{group_category_name}' 分组后，唯一有效组别少于2个，无法比较。")

# 2.3.2 分析环境责任感高低对激励偏好的影响
print("\n\n--- 2.3.2 环境责任感高低对激励偏好的影响 ---")
env_resp_mean_col = '环境责任_平均分'

if env_resp_mean_col not in df.columns or df[env_resp_mean_col].isnull().all():
    print(f"警告: '{env_resp_mean_col}' 列不存在或全为空，无法执行环境责任感与激励偏好的分析。")
else:
    median_env_responsibility = df[env_resp_mean_col].median()
    if pd.isna(median_env_responsibility):
        print("环境责任感数据不足，无法计算中位数进行分组。")
    else:
        # 创建分组列，确保不修改原始df太多次，这里直接用条件选择
        low_resp_mask = df[env_resp_mean_col] < median_env_responsibility
        high_resp_mask = df[env_resp_mean_col] >= median_env_responsibility

        print(f"根据环境责任感中位数 ({median_env_responsibility:.2f}) 分为高低两组。")

        for incentive_col_name in incentive_item_cols_all:
            incentive_name_simple = incentive_col_name.split(':')[-1]
            if incentive_col_name not in df.columns or df[incentive_col_name].isnull().all():
                print(f"\n激励方案 '{incentive_name_simple}' 数据不足，跳过。")
                continue

            print(f"\n对激励方案 '{incentive_name_simple}' 的偏好差异 (按环境责任感高低):")

            low_resp_incentive_data = df.loc[low_resp_mask, incentive_col_name].dropna()
            high_resp_incentive_data = df.loc[high_resp_mask, incentive_col_name].dropna()

            if len(low_resp_incentive_data) >= 2 and len(high_resp_incentive_data) >= 2:
                stat, p_value = ttest_ind(low_resp_incentive_data, high_resp_incentive_data, equal_var=False)
                print(
                    f"  低责任感组 (均值={low_resp_incentive_data.mean():.2f}, N={len(low_resp_incentive_data)}) vs 高责任感组 (均值={high_resp_incentive_data.mean():.2f}, N={len(high_resp_incentive_data)}):")
                print(f"  T统计量={stat:.2f}, P值={p_value:.3f}{' *' if p_value < 0.05 else ''}")
                if p_value < 0.05:
                    print("    结论: 两组在该激励方案偏好上存在显著差异。")
                else:
                    print("    结论: 两组在该激励方案偏好上无显著差异。")
            else:
                print(f"  环境责任感分组后，'{incentive_name_simple}' 的数据不足以进行t-检验 (某组少于2个有效值)。")

print("\n" + "=" * 60)
print("--- 相关性分析脚本执行完毕 ---")
print(f"如果生成了图表，已尝试保存到 '{output_folder_corr}' 文件夹中。")
print("请检查控制台输出的统计结果。标记 '*' 的P值表示在alpha=0.05水平上显著。")
