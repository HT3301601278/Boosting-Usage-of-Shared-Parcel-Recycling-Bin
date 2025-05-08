import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr, mannwhitneyu, kruskal, ttest_ind
import warnings
import os

# --- 创建图表保存文件夹 ---
output_folder = "相关性分析文件夹"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"文件夹 '{output_folder}' 已创建。")
else:
    print(f"文件夹 '{output_folder}' 已存在。")

# Matplotlib中文显示设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore', category=UserWarning, module='scipy')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- 1. 数据加载和预处理 (与之前相同，此处省略以保持简洁) ---
try:
    df = pd.read_csv('数据详情值.csv')
except FileNotFoundError:
    print("错误：未找到 '数据详情值.csv' 文件。请确保文件在脚本同目录下。")
    exit()

# ... (省略数据预处理代码，与你之前提供的版本一致) ...
# Likert量表映射 (统一处理A/B/C/D/E 和 A./B./C./D./E.的情况)
likert_mapping = {
    'A. 非常不同意': 1, '非常不同意': 1,
    'B. 不同意': 2, '不同意': 2,
    'C. 一般': 3, '一般': 3,
    'D. 同意': 4, '同意': 4,
    'E. 非常同意': 5, '非常同意': 5
}

# 清理列名中的特殊字符，方便访问
df.columns = df.columns.str.replace('\*', '', regex=True) # 移除星号
df.columns = df.columns.str.replace('\n', '', regex=False) # 移除换行符
df.columns = df.columns.str.strip() # 移除前后空格

# 定义需要转换的Likert量表列
likert_cols_prefixes = [
    "11.感知有用性:", "12.感知便利性:", "13.感知风险性:", "14.感知趣味性:",
    "15.信任程度:", "16.主观规范:", "17.环境责任:", "18.使用习惯:",
    "19.激励方案兴趣度:", "20.未来使用意向:"
]

all_likert_cols = []
for col in df.columns:
    for prefix in likert_cols_prefixes:
        if col.startswith(prefix):
            all_likert_cols.append(col)
            break

for col in all_likert_cols:
    df[col] = df[col].astype(str).str.replace(r'^[A-E]\.\s*', '', regex=True).map(likert_mapping)


# 人口统计学特征的映射
df['7.您的性别'] = df['7.您的性别'].map({'A.男': 1, 'B.女': 2})
age_mapping = {'A.15-20': 1, 'B.21-25': 2, 'C.26-30': 3, 'D.31-35': 4, 'E.36-45': 5, 'F.46及以上': 6}
df['8.您的年龄_numeric'] = df['8.您的年龄'].map(age_mapping)
edu_mapping = {'A.初中及以下': 1, 'B.高中/中专': 2, 'C.大专': 3, 'D.本科': 4, 'E.硕士': 5, 'F.博士': 6}
df['9.您的最高学历_numeric'] = df['9.您的最高学历'].map(edu_mapping)
occupation_mapping = {
    'A.学生': 1, 'B.全职上班族': 2, 'C.自由职业': 3, 'D.退休': 4,
    'E.事业单位/公务员': 5,
    'F.其他': 6
}
df['10.您的职业_numeric'] = df['10.您的职业'].map(occupation_mapping)


usage_freq_mapping = {
    'A.0次': 0, 'B.1-2次': 1.5, 'C.3-5次': 4, 'D.6-10次': 8, 'E.10次以上': 12
}
df['使用频率_numeric'] = df['3.过去30天您使用回收箱的次数'].map(usage_freq_mapping)

obstacle_cols = [col for col in df.columns if col.startswith("5.您在使用回收箱时遇到的主要障碍:")]
for col in obstacle_cols:
    df[col] = df[col].notna().astype(int)

psych_factors_def = {
    '感知有用性': [col for col in df.columns if col.startswith("11.感知有用性:")],
    '感知便利性': [col for col in df.columns if col.startswith("12.感知便利性:")],
    '感知风险性': [col for col in df.columns if col.startswith("13.感知风险性:")],
    '感知趣味性': [col for col in df.columns if col.startswith("14.感知趣味性:")],
    '信任程度': [col for col in df.columns if col.startswith("15.信任程度:")],
    '主观规范': [col for col in df.columns if col.startswith("16.主观规范:")],
    '环境责任': [col for col in df.columns if col.startswith("17.环境责任:")],
    '使用习惯': [col for col in df.columns if col.startswith("18.使用习惯:")],
}

psych_avg_cols = []
for factor_name, cols in psych_factors_def.items():
    avg_col_name = f'{factor_name}_avg'
    if cols:
        df[avg_col_name] = df[cols].mean(axis=1, skipna=True)
        psych_avg_cols.append(avg_col_name)
    else:
        print(f"警告：心理因素 '{factor_name}' 对应的列未在数据中找到。")

print("--- 数据预处理完成 ---\n")


# --- 2.1 使用行为影响因素分析 ---
print("--- 2.1 使用行为影响因素分析 ---")

demographics_to_analyze = {
    '性别': '7.您的性别',
    '年龄段': '8.您的年龄',
    '学历': '9.您的最高学历',
    '职业': '10.您的职业'
}

print("\n人口统计特征与平均使用频率：")
for name, col in demographics_to_analyze.items():
    if col in df.columns and '使用频率_numeric' in df.columns:
        temp_df = df.dropna(subset=['使用频率_numeric', col])
        if not temp_df.empty:
            mean_usage_by_group = temp_df.groupby(col)['使用频率_numeric'].mean()
            print(f"\n按{name}分组的平均使用频率:")
            print(mean_usage_by_group)

            plt.figure(figsize=(8, 5))
            mean_usage_by_group.plot(kind='bar')
            plt.title(f'按{name}分组的平均使用频率')
            plt.ylabel('平均使用频率 (次数/30天)')
            plt.xlabel(name)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            # 保存图表
            filename = os.path.join(output_folder, f'平均使用频率_按{name}分组.png')
            plt.savefig(filename)
            print(f"图表已保存到: {filename}")
            plt.show() # 仍然显示图表

            if name == '年龄段' and '8.您的年龄_numeric' in temp_df.columns:
                age_groups = [group_data['使用频率_numeric'].dropna().values for _, group_data in temp_df.groupby('8.您的年龄_numeric')]
                age_groups = [g for g in age_groups if len(g) > 0]
                if len(age_groups) > 1:
                    try:
                        stat, p = kruskal(*age_groups)
                        print(f"Kruskal-Wallis检验 ({name} vs 使用频率): H-statistic={stat:.3f}, p-value={p:.3f}")
                        if p < 0.05: print("结论：不同年龄段的使用频率存在显著差异。")
                        else: print("结论：不同年龄段的使用频率无显著差异。")
                    except ValueError as e:
                         print(f"Kruskal-Wallis检验 ({name} vs 使用频率) 无法执行: {e}")
        else:
            print(f"警告：对于特征 '{name}' 或使用频率，数据不足或存在过多缺失值。")
    else:
        print(f"警告：列 '{col}' 或 '使用频率_numeric' 不在DataFrame中。")

# ... (心理因素与使用频率的相关系数部分不变，因为它不直接生成图表，只打印文本) ...
print("\n心理因素与使用频率的Spearman相关系数：")
correlations_psych_usage = {}
if '使用频率_numeric' in df.columns:
    target_usage_freq = df['使用频率_numeric'].dropna()
    for factor_avg_col in psych_avg_cols:
        if factor_avg_col in df.columns:
            factor_data = df[factor_avg_col].dropna()
            common_index = target_usage_freq.index.intersection(factor_data.index)
            if len(common_index) > 1:
                corr, p_value = spearmanr(target_usage_freq.loc[common_index], factor_data.loc[common_index])
                correlations_psych_usage[factor_avg_col] = (corr, p_value)
                print(f"{factor_avg_col.replace('_avg','')} vs 使用频率: Correlation={corr:.3f}, P-value={p_value:.3f}")
            else:
                print(f"警告：{factor_avg_col.replace('_avg','')} vs 使用频率：数据不足以计算相关性。")
        else:
            print(f"警告：心理因素平均分列 '{factor_avg_col}' 不存在。")
else:
    print("警告：'使用频率_numeric' 列不存在，无法计算心理因素与使用频率的相关性。")


# --- 2.2 障碍与态度关系分析 ---
print("\n--- 2.2 障碍与态度关系分析 ---")
# ... (障碍与使用意愿关联分析部分不变，因为它不直接生成图表，只打印文本) ...
usage_intention_col = '20.未来使用意向:每月持续使用'
print(f"\n主要障碍因素与使用意愿 ({usage_intention_col}) 的关联 (Mann-Whitney U test):")
if usage_intention_col in df.columns and df[usage_intention_col].notna().sum() > 0:
    for obs_col in obstacle_cols:
        if obs_col in df.columns and df[obs_col].isin([0,1]).all():
            group_has_obstacle = df[df[obs_col] == 1][usage_intention_col].dropna()
            group_no_obstacle = df[df[obs_col] == 0][usage_intention_col].dropna()
            if len(group_has_obstacle) >= 5 and len(group_no_obstacle) >= 5:
                try:
                    stat, p_value = mannwhitneyu(group_has_obstacle, group_no_obstacle, alternative='two-sided')
                    obstacle_name = obs_col.split(':')[-1]
                    print(f"障碍 '{obstacle_name}':")
                    print(f"  - 遇到障碍者平均意愿: {group_has_obstacle.mean():.2f} (N={len(group_has_obstacle)})")
                    print(f"  - 未遇到障碍者平均意愿: {group_no_obstacle.mean():.2f} (N={len(group_no_obstacle)})")
                    print(f"  - Mann-Whitney U: statistic={stat:.2f}, p-value={p_value:.3f}")
                    if p_value < 0.05: print("  - 结论：该障碍与使用意愿显著相关。")
                    else: print("  - 结论：该障碍与使用意愿无显著相关。")
                except ValueError as e:
                    print(f"  - 对于障碍 '{obstacle_name}', Mann-Whitney U检验无法执行: {e}")
            else:
                obstacle_name = obs_col.split(':')[-1]
                print(f"障碍 '{obstacle_name}': 数据不足，无法进行有效比较。")
        else:
            print(f"警告：障碍列 '{obs_col}' 格式不正确或不存在。")
else:
    print(f"警告：使用意愿列 '{usage_intention_col}' 不存在或无有效数据，无法分析障碍与态度的关系。")


# 心理因素之间的相关矩阵
print("\n心理因素之间的Spearman相关矩阵：")
if psych_avg_cols and all(col in df.columns for col in psych_avg_cols):
    psych_df_for_corr = df[psych_avg_cols].dropna()
    if not psych_df_for_corr.empty and len(psych_df_for_corr) > 1:
        corr_matrix_psych = psych_df_for_corr.corr(method='spearman')
        print(corr_matrix_psych)

        plt.figure(figsize=(12, 10)) # 稍微调整了尺寸以适应更多标签
        sns.heatmap(corr_matrix_psych, annot=True, cmap='coolwarm', fmt=".2f",
                    xticklabels=[col.replace('_avg','').replace('感知','').replace('程度','').replace('责任','') for col in psych_avg_cols], # 简化标签
                    yticklabels=[col.replace('_avg','').replace('感知','').replace('程度','').replace('责任','') for col in psych_avg_cols])
        plt.title('心理因素之间的Spearman相关矩阵', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        # 保存图表
        filename = os.path.join(output_folder, '心理因素相关矩阵_热力图.png')
        plt.savefig(filename)
        print(f"图表已保存到: {filename}")
        plt.show()
    else:
        print("警告：用于计算心理因素相关矩阵的数据不足或全为NaN。")
else:
    print("警告：部分或全部心理因素平均分列不存在，无法计算相关矩阵。")


# --- 2.3 激励偏好分析 ---
print("\n--- 2.3 激励偏好分析 ---")
# ... (这部分主要打印文本，如果后续需要针对激励偏好生成特定图表，也按类似方式添加保存逻辑) ...
incentive_cols = [col for col in df.columns if col.startswith("19.激励方案兴趣度:")]
incentive_names_clean = [col.split(':')[-1] for col in incentive_cols]

print("\n不同年龄段对各类激励方案的偏好差异 (Kruskal-Wallis H-test):")
if '8.您的年龄_numeric' in df.columns and incentive_cols:
    age_col_numeric = '8.您的年龄_numeric'
    age_col_categorical = '8.您的年龄'
    for i, inc_col in enumerate(incentive_cols):
        incentive_name = incentive_names_clean[i]
        print(f"\n激励方案: {incentive_name}")
        if inc_col in df.columns:
            temp_df_inc = df.dropna(subset=[inc_col, age_col_numeric, age_col_categorical])
            if not temp_df_inc.empty:
                mean_pref_by_age = temp_df_inc.groupby(age_col_categorical)[inc_col].mean().sort_index()
                print("各年龄段平均偏好度:")
                print(mean_pref_by_age)
                age_groups_data = [
                    group_data[inc_col].dropna().values
                    for name, group_data in temp_df_inc.groupby(age_col_numeric)
                ]
                age_groups_data = [g for g in age_groups_data if len(g) > 0]
                if len(age_groups_data) > 1:
                    try:
                        stat, p = kruskal(*age_groups_data)
                        print(f"Kruskal-Wallis检验: H-statistic={stat:.3f}, p-value={p:.3f}")
                        if p < 0.05: print("结论：不同年龄段对该激励方案的偏好存在显著差异。")
                        else: print("结论：不同年龄段对该激励方案的偏好无显著差异。")
                    except ValueError as e:
                        print(f"Kruskal-Wallis检验无法执行: {e}")
                else:
                    print("数据不足以进行Kruskal-Wallis检验 (少于2个年龄组有数据)。")
            else:
                print(f"激励方案 '{incentive_name}' 或年龄数据不足。")
        else:
            print(f"警告：激励方案列 '{inc_col}' 不存在。")
else:
    print("警告：年龄列 '8.您的年龄_numeric' 或激励方案列不存在，无法进行分析。")

print("\n环境责任感高低对激励偏好的影响 (Mann-Whitney U test):")
env_resp_avg_col = '环境责任_avg'
if env_resp_avg_col in df.columns and df[env_resp_avg_col].notna().sum() > 0 and incentive_cols:
    median_env_resp = df[env_resp_avg_col].median()
    df['环境责任感_分组'] = df[env_resp_avg_col].apply(lambda x: '高' if x >= median_env_resp else ('低' if pd.notna(x) else np.nan))
    for i, inc_col in enumerate(incentive_cols):
        incentive_name = incentive_names_clean[i]
        print(f"\n激励方案: {incentive_name}")
        if inc_col in df.columns and df[inc_col].notna().sum() > 0 :
            temp_df_env = df.dropna(subset=[inc_col, '环境责任感_分组'])
            if not temp_df_env.empty:
                group_high_resp = temp_df_env[temp_df_env['环境责任感_分组'] == '高'][inc_col].dropna()
                group_low_resp = temp_df_env[temp_df_env['环境责任感_分组'] == '低'][inc_col].dropna()
                if len(group_high_resp) >= 5 and len(group_low_resp) >= 5:
                    try:
                        stat, p_value = mannwhitneyu(group_high_resp, group_low_resp, alternative='two-sided')
                        print(f"  - 高责任感组平均偏好: {group_high_resp.mean():.2f} (N={len(group_high_resp)})")
                        print(f"  - 低责任感组平均偏好: {group_low_resp.mean():.2f} (N={len(group_low_resp)})")
                        print(f"  - Mann-Whitney U: statistic={stat:.2f}, p-value={p_value:.3f}")
                        if p_value < 0.05: print("  - 结论：环境责任感高低对该激励方案的偏好存在显著差异。")
                        else: print("  - 结论：环境责任感高低对该激励方案的偏好无显著差异。")
                    except ValueError as e:
                        print(f"  - Mann-Whitney U检验无法执行: {e}")
                else:
                    print("  - 数据不足（高/低责任感组样本量不足），无法进行有效比较。")
            else:
                print(f"  - 激励方案 '{incentive_name}' 或环境责任感分组数据不足。")
        else:
            print(f"警告：激励方案列 '{inc_col}' 不存在或无有效数据。")
else:
    print("警告：'环境责任_avg'列不存在，或无有效环境责任数据，或无激励方案列，无法分析环境责任感对激励偏好的影响。")

print("\n--- 相关性分析脚本执行完毕 ---")