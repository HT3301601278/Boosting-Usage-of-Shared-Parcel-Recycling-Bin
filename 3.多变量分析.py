import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

# --- 全局设置与文件夹创建 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

output_folder_multi = "多变量分析文件夹"
if not os.path.exists(output_folder_multi):
    os.makedirs(output_folder_multi)
    print(f"已创建文件夹: {output_folder_multi}")

# --- 数据加载与预处理 ---
print("--- 正在加载和预处理数据... ---")
try:
    df = pd.read_csv("数据详情值.csv")
except FileNotFoundError:
    print("错误：未找到 '数据详情值.csv' 文件。")
    exit()

df.dropna(how='all', inplace=True)
df.reset_index(drop=True, inplace=True)


def clean_prefix(x):
    if isinstance(x, str) and len(x) > 2 and x[1] == '.' and x[0].isalpha():
        return x[2:]
    return x


# --- 列名定义 ---
usage_frequency_col_actual = '3.**过去30天**您使用回收箱的次数'
gender_col = '7.您的性别'
age_col = '8.您的年龄'
education_col = '9.您的最高学历'
occupation_col = '10.您的职业'
likert_q_prefixes = [f"{i}." for i in range(11, 21)]
obstacle_cols_raw_prefix = '5.您在使用回收箱时遇到的**主要障碍**（可多选）:'
obstacle_items = [
    '位置不便', '排队/拥堵', '不清楚可回收物', '无奖励',
    '卫生差', '担心信息泄露', '其他'
]
obstacle_cols_actual = [f"{obstacle_cols_raw_prefix}{item}" for item in obstacle_items]

# --- 应用前缀清洗 ---
cols_to_clean_prefix = [
                           usage_frequency_col_actual, gender_col, age_col, education_col, occupation_col,
                           '1.您所在社区目前**是否已设置**“快递包装共享回收箱”？',
                           '2.您**每月收到的快递数量**',
                           '6.若将回收服务成本计入物业费，您是否接受每月物业费**相比当前增加不超过2元**？'
                       ] + [col for col in df.columns if
                            any(col.startswith(prefix) for prefix in likert_q_prefixes)]  # Likert列

for col in cols_to_clean_prefix:
    if col in df.columns:
        df[col] = df[col].apply(clean_prefix)

# --- 数值编码 ---
# 使用频率
usage_freq_mapping = {"0次": 0, "1-2次": 1, "3-5次": 2, "6-10次": 3, "10次以上": 4}
if usage_frequency_col_actual in df.columns:
    df['使用频率_数值'] = df[usage_frequency_col_actual].map(usage_freq_mapping)
else:
    df['使用频率_数值'] = np.nan
    print(f"警告: 列 '{usage_frequency_col_actual}' 未找到，'使用频率_数值' 将全为NaN。")

# 年龄
age_mapping = {"15-20": 0, "21-25": 1, "26-30": 2, "31-35": 3, "36-45": 4, "46及以上": 5}
if age_col in df.columns:
    df['年龄_数值'] = df[age_col].map(age_mapping)

# 学历
education_mapping = {"初中及以下": 0, "高中/中专": 1, "大专": 2, "本科": 3, "硕士": 4, "博士": 5}
if education_col in df.columns:
    df['学历_数值'] = df[education_col].map(education_mapping)

# 性别 (回归中用0/1)
if gender_col in df.columns:
    df['性别_二元'] = df[gender_col].map({'男': 0, '女': 1})

# 职业 (创建哑变量 for regression)
if occupation_col in df.columns:
    df = pd.get_dummies(df, columns=[occupation_col], prefix='职业', dummy_na=False,
                        drop_first=True)  # drop_first 避免多重共线性

# Likert量表数据处理
likert_mapping = {"非常不同意": 1, "不同意": 2, "一般": 3, "同意": 4, "非常同意": 5}
likert_cols_from_csv = [col for col in df.columns if any(col.startswith(prefix) for prefix in likert_q_prefixes)]
for col in likert_cols_from_csv:
    if col in df.columns:
        df[col] = df[col].map(likert_mapping)  # 假设前缀已通过上面的循环清洗
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
psych_dim_mean_cols = []
for dim_name, item_cols in psychological_dimensions_def.items():
    actual_item_cols = [col for col in item_cols if col in df.columns]
    dim_mean_col_name = f"{dim_name}_平均分"
    psych_dim_mean_cols.append(dim_mean_col_name)
    if actual_item_cols:
        df[dim_mean_col_name] = df[actual_item_cols].mean(axis=1)
    else:
        df[dim_mean_col_name] = np.nan

# --- 创建障碍感知二元变量 ---
obstacle_binary_cols = []
for item in obstacle_items:
    original_col_name = f"{obstacle_cols_raw_prefix}{item}"
    binary_col_name = f"障碍_{item.replace('/', '_')}_提及"  # 清理特殊字符
    obstacle_binary_cols.append(binary_col_name)
    if original_col_name in df.columns:
        df[binary_col_name] = df[original_col_name].notna().astype(int)
    else:
        df[binary_col_name] = 0  # 如果原始列不存在，则认为未提及
        print(f"警告: 原始障碍列 '{original_col_name}' 未找到，对应二元变量将全为0。")

print("--- 数据加载与预处理完成 ---")
print(f"总样本数: {len(df)}")
if len(df) < 100:
    print(f"注意: 当前样本量为 {len(df)}。")
print("\n" + "=" * 60 + "\n")

# ==============================================================================
# --- 1. 行为预测模型 (多元回归) ---
# ==============================================================================
print("--- 1. 行为预测模型 (多元回归) ---")
dependent_var_behavior = '使用频率_数值'

# 确定自变量
independent_vars_behavior = []
# 用户特征
user_feature_cols = ['年龄_数值', '学历_数值', '性别_二元']  # 原始职业列已被哑变量替代
for col in user_feature_cols:
    if col in df.columns:
        independent_vars_behavior.append(col)
# 职业哑变量 (除了被drop_first的那个)
occupation_dummy_cols = [col for col in df.columns if col.startswith('职业_')]
independent_vars_behavior.extend(occupation_dummy_cols)

# 心理因素
independent_vars_behavior.extend(psych_dim_mean_cols)
# 障碍感知变量
independent_vars_behavior.extend(obstacle_binary_cols)

# 确保所有选定的自变量列都存在于DataFrame中
independent_vars_behavior = [col for col in independent_vars_behavior if col in df.columns]

if dependent_var_behavior in df.columns and independent_vars_behavior:
    df_regression_behavior = df[[dependent_var_behavior] + independent_vars_behavior].copy()
    df_regression_behavior.dropna(inplace=True)  # Listwise deletion for missing data

    if not df_regression_behavior.empty and len(df_regression_behavior) > len(
            independent_vars_behavior) + 1:  # 保证有足够数据点
        Y_behavior = df_regression_behavior[dependent_var_behavior]
        X_behavior = df_regression_behavior[independent_vars_behavior]
        X_behavior = sm.add_constant(X_behavior)  # 添加常数项

        try:
            model_behavior = sm.OLS(Y_behavior, X_behavior).fit()
            print("\n--- 使用频率预测模型 (OLS回归) 结果 ---")
            print(model_behavior.summary())
            print("\n注意: '使用频率_数值' 是有序变量，OLS结果的解释需谨慎。更优方法可能是序数逻辑回归。")
        except Exception as e:
            print(f"行为预测模型回归分析失败: {e}")
            print("可能原因：数据中存在完全共线性，或数据量相对于变量数过少，或数据类型问题。")
            print("检查自变量列表:", independent_vars_behavior)
            # print("检查X_behavior数据的前几行和描述统计:\n", X_behavior.head())
            # print(X_behavior.describe(include='all'))


    else:
        print("行为预测模型：移除缺失值后数据不足或自变量不足，无法进行回归分析。")
        print(f"  移除缺失值后剩余样本数: {len(df_regression_behavior)}")
        print(f"  自变量数量: {len(independent_vars_behavior)}")

else:
    print(f"行为预测模型：因变量 '{dependent_var_behavior}' 或自变量列表为空，无法进行回归分析。")
    print(f"  选定的自变量: {independent_vars_behavior}")

# ==============================================================================
# --- 2. 用户分群分析 (K-Means 聚类) ---
# ==============================================================================
print("\n" + "=" * 60 + "\n")
print("--- 2. 用户分群分析 (K-Means 聚类) ---")

# 选择聚类特征：使用行为和态度（心理因素平均分）
features_for_clustering = ['使用频率_数值'] + psych_dim_mean_cols
features_for_clustering = [col for col in features_for_clustering if col in df.columns]  # 确保列存在

if features_for_clustering and '使用频率_数值' in features_for_clustering:
    df_clustering_raw = df[features_for_clustering].copy()
    df_clustering_raw.dropna(inplace=True)  # 处理缺失值

    if not df_clustering_raw.empty and len(df_clustering_raw) > 1:
        # 数据标准化
        scaler = StandardScaler()
        df_clustering_scaled = scaler.fit_transform(df_clustering_raw)
        df_clustering_scaled = pd.DataFrame(df_clustering_scaled, columns=df_clustering_raw.columns,
                                            index=df_clustering_raw.index)

        # 确定最佳聚类数 (Elbow Method & Silhouette Score)
        print("\n--- 确定最佳聚类数 ---")
        # Elbow Method
        print("正在计算Elbow曲线...")
        elbow_model = KMeans(random_state=42, n_init='auto')
        visualizer_elbow = KElbowVisualizer(elbow_model, k=(2, 11), metric='distortion', timings=False)
        try:
            visualizer_elbow.fit(df_clustering_scaled)
            elbow_plot_path = os.path.join(output_folder_multi, "kmeans_elbow_plot.png")
            visualizer_elbow.show(outpath=elbow_plot_path, clear_figure=True)  # 保存并显示
            print(f"Elbow曲线图已保存到: {elbow_plot_path}")
            print(f"根据Elbow图，建议的K值为: {visualizer_elbow.elbow_value_}")
            suggested_k_elbow = visualizer_elbow.elbow_value_ if visualizer_elbow.elbow_value_ else 3  # 默认值
        except Exception as e:
            print(f"生成Elbow图失败: {e}")
            suggested_k_elbow = 3  # 默认值

        # Silhouette Scores
        print("\n计算不同K值的轮廓系数:")
        silhouette_scores = {}
        for k_test in range(2, 7):  # 测试2到6个簇
            if k_test < len(df_clustering_scaled):  # K不能大于样本数
                kmeans_test = KMeans(n_clusters=k_test, random_state=42, n_init='auto')
                cluster_labels_test = kmeans_test.fit_predict(df_clustering_scaled)
                try:
                    score = silhouette_score(df_clustering_scaled, cluster_labels_test)
                    silhouette_scores[k_test] = score
                    print(f"  K={k_test}, 轮廓系数={score:.3f}")
                except ValueError:
                    print(f"  K={k_test}, 轮廓系数计算失败 (可能是由于簇内样本过少)")
            else:
                print(f"  K={k_test}, 样本量不足以形成这么多簇。")

        # 选择K (用户指定2-3类，这里可以结合Elbow和轮廓系数)
        # 优先考虑用户指定的2-3，然后看Elbow和轮廓系数哪个在这个范围内表现好
        # 如果Elbow建议的K在2-3之间，用它；否则，在2和3中选轮廓系数较高的
        final_k = suggested_k_elbow
        if not (2 <= final_k <= 3):
            if 3 in silhouette_scores and 2 in silhouette_scores:
                final_k = 3 if silhouette_scores.get(3, -1) >= silhouette_scores.get(2, -1) else 2
            elif 3 in silhouette_scores:
                final_k = 3
            elif 2 in silhouette_scores:
                final_k = 2
            else:  # 如果轮廓系数都没算出来，就用Elbow的建议，或者默认3
                final_k = suggested_k_elbow if suggested_k_elbow else 3

        if final_k >= len(df_clustering_scaled):  # 再次检查K值不能大于样本数
            final_k = max(2, len(df_clustering_scaled) - 1) if len(df_clustering_scaled) > 1 else 1
            print(f"警告: 调整K值为 {final_k} 以适应小样本量。")

        if final_k <= 1:
            print("样本量过小或数据问题，无法进行有意义的聚类。")
        else:
            print(f"\n--- 执行K-Means聚类 (K={final_k}) ---")
            kmeans = KMeans(n_clusters=final_k, random_state=42, n_init='auto')
            df_clustering_raw['Cluster'] = kmeans.fit_predict(df_clustering_scaled)
            df['Cluster_Multivariate'] = df_clustering_raw['Cluster']  # 将聚类结果合并回主df (基于共同索引)

            print("\n各用户群体的特征 (各特征均值):")
            cluster_profiles = df_clustering_raw.groupby('Cluster').mean()
            print(cluster_profiles)

            # 可视化聚类特征 (示例：使用频率)
            if '使用频率_数值' in cluster_profiles.columns:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x='Cluster', y='使用频率_数值', data=df_clustering_raw)
                plt.title(f'各聚类在使用频率上的分布 (K={final_k})')
                cluster_profile_plot_path = os.path.join(output_folder_multi, "cluster_usage_frequency_boxplot.png")
                plt.savefig(cluster_profile_plot_path)
                print(f"聚类使用频率箱线图已保存到: {cluster_profile_plot_path}")
                plt.show()
                plt.close()

            # 可视化更多特征的均值对比
            try:
                cluster_profiles_T = cluster_profiles.T  # 转置，方便绘图
                num_features_to_plot = len(cluster_profiles_T.index)
                if num_features_to_plot > 0:
                    cluster_profiles_T.plot(kind='bar', figsize=(15, 8), subplots=False)  # 所有簇画在一张图上对比
                    plt.title(f'各用户群体特征均值对比 (K={final_k})')
                    plt.ylabel('特征均值 (标准化前)')
                    plt.xticks(rotation=45, ha='right')
                    plt.legend(title='用户群体 (Cluster)')
                    plt.tight_layout()
                    cluster_profiles_bar_path = os.path.join(output_folder_multi, "cluster_profiles_barchart.png")
                    plt.savefig(cluster_profiles_bar_path)
                    print(f"聚类特征均值对比图已保存到: {cluster_profiles_bar_path}")
                    plt.show()
                    plt.close()
            except Exception as e_plot:
                print(f"绘制聚类特征均值对比图失败: {e_plot}")

    else:
        print("用户分群：移除缺失值后数据不足，无法进行聚类分析。")
else:
    print("用户分群：用于聚类的特征列缺失，无法进行分析。")

# ==============================================================================
# --- 3. 激励机制效果预测模型 (多元回归) ---
# ==============================================================================
print("\n" + "=" * 60 + "\n")
print("--- 3. 激励机制效果预测模型 (多元回归) ---")
dependent_var_incentive = '未来使用意向_平均分'

# 自变量：各激励方案的兴趣度评分 (Likert 1-5)
incentive_interest_cols = [col for col in likert_cols_from_csv if col.startswith('19.激励方案兴趣度:')]
incentive_interest_cols = [col for col in incentive_interest_cols if col in df.columns]  # 确保列存在

if dependent_var_incentive in df.columns and incentive_interest_cols:
    df_regression_incentive = df[[dependent_var_incentive] + incentive_interest_cols].copy()
    df_regression_incentive.dropna(inplace=True)

    if not df_regression_incentive.empty and len(df_regression_incentive) > len(incentive_interest_cols) + 1:
        Y_incentive = df_regression_incentive[dependent_var_incentive]
        X_incentive = df_regression_incentive[incentive_interest_cols]
        X_incentive = sm.add_constant(X_incentive)

        try:
            model_incentive = sm.OLS(Y_incentive, X_incentive).fit()
            print("\n--- 激励机制效果预测模型 (OLS回归) 结果 ---")
            print(f"因变量: {dependent_var_incentive}")
            print(model_incentive.summary())
            print("\n模型解释:")
            print("  - R-squared 表示模型解释的因变量方差的百分比。")
            print("  - coef 列显示了各个激励方案兴趣度对'未来使用意向平均分'的边际效应。")
            print("  - P>|t| 列显示了系数的显著性。P值小于0.05通常认为显著。")
            print("  - 显著的正系数表明，对该激励方案的兴趣度越高，未来使用意向得分也越高。")
            print("  - '最有效的激励机制组合'的判断较为复杂，此模型主要显示各方案兴趣度的独立关联强度。")
        except Exception as e:
            print(f"激励机制效果预测模型回归分析失败: {e}")

    else:
        print("激励机制效果预测：移除缺失值后数据不足或自变量不足，无法进行回归分析。")
else:
    print(f"激励机制效果预测：因变量 '{dependent_var_incentive}' 或自变量列表为空，无法进行回归分析。")
    print(f"  选定的激励方案兴趣度列: {incentive_interest_cols}")

print("\n" + "=" * 60)
print("--- 多变量分析脚本执行完毕 ---")
print(f"如果生成了图表，已尝试保存到 '{output_folder_multi}' 文件夹中。")
