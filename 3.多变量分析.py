import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- 0. 数据加载与通用预处理函数 ---
try:
    df_original = pd.read_csv('数据详情值.csv')
except FileNotFoundError:
    print("错误：数据详情值.csv 文件未找到。请确保文件路径正确。")
    exit()

df_main_analysis = df_original.copy()
print(f"原始数据行数: {len(df_main_analysis)}, 列数: {len(df_main_analysis.columns)}")

likert_map_positive = {'A.非常不同意': 1, 'B.不同意': 2, 'C.一般': 3, 'D.同意': 4, 'E.非常同意': 5}

df_main_analysis.columns = df_main_analysis.columns.str.replace('\s+', '', regex=True)
df_main_analysis.columns = df_main_analysis.columns.str.replace('[*\n]', '', regex=True)
df_main_analysis.columns = df_main_analysis.columns.str.replace('（可多选）', '', regex=False)

# --- 图片保存设置 ---
output_folder = "多变量分析图表"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"已创建文件夹: {output_folder}")

# --- 1. 行为预测模型 ---
print("\n--- 1. 行为预测模型 (多元回归) ---")
print("注意：当前模型包含大量自变量，可能导致过拟合和不稳定的结果。强烈建议进行特征选择或降维。")
# 1.1 因变量 Y: 使用频率 (3.过去30天您使用回收箱的次数)
usage_freq_map = {'A.0次': 0, 'B.1-2次': 1.5, 'C.3-5次': 4, 'D.6-10次': 8, 'E.10次以上': 12}
df_main_analysis['使用频率_数值'] = df_main_analysis['3.过去30天您使用回收箱的次数'].map(usage_freq_map)

# 处理因变量缺失值
missing_y_count = df_main_analysis['使用频率_数值'].isnull().sum()
if missing_y_count > 0:
    print(f"警告: 因变量 '使用频率_数值' 存在 {missing_y_count} 个缺失值。")
    df_main_analysis.dropna(subset=['使用频率_数值'], inplace=True)
    print(f"已删除包含缺失因变量的行，剩余样本数: {len(df_main_analysis)}")

if len(df_main_analysis) < 50:
    print("警告：处理因变量缺失值后，有效样本量过少，可能不适合进行可靠的回归分析。")

y_reg = df_main_analysis['使用频率_数值'].copy()


# 1.2 自变量 X:
user_features_categorical = ['7.您的性别', '8.您的年龄', '9.您的最高学历', '10.您的职业']
psychological_cols = []
for i in range(11, 19):
    cols_for_q = [col for col in df_main_analysis.columns if col.startswith(f'{i}.') and ':' in col]
    psychological_cols.extend(cols_for_q)

for col in psychological_cols:
    df_main_analysis[col + '_数值'] = df_main_analysis[col].map(likert_map_positive)

obstacle_cols_prefix = '5.您在使用回收箱时遇到的主要障碍'
obstacle_cols = [col for col in df_main_analysis.columns if col.startswith(obstacle_cols_prefix) and ':' in col]

for col in obstacle_cols:
    df_main_analysis[col + '_二元'] = df_main_analysis[col].notna().astype(int)

# 构建自变量DataFrame
X_reg_num_cols = [col + '_数值' for col in psychological_cols] + \
                 [col + '_二元' for col in obstacle_cols]
X_reg_cat_cols = user_features_categorical

X_reg_df = df_main_analysis[X_reg_cat_cols + X_reg_num_cols].copy()

# 缺失值处理 (自变量):
for col in X_reg_num_cols:
    if X_reg_df[col].isnull().any():
        X_reg_df[col] = X_reg_df[col].fillna(X_reg_df[col].median())
for col in X_reg_cat_cols:
    if X_reg_df[col].isnull().any():
        X_reg_df[col] = X_reg_df[col].fillna(X_reg_df[col].mode()[0])

# 创建预处理管道
preprocessor_reg = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), X_reg_cat_cols)
    ],
    remainder='passthrough'
)

X_reg_processed_array = preprocessor_reg.fit_transform(X_reg_df)

try:
    feature_names_onehot = list(preprocessor_reg.named_transformers_['onehot'].get_feature_names_out(X_reg_cat_cols))
except AttributeError:
    feature_names_onehot = []
    encoder = preprocessor_reg.named_transformers_['onehot']
    for i, col_name in enumerate(X_reg_cat_cols):
        categories = encoder.categories_[i]
        if encoder.drop is None or (hasattr(encoder, 'drop_idx_') and encoder.drop_idx_ is None) or (not hasattr(encoder, 'drop_idx_') and encoder.drop is None): # No dropping or 'if_binary' and not binary
             names = [f"{col_name}_{cat}" for cat in categories]
        elif (hasattr(encoder, 'drop_idx_') and encoder.drop_idx_ is not None and encoder.drop_idx_[i] == 0 and encoder.drop == 'first') or \
             (not hasattr(encoder, 'drop_idx_') and encoder.drop == 'first'): # drop='first'
             names = [f"{col_name}_{cat}" for cat in categories[1:]]
        else: # Manual handling for specific drop indices if any, or other drop strategies
             dropped_indices = encoder.drop_idx_[i] if hasattr(encoder, 'drop_idx_') and isinstance(encoder.drop_idx_[i], (list, np.ndarray)) else ([encoder.drop_idx_[i]] if hasattr(encoder, 'drop_idx_') and encoder.drop_idx_[i] is not None else [])
             names = [f"{col_name}_{cat}" for idx, cat in enumerate(categories) if idx not in dropped_indices]
        feature_names_onehot.extend(names)


feature_names_reg = feature_names_onehot + X_reg_num_cols
X_reg_processed_df = pd.DataFrame(X_reg_processed_array, columns=feature_names_reg, index=y_reg.index)

X_reg_sm = sm.add_constant(X_reg_processed_df.astype(float))
common_index = y_reg.index.intersection(X_reg_sm.index)
y_reg_aligned = y_reg.loc[common_index]
X_reg_sm_aligned = X_reg_sm.loc[common_index]

if len(y_reg_aligned) < len(feature_names_reg) + 1:
     print(f"警告: 行为预测模型的样本量 ({len(y_reg_aligned)}) 过少，相对于参数数量 ({len(feature_names_reg)+1})，回归结果可能不可靠。")
else:
    try:
        model_reg_sm = sm.OLS(y_reg_aligned.astype(float), X_reg_sm_aligned.astype(float)).fit()
        print(model_reg_sm.summary(xname=['const'] + feature_names_reg))
        print("\n解读提示：关注R-squared, Adj. R-squared, F-statistic P-value, 以及各变量的coef和P>|t|。")
    except Exception as e:
        print(f"行为预测模型回归分析失败: {e}")

# --- 2. 用户分群分析 (K-Means聚类) ---
print("\n--- 2. 用户分群分析 (K-Means聚类) ---")

df_for_cluster = df_original.copy()

df_for_cluster.columns = df_for_cluster.columns.str.replace('\s+', '', regex=True)
df_for_cluster.columns = df_for_cluster.columns.str.replace('[*\n]', '', regex=True)
df_for_cluster.columns = df_for_cluster.columns.str.replace('（可多选）', '', regex=False)

df_for_cluster['使用频率_数值'] = df_for_cluster['3.过去30天您使用回收箱的次数'].map(usage_freq_map)
if df_for_cluster['使用频率_数值'].isnull().any():
    df_for_cluster['使用频率_数值'] = df_for_cluster['使用频率_数值'].fillna(df_for_cluster['使用频率_数值'].median())

psychological_cols_cluster_names = []
for i in range(11, 19):
    cols_for_q = [col for col in df_for_cluster.columns if col.startswith(f'{i}.') and ':' in col]
    psychological_cols_cluster_names.extend(cols_for_q)

for col in psychological_cols_cluster_names:
    df_for_cluster[col + '_数值'] = df_for_cluster[col].map(likert_map_positive)

cluster_features_list = ['使用频率_数值'] + [col + '_数值' for col in psychological_cols_cluster_names]
df_cluster_data = df_for_cluster[cluster_features_list].copy()

for col in cluster_features_list:
    if df_cluster_data[col].isnull().any():
        df_cluster_data[col] = df_cluster_data[col].fillna(df_cluster_data[col].median())

scaler_cluster = StandardScaler()
df_cluster_scaled = scaler_cluster.fit_transform(df_cluster_data)

inertia = []
silhouette_scores = []
k_range = range(2, 8)

for k_val in k_range:
    kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
    kmeans.fit(df_cluster_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_cluster_scaled, kmeans.labels_))

# 绘制并保存图表
fig_kmeans_k_selection, axs = plt.subplots(1, 2, figsize=(12, 5)) # <<< 修改：获取figure和axes对象

axs[0].plot(k_range, inertia, marker='o')
axs[0].set_title('Elbow Method for Optimal K')
axs[0].set_xlabel('Number of clusters (K)')
axs[0].set_ylabel('Inertia')

axs[1].plot(k_range, silhouette_scores, marker='o')
axs[1].set_title('Silhouette Score for Optimal K')
axs[1].set_xlabel('Number of clusters (K)')
axs[1].set_ylabel('Silhouette Score')

plt.tight_layout()
chart_path_kmeans_k = os.path.join(output_folder, "kmeans_k_selection.png") # <<< 新增：定义图表保存路径
plt.savefig(chart_path_kmeans_k) # <<< 新增：保存图表
print(f"K-Means K值选择图表已保存至: {chart_path_kmeans_k}")
plt.show() # 继续显示图表

chosen_k_input = input("根据图表，请输入选择的K值 (例如 3): ")
try:
    chosen_k = int(chosen_k_input)
    if chosen_k < 2 : chosen_k = 3
except ValueError:
    print("输入无效，默认使用 K=3")
    chosen_k = 3

print(f"\n选择 K = {chosen_k} 进行聚类。")

kmeans_final = KMeans(n_clusters=chosen_k, random_state=42, n_init='auto')
df_for_cluster['Cluster'] = kmeans_final.fit_predict(df_cluster_scaled)

cluster_summary_features = df_for_cluster.groupby('Cluster')[cluster_features_list].mean()
print("\n各用户群体在聚类特征上的均值:")
print(cluster_summary_features)

demographic_cols_for_summary = ['7.您的性别', '8.您的年龄', '9.您的最高学历', '10.您的职业']
for col_demo in demographic_cols_for_summary:
    print(f"\n--- {col_demo} 在各簇中的分布 (%) ---")
    distribution = df_for_cluster.groupby('Cluster')[col_demo].value_counts(normalize=True).mul(100).round(1).unstack(fill_value=0)
    print(distribution)

# --- 3. 激励机制效果预测模型 ---
print("\n--- 3. 激励机制效果预测模型 (多元回归) ---")
# ... (激励机制效果预测模型的代码保持不变，此处省略以减少篇幅) ...
future_intention_col_prefix = '20.未来使用意向'
target_intention_col_name_part = "每月持续使用" # 或者选择其他意向指标
potential_cols_intent = [c for c in df_main_analysis.columns if c.startswith(future_intention_col_prefix) and target_intention_col_name_part in c]

if not potential_cols_intent:
    print(f"错误: 找不到包含 '{target_intention_col_name_part}' 的未来使用意向列。请检查列名。")
else:
    target_intention_col = potential_cols_intent[0]
    df_main_analysis[target_intention_col + '_数值'] = df_main_analysis[target_intention_col].map(likert_map_positive)
    y_incentive = df_main_analysis[target_intention_col + '_数值'].copy()

    if y_incentive.isnull().any():
        y_incentive = y_incentive.fillna(y_incentive.median())

    incentive_interest_cols_prefix = '19.激励方案兴趣度'
    incentive_interest_cols = [col for col in df_main_analysis.columns if col.startswith(incentive_interest_cols_prefix) and ':' in col]

    X_incentive_df = pd.DataFrame(index=df_main_analysis.index)
    for col in incentive_interest_cols:
        X_incentive_df[col + '_数值'] = df_main_analysis[col].map(likert_map_positive)

    for col in X_incentive_df.columns:
        if X_incentive_df[col].isnull().any():
            X_incentive_df[col] = X_incentive_df[col].fillna(X_incentive_df[col].median())

    X_incentive_sm = sm.add_constant(X_incentive_df.astype(float))
    common_index_incentive = y_incentive.index.intersection(X_incentive_sm.index)
    y_incentive_aligned = y_incentive.loc[common_index_incentive]
    X_incentive_sm_aligned = X_incentive_sm.loc[common_index_incentive]

    if len(y_incentive_aligned) < len(X_incentive_sm_aligned.columns):
        print(f"警告: 激励机制效果预测模型的样本量 ({len(y_incentive_aligned)}) 过少，相对于参数数量 ({len(X_incentive_sm_aligned.columns)})，回归结果可能不可靠。")
    else:
        try:
            model_incentive_sm = sm.OLS(y_incentive_aligned.astype(float), X_incentive_sm_aligned.astype(float)).fit()
            print(model_incentive_sm.summary(xname=['const'] + list(X_incentive_df.columns)))

            results_summary_incentive = model_incentive_sm.summary2().tables[1]
            significant_incentives = results_summary_incentive[results_summary_incentive['P>|t|'] < 0.05].sort_values(by='Coef.', ascending=False)
            print("\n显著影响未来使用意向的激励方案 (按影响程度降序排列):")
            if not significant_incentives.empty:
                print(significant_incentives[['Coef.', 'P>|t|']])
            else:
                print("在此模型中，没有找到在0.05水平下显著影响未来使用意向的激励方案。")
        except Exception as e:
            print(f"激励机制效果预测模型回归分析失败: {e}")

print("\n--- 多变量分析完成 ---")