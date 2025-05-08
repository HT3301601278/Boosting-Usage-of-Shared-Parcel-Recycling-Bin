import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# from yellowbrick.cluster import KElbowVisualizer # 可选，用于更复杂的K值选择

# --- 全局设置与文件夹创建 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

output_folder_viz = "可视化图表文件夹"
if not os.path.exists(output_folder_viz):
    os.makedirs(output_folder_viz)
    print(f"已创建文件夹: {output_folder_viz}")

# --- 数据加载与预处理 ---
print("--- 正在加载和预处理数据... ---")
try:
    df = pd.read_csv("数据详情值.csv")
except FileNotFoundError:
    print("错误：未找到 '数据详情值.csv' 文件。请确保文件与脚本在同一目录下，或提供正确路径。")
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
obstacle_items = ['位置不便', '排队/拥堵', '不清楚可回收物', '无奖励', '卫生差', '担心信息泄露', '其他']
incentive_interest_cols_prefix = '19.激励方案兴趣度:'
geo_province_col = '地理位置省'
geo_city_col = '地理位置市'

# --- 应用前缀清洗 ---
cols_to_clean_prefix = [
                           usage_frequency_col_actual, gender_col, age_col, education_col, occupation_col,
                       ] + [col for col in df.columns if any(col.startswith(prefix) for prefix in likert_q_prefixes)]

for col in cols_to_clean_prefix:
    if col in df.columns:
        df[col] = df[col].apply(clean_prefix)

# --- 数值编码与变量计算 ---
# 使用频率
usage_freq_mapping = {"0次": 0, "1-2次": 1, "3-5次": 2, "6-10次": 3, "10次以上": 4}
if usage_frequency_col_actual in df.columns:
    df['使用频率_数值'] = df[usage_frequency_col_actual].map(usage_freq_mapping)
else:
    df['使用频率_数值'] = np.nan

# 年龄 (用于散点图)
age_mapping = {"15-20": 0, "21-25": 1, "26-30": 2, "31-35": 3, "36-45": 4, "46及以上": 5}
age_category_order = ["15-20", "21-25", "26-30", "31-35", "36-45", "46及以上"]
if age_col in df.columns:
    df['年龄_数值'] = df[age_col].map(age_mapping)
    df[age_col] = pd.Categorical(df[age_col], categories=age_category_order, ordered=True)

# Likert量表映射
likert_mapping = {"非常不同意": 1, "不同意": 2, "一般": 3, "同意": 4, "非常同意": 5}
likert_cols_from_csv = [col for col in df.columns if any(col.startswith(prefix) for prefix in likert_q_prefixes)]
for col in likert_cols_from_csv:
    if col in df.columns:
        df[col] = df[col].map(likert_mapping)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 心理因素各维度平均分
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

# 激励方案兴趣度平均分
incentive_cols_actual = [col for col in df.columns if col.startswith(incentive_interest_cols_prefix)]
incentive_mean_scores = {}
if incentive_cols_actual:
    for col in incentive_cols_actual:
        incentive_name = col.split(':')[-1]
        incentive_mean_scores[incentive_name] = df[col].mean()

print("--- 数据加载与预处理完成 ---")
print(f"总样本数: {len(df)}")
if len(df) < 30:
    print(f"警告: 当前样本量 ({len(df)}) 非常小，部分图表可能无法有效生成或缺乏统计意义。")
print("\n" + "=" * 60 + "\n")

# ==============================================================================
# --- 1. 基础图表 ---
# ==============================================================================
print("--- 正在生成基础图表 ---")

# 1.1 人口统计学特征饼图
print("  生成人口统计学特征饼图...")
demographic_pie_charts = {
    gender_col: '受访者性别分布',
    age_col: '受访者年龄分布',
    education_col: '受访者学历分布'
}
for col_name, title in demographic_pie_charts.items():
    if col_name in df.columns and not df[col_name].dropna().empty:
        plt.figure(figsize=(8, 8))
        counts = df[col_name].value_counts()
        # 确保标签和计数的顺序一致，特别是对于已排序的分类变量（如年龄）
        if isinstance(df[col_name].dtype, pd.CategoricalDtype) and df[col_name].cat.ordered:
            counts = counts.reindex(df[col_name].cat.categories)
        plt.pie(counts.dropna(), labels=counts.dropna().index, autopct='%1.1f%%', startangle=140,
                textprops={'fontsize': 10})
        plt.title(title, fontsize=15)
        plt.axis('equal')
        filepath = os.path.join(output_folder_viz, f"{title.replace(' ', '_')}.png")
        plt.savefig(filepath)
        print(f"    图表已保存: {filepath}")
        plt.show()
        plt.close()
    else:
        print(f"    无法生成图表 '{title}'：列 '{col_name}' 不存在或数据为空。")

# 1.2 使用障碍因素条形图
print("\n  生成使用障碍因素条形图...")
obstacle_counts = {}
for item in obstacle_items:
    col_name = f"{obstacle_cols_raw_prefix}{item}"
    if col_name in df.columns:
        obstacle_counts[item] = df[col_name].count()

if obstacle_counts:
    obstacle_series = pd.Series(obstacle_counts).sort_values(ascending=False)
    if not obstacle_series.empty:
        plt.figure(figsize=(12, 7))
        obstacle_series.plot(kind='bar', color='skyblue')
        plt.title('主要使用障碍因素提及频次', fontsize=15)
        plt.xlabel('障碍因素', fontsize=12)
        plt.ylabel('提及人次', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        filepath = os.path.join(output_folder_viz, "使用障碍因素条形图.png")
        plt.savefig(filepath)
        print(f"    图表已保存: {filepath}")
        plt.show()
        plt.close()
    else:
        print("    无法生成使用障碍因素条形图：无有效的障碍数据。")
else:
    print("    无法生成使用障碍因素条形图：未找到障碍因素相关列。")

# 1.3 激励偏好排序柱状图
print("\n  生成激励偏好排序柱状图...")
if incentive_mean_scores:
    incentive_pref_series = pd.Series(incentive_mean_scores).dropna().sort_values(ascending=True)
    if not incentive_pref_series.empty:
        plt.figure(figsize=(12, 8))
        incentive_pref_series.plot(kind='barh', color='lightcoral')
        plt.title('不同激励方案的平均兴趣度排序', fontsize=15)
        plt.xlabel('平均兴趣度 (1-5分)', fontsize=12)
        plt.ylabel('激励方案', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        filepath = os.path.join(output_folder_viz, "激励偏好排序柱状图.png")
        plt.savefig(filepath)
        print(f"    图表已保存: {filepath}")
        plt.show()
        plt.close()
    else:
        print("    无法生成激励偏好排序柱状图：无有效的激励方案平均兴趣度数据。")
else:
    print("    无法生成激励偏好排序柱状图：未计算激励方案平均兴趣度。")

# 1.4 使用频率分布图 (分类条形图)
print("\n  生成使用频率分布图...")
if usage_frequency_col_actual in df.columns and not df[usage_frequency_col_actual].dropna().empty:
    plt.figure(figsize=(10, 6))
    usage_order = ["0次", "1-2次", "3-5次", "6-10次", "10次以上"]
    df_usage_temp = df[[usage_frequency_col_actual]].copy()
    df_usage_temp[usage_frequency_col_actual] = pd.Categorical(
        df_usage_temp[usage_frequency_col_actual],
        categories=usage_order,
        ordered=True
    )
    usage_counts_ordered = df_usage_temp[usage_frequency_col_actual].value_counts().sort_index()

    usage_counts_ordered.plot(kind='bar', color='mediumseagreen')
    plt.title('回收箱使用频率分布', fontsize=15)
    plt.xlabel('使用次数 (过去30天)', fontsize=12)
    plt.ylabel('受访者人数', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    filepath = os.path.join(output_folder_viz, "使用频率分布图.png")
    plt.savefig(filepath)
    print(f"    图表已保存: {filepath}")
    plt.show()
    plt.close()
else:
    print("    无法生成使用频率分布图：使用频率数据列不存在或为空。")

# ==============================================================================
# --- 2. 关系图表 ---
# ==============================================================================
print("\n" + "=" * 60 + "\n")
print("--- 正在生成关系图表 ---")

# 2.1 年龄-使用频率散点图 (stripplot)
print("\n  生成年龄-使用频率散点图...")
if age_col in df.columns and '使用频率_数值' in df.columns:  # 使用原始年龄分类列 age_col
    df_scatter = df[[age_col, '使用频率_数值']].copy().dropna()
    if not df_scatter.empty:
        plt.figure(figsize=(10, 7))
        # 按照 FutureWarning 的建议修改 stripplot 调用
        sns.stripplot(x=age_col, y='使用频率_数值', data=df_scatter,
                      order=age_category_order,  # 确保x轴顺序
                      hue=age_col,  # 将x变量赋给hue
                      palette="viridis",  # 调色板
                      jitter=0.2,
                      alpha=0.7,
                      legend=False  # 关闭图例，因为hue与x相同
                      )
        plt.title('年龄与使用频率关系图', fontsize=15)  # 保持散点图的称呼，但用stripplot实现
        plt.xlabel('年龄段', fontsize=12)
        plt.ylabel('使用频率 (编码值)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(ticks=sorted(df['使用频率_数值'].dropna().unique()), fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        filepath = os.path.join(output_folder_viz, "年龄_使用频率散点图.png")
        plt.savefig(filepath)
        print(f"    图表已保存: {filepath}")
        plt.show()
        plt.close()
    else:
        print("    无法生成年龄-使用频率散点图：数据不足。")
else:
    print("    无法生成年龄-使用频率散点图：年龄或使用频率数值列不存在。")

# 2.2 心理因素与使用意愿关系热力图
print("\n  生成心理因素与使用意愿关系热力图...")
cols_for_heatmap = [col for col in psych_dim_mean_cols if col in df.columns and col != '未来使用意向_平均分']
if '未来使用意向_平均分' in df.columns and df['未来使用意向_平均分'].notna().any():
    cols_for_heatmap.append('未来使用意向_平均分')

if len(cols_for_heatmap) > 1:
    df_heatmap_data = df[cols_for_heatmap].dropna()
    if not df_heatmap_data.empty and len(df_heatmap_data) >= 2:
        correlation_matrix = df_heatmap_data.corr(method='spearman')
        plt.figure(figsize=(12, 10))
        cleaned_labels_heatmap = [label.replace('_平均分', '') for label in correlation_matrix.columns]
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                    xticklabels=cleaned_labels_heatmap, yticklabels=cleaned_labels_heatmap,
                    annot_kws={"size": 8})
        plt.title('心理因素与未来使用意向的相关性热力图', fontsize=15)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        filepath = os.path.join(output_folder_viz, "心理因素_使用意愿热力图.png")
        plt.savefig(filepath)
        print(f"    图表已保存: {filepath}")
        plt.show()
        plt.close()
    else:
        print("    无法生成心理因素与使用意愿关系热力图：数据不足。")
else:
    print("    无法生成心理因素与使用意愿关系热力图：有效的心理因素或使用意愿列不足。")

# 2.3 不同激励方案效果比较雷达图 (基于平均兴趣度)
print("\n  生成不同激励方案效果比较雷达图...")
if incentive_mean_scores:  # 确保 incentive_mean_scores 已被计算且非空
    # 过滤掉NaN或无效的平均分
    valid_incentive_scores = {k: v for k, v in incentive_mean_scores.items() if pd.notna(v)}

    if valid_incentive_scores and len(valid_incentive_scores) >= 3:
        labels_radar_incentive = list(valid_incentive_scores.keys())
        stats_radar_incentive = list(valid_incentive_scores.values())

        angles = np.linspace(0, 2 * np.pi, len(labels_radar_incentive), endpoint=False).tolist()
        stats_closed = np.concatenate((stats_radar_incentive, [stats_radar_incentive[0]]))
        angles_closed = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.plot(angles_closed, stats_closed, linewidth=2, linestyle='solid', color='crimson', label='平均兴趣度')
        ax.fill(angles_closed, stats_closed, 'crimson', alpha=0.25)
        ax.set_thetagrids(np.degrees(angles), labels_radar_incentive, fontsize=10)
        ax.set_title('不同激励方案平均兴趣度比较雷达图', size=16, y=1.1)
        ax.set_yticks(np.arange(1, 6, 1))
        ax.set_ylim(0, 5)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        filepath = os.path.join(output_folder_viz, "激励方案效果雷达图.png")
        plt.savefig(filepath)
        print(f"    图表已保存: {filepath}")
        plt.show()
        plt.close()
    else:
        print("    无法生成激励方案效果雷达图：有效的激励方案不足3个或数据为空。")
else:
    print("    无法生成激励方案效果雷达图：未计算激励方案平均兴趣度或结果为空。")

# 2.4 用户分群特征雷达图
print("\n  生成用户分群特征雷达图...")
features_for_clustering_viz = ['使用频率_数值'] + [col for col in psych_dim_mean_cols if
                                                   col in df.columns and col != '未来使用意向_平均分']
df_for_k_means = df[features_for_clustering_viz].copy().dropna()

if not df_for_k_means.empty and len(df_for_k_means) >= 3:
    scaler_viz = StandardScaler()
    df_k_means_scaled_array = scaler_viz.fit_transform(df_for_k_means)

    k_clusters = 3
    if len(df_for_k_means) < k_clusters:
        k_clusters = max(2, len(df_for_k_means))
        print(f"    聚类警告: 样本量不足以分为3个簇，调整为K={k_clusters}")

    if k_clusters >= 2:
        kmeans_viz = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans_viz.fit_predict(df_k_means_scaled_array)
        df_for_k_means['用户分群'] = cluster_labels

        cluster_profiles_viz = df_for_k_means.groupby('用户分群').mean()

        radar_labels_cluster = cluster_profiles_viz.columns.tolist()
        radar_labels_cluster_cleaned = [label.replace('_平均分', '').replace('_数值', '') for label in
                                        radar_labels_cluster]

        if len(radar_labels_cluster) >= 3:
            angles_cluster = np.linspace(0, 2 * np.pi, len(radar_labels_cluster), endpoint=False).tolist()
            angles_cluster_closed = angles_cluster + [angles_cluster[0]]

            fig_cluster, ax_cluster = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
            colors = ['royalblue', 'forestgreen', 'darkorange', 'purple', 'brown']

            for i, (cluster_id, row) in enumerate(cluster_profiles_viz.iterrows()):
                stats_cluster = row.values.flatten().tolist()
                stats_cluster_closed = np.concatenate((stats_cluster, [stats_cluster[0]]))
                ax_cluster.plot(angles_cluster_closed, stats_cluster_closed, linewidth=2, linestyle='solid',
                                label=f'用户群 {cluster_id} (N={df_for_k_means["用户分群"].value_counts().get(cluster_id, 0)})',
                                color=colors[i % len(colors)])
                ax_cluster.fill(angles_cluster_closed, stats_cluster_closed, alpha=0.15, color=colors[i % len(colors)])

            ax_cluster.set_thetagrids(np.degrees(angles_cluster), radar_labels_cluster_cleaned, fontsize=10)
            ax_cluster.set_title(f'用户分群特征雷达图 (K={k_clusters}, 均值为原始值)', size=16, y=1.1)

            # 动态调整Y轴范围，确保所有数据点可见
            all_profile_values = cluster_profiles_viz.values.flatten()
            y_min = min(0, np.nanmin(all_profile_values)) if len(all_profile_values) > 0 else 0
            y_max = max(5, np.nanmax(all_profile_values)) if len(all_profile_values) > 0 else 5
            ax_cluster.set_yticks(np.linspace(y_min, y_max, 6))
            ax_cluster.set_ylim(y_min, y_max)

            plt.legend(loc='upper right', bbox_to_anchor=(0.15, 0.1), fontsize=10)
            filepath = os.path.join(output_folder_viz, "用户分群特征雷达图.png")
            plt.savefig(filepath)
            print(f"    图表已保存: {filepath}")
            plt.show()
            plt.close()
        else:
            print("    无法生成用户分群雷达图：用于聚类的特征不足3个。")
    else:
        print("    无法生成用户分群雷达图：聚类后K值小于2。")
else:
    print("    无法生成用户分群雷达图：用于聚类的数据不足。")

# ==============================================================================
# --- 3. 地理分布 ---
# ==============================================================================
print("\n" + "=" * 60 + "\n")
print("--- 正在生成地理分布图表 ---")

# 3.1 基于问卷地理位置数据的使用率差异简图 (按省份)
print("\n  生成按省份的平均使用频率图...")
if geo_province_col in df.columns and '使用频率_数值' in df.columns:
    df_geo = df[[geo_province_col, '使用频率_数值']].copy().dropna()
    if not df_geo.empty:
        province_counts = df_geo[geo_province_col].value_counts()
        provinces_to_include = province_counts[province_counts >= 5].index.tolist()  # 阈值设为5

        if provinces_to_include:
            df_geo_filtered = df_geo[df_geo[geo_province_col].isin(provinces_to_include)]
            avg_usage_by_province = df_geo_filtered.groupby(geo_province_col)['使用频率_数值'].mean().sort_values(
                ascending=True)

            if not avg_usage_by_province.empty:
                plt.figure(figsize=(10, max(6, len(avg_usage_by_province) * 0.5)))
                avg_usage_by_province.plot(kind='barh', color='teal')
                plt.title('各省份平均回收箱使用频率 (样本量≥5的省份)', fontsize=15)
                plt.xlabel('平均使用频率 (编码值)', fontsize=12)
                plt.ylabel('省份', fontsize=12)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.tight_layout()
                filepath = os.path.join(output_folder_viz, "各省份平均使用频率图.png")
                plt.savefig(filepath)
                print(f"    图表已保存: {filepath}")
                plt.show()
                plt.close()
            else:
                print("    无法生成按省份的平均使用频率图：筛选后无有效省份数据。")
        else:
            print("    无法生成按省份的平均使用频率图：所有省份的样本量均小于5。")
    else:
        print("    无法生成按省份的平均使用频率图：地理位置或使用频率数据不足。")
else:
    print(f"    无法生成按省份的平均使用频率图：列 '{geo_province_col}' 或 '使用频率_数值' 不存在。")

print("\n" + "=" * 60)
print("--- 可视化展示计划脚本执行完毕 ---")
print(f"所有图表已尝试保存到 '{output_folder_viz}' 文件夹中。")
