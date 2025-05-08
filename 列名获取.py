import pandas as pd

df = pd.read_csv('数据详情值.csv')
print("CSV中的所有列名:", df.columns.tolist()) # 添加这行