"""计算纹饰、类型、颜色属性与表面风化属性的斯皮尔曼相关系数"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

filename = 'data/附件.xlsx'
info = pd.read_excel(filename, sheet_name='表单1')

# 特征编码
for col in ['类型', '表面风化']:
    le = LabelEncoder()
    info[col] = le.fit_transform(info[col])

# 计算类型属性与表面风化属性的斯皮尔曼相关系数
print('类型', spearmanr(info['表面风化'], info['类型']))

# 特征离散化
lag_onehot = pd.get_dummies(info['纹饰'])
# 计算纹饰属性与表面风化属性的斯皮尔曼相关系数
for col in lag_onehot.columns:
    print(col, spearmanr(info['表面风化'], lag_onehot[col]))

# 删除缺失值
info_dropna = info.dropna()
color_onehot = pd.get_dummies(info_dropna['颜色'])
# 计算颜色属性与表面风化属性的斯皮尔曼相关系数
for col in color_onehot.columns:
    print(col, spearmanr(info_dropna['表面风化'], color_onehot[col]))



