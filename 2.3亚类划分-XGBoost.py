import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.globals import ThemeType
from xgboost import XGBClassifier, plot_tree

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.style.use('ggplot')


filename = 'data/附件.xlsx'
info = pd.read_excel(filename, sheet_name='表单1')
chem = pd.read_excel(filename, sheet_name='表单2')
chem.fillna(0, inplace=True)
chem['累加和'] = chem.iloc[:, 1:].apply(lambda x: x.sum(), axis=1)
chem = chem[chem['累加和'] <= 105]
chem = chem[chem['累加和'] >= 85]
chem['文物编号'] = chem['文物采样点'].apply(lambda x: int(x[:2]))
chem['采样点'] = chem['文物采样点'].apply(lambda x: x[2:])
# chem = chem[chem['采样点'] != '严重风化点']
chem.index = list(range(len(chem)))
chem['类型'] = [0] * len(chem)
chem['表面风化'] = [0] * len(chem)
for i in chem.index:
    chem['类型'][i] = info[info['文物编号'] == chem['文物编号'][i]]['类型'].tolist()[0]
    chem['表面风化'][i] = info[info['文物编号'] == chem['文物编号'][i]]['表面风化'].tolist()[0]
    # if '未风化点' in chem['采样点'][i]:
    #     chem['表面风化'][i] = '无风化'

weather = pd.get_dummies(chem['表面风化'])
chem = pd.concat([chem, weather], axis=1)

highK = chem[chem['类型'] == '高钾']
highK_X = highK.iloc[:, list(range(1, 15))]
highK_X = StandardScaler().fit_transform(highK_X)
kmeans = KMeans(n_clusters=6)
kmeans.fit(highK_X)
highK['亚类'] = kmeans.labels_
print(highK.loc[:, ['二氧化硅(SiO2)', '二氧化硫(SO2)', '氧化钙(CaO)', '氧化钡(BaO)', '氧化镁(MgO)', '亚类']])

highK_X = highK.iloc[:, list(range(1, 15))]
highK_y = highK['亚类']
print(highK_X)
print(highK_y)
# 初始化决策树分类器
xgbc = XGBClassifier()
xgbc.fit(highK_X, highK_y)
plt.gcf()
plot_tree(xgbc)
plt.show()
# plt.savefig('image/高钾玻璃xgboost.png', dpi=600)


# PbBa = chem[chem['类型'] == '铅钡']
# PbBa_X = PbBa.iloc[:, list(range(1, 15))]
# PbBa_X = StandardScaler().fit_transform(PbBa_X)
# kmeans = KMeans(n_clusters=8)
# kmeans.fit(PbBa_X)
# PbBa['亚类'] = kmeans.labels_
# PbBa_X = PbBa.iloc[:, list(range(1, 15))]
# PbBa_y = PbBa['亚类']
# # 初始化决策树分类器
# dtc = DecisionTreeClassifier(random_state=0)
# dtc.fit(PbBa_X, PbBa_y)
# plot_tree(dtc,
#           feature_names=PbBa_X.columns,
#           class_names=['亚类Ⅰ', '亚类Ⅱ', '亚类Ⅲ', '亚类Ⅳ', '亚类Ⅴ', '亚类Ⅵ', '亚类Ⅶ', '亚类Ⅷ'],
#           filled=True,
#           rounded=True)
# plt.savefig('image/铅钡玻璃决策树.png', dpi=600)
#
