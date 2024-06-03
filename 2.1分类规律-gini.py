import pandas as pd
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.globals import ThemeType
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, plot_tree

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


# print(chem.loc[:, ('类型', '氧化铅(PbO)')])
# X = pd.concat([chem.iloc[:, 1:-5], chem['表面风化']], axis=1)
X = chem.iloc[:, 1:-5]
# print(X)
y = chem['类型']

dtc = DecisionTreeClassifier(criterion='gini')
dtc.fit(X, y)
print(dtc.tree_.feature)
print(dtc.tree_.impurity)

# for col in X.columns:
#     dtc = DecisionTreeClassifier(criterion='entropy')
#     dtc.fit(X[col].to_numpy().reshape(-1, 1), y)
#     # plot_tree(dtc,
#     #           filled=True,
#     #           rounded=True)
#     print(dtc.tree_.feature)
#     print(dtc.tree_.impurity)
