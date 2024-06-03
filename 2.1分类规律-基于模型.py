import pandas as pd
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts.globals import ThemeType
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

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


print(chem.loc[:, ('类型', '氧化铅(PbO)')])
# X = pd.concat([chem.iloc[:, 1:-5], chem['表面风化']], axis=1)
X = chem.iloc[:, 1:-5]
y = chem['类型']
print(y)
# print(X)
# print(y)

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
# 带L1惩罚项的逻辑回归作为基模型的特征选择
sf_model: SelectFromModel = SelectFromModel(LogisticRegression())
sf_model.fit(X, y)
# 显示保留的特征
print("select feature: ", X.columns[sf_model.get_support()])
print(sf_model.estimator_.coef_)


from sklearn.tree import DecisionTreeClassifier, plot_tree
sf_dtc = SelectFromModel(DecisionTreeClassifier())
sf_dtc.fit(X, y)
print("select feature: ", X.columns[sf_model.get_support()])

# # 使用scikit-learn.feature_extraction中的特征转换器
# from sklearn.feature_extraction import DictVectorizer
#
# vec = DictVectorizer(sparse=True)
# # 转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变。
# X = vec.fit_transform(X.to_dict(orient='record'))
# print(vec.feature_names_)
#
# # 从sklearn.tree中导入决策树分类器
#
# # 初始化决策树分类器
# dtc = DecisionTreeClassifier(random_state=0)
# dtc.fit(X, y)
#
# plot_tree(dtc,
#           feature_names=vec.feature_names_,
#           class_names=['铅钡玻璃', '高钾玻璃'],
#           filled=True,
#           rounded=True)
#
# plt.savefig('output/tree_visualization.png', dpi=600)
#
#
# highK = chem[chem['类型'] == '高钾']
# highK_weat = highK[highK['表面风化'] == '风化']
# highK_unweat = highK[highK['表面风化'] == '无风化']
# highK_weat_mean = highK_weat.iloc[:, 1:-4].mean()
#
# # print(highK_weat)
# # print(highK_unweat)
#
#
# PbBa = chem[chem['类型'] == '铅钡']
# PbBa_weat = PbBa[PbBa['表面风化'] == '风化']
# PbBa_unweat = PbBa[PbBa['表面风化'] == '无风化']
# # print(PbBa_weat)
# # print(PbBa_unweat)

