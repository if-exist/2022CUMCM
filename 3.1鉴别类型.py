"""使用逻辑斯蒂回归模型训练表二数据，对表三的未知数据进行类型分类，并对分类模型进行敏感度测试"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


def plot_line(y, y_label):
    x = [str(i) + "%" for i in range(1, 41)]

    trace = go.Scatter(
        x=x,
        y=y,
        mode='lines',
    )

    layout = go.Layout(
        xaxis=dict(
            title='随机波动幅度',
            ticks='outside',
        ),
        yaxis=dict(
            title=y_label,
            ticks='outside',
        ),
    )

    fig = go.Figure(data=trace, layout=layout)
    fig.show()
    filename = "image/logistic" + y_label + ".png"
    fig.write_image(filename, scale=6)
    Image.open(filename).show()


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


filename = 'data/附件.xlsx'
info = pd.read_excel(filename, sheet_name='表单1')
chem = pd.read_excel(filename, sheet_name='表单2')
chem.fillna(0, inplace=True)
chem['累加和'] = chem.iloc[:, 1:].apply(lambda x: x.sum(), axis=1)
chem = chem[chem['累加和'] <= 105]
chem = chem[chem['累加和'] >= 85]
chem['文物编号'] = chem['文物采样点'].apply(lambda x: int(x[:2]))
chem['采样点'] = chem['文物采样点'].apply(lambda x: x[2:])
chem.index = list(range(len(chem)))
chem['类型'] = [0] * len(chem)
chem['表面风化'] = [0] * len(chem)
for i in chem.index:
    chem['类型'][i] = info[info['文物编号'] == chem['文物编号'][i]]['类型'].tolist()[0]
    chem['表面风化'][i] = info[info['文物编号'] == chem['文物编号'][i]]['表面风化'].tolist()[0]

unknown = pd.read_excel(filename, sheet_name='表单3')
unknown.fillna(0, inplace=True)
unknown_X = unknown.iloc[:, 2:]
X = chem.iloc[:, 1:-5]
y = chem['类型']


lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=0)
lr.fit(X, y)
print(lr.coef_)
print(lr.intercept_)
unknown_pred = lr.predict(unknown_X)
print(unknown_pred)
print(lr.predict_proba(unknown_X))
print(lr.decision_function(unknown_X))


unknown_X = unknown_X.values
scores = []
pred_probas = []
decisions = []
for _ in range(1000):
    score_round = []
    proba_round = []
    decision_round = []
    for i in range(1, 41):
        unknown_X_wave = unknown_X * (np.random.uniform(low=(100-i)/100, high=(100+i)/100, size=(8, 14)))
        score_round.append(lr.score(unknown_X_wave, unknown_pred))
        proba_round.append(lr.predict_proba(unknown_X_wave[[0], :]))
        decision_round.append(lr.decision_function(unknown_X_wave[[0], :]))
    scores.append(score_round)
    pred_probas.append(proba_round)
    decisions.append(decision_round)
scores = np.array(scores).mean(axis=0)
pred_probas = np.array(pred_probas).mean(axis=0)
decisions = np.array(decisions).mean(axis=0)
pred_probas = [proba[0][1] for proba in pred_probas]
decisions = decisions.ravel()

print(scores)
print(pred_probas)
print(decisions)

plot_line(scores, '精确度')
plot_line(pred_probas, '预测概率值')
plot_line(decisions, '置信度')



