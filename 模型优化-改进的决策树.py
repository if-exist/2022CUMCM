"""使用改进的决策树算法进行亚类划分，并对分裂结果进行敏感性测试"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


def plot_line_accu(y, y_label, title):
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
    filename = "image/改进的决策树" + title + y_label + ".png"
    fig.write_image(filename, scale=6)
    Image.open(filename).show()


def sens_analyse(model, X, y_true, title):
    scores = []
    for _ in range(1000):
        score_round = []
        for i in range(1, 41):
            X_wave = X * (np.random.uniform(low=(100 - i) / 100, high=(100 + i) / 100, size=(len(X), 14)))
            score_round.append(model.score(X_wave, y_true))
        scores.append(score_round)
    scores = np.array(scores).mean(axis=0)
    scores = [round(score, 2) for score in scores]
    print(scores)
    plot_line_accu(scores, '精确度', title)


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


highK = chem[chem['类型'] == '高钾']
highK_X = highK.iloc[:, list(range(1, 15))]
highK_X = StandardScaler().fit_transform(highK_X)
kmeans = KMeans(n_clusters=6)
kmeans.fit(highK_X)
highK['亚类'] = kmeans.labels_
highK_X = highK.iloc[:, list(range(1, 15))]
highK_y = highK['亚类']

rfc = RandomForestClassifier(random_state=0)
rfc.fit(highK_X, highK_y)
highK_X = highK_X.to_numpy()
sens_analyse(rfc, highK_X, highK['亚类'], "高钾玻璃")


PbBa = chem[chem['类型'] == '铅钡']
PbBa_X = PbBa.iloc[:, list(range(1, 15))]
PbBa_X = StandardScaler().fit_transform(PbBa_X)
kmeans = KMeans(n_clusters=6)
kmeans.fit(PbBa_X)
PbBa['亚类'] = kmeans.labels_
PbBa_X = PbBa.iloc[:, list(range(1, 15))]
PbBa_y = PbBa['亚类']
rfc = RandomForestClassifier(random_state=0)
rfc.fit(PbBa_X, PbBa_y)
PbBa_X = PbBa_X.to_numpy()
sens_analyse(rfc, PbBa_X, PbBa['亚类'], "铅钡玻璃")

