"""根据风化点检测数据，预测其风化前的化学成分含量"""
import pandas as pd

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
chem = chem[chem['采样点'] != '严重风化点']
chem.index = list(range(len(chem)))
chem['类型'] = [0] * len(chem)
chem['表面风化'] = [0] * len(chem)
for i in chem.index:
    chem['类型'][i] = info[info['文物编号'] == chem['文物编号'][i]]['类型'].tolist()[0]
    chem['表面风化'][i] = info[info['文物编号'] == chem['文物编号'][i]]['表面风化'].tolist()[0]
    if '未风化点' in chem['采样点'][i]:
        chem['表面风化'][i] = '无风化'


highK = chem[chem['类型'] == '高钾']
highK_weat = highK[highK['表面风化'] == '风化']
highK_unweat = highK[highK['表面风化'] == '无风化']
highK_weat_mean = highK_weat.iloc[:, 1:-5].mean()
highK_unweat_mean = highK_unweat.iloc[:, 1:-5].mean()
intercept = highK_unweat_mean - highK_weat_mean
highK_pred = highK_weat.iloc[:, 1:-5] + intercept
highK_pred = highK_pred.applymap(lambda x: round(max(x, 0), 2))
highK_pred = pd.concat([highK_weat['文物编号'], highK_pred], axis=1)
print(highK_pred)
highK_pred.to_csv('data/高钾玻璃风化前的化学成分含量预测.csv', index=False)


PbBa = chem[chem['类型'] == '铅钡']
PbBa_weat = PbBa[PbBa['表面风化'] == '风化']
PbBa_unweat = PbBa[PbBa['表面风化'] == '无风化']
PbBa_weat_mean = PbBa_weat.iloc[:, 1:-5].mean()
PbBa_unweat_mean = PbBa_unweat.iloc[:, 1:-5].mean()
intercept = PbBa_unweat_mean - PbBa_weat_mean
PbBa_pred = PbBa_weat.iloc[:, 1:-5] + intercept
PbBa_pred = PbBa_pred.applymap(lambda x: round(max(x, 0), 2))
PbBa_pred = pd.concat([PbBa_weat['文物编号'], PbBa_pred], axis=1)
print(PbBa_pred)
PbBa_pred.to_csv('data/铅钡玻璃风化前的化学成分含量预测.csv', index=False)
