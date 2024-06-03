"""使用基于聚类的离散化方法处理连续属性，对离散化后的特征进行关联规则分析"""
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.globals import ThemeType

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

highK = chem[chem['类型'] == '高钾']
highK = highK.iloc[:, 1:-5]

highK_kbd = pd.DataFrame()
for col in highK.columns:
    kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
    highK_kbd[col] = kbd.fit_transform(highK[[col]]).ravel()
    highK_kbd[col] = kbd.inverse_transform(highK_kbd[[col]]).ravel()
highK_kbd = highK_kbd.applymap(lambda x: round(x, 2))
highK_kbd.columns = ['SiO2', 'Na2O', 'K2O', 'CaO', 'MgO', 'Al2O3', 'Fe2O3', 'CuO', 'PbO', 'BaO', 'P2O5', 'SrO', 'SnO2', 'SO2']
for col in highK_kbd.columns:
    value = highK_kbd[col].drop_duplicates()
    value = value.sort_values(ascending=True)
    value.index = list(range(len(value)))
    value = value.to_dict()
    rep_rule = dict(zip(value.values(), value.keys()))
    highK_kbd[col] = highK_kbd[col].replace(rep_rule)
    highK_kbd[col] = highK_kbd[col].apply(lambda x: col + "," + str(int(x)))

highK_kbd = highK_kbd.to_numpy()

# 独热编码
te = TransactionEncoder()
df_tf = te.fit_transform(highK_kbd)
df = pd.DataFrame(df_tf, columns=te.columns_)

# 设置支持度求频繁项集
frequent_itemsets = apriori(df, min_support=0.1, max_len=2, use_colnames=True)
# 求关联规则, 设置最小置信度为0.1
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.1)
# 按支持度降序排列
rules = rules.sort_values(by='confidence', ascending=False, axis=0)  # axis=0，by列名；axis=1，by行名
# 设置最小提升度
rules = rules.drop(rules[rules.lift < 1.0].index)

output = rules.copy(deep=True)
output.iloc[:, 2:] = output.iloc[:, 2:].applymap(lambda x: round(x, 2))
output.to_csv("data/高钾玻璃关联规则.csv", index=False)

print(rules)

rules.rename(columns={'antecedents': 'from', 'consequents': 'to', 'support': 'sup', 'confidence': 'conf'}, inplace=True)
rules = rules[['from', 'to', 'sup', 'conf', 'lift']]

highK_from = rules['from'].value_counts().index.to_list()
highK_to = rules['to'].value_counts().index.to_list()
print(highK_from)
print(highK_to)

highK_conf = [[0 for i in range(len(highK_to))] for j in range(len(highK_from))]
highK_lift = [[0 for i in range(len(highK_to))] for j in range(len(highK_from))]
for i in range(len(highK_from)):
    for j in range(len(highK_to)):
        conf = rules[(rules['from'] == highK_from[i]) & (rules['to'] == highK_to[j])]['conf'].values
        if conf.size != 0:
            highK_conf[i][j] = conf[0]
        lift = rules[(rules['from'] == highK_from[i]) & (rules['to'] == highK_to[j])]['lift'].values
        if lift.size != 0:
            highK_lift[i][j] = lift[0]

highK_conf_value = []
highK_lift_value = []
for j in range(len(highK_to)):
    for i in range(len(highK_from)):
        if highK_conf[i][j] != 0:
            highK_conf_value.append([j, i, round(highK_conf[i][j], 2)])
        if highK_lift[i][j] != 0:
            highK_lift_value.append([j, i, round(highK_lift[i][j], 2)])


def plot_heatmap(x_label, y_label, value, title):
    c = (
        HeatMap(init_opts=opts.InitOpts(theme=ThemeType.WHITE))
        .add_xaxis(x_label)
        .add_yaxis(
            "",
            y_label,
            value,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="",
                pos_left="center",
            ),
            visualmap_opts=opts.VisualMapOpts(
                max_=max(i[2] for i in value),
                min_=min(i[2] for i in value),
                pos_right="right",
                pos_top="center",
            ),
            xaxis_opts=opts.AxisOpts(
                type_='category',
                axislabel_opts=opts.LabelOpts(
                    rotate=90,
                ),
                position='top',
                name="consequents",
                name_location="end",
                name_rotate=90,
                name_gap=30,
                name_textstyle_opts=opts.TextStyleOpts(
                    font_size=16,
                    font_weight="bold",
                )
            ),
            yaxis_opts=opts.AxisOpts(
                type_='category',
                name="antecedents",
                name_location="middle",
                name_gap=70,
                name_textstyle_opts=opts.TextStyleOpts(
                    font_size=16,
                    font_weight="bold",
                )
            ),
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                        pixel_ratio=6,  # 设置保存图片的分辨率
                        background_color='white',  # 设置导出图片的背景颜色
                    )
                )
            ),
        )
        .render("output/" + title + "heatmap.html")
    )


x_label = ['SnO2,0', 'SiO2,0', 'Na2O,0', 'SO2,0', 'PbO,0', 'P2O5,1', 'SrO,0', 'BaO,0', 'Fe2O3,1', 'MgO,1', 'CuO,1', 'P2O5,0', 'K2O,0', 'MgO,0', 'MgO,2', 'CuO,0', 'Fe2O3,0', 'CaO,2', 'CaO,0', 'Al2O3,1', 'K2O,2', 'SiO2,2', 'Al2O3,0', 'SrO,2', 'K2O,1', 'CaO,1', 'BaO,1', 'P2O5,2', 'SrO,1', 'Al2O3,2', 'CuO,2', 'SO2,1', 'BaO,2', 'Na2O,2', 'PbO,2', 'SiO2,1']
y_label = ['SnO2,0', 'SiO2,0', 'Na2O,0', 'P2O5,1', 'PbO,0', 'SrO,0', 'SO2,0', 'BaO,0', 'MgO,1', 'Fe2O3,1', 'CuO,1', 'MgO,0', 'MgO,2', 'P2O5,0', 'K2O,0', 'SiO2,2', 'K2O,2', 'CaO,0', 'Al2O3,1', 'CuO,0', 'Fe2O3,0', 'CaO,2', 'Al2O3,0', 'CaO,1', 'SrO,2', 'K2O,1', 'P2O5,2', 'BaO,1', 'SrO,1', 'Al2O3,2', 'CuO,2', 'BaO,2', 'SO2,1', 'Na2O,2', 'PbO,2', 'SiO2,1']
plot_heatmap(y_label, x_label, highK_conf_value, "高钾玻璃置信度")
plot_heatmap(y_label, x_label, highK_lift_value, "高钾玻璃提升度")


"""对铅钡玻璃进行关联分析"""
PbBa = chem[chem['类型'] == '铅钡']
PbBa = PbBa.iloc[:, 1:-5]

PbBa_kbd = pd.DataFrame()
for col in PbBa.columns:
    kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
    PbBa_kbd[col] = kbd.fit_transform(PbBa[[col]]).ravel()
    PbBa_kbd[col] = kbd.inverse_transform(PbBa_kbd[[col]]).ravel()
PbBa_kbd = PbBa_kbd.applymap(lambda x: round(x, 2))
PbBa_kbd.columns = ['SiO2', 'Na2O', 'K2O', 'CaO', 'MgO', 'Al2O3', 'Fe2O3', 'CuO', 'PbO', 'BaO', 'P2O5', 'SrO', 'SnO2', 'SO2']
for col in PbBa_kbd.columns:
    value = PbBa_kbd[col].drop_duplicates()
    value = value.sort_values(ascending=True)
    value.index = list(range(len(value)))
    value = value.to_dict()
    rep_rule = dict(zip(value.values(), value.keys()))
    PbBa_kbd[col] = PbBa_kbd[col].replace(rep_rule)
    PbBa_kbd[col] = PbBa_kbd[col].apply(lambda x: col + "_" + str(int(x)))
PbBa_kbd = PbBa_kbd.to_numpy()

te = TransactionEncoder()
df_tf = te.fit_transform(PbBa_kbd)
df = pd.DataFrame(df_tf, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.1, max_len=2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.1)
rules = rules.sort_values(by='confidence', ascending=False, axis=0)  # axis=0，by列名；axis=1，by行名
rules = rules.drop(rules[rules.lift < 1.0].index)

output = rules.copy(deep=True)
output.iloc[:, 2:] = output.iloc[:, 2:].applymap(lambda x: round(x, 2))
output.to_csv("data/铅钡玻璃关联规则.csv", index=False)

rules.rename(columns={'antecedents': 'from', 'consequents': 'to', 'support': 'sup', 'confidence': 'conf'}, inplace=True)
rules = rules[['from', 'to', 'sup', 'conf', 'lift']]

PbBa_from = rules['from'].value_counts().index.to_list()
PbBa_to = rules['to'].value_counts().index.to_list()
print(PbBa_from)
print(PbBa_to)

PbBa_conf = [[0 for i in range(len(PbBa_to))] for j in range(len(PbBa_from))]
PbBa_lift = [[0 for i in range(len(PbBa_to))] for j in range(len(PbBa_from))]
for i in range(len(PbBa_from)):
    for j in range(len(PbBa_to)):
        conf = rules[(rules['from'] == PbBa_from[i]) & (rules['to'] == PbBa_to[j])]['conf'].values
        if conf.size != 0:
            PbBa_conf[i][j] = conf[0]
        lift = rules[(rules['from'] == PbBa_from[i]) & (rules['to'] == PbBa_to[j])]['lift'].values
        if lift.size != 0:
            PbBa_lift[i][j] = lift[0]

PbBa_conf_value = []
PbBa_lift_value = []
for j in range(len(PbBa_to)):
    for i in range(len(PbBa_from)):
        if PbBa_conf[i][j] != 0:
            PbBa_conf_value.append([j, i, round(PbBa_conf[i][j], 2)])
        if PbBa_lift[i][j] != 0:
            PbBa_lift_value.append([j, i, round(PbBa_lift[i][j], 2)])


x_label = ['Na2O_0', 'SO2_0', 'CuO_0', 'SnO2_0', 'K2O_0', 'PbO_1', 'SrO_0', 'Fe2O3_0', 'Fe2O3_1', 'MgO_0', 'Al2O3_1', 'BaO_0', 'MgO_1', 'P2O5_0', 'Al2O3_0', 'CaO_1', 'SiO2_2', 'SiO2_0', 'K2O_1', 'P2O5_1', 'Na2O_1', 'CaO_2', 'CaO_0', 'CuO_1', 'SrO_1', 'SiO2_1', 'PbO_0', 'PbO_2', 'BaO_2', 'BaO_1']
y_label = ['Na2O_0', 'SO2_0', 'Fe2O3_0', 'CuO_0', 'Fe2O3_1', 'PbO_1', 'SrO_0', 'K2O_0', 'SnO2_0', 'MgO_0', 'BaO_0', 'Al2O3_1', 'Al2O3_0', 'P2O5_0', 'MgO_1', 'SiO2_0', 'CaO_1', 'SiO2_2', 'P2O5_1', 'K2O_1', 'Na2O_1', 'SrO_1', 'CaO_0', 'CuO_1', 'CaO_2', 'PbO_0', 'SiO2_1', 'PbO_2', 'BaO_2', 'BaO_1']
plot_heatmap(y_label, x_label, PbBa_conf_value, "铅钡玻璃置信度")
plot_heatmap(y_label, x_label, PbBa_lift_value, "铅钡玻璃提升度")
