import pandas as pd
import numpy as np
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
# chem = chem[chem['采样点'] != '严重风化点']
chem.index = list(range(len(chem)))
chem['类型'] = [0] * len(chem)
chem['表面风化'] = [0] * len(chem)
for i in chem.index:
    chem['类型'][i] = info[info['文物编号'] == chem['文物编号'][i]]['类型'].tolist()[0]
    chem['表面风化'][i] = info[info['文物编号'] == chem['文物编号'][i]]['表面风化'].tolist()[0]

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
# print(PbBa_kbd)
PbBa_kbd = PbBa_kbd.to_numpy()
# print(PbBa_kbd)

# 独热编码
te = TransactionEncoder()
df_tf = te.fit_transform(PbBa_kbd)
df = pd.DataFrame(df_tf, columns=te.columns_)

# 设置支持度求频繁项集
# df：数据集。
# min_support：给定的最小支持度。
# use_colnames：默认False，则返回的物品组合用编号显示，为True的话直接显示物品名称。
# max_len：最大物品组合数，默认是None，不做限制。如果只需要计算两个物品组合的话，便将这个值设置为2。
frequent_itemsets = apriori(df, min_support=0.1, max_len=2, use_colnames=True)
# 求关联规则, 设置最小置信度为0.15
# df： Apriori 计算后的频繁项集。
# metric：可选值['support','confidence','lift','leverage','conviction']。里面比较常用的就是置信度和支持度。这个参数和下面的min_threshold参数配合使用。
# min_threshold：参数类型是浮点型，根据 metric 不同可选值有不同的范围，
#     metric = 'support'  => 取值范围 [0,1]
#     metric = 'confidence'  => 取值范围 [0,1]
#     metric = 'lift'  => 取值范围 [0, inf]
# support_only：默认是 False。仅计算有支持度的项集，若缺失支持度则用 NaNs 填充。
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.1)
# 按支持度降序排列
rules = rules.sort_values(by='confidence', ascending=False, axis=0)  # axis=0，by列名；axis=1，by行名
# 设置最小提升度
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
# PbBa_conf = np.empty(shape=(len(PbBa_from), len(PbBa_to)))
# PbBa_lift = np.empty(shape=(len(PbBa_from), len(PbBa_to)))
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


def plot_heatmap(x_label, y_label, value, title):
    c = (
        HeatMap(init_opts=opts.InitOpts(theme=ThemeType.WHITE))
        .add_xaxis(x_label)
        .add_yaxis(
            "",
            y_label,
            value,
            # label_opts=opts.LabelOpts(is_show=True, position="inside"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="",
                pos_left="center",
            ),
            visualmap_opts=opts.VisualMapOpts(
                # is_show=False,
                max_=max(i[2] for i in value),
                min_=min(i[2] for i in value),
                # range_color=['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58',
                #              '#b5de2b', '#fde725'],
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
                    # color="#d14a61",
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
                    # color="#675bba",
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


x_label = ['Na2O_0', 'SO2_0', 'CuO_0', 'SnO2_0', 'K2O_0', 'PbO_1', 'SrO_0', 'Fe2O3_0', 'Fe2O3_1', 'MgO_0', 'Al2O3_1', 'BaO_0', 'MgO_1', 'P2O5_0', 'Al2O3_0', 'CaO_1', 'SiO2_2', 'SiO2_0', 'K2O_1', 'P2O5_1', 'Na2O_1', 'CaO_2', 'CaO_0', 'CuO_1', 'SrO_1', 'SiO2_1', 'PbO_0', 'PbO_2', 'BaO_2', 'BaO_1']
y_label = ['Na2O_0', 'SO2_0', 'Fe2O3_0', 'CuO_0', 'Fe2O3_1', 'PbO_1', 'SrO_0', 'K2O_0', 'SnO2_0', 'MgO_0', 'BaO_0', 'Al2O3_1', 'Al2O3_0', 'P2O5_0', 'MgO_1', 'SiO2_0', 'CaO_1', 'SiO2_2', 'P2O5_1', 'K2O_1', 'Na2O_1', 'SrO_1', 'CaO_0', 'CuO_1', 'CaO_2', 'PbO_0', 'SiO2_1', 'PbO_2', 'BaO_2', 'BaO_1']
plot_heatmap(y_label, x_label, PbBa_conf_value, "铅钡玻璃置信度")
plot_heatmap(y_label, x_label, PbBa_lift_value, "铅钡玻璃提升度")
PbBa_conf = pd.DataFrame(data=PbBa_conf, columns=PbBa_from)
PbBa_lift = pd.DataFrame(data=PbBa_lift, columns=PbBa_to)
