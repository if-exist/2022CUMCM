import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pyecharts.globals import ThemeType
from sklearn.preprocessing import MinMaxScaler


def plot_heatmap(x_label, y_label, value, title):
    c = (
        HeatMap(init_opts=opts.InitOpts(theme=ThemeType.WHITE))
        .add_xaxis(x_label)
        .add_yaxis(
            "",
            y_label,
            value,
            label_opts=opts.LabelOpts(is_show=True, position="inside"),
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
                name="",
                name_location="middle",
                name_gap=25,
                name_textstyle_opts=opts.TextStyleOpts(
                    # color="#d14a61",
                    font_size=16,
                    font_weight="bold",
                )
            ),
            yaxis_opts=opts.AxisOpts(
                name="",
                name_location="middle",
                name_gap=35,
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

highK = chem[chem['类型'] == '高钾']
highK = highK.iloc[:, 1:-5]
highK = MinMaxScaler().fit_transform(highK)
highK = pd.DataFrame(highK)
highK_cor = highK.corr(method='pearson').applymap(lambda x: round(x, 2))
print(highK_cor)
highK_cor = highK_cor.to_numpy()
label = ['SiO2', 'Na2O', 'K2O', 'CaO', 'MgO', 'Al2O3', 'Fe2O3', 'CuO', 'PbO', 'BaO', 'P2O5', 'SrO', 'SnO2', 'SO2']
highK_value = [[i, j, highK_cor[i][j]] for i in range(14) for j in range(14)]
plot_heatmap(label, label, highK_value, "高钾玻璃相关系数")

PbBa = chem[chem['类型'] == '铅钡']
PbBa = PbBa.iloc[:, 1:-5]
PbBa_cor = PbBa.corr(method='pearson').applymap(lambda x: round(x, 2))
print(PbBa_cor)
PbBa_cor = PbBa_cor.to_numpy()
label = ['SiO2', 'Na2O', 'K2O', 'CaO', 'MgO', 'Al2O3', 'Fe2O3', 'CuO', 'PbO', 'BaO', 'P2O5', 'SrO', 'SnO2', 'SO2']
PbBa_value = [[i, j, PbBa_cor[i][j]] for i in range(14) for j in range(14)]
# plot_heatmap(label, label, PbBa_value, "铅钡玻璃相关系数")