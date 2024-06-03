"""对纹饰、类别和颜色属性分别与表面风化属性进行卡方检验"""
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

# 纹饰属性与有无风化属性的卡方检验
data = np.array([[11, 11], [6, 0], [17, 13]])
df = pd.DataFrame(data, index=['A', 'B', 'C'], columns=['风化', '无风化'])
kt = chi2_contingency(df)
print('纹饰\n卡方值=%.4f, p值=%.4f, 自由度=%i expected_frep=%s\n'%kt)

# 类型属性与有无风化属性的卡方检验
data = np.array([[6, 12], [28, 12]])
df = pd.DataFrame(data, index=['高钾', '铅钡'], columns=['风化', '无风化'])
kt = chi2_contingency(df)
print('类型\n卡方值=%.4f, p值=%.4f, 自由度=%i expected_frep=%s\n'%kt)

# 颜色属性与有无风化属性的卡方检验
data = np.array([[1, 2], [12, 8], [4, 3], [0, 2], [2, 2], [0, 1], [9, 6], [2, 0]])
df = pd.DataFrame(data, index=['浅绿', '浅蓝', '深绿', '深蓝', '紫', '绿', '蓝绿', '黑'], columns=['风化', '无风化'])
kt = chi2_contingency(df)
print('颜色\n卡方值=%.4f, p值=%.4f, 自由度=%i expected_frep=%s\n'%kt)