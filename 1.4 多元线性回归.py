from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('./客户价值数据表.xlsx')

# print(df.head())

x = df[['历史贷款金额', '贷款次数', '学历', '月收入','性别']]
y = df['客户价值']

regr = LinearRegression()
regr.fit(x,y)

print(regr.coef_)
print(regr.intercept_)

import statsmodels.api as sm
X2 = sm.add_constant(x)  # 给特征变量添加常数项
est = sm.OLS(y, X2).fit()    # OLS()和fit()函数对y和x_进行线性回归方程搭建
print(est.summary())

# print(x,regr.predict(x))



