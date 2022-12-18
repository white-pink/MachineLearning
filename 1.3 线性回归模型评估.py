from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("./IT行业收入表.xlsx")

x = data[["工龄"]]
y = data["薪水"]

poly_reg = PolynomialFeatures(degree=2)
x_ = poly_reg.fit_transform(x)   # 将x转化为二维数组

regr = LinearRegression()
regr.fit(x_,y)

plt.scatter(x,y)
plt.plot(x,regr.predict(x_))
plt.show()

# print(regr.coef_,regr.intercept_)

# 方法一：直接调用sklearn中的LinearRegression().score 方法求R2
R2 = regr.score(x_, y)
print(R2)

# 方法二：引用评估线性回归模型statsmodels求R2
import statsmodels.api as sm
X2 = sm.add_constant(x_)  # 给特征变量添加常数项
est = sm.OLS(y, X2).fit()    # OLS()和fit()函数对y和x_进行线性回归方程搭建
print(est.summary())

