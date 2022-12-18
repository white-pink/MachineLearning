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

print(regr.coef_,regr.intercept_)

