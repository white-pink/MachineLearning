from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_excel("./IT行业收入表.xlsx")
print(file.head(5))

x = file[["工龄"]]   # x需要写成二维结构
y = file["薪水"]

regr = LinearRegression()
regr.fit(x,y)

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文坐标轴标签
plt.scatter(x,y)
plt.xlabel("工龄")
plt.ylabel("薪水")
plt.plot(x,regr.predict(x), color ='red')

plt.show()


# print(file)