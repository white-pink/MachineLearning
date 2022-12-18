from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import numpy as np

linear = LinearRegression()
logis = LogisticRegression()

x = [[1,0],[5,1],[6,4],[4,2],[3,2]]
y = [0,1,1,0,0]

linear.fit(x,y)
logis.fit(x,y)

a = linear.predict([[0,1]])
b = logis.predict([[0,1]])

k1 = linear.coef_
b1 = linear.intercept_

k2 = logis.coef_
b2 = logis.intercept_

print(a,b)

#-------预测特征值的概率值计算---------

y_pred_proba = logis.predict_proba([[1,1]])
import pandas as pd
df = pd.DataFrame(y_pred_proba)
print(df)