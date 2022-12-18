from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_excel('./产品定价模型.xlsx')

le = LabelEncoder()   # 文字分类进行数值化处理
df['类别'] = le.fit_transform(df['类别'])
df['纸张'] = le.fit_transform(df['纸张'])
# print(df.head())

x = df.drop(columns='价格')
y = df['价格']
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=123)

model = GradientBoostingRegressor()
model.fit(x_train,y_train)

y_pre = model.predict(x_test)

a = pd.DataFrame()
a['预测'] = list(y_pre)
a['实际'] = list(y_test)
print(a)

score = model.score(x_test,y_test)
print(score)




