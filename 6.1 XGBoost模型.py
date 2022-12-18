import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_excel('./信用卡交易数据.xlsx')
# print(df)

x = df.drop(columns='欺诈标签')
y = df['欺诈标签']

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=123)

model = XGBClassifier(n_estimators=100, learning_rate=0.05)
model.fit(x_train,y_train)
y_pre = model.predict(x_test)
score = model.score(x_test,y_test)
print(score)

