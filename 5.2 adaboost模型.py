from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_excel('./信用卡精准营销模型.xlsx')
# print(df.head())

x = df.drop(columns='响应')
y = df['响应']

x_test, x_train, y_test, y_train = train_test_split(x,y,test_size=0.2)

model = AdaBoostClassifier(random_state=123)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
a = pd.DataFrame()
a['预测'] = list(y_pred)
a['实际'] = list(y_test)
score = model.score(x_test,y_test)

# print(a)
print(score)

# 重要程度


importance = model.feature_importances_
b = pd.DataFrame()
items = x.columns
b['特证名称'] = list(items)
b['特征重要度'] = list(importance)
b.sort_values(by='特征重要度', inplace=True,ascending=False)
print(b)