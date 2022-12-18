from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_excel('./员工离职预测模型.xlsx')
#print(df.head())
df['工资'].replace({'低':0, '中':1, '高':2},inplace=True)
x = df.drop(columns='离职')
y = df['离职']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=123)

model = RandomForestClassifier(n_estimators=20, max_depth=3, min_samples_leaf=10)
model.fit(x_train,y_train)
y_pre = model.predict(x_test)
a = pd.DataFrame()
a['RandomForecest'] = list(y_pre)
a['test'] = list(y_test)
print(a)

score1 = model.score(x_test,y_test)
print(score1)

features = x.columns
importance = model.feature_importances_
b = pd.DataFrame()
b['特征'] = list(features)
b['特征重要度'] = list(importance)
print(b)

# 参数调优

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[10,20,30],'max_depth':[3,5,7],'min_samples_leaf':[10,20,30]}

new_model = RandomForestClassifier(random_state=123)

grid = GridSearchCV(new_model,parameters,cv=6,scoring='accuracy')
grid.fit(x_train,y_train)
bestParameter = grid.best_params_
print(bestParameter)
max_depth = bestParameter['max_depth']
min_samples_leaf = bestParameter['min_samples_leaf']
n_estimators = bestParameter['n_estimators']
new_model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
new_model.fit(x_train,y_train)
score2 = new_model.score(x_test,y_test)
print(score1,score2)