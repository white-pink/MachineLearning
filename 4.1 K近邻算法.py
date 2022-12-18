from sklearn.neighbors import KNeighborsClassifier as KNN
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel('./手写字体识别.xlsx')
# print(df.head())
x = df.drop(columns='对应数字')
y = df['对应数字']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=123)

# 网格搜索寻优
from sklearn.model_selection import GridSearchCV
parameter = {'n_neighbors':[1,3,5,7,9,11]}
model = KNN()
gridSearch = GridSearchCV(model,parameter,cv=5)
gridSearch.fit(x_train,y_train)
best_para = gridSearch.best_params_
print(best_para)

model = KNN(n_neighbors=best_para['n_neighbors'])
model.fit(x_train,y_train)
y_pre = model.predict(x_test)

a = pd.DataFrame()
a['pre'] = list(y_pre)
a['test'] = list(y_test)
# print(a.head())

score = model.score(x_test,y_test) # 计算模型准确度
print(score)

