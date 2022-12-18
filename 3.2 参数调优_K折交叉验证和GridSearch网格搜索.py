from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_excel('./员工离职预测模型.xlsx')
df['工资'].replace({'低':0, '中':1, '高':2},inplace=True)
x = df.drop(columns='离职')
y = df['离职']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=123)

# K折交叉验证
from sklearn.model_selection import cross_val_score
model = DecisionTreeClassifier()
score1 = cross_val_score(model,x,y,cv=5,scoring='roc_auc')   # cv是交叉验证的次数
# print(score1)

# GridSearch 参数调优

from sklearn.model_selection import GridSearchCV

# 定义决策树模型中待调优参数
parameter = {'max_depth': [1,3,5,7,9,11,13], 'criterion':['gini','entropy'], 'min_samples_split':[5,7,9,11,13,15]}
model2 = DecisionTreeClassifier()

grid_search = GridSearchCV(model2,parameter,scoring='roc_auc',cv=5)
grid_search.fit(x_train,y_train)

score2 = grid_search.best_score_
print(score2)






