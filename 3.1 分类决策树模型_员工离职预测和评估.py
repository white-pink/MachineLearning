import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_excel('./员工离职预测模型.xlsx')
df['工资'].replace({'低':0,'中':1,'高':2},inplace=True)
print(df.head())

x = df.drop(columns='离职')
y = df['离职']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=123)

model = DecisionTreeClassifier(max_depth=3,random_state=123)
model.fit(x_train,y_train)

# 模型预测不离职和离职概率
y_pred = model.predict(x_test)   #模型预测正确率
y_pred_proba = model.predict_proba(x_test)  #模型分别预测不离职和离职概率
proba = pd.DataFrame(y_pred_proba,columns=['不离职概率','离职概率'])
print(proba)

# 模型预测效果评估
fpr,tpr,threshold = roc_curve(y_test,y_pred_proba[:,1])  # 其中y_pred_proba是每个样本为True的预测结果
roc_ = pd.DataFrame()
roc_['阈值'] = list(threshold[1:])
roc_['fpr'] = list(fpr[1:])
roc_['tpr'] = list(tpr[1:])
print(roc_)

plt.plot(fpr,tpr)
plt.show()

score = roc_auc_score(y_test,y_pred_proba[:,1])
print(score)

# 特征重要性评估，评估各个特征变量的重要程度

features = x.columns
importance = model.feature_importances_
df_importance = pd.DataFrame()
df_importance['特征变量'] = list(features)
df_importance['重要程度'] = list(importance)
df_importance.sort_values('重要程度',ascending=False,inplace=True)

print(df_importance)






