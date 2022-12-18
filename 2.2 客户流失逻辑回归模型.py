from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_excel('./股票客户流失.xlsx')
# print(df.head(5))

x = df.drop(['是否流失'],axis=1)
y = df['是否流失']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)  # 数据分成训练集和测试集

logis = LogisticRegression()
logis.fit(x_train,y_train)

#-------模型评估------
#方法一：调用metrics中accuracy_score方法
from sklearn.metrics import accuracy_score
y_pred = logis.predict(x_test)
score1 = accuracy_score(y_test,y_pred)

#方法二：直接用score方法,输入的是x_test和y_test, 因为logis已经拟合了，score方法会根据输入的x_test计算y_pred与y_test对比
score2 = logis.score(x_test,y_test)

print(score1,score2)

#------预测概率-------
y_pred_proba = logis.predict_proba(x_test)
a = pd.DataFrame(y_pred_proba, columns=['不流失概率','流失概率'])
print(a)