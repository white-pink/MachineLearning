from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('./股票客户流失.xlsx')
# print(df.head())

x = df.drop(columns='是否流失')
y = df['是否流失']

x_test, x_train, y_test, y_train = train_test_split(x,y,test_size=0.2,random_state=1)

logis = LogisticRegression()
logis.fit(x_test,y_test)
y_pre = logis.predict(x_train)
y_pre_proba = logis.predict_proba(x_test)

fpr, tpr, threshold = roc_curve(y_test,y_pre_proba[:,1])
df_roc = pd.DataFrame()
df_roc['fpr'] = list(fpr)
df_roc['tpr'] = list(tpr)
df_roc['threshold'] = list(threshold)

# KS值就是 max(tpr-fpr)
print(df_roc)

plt.plot(threshold[1:], fpr[1:])
plt.plot(threshold[1:], tpr[1:])
plt.plot(threshold[1:], tpr[1:]-fpr[1:])
plt.xlabel = 'threshold'
plt.legend(['fpr','tpr','tpr-fpr'])
plt.show()
