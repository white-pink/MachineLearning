from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = [[1],[2],[4],[5]]
y = [2,4,6,8]

regr = LinearRegression()

obj = regr.fit(X=x,y=y)
a = float(obj.coef_[0])
b= float(obj.intercept_)

print(a,b)
print(y)

plt.scatter(x,y)
plt.plot(x,obj.predict(x))
plt.show()


