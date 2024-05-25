import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


data = pd.read_csv('real_estate_price_size.csv')

# print(data.describe())

y = data['price']
x = data['size']

plt.scatter(x, y)
yhat = 0.0017*x+0.275
fig=plt.plot(x,yhat,lw=4,c='red',label='regression line')
plt.xlabel('size', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.show()
sns.set_theme

x = sm.add_constant(x)
results = sm.OLS(y, x).fit()
print(results.summary())
