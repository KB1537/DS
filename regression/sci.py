import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import sklearn 
from sklearn.linear_model import LinearRegression
sns.set_theme

df=pd.read_csv('data.csv')

x=df['size']
y= df['price']

x_matrix= x.values.reshape(-1,1)

reg=LinearRegression()
print(reg.fit(x_matrix,y))
print(reg.score(x_matrix,y))
print(reg.coef_)
print(reg.intercept_)  

#create summary table
#reg_summary = pd.dataframe(data= x.columns.values, columns=['Features'])
#reg_summary
#reg_summary['coefficients'] =reg.coef_
#reg_summary['p-value']=p_values.round(3)




