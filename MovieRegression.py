import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np


df=pd.read_csv("IMDB_Movies.csv")
print(df.dtypes)
le = LabelEncoder()
fig, axes = plt.subplots(nrows=8, ncols=4, figsize=(15,20))
fig.subplots_adjust(hspace =.2, wspace=.5)
axes = axes.ravel()


df['genres'] = le.fit_transform(df['genres'])
df['content_rating']=le.fit_transform(df['content_rating'])
df=df.dropna()
df=df.drop(['color'],axis="columns")
df=df.select_dtypes(exclude='object')

def IQR_outliers(column):
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR = Q3-Q1
    df_outliers=df[((df[column]<(Q1-1.5*IQR)) | (df[column]>(Q3+1.5*IQR)))]
    return df_outliers

#delete outliers
def replace_outliers(df):
    for column in df:
        df_outliers = IQR_outliers(column)
        for i in range((df_outliers.shape[0])):
            label_index = df_outliers.index[i]
            df.loc[label_index, column] = df[column].median()
    return df

df=replace_outliers(df)

x=df.drop(['imdb_score'],axis="columns")
y=df['imdb_score']


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train, y_train)
scores = np.average(cross_val_score(model, x, y, cv=5, scoring="neg_mean_squared_error"))
print(abs(scores))



print(pd.DataFrame(zip(x.columns, model.coef_)))









