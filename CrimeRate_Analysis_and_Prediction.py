#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


data_set=pd.read_csv('Crime_Data.csv')


# In[21]:


data_set.head()


# In[22]:


data_set


# In[23]:


data_set.describe()


# In[24]:


data_set.info()


# In[25]:


data_set.columns


# In[26]:


data_set.drop(columns=['ATTEMPT TO MURDER','CULPABLE HOMICIDE NOT AMOUNTING TO MURDER','CUSTODIAL RAPE','OTHER RAPE','KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS','KIDNAPPING AND ABDUCTION OF OTHERS','PREPARATION AND ASSEMBLY FOR DACOITY'  , 'BURGLARY', 'AUTO THEFT','OTHER THEFT','RIOTS','CRIMINAL BREACH OF TRUST' ,'COUNTERFIETING','ARSON','HURT/GREVIOUS HURT','ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY' ,'INSULT TO MODESTY OF WOMEN', 'CRUELTY BY HUSBAND OR HIS RELATIVES' ],inplace=True)


# In[27]:


data_set.columns


# In[28]:


data_set


# In[29]:


data_set.info()


# data_set.dropna(inplace=True)

# In[30]:


data_set.head()


# # ONE HOT ENCODING ON STATE/UT
# 

# In[50]:


from sklearn.impute import SimpleImputer
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 2:31])
X[:, 2:31] = imputer.transform(X[:, 2:31])

print(X)


# In[51]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from seaborn import load_dataset
import pandas as pd

df = X

transformer = make_column_transformer(
    (OneHotEncoder(), [0]),
    remainder='passthrough')

transformed = transformer.fit_transform(df)
X = transformed
print(X)


# # visualization and analysis

# In[41]:


#STATE VS TOTAL CRIME OVER 10 YEARS
df_sum_by_state = data_set.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().reset_index()
states = df_sum_by_state['STATE/UT']
sum = df_sum_by_state['TOTAL IPC CRIMES']
print(df_sum_by_state)
fig, ax = plt.subplots()
plt.xticks(rotation=90, ha='right')
ax.bar(states, sum)
plt.show()


# In[72]:


#TYPE OF CRIME V/S RATE OF THAT CRIME
sum_column = data_set.sum(axis=0)
sum_col = sum_column
f = np.array(sum_col[2:30])
crimes = data_set.columns.values[2:30]

fig, ax = plt.subplots()
plt.xticks(rotation=90, ha='right')
ax.bar(crimes, f)

plt.show()


# In[44]:


#PIECHART OF CRIME RATE PER STATE
df_sum_by_state = data_set.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().reset_index()
states = df_sum_by_state['STATE/UT']
sum = df_sum_by_state['TOTAL IPC CRIMES']

x = states
y = sum
colors = ['yellowgreen','red','gold','lightskyblue','violet','lightcoral','blue','pink', 'darkgreen','yellow','grey','violet','magenta','cyan']
porcent = 100.*y/y.sum()

patches, texts = plt.pie(y, colors=colors, startangle=90, radius=1.2)
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]

sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                          key=lambda x: x[2],
                                          reverse=True))

plt.legend(patches, labels, loc='lower center', bbox_to_anchor=(-0.1, 1.),
           fontsize=8)

plt.savefig('piechart.png', bbox_inches='tight')


# # K-MEANS

# In[54]:


y = np.array(y)
Z = y
U = (X[:, 37:])
A = X[:, 36:]
print(X[:, 36:])


# In[55]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42, n_init=10)
    kmeans.fit(A)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[56]:


kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42, n_init = 10)
y_kmeans = kmeans.fit_predict(A)
print(y_kmeans)


# In[57]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y_kmeans)
print(y)


# # splitting data

# In[58]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
print(X_train.shape)


# In[62]:


data_set.describe()


# In[63]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# In[64]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[65]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# # Logistic Regression

# In[67]:


# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# # Random Forest Regression

# In[68]:


# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# # RESULTS

# In[70]:


#States with higher bars are unsafe
print(y_kmeans)
data_set['category'] = y_kmeans
df_sum_by_state = data_set.groupby('STATE/UT')['category'].sum().reset_index()
states = df_sum_by_state['STATE/UT']
sum = df_sum_by_state['category']
#print(df_sum_by_state)
fig, ax = plt.subplots()
plt.xticks(rotation=90, ha='right')
ax.bar(states, sum)
# Display the resulting DataFrame
plt.show()
print(data_set.head(10))


# In[71]:


import seaborn as sns
import numpy as np
import pandas as pd

states = (data_set.iloc[:, 0].unique())
for state in states:
  print( state, " : \n")
  data = data_set[data_set['STATE/UT'] == state]
  grouped = data.groupby('YEAR').agg('TOTAL IPC CRIMES').sum()
  arr = np.array(grouped)
  year = (data_set.iloc[:, 1].unique())
  plt.figure()
  plt.plot(year, arr)
  plt.show()
  print("\n")


# In[ ]:




