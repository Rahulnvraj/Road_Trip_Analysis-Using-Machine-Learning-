#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import seaborn as sns
import numpy as np 
import math
import datetime as dt
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.api import Holt


# In[5]:


from sklearn.tree import DecisionTreeClassifier


# In[6]:


from sklearn.ensemble import RandomForestClassifier


# In[7]:


route = pd.read_csv("bestroute.csv")
route.head()


# In[8]:


print("size/shape of the datasrt", route.shape)
print("checking for null values", route.isnull().sum())
print("checking Data.Type", route.dtypes)


# In[9]:


sns.countplot(x="Taken", hue="transport(1,2)", data=route)


# In[10]:


df=sns.countplot("Taken", hue="safety(1,2,3)", data=route)


# In[11]:


sns.countplot(x="Taken", hue="traffic(0,1,2)", data=route)


# In[12]:


sns.countplot(x="Taken", hue="network avable(0,1,2)", data=route)


# In[13]:


sns.countplot(x="Taken", hue="diversions(0,1,2,3,4,5)", data=route)


# In[14]:


route["distance"] = route["distance"].astype(float)

#route["distance"].plot.hist()


# In[15]:


route["safety(1,2,3)"].plot.hist()


# In[16]:


route.info(5)


# In[17]:


sns.boxplot(x="traffic(0,1,2)", y="safety(1,2,3)", data=route)


# In[18]:


route.isnull()


# In[19]:


route.isnull().sum()


# In[20]:


sns.heatmap(route.isnull(), cmap="viridis")


# In[21]:


route.head(5)


# In[22]:


sns.heatmap(route.isnull(), yticklabels=False, cbar=False)


# In[23]:


route.drop("accident", axis=1, inplace=True)


# In[24]:


route.drop("diversions(0,1,2,3,4,5)", axis=1, inplace=True)


# In[25]:


route.head(5)


# In[26]:


sns.heatmap(route.isnull(), yticklabels=False, cbar=False)


# In[27]:


route.isnull().sum()


# In[28]:


route.drop(["distance"], axis=1, inplace=True)
route.drop("via", axis=1, inplace=True)
route.drop("pickup", axis=1, inplace=True)
route.drop("drop", axis=1, inplace=True)


# In[29]:


safety=pd.get_dummies(route['safety(1,2,3)'])


# In[30]:


safety.head(5)


# In[31]:


safety=pd.get_dummies(route['safety(1,2,3)'], drop_first=True)
safety.head(5)


# In[32]:


network=pd.get_dummies(route['network avable(0,1,2)'])
network.head(5)


# In[33]:


network=pd.get_dummies(route['network avable(0,1,2)'], drop_first=True)
network.head(5)


# In[34]:


road=pd.get_dummies(route['road_type(2,3,4,6)'])
road.head(5)


# In[35]:


road=pd.get_dummies(route['road_type(2,3,4,6)'], drop_first=True)
road.head(5)


# In[36]:


traffic=pd.get_dummies(route['traffic(0,1,2)'])
traffic.head(5)


# In[37]:


traffic=pd.get_dummies(route['traffic(0,1,2)'], drop_first=True)
traffic.head(5)


# In[38]:


route=pd.concat([route, traffic, road, network, safety], axis=1)


# In[39]:


route.head(5)


# In[40]:


x=route.drop("Taken", axis=1)
y=route["Taken"]


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)


# In[43]:


from sklearn.linear_model import LogisticRegression


# In[44]:


logmodel=LogisticRegression(max_iter=1000)


# In[45]:


route.head(5)


# In[46]:


logmodel.fit(x_train, y_train)


# In[47]:


predictions=logmodel.predict(x_test)


# In[48]:


from sklearn.metrics import classification_report


# In[49]:


classification_report(y_test, predictions)


# In[50]:


from sklearn.metrics import confusion_matrix


# In[51]:


confusion_matrix(y_test, predictions)


# In[52]:


from sklearn.metrics import accuracy_score


# In[53]:


accuracy_score(y_test, predictions)


# In[54]:


decision_tree=DecisionTreeClassifier()


# In[55]:


decision_tree.fit(x_train, y_train)


# In[56]:


x_pred=decision_tree.predict(x_test)


# In[57]:


acc_decision_tree=round(decision_tree.score(x_train, y_train)*100, 2)


# In[58]:


acc_decision_tree


# In[59]:


route.pivot_table('Taken', index='transport(1,2)', columns='safety(1,2,3)').plot()


# In[60]:


route.pivot_table('Taken', index='network avable(0,1,2)', columns='safety(1,2,3)').plot()


# In[61]:


route.pivot_table('Taken', index='traffic(0,1,2)', columns='safety(1,2,3)').plot()


# In[62]:


def models(x_train, y_train):
       from sklearn.linear_model import LogisticRegression
       log = LogisticRegression(random_state=0)
       log.fit(x_train, y_train)
       
       from sklearn.tree import DecisionTreeClassifier
       tree=DecisionTreeClassifier(criterion='entropy', random_state=0)
       tree.fit(x_train, y_train)
       
       print('[0]Logistic Regression Training Accuracy:', log.score(x_train, y_train))
       print('[1]Decision Tree Classifier Training Accuracy:', tree.score(x_train, y_train))
       
       return log, tree


# In[63]:


model=models(x_train, y_train)


# In[70]:


picpup=(input("pickup location latitude:-"));
drop=(input("drop location latitude:-    "));
via=(input("via location latitude:-      "));

transport=int(input("1 for 2 wheel, 2 for 4 wheel:-"));
tot_dist=int(input("distance in km:- "));
safety=int(input("safety level measure in 1,2,3:-"));
network_avable=int(input("network availability rating (1,2,3):-"));
road_type=int(input("2-two len, 3-three len, 4-four len, 6-six len:- "));
diversions=int(input("number of diversion:- "));
traffic=int(input("traffic measures level(1,2,3):- "));
accident=int(input("no. of accidents recorded:- "));
people_travelled= int(input("no. of people travelled:-"));


# In[69]:


road_taken=[[pickup, drop, via, transport, distance, safety(1,2,3), network_avable, road_type(2,3,4,6), diversions, traffic, accident, people_travelled]]

pred=model[0].predict(road_taken)
print(pred)

if pred == 0:
    print("the road is taken")
else:
    print("the road is not taken")


# In[ ]:




