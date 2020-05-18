#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df=pd.read_csv("Amazon_Unlocked_Mobile.csv")
df.head()


# In[2]:


df.dropna(inplace=True)


# In[3]:


df=df[df['Rating']!=3]


# In[4]:


df['Positive Review'] = np.where(df['Rating']>3 , 1, 0)
df.head(10)


# In[6]:


df['Positive Review'].mean()


# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['Reviews'],
                                                   df['Positive Review'],
                                                   random_state=0)


# In[9]:


print('X_train first entry:\n\n',X_train.iloc[0])
print('\n\nX_train shape:',X_train.shape)


# In[12]:


CountVectorizer


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer().fit(X_train)


# In[14]:


vect.get_feature_names()[::2000]


# In[15]:


len(vect.get_feature_names())


# In[16]:


X_train_vectorized = vect.transform(X_train)
X_train_vectorized


# In[18]:


from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X_train_vectorized,y_train)


# In[20]:


from sklearn.metrics import roc_auc_score

predictions = model.predict(vect.transform(X_test))
print('AUC: ',roc_auc_score(y_test,predictions))


# In[31]:


features_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()

print('Smallest coefs:\n{}\n'.format(features_names[sorted_coef_index[:10]]))

print('Largest coefs:\n{}\n'.format(features_names[sorted_coef_index[:11:-1]]))


# In[32]:


#Tfidf


# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=5).fit(X_train)
len(vect.get_feature_names())


# In[43]:


X_train_vectorized = vect.transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectorized,y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC Score: ',roc_auc_score(y_test,predictions))


# In[48]:


feature_names = np.array(vect.get_feature_names())

sorted_tfidf_values = X_train_vectorized.max(0).toarray()[0].argsort()
#sorted_tfidf_values

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_values[:10]]))
print('Largest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_values[:-11:-1]]))


# In[52]:


sorted_coef_index = model.coef_[0].argsort()
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[53]:


# These reviews are treated the same by our current model
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))


# In[54]:


#n-grams


# In[55]:


# Fit the CountVectorizer to the training data specifiying a minimum 
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names())


# In[56]:


model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))


# In[57]:


feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[58]:


# These reviews are now correctly identified
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))


# In[ ]:




