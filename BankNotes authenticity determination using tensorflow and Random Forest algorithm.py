
# coding: utf-8

# # Determination of authenticity of Bank Notes' image
# 
# The project's purpose is to determine if the bank note is fake(0) or orignal on the basis of image features like entropy of image, skewness, curtosis variance etc
# TensorFlow's DNNClassifier() will be used for classification model 
# 
# DataSet: [Bank Authentication Data Set](https://archive.ics.uci.edu/ml/datasets/banknote+authentication) from the UCI repository.
# 
# The data consists of 5 columns:
# 
# * variance of Wavelet Transformed image (continuous)
# * skewness of Wavelet Transformed image (continuous)
# * curtosis of Wavelet Transformed image (continuous)
# * entropy of image (continuous)
# * class (integer)
# 
# Where class indicates whether or not a Bank Note was authentic.
# 
# This sort of task is perfectly suited for Neural Networks and Deep Learning! Just follow the instructions below to get started!

# ## Getting the Data
# 
# ** Using pandas to read in the bank_note_data.csv file **

# In[1]:


import pandas as pd


# In[3]:


data = pd.read_csv('bank_note_data.csv')


# ** Check the head of the Data **

# In[61]:


data.head()


# ## Explainatory Data Analysis
# 
# We'll just do a few quick plots of the data.
# 
# ** Import seaborn and set matplolib inline for viewing **

# In[67]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Creating a Countplot of the Classes (Authentic 1 vs Fake 0) **

# In[68]:


sns.countplot(x='Class',data=data)


# ** Creating a PairPlot of the Data with Seaborn, set Hue to Class **

# In[69]:


sns.pairplot(data,hue='Class')


# ## Data Preparation 
# 
# When using Neural Network and Deep Learning based systems, it is usually a good idea to Standardize your data, this step isn't actually necessary for our particular data set, but let's run through it for practice!
# 
# ### Standard Scaling
# 
# ** 

# In[71]:


from sklearn.preprocessing import StandardScaler


# **Creating a StandardScaler() object called scaler.**

# In[72]:


scaler = StandardScaler()


# **Fit scaler to the features.**

# In[73]:


scaler.fit(data.drop('Class',axis=1))


# **Using the .transform() method to transform the features to a scaled version.**

# In[74]:


scaled_features = scaler.fit_transform(data.drop('Class',axis=1))


# **Converting the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[77]:


df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()


# ## Train Test Split
# 
# ** Creating two objects X and y which are the scaled feature values and labels respectively.**

# In[79]:


X = df_feat


# In[80]:


y = data['Class']


# ** Using the .as_matrix() method on X and Y and reset them equal to this result. We need to do this in order for TensorFlow to accept the data in Numpy array form instead of a pandas series. **

# In[81]:


X = X.as_matrix()
y = y.as_matrix()


# ** Using SciKit Learn to create training and testing sets of the data as we've done in previous lectures:**

# In[45]:


from sklearn.cross_validation import train_test_split


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# # Contrib.learn
# 
# ** Importing tensorflow.contrib.learn.python.learn as learn**

# In[82]:


import tensorflow.contrib.learn.python.learn as learn


# ** Creating an object called classifier which is a DNNClassifier from learn. Set it to have 2 classes and a [10,20,10] hidden unit layer structure:**

# In[83]:


classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2)


# ** We will fit classifier to the training data. Use steps=200 with a batch_size of 20. **
# 
# 

# In[94]:


classifier.fit(X_train, y_train, steps=200, batch_size=20)


# ## Model Evaluation
# 
# ** Using the predict method from the classifier model to create predictions from X_test **

# In[95]:


note_predictions = classifier.predict(X_test)


# ** Now create a classification report and a Confusion Matrix. Does anything stand out to you?**

# In[96]:


from sklearn.metrics import classification_report,confusion_matrix


# In[97]:


print(confusion_matrix(y_test,note_predictions))


# In[98]:


print(classification_report(y_test,note_predictions))


# ## Comparison with random forest method
# 
# ** The results produced ny tensorflow will be extremely accurate, more accurate the random tree forest method**
# 
# **Using SciKit Learn to Create a Random Forest Classifier and compare the confusion matrix and classification report to the DNN model**

# In[99]:


from sklearn.ensemble import RandomForestClassifier


# In[100]:


rfc = RandomForestClassifier(n_estimators=200)


# In[101]:


rfc.fit(X_train,y_train)


# In[102]:


rfc_preds = rfc.predict(X_test)


# In[103]:


print(classification_report(y_test,rfc_preds))


# In[104]:


print(confusion_matrix(y_test,rfc_preds))


# ** It should have also done very well, but not quite as good as the DNN model. Hopefully you have seen the power of DNN! **

# # Great Job!
