
# coding: utf-8

# In[259]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# For detailed explaination: https://www.youtube.com/watch?v=h9gpufJFF-0

# In[260]:


pd.read_csv('u.data').head()


# In[261]:


df=pd.read_csv("u.data", sep="\t",names=['user_id', 'item_id', 'rating', 'timestamp'])


# In[262]:


df.head()


# In[263]:


movie_titles=pd.read_csv("Movie_Id_Titles")


# In[264]:


movie_titles.head()


# In[265]:


print ("Number of unque users {} !".format(df["user_id"].nunique()))
print ("Number of movies {} !".format(df["item_id"].nunique()))


# In[274]:


from sklearn.cross_validation import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)


# In[89]:


for line in train_data.itertuples():
    print(line)
    print ("\n")
    
#See, each line has user_id, and item_id and rating present at indexes 1,2 and 3 respectively, Index present at 0


# In[13]:


for line in train_data.itertuples():
    print(line[1])   #user_ids
    print ("\n")


# In[14]:


for line in train_data.itertuples():
    print(line[2])     #items ids
    print ("\n")


# In[275]:


#Create two user-item matrices, one for training and another for testing
n_users = df.user_id.nunique()
n_items = df.item_id.nunique()
train_data_matrix = np.zeros((n_users, n_items))
test_data_matrix=np.zeros((n_users,n_items))


# In[276]:


train_data_matrix #A matrix of (user_ids, items/movies), containing zeros


# In[277]:


test_data_matrix


# In[278]:


for line in train_data.itertuples():         #User item matrix for training, containing rating at the cordinates(user_id, item_id)
    train_data_matrix[line[1]-1, line[2]-1]=line[3]


# In[279]:



for line in test_data.itertuples():         #User item matrix for testing, containing rating at the cordinates(user_id, item_id)
    test_data_matrix[line[1]-1, line[2]-1]=line[3]


# In[280]:


train_data_matrix   # train data containing rate at cordinate(user_id, item_id), values filled up 


# In[281]:


test_data_matrix


# In[282]:


rowsum=train_data_matrix.sum(axis=1)      
normalised_matrix=train_data_matrix/rowsum[:,np.newaxis] 
normalised_matrix                                         
 #Normalising the values of rating for each users, by taking ratio of the rate by total sum of all the rates in the row 
#This ensures better user wise comparison for each movie
#This normalised matrix for estimating the user similarites in user similarity matrix
#Normalised matrix is being taken on the training data and hance, our algorithm will be trained on training set


# In[283]:


from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import  cosine_similarity
user_similarity=cosine_similarity(normalised_matrix)  
item_similarity=cosine_similarity(normalised_matrix.T)  #Taking transpose


# In[284]:


user_similarity


# In[285]:


fun=lambda x: "user {}".format(x)
user_list=list(map(fun,range(944)))
user_similarity_df=pd.DataFrame(user_similarity, index=user_list, columns=user_list)


# In[286]:


user_similarity_df.head() 
#A visualisation of similarities between users
# A user is related to himself in best possible way, hence daigonal values=1
#Example, user 0, and user 1 are related to each other as 0.14 out of 1


# In[287]:


user_similarity


# In[288]:


item_similarity


# In[289]:


train_data_matrix


# In[290]:


similarity_sum=user_similarity.sum(axis=1)
user_prediction_test=user_similarity.dot(test_data_matrix)/similarity_sum[:, np.newaxis]
user_prediction_train=user_similarity.dot(train_data_matrix)/similarity_sum[:, np.newaxis]
#taking weighted mean of the ratings of other users for a movie to predict the rating of a user for that particular movie
# Weights are nothing but the values in user similarity matrix.
# Example, if the rating for user 1 is to be found out for a movie, the weight of user 0:0.1418, weight of user 3:0.1439


# In[291]:


user_prediction_train_df=pd.DataFrame(user_prediction_train, index=user_list, columns=movie_titles.values)
user_prediction_test_df=pd.DataFrame(user_prediction_test, index=user_list, columns=movie_titles.values)


# In[292]:


user_prediction_test_df
#ratings/predictionfactors predicted for each each movie bu each user according to the ratings of testing rating dataset/matrix 


# In[293]:


user_prediction_train_df
#ratings/prediction predicted for each each movie bu each user according to the ratings of training rating dataset/matrix 


# In[294]:


pred0=pd.DataFrame(np.arange(1,1683),columns=["item_id"])
pred0["predictionfactor_test"]=user_prediction_test[0]
pred0["predictionfactor_train"]=user_prediction_train[0]
pred0=pd.merge(pred0,movie_titles,on="item_id")

#Suppose, we want to predict the ratings and hence find the recommendations of movies according to other users.
# say, we want recommendations for user with user_id=0
#The ratings predicted or prediction factors will be the basis of sorting out movies for recommendations
#We would compare the movie order for both test and train ratings and see how or similar they are to evaluate


# In[295]:


pred0


# In[296]:


recommendation_test=pred0.sort_values(by="predictionfactor_test", ascending=False)["title"]


# In[297]:


recommendation_train=pred0.sort_values(by="predictionfactor_train", ascending=False)["title"]


# In[298]:


recommendation_test.head()


# In[299]:


recommendation_train.values


# In[300]:


pred0["recommendation order according to training dataset"]=recommendation_train.values
pred0["recommendation order according to testing dataset"]=recommendation_test.values


# In[301]:


pred0


# # Using Item-Item similarity

# In[246]:


train_data_matrix=train_data_matrix.T


# In[247]:


test_data_matrix=test_data_matrix.T


# In[250]:


similarity_sum=item_similarity.sum(axis=1)
item_prediction_test=item_similarity.dot(test_data_matrix)/similarity_sum[:, np.newaxis]
item_prediction_train=item_similarity.dot(train_data_matrix)/similarity_sum[:, np.newaxis]


# In[251]:


item_prediction_train_df=pd.DataFrame(item_prediction_train, index=movie_titles.values, columns=user_list)
item_prediction_test_df=pd.DataFrame(item_prediction_test, index=movie_titles.values, columns=user_list)


# In[254]:


item_prediction_train_df
#Rating of each item based on the rating of all the other items, weighted according to item similarities


# In[256]:


pred0=pd.DataFrame(np.arange(1,1683),columns=["item_id"])
pred0["predictionfactor_test"]=item_prediction_test.T[0]
pred0["predictionfactor_train"]=item_prediction_train.T[0]
pred0=pd.merge(pred0,movie_titles,on="item_id")


# In[257]:


pred0


# In[258]:


recommendation_test=pred0.sort_values(by="predictionfactor_test", ascending=False)["title"]
recommendation_train=pred0.sort_values(by="predictionfactor_train", ascending=False)["title"]
pred0["recommendation order according to training dataset"]=recommendation_train.values
pred0["recommendation order according to testing dataset"]=recommendation_test.values
pred0

