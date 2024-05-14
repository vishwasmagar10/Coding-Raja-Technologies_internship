#!/usr/bin/env python
# coding: utf-8

# ## Content Based Recommender System

# In[1]:


import numpy as np 
import pandas as pd
import warnings


# In[2]:


movies = pd.read_csv(r"C:\Users\vishw\OneDrive\Documents\archive\tmdb_5000_movies.csv")
credits = pd.read_csv(r"C:\Users\vishw\OneDrive\Documents\archive\tmdb_5000_credits.csv")


# In[3]:


movies.head(2)


# In[4]:


movies.shape


# In[5]:


credits.head()


# In[6]:


credits.shape


# In[7]:


movies = movies.merge(credits,on='title')


# In[8]:


movies.head(2)


# In[9]:


movies.shape


# In[10]:


# Keeping important columns for recommendation
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[11]:


movies.head(2)


# In[12]:


movies.shape


# In[13]:


movies.isnull().sum()


# In[14]:


movies.dropna(inplace=True)


# In[15]:


movies.isnull().sum()


# In[16]:


movies.shape


# In[17]:


movies.duplicated().sum()


# In[18]:


# handle genres

movies.iloc[0]['genres']


# In[19]:


import ast #for converting str to list

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L


# In[20]:


movies['genres'] = movies['genres'].apply(convert)


# In[21]:


movies.head()


# In[22]:


# handle keywords
movies.iloc[0]['keywords']


# In[23]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[24]:


# handle cast
movies.iloc[0]['cast']


# In[25]:


# Here i am just keeping top 3 cast

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L


# In[26]:


movies['cast'] = movies['cast'].apply(convert_cast)
movies.head()


# In[27]:


# handle crew

movies.iloc[0]['crew']


# In[28]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[29]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[30]:


movies.head()


# In[31]:


# handle overview (converting to list)

movies.iloc[0]['overview']


# In[32]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(4)


# In[33]:


movies.iloc[0]['overview']


# In[34]:


# now removing space like that 
'Anna Kendrick'
'AnnaKendrick'

def remove_space(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[35]:



movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)


# In[36]:


movies.head()


# In[37]:


# Concatinate all
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[38]:


movies.head()


# In[39]:


movies.iloc[0]['tags']


# In[40]:


# droping those extra columns
new_df = movies[['movie_id','title','tags']]


# In[41]:


new_df.head()


# In[42]:


# Converting list to str
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df.head()


# In[43]:


new_df.iloc[0]['tags']


# In[44]:


# Converting to lower case
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[45]:


new_df.head()


# In[46]:


new_df.iloc[0]['tags']


# In[47]:


import nltk
from nltk.stem import PorterStemmer


# In[48]:


ps = PorterStemmer()


# In[49]:


def stems(text):
    T = []
    
    for i in text.split():
        T.append(ps.stem(i))
    
    return " ".join(T)


# In[50]:


new_df['tags'] = new_df['tags'].apply(stems)


# In[51]:


new_df.iloc[0]['tags']


# In[52]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[53]:


vector = cv.fit_transform(new_df['tags']).toarray()


# In[54]:


vector[0]


# In[55]:


vector.shape


# In[ ]:





# In[56]:


from sklearn.metrics.pairwise import cosine_similarity


# In[57]:


similarity = cosine_similarity(vector)


# In[58]:


similarity.shape


# In[59]:


# similarity


# In[60]:


new_df[new_df['title'] == 'The Lego Movie'].index[0]


# In[61]:


def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)


# In[62]:


recommend('Spider-Man 2')


# In[69]:
import pickle
pickle.dump(new_df,open(r'C:\task1\artificats/movie_list.pkl','wb'))
pickle.dump(similarity,open(r'C:\task1\artificats/similarity.pkl','wb'))




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




