#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction Model

# In[1]:


# importing libraries
import pandas as pd


# In[2]:


# reading dataset
housing = pd.read_csv("data.csv")


# In[3]:


# printing top 5 rows
housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# For plotting histogram
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))


# # Train Test Splitting on dataset

# In[9]:


# dividing dataset into two parts training and testing
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:] 
    return data.iloc[train_indices], data.iloc[test_indices]


# In[10]:


# importing sklearn for train test splitting
from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[11]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[12]:


strat_test_set['CHAS'].value_counts()


# In[13]:


strat_train_set['CHAS'].value_counts()


# In[14]:


housing = strat_train_set.copy()


# # Looking for Correlations

# In[15]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[16]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))


# In[17]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


# # Try out attribute combinations

# In[18]:


housing["TAXRM"] = housing['TAX']/housing['RM']


# In[19]:


housing.head()


# In[20]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[21]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[22]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# # Missing Attribute 

# In[23]:


median = housing["RM"].median()
housing["RM"].fillna(median)
# Note that the original housing dataframe will remain unchanged


# In[24]:


housing.shape


# In[25]:


housing.describe()


# In[26]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[27]:


imputer.statistics_


# In[28]:


X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)
housing_tr.describe()


# # Creating a pipeline

# In[29]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# In[30]:


housing_num_tr = my_pipeline.fit_transform(housing)
housing_num_tr.shape


# In[31]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[32]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]


# In[33]:


prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[34]:


list(some_labels)


# # Evaluating the model

# In[35]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[36]:


print(rmse)


# # Cross Validation Technique

# In[37]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[38]:


rmse_scores


# In[39]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[40]:


print_scores(rmse_scores)


# # Saving the Model

# In[41]:


from joblib import dump, load
dump(model, 'Dragon.joblib') 


# # Testing the model on test data

# In[42]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions, list(Y_test))


# In[43]:


final_rmse


# In[44]:


prepared_data[0]


# # Using the model

# In[45]:


from joblib import dump, load
import numpy as np
model = load('Dragon.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)

