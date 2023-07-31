#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import os
from dataprep.eda import create_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from scipy import stats


# In[2]:


data_frame = pd.read_csv('D:/Documents Backup/DIC Spring 23/Project/Phase2/flight.csv')


# In[3]:


data_frame.head()


# In[4]:


data_frame


# In[5]:


# Print the number of rows in the dataset
print("Number of rows in the dataset:", len(data_frame))
# Print the number of features in the dataset
print("Number of features in the dataset:", len(data_frame.columns))
# Print the features of the dataset
print(data_frame.columns)


# In[6]:


# data cleaning-1  (removing column : Unnamed: 0)
data_frame = data_frame.drop('Unnamed: 0', axis=1)


# In[7]:


data_frame


# In[8]:


# data cleaning-2   (removing column : flight) as it is irrelevent
data_frame = data_frame.drop('flight', axis=1)


# In[9]:


data_frame


# In[10]:


# EDA-1 (Visualizing Heatmap to check Null values in Rows)

sns.heatmap(data_frame.isnull(),cbar=False,cmap='viridis')


# In[11]:


# data cleaning-3 (Removing NA values from dataset)
data_frame = data_frame.dropna()


# In[12]:


data_frame.shape


# In[13]:


# Data Cleaning-4

# Convert departure_time column to string and remove leading/trailing white spaces
data_frame['departure_time'] = data_frame['departure_time'].astype(str).str.strip()

# Remove leading and trailing white spaces from the airline column
data_frame['airline'] = data_frame['airline'].str.strip()


# In[14]:


# Data Cleaning-5 
data_frame = data_frame[data_frame['duration']>3]


# In[15]:


data_frame.shape


# In[16]:


# EDA-2
sns.heatmap(data_frame.isnull(),cbar=False,cmap='viridis')


# In[17]:


# Displaying the Duplicate entries
print(data_frame.duplicated().any())


# In[18]:


# data cleaning-6 ( Removing The Duplicate Values from dataset )
data_frame = data_frame.drop_duplicates()


# In[19]:


#Validating 
print(data_frame.duplicated().any())


# In[20]:


# Data Cleaning-7

# Identify negative values in the DataFrame
negative_prices = data_frame[data_frame['price'] < 0].count()

# Print the count of negative prices
print(negative_prices)

# Remove rows with negative prices
data_frame = data_frame[data_frame['price'] >= 0].dropna()


# In[21]:


# Data Cleaning-8

# Convert the 'source' and 'destination' columns to categorical data type
data_frame[['source_city', 'destination_city']] = data_frame[['source_city', 'destination_city']].astype('category')


# In[22]:


data_frame


# In[23]:


data_frame['airline'].unique()


# In[24]:


# EDA-3
airline_counts = data_frame['airline'].value_counts().sort_values(ascending=True)
sns.set_style("whitegrid")
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

# Create horizontal bar chart of airline counts
airline_counts.plot(kind='barh', color=colors)
plt.title("Customer Count by Airline")
plt.xlabel("Count")
plt.ylabel("Airline")
plt.show()


# In[25]:


# EDA-4
avg_price = data_frame.groupby('airline')['price'].mean().reset_index()
avg_price = avg_price.sort_values(by='price',ascending=False)
sns.barplot(x='airline', y='price', data=avg_price)

plt.xlabel('Airline')
plt.ylabel('Average Price')
plt.show()


# In[26]:


# EDA-5
class_counts = data_frame['source_city'].value_counts()
colors = ['#FFD700', '#ed8e51','#FF0000', '#00FF00','#3399ff','#800080']
class_counts.plot(kind='pie', autopct='%1.1f%%',colors=colors)
plt.title("Fliers from different cities")
plt.ylabel('')
plt.show()


# In[27]:


# EDA-6
class_prices = data_frame.groupby('class')['price'].mean()
sns.set_style("whitegrid")
class_prices.plot(kind='bar', color=['#4C72B0', '#55A868'])
plt.title("Average Ticket Price by Airplane Class")
plt.xlabel("Class")
plt.ylabel("Price)")
plt.show()


# In[28]:


# EDA-7
plt.scatter(data_frame['duration'], data_frame['price'], s=2, color= '#ed8e51')

plt.title("Flight Duration vs Ticket Price")
plt.xlabel("Duration of Flight")
plt.ylabel("Ticket Price")
plt.show()


# In[29]:


# EDA-8

# Create box plot of number of stops vs ticket price
data_frame.boxplot(column='price', by='stops')

plt.title("")
plt.xlabel("Number of Stops")
plt.ylabel("Ticket Price")
plt.show()


# In[30]:


cat_cols = list(data_frame.select_dtypes(include=['object']).columns)
print(f"Number of categorical columns: {len(cat_cols)}")
print(f"Categorical columns:\n{cat_cols}")


# In[31]:


# Data Cleaning-9
te = ce.TargetEncoder(cols=cat_cols)
data_frame = te.fit_transform(data_frame, data_frame['price'])


# In[32]:


data_frame


# In[33]:


features_with_na = [col for col in data_frame.columns if data_frame[col].isna().sum() > 0]

missing_values_df = pd.DataFrame(data_frame[features_with_na].isnull().mean().sort_values(ascending=False), columns=["percentage"])
missing_values_df.head(10)


# In[34]:


# Data Cleaning-10

# Identify non-numeric columns
non_numeric_cols = data_frame.select_dtypes(exclude='number').columns

# Create a new DataFrame with numeric columns only
data_numeric = data_frame.drop(columns=non_numeric_cols)

# Apply MinMaxScaler to numeric columns only
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = pd.DataFrame(scaler.fit_transform(data_numeric), columns=data_numeric.columns)

# Combine scaled numeric columns with non-numeric columns
data_final = pd.concat([data_scaled, data_frame[non_numeric_cols]], axis=1)


# In[35]:


# EDA-9

plt.figure(figsize=(25,10))
cor = data_scaled.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[36]:


# EDA-10

#Correlation with target variable price
cor_target = abs(cor["price"])

relevant_features = cor_target
relevant_features 

plt.figure(figsize=(10,5))
plt.bar(x=cor_target.index, height=cor_target.values)
plt.xticks(rotation=90)
plt.title("Correlation with Price")
plt.xlabel("Features")
plt.ylabel("Correlation Coefficient")
plt.show()


# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set 'price' as the target variable
y = data_scaled['price']

# Extract the input features
X_data = data_scaled.drop(['price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)


# In[38]:


# Linear Regression


# In[39]:


linreg = LinearRegression()

# Fit the model to the training data
linreg.fit(X_train, y_train)

# Make predictions
y_pred = linreg.predict(X_test)

# Evaluate the model on the testing data
score = linreg.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Accuracy of model :", score)
print("Mean squared error:", mse)
print("R-squared:", r2)


# In[40]:


import matplotlib.pyplot as plt
import numpy as np

# Plot the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Linear Regression Model")

# Plot the line of best fit
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot(np.arange(xmin, xmax, 0.1), np.arange(xmin, xmax, 0.1), 'k--', linewidth=2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()


# In[41]:


# Decision Tree


# In[42]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=5, min_samples_split=10)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

dt_score = dt.score(X_test, y_test)
dt_mse = mean_squared_error(y_test, y_pred)
dt_r2 = r2_score(y_test, y_pred)

print("Accuracy of model :", dt_score)
print("Mean squared error:", dt_mse)
print("R-squared:", dt_r2)


# In[43]:


import matplotlib.pyplot as plt
import numpy as np

# Plot the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Decision Tree Model")

# Plot the line of best fit
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
plt.plot(np.arange(xmin, xmax, 0.1), np.arange(xmin, xmax, 0.1), 'k--', linewidth=2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()


# In[44]:


# from imblearn import under_sampling, over_sampling
# from imblearn.over_sampling import SMOTE


# In[45]:


# counter=Counter(y_train)
# print('Before',counter)
# # over sampling the train dataset using SMOTE

# smt=SMOTE()
# x_train_sm , y_train_sm= smt.fit_resample(X_train,y_train)
# counter=Counter(y_train_sm)
# print('After', counter)


# In[46]:


# Random Forest


# In[51]:


from sklearn.ensemble import RandomForestRegressor


rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rf_score = rf.score(X_test, y_test)
rf_mse = mean_squared_error(y_test, y_pred)
rf_r2 = r2_score(y_test, y_pred)

print("Accuracy of model :", rf_score)
print("Mean squared error:", rf_mse)
print("R-squared:", rf_r2)


# In[52]:


# rf = RandomForestRegressor(n_estimators=100, max_depth=2, min_samples_split=2,random_state=1)


# In[53]:


import matplotlib.pyplot as plt
import numpy as np

# Scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Random Forest Regressor")
plt.show()

# Plot feature importances
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[54]:


# pip install xgboost


# In[55]:


# XGBoost


# In[56]:


import xgboost as xgb

XGB = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = 10, seed = 42)

XGB.fit(X_train, y_train)

y_pred = XGB.predict(X_test)

XGB_score = XGB.score(X_test, y_test)
XGB_mse = mean_squared_error(y_test, y_pred)
XGB_r2 = r2_score(y_test, y_pred)

print("Accuracy of model :", XGB_score)
print("Mean squared error:", XGB_mse)
print("R-squared:", XGB_r2)


# In[57]:


import matplotlib.pyplot as plt

# Scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("XGBoost Regressor")
plt.show()

# Plot feature importances
xgb.plot_importance(XGB)
plt.show()


# In[58]:


# AdaBoost


# In[59]:


from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=42)

ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)

ada_score = ada.score(X_test, y_test)
ada_mse = mean_squared_error(y_test, y_pred)
ada_r2 = r2_score(y_test, y_pred)

print("Accuracy of model :", ada_score)
print("Mean squared error:", ada_mse)
print("R-squared:", ada_r2)


# In[60]:


import matplotlib.pyplot as plt

# Scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("AdaBoost Regressor")
plt.show()

# Plot feature importances
importances = ada.feature_importances_
std = np.std([tree.feature_importances_ for tree in ada.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[ ]:


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    ada, X_train, y_train, cv=10, scoring='neg_mean_squared_error', 
    train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', label="Training error")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation error")
plt.xlabel("Training size")
plt.ylabel("Mean squared error")
plt.title("Learning Curve")
plt.legend()
plt.show()


# In[ ]:




