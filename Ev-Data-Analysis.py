#!/usr/bin/env python
# coding: utf-8

# # Electric Vehicle Data Analysis Assignment

# # Dataset Information

# This dataset shows the Battery Electric Vehicles (BEVs) and Plug-in Hybrid Electric Vehicles (PHEVs) that are currently registered through the Washington State
# Department of Licensing (DOL).
# 

#  Column Descriptions:

#     - VIN (1-10): First 10 chars of Vehicle Identification Number
#     - County, City, State, Postal Code: Vehicle registration location details
#     - Model Year, Make, Model: Vehicle specifics
#     - Electric Vehicle Type: Type of EV (BEV)
#     - CAF Eligibility: Qualifies for clean energy incentives?
#     - Electric Range: Range in miles on full charge
#     - Base MSRP: Manufacturer's Suggested Retail Price
#     - Legislative District: District where vehicle is registered
#     - DOL Vehicle ID: Unique ID by Dept of Licensing (Washington)
#     - Vehicle Location: GPS coords of registered location
#     - Electric Utility: Energy provider for the area
#     - 2020 Census Tract: For demographic/geographic analysis
# 
# 

# # Assignment Questions

# Load the dataset

# In[444]:


get_ipython().system('pip install seaborn')


# In[445]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[446]:


df=pd.read_csv('EVD[1].csv')


# In[447]:


df


# In[448]:


df.shape #shape


# In[449]:


df.head() #head


# In[450]:


df.tail() #tail


# In[451]:


df.info() #info


# In[452]:


df.describe() #describe


# # Data Cleaning:

# 1.How many missing values exist in the dataset, and in which columns?

# In[453]:


print("Missing values per column:")
print(df.isnull().sum())


# 2.How should missing or zero values in the Base MSRP and Electric Range columns be handled?

# In[454]:


mean_msrp=df['Base MSRP'].mean()
df['Base MSRP']=df['Base MSRP'].fillna(mean_msrp)
mean_msrp=df['Electric Range'].mean()
df['Electric Range']=df['Electric Range'].fillna(mean_range)


# In[455]:


df.isnull().sum()


# 3.Are there duplicate records in the dataset?if so, how should they be manged?

# In[456]:


# Remove duplicates (by Vehicle ID)
df = df.drop_duplicates(subset=['DOL Vehicle ID'])


# 4.How can VINs be anonymized while maintaining uniqueness?

# In[457]:


df['VIN_Anon'] = df['VIN (1-10)'].astype(str).apply(lambda x: hash(x) % 10**8)
df['VIN_Anon']


# 5.How can Vehicle Location (GPS coordinates) be cleaned or converted for better readability?

# In[458]:


df[['lat','lon']] = df['Vehicle Location'].str.extract(r'POINT \(([-\d\.]+) ([-\d\.]+)\)').astype(float)

df[['lat','lon']]


# # 2.Data Exploration

# 1.What are the top 5 most common EV makes and models in the dataset?

# In[459]:


# Top 5 Makes
print(df['Make'].value_counts().head(5))


# In[460]:


# Top 5 Models
print(df['Model'].value_counts().head(5))


# 2.What is the distribution of EVs by county? Which county has the most registrations?

# In[461]:


# County distribution
print(df['County'].value_counts().head(5))


# 3.How has EV adoption changed over different model years?

# In[462]:


# Trend by Model Year
print(df['Model Year'].value_counts().sort_index().tail(10))


# 4.What is the average electric range of EVs in the dataset?

# In[463]:


# Average range
print("Average Range:", df['Electric Range'].mean())


# 5.What percentage of EVs are eligible for Clean Alternative Fuel Vehicle (CAFV) incentives?

# In[464]:


# CAFV Eligibility %
eligible = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].str.contains("Eligible", na=False).sum()
print("CAFV %:", eligible/len(df)*100)


# 6.How does the electric range vary across different makes and models?

# In[465]:


range_by_make_model=df.groupby(['Make','Model'])['Electric Range'].mean().reset_index()


# In[466]:


print(range_by_make_model.sort_values('Electric Range',ascending=False).head(5))


# 7.What is the average Base MSRP for each EV model?

# In[467]:


# Average MSRP per Model (top 5)
print(df.groupby('Model')['Base MSRP'].mean().sort_values(ascending=False).head(5))


# 8.Are there any regional trends in EV adoption (e.g., urban vs. rural areas)?

# In[468]:


regional_trends=df.groupby('State')['VIN (1-10)'].count().reset_index()


# In[469]:


regional_trends


# # Data Visualization

# In[470]:


import matplotlib.pyplot as plt


# 1.Create a bar chart showing the top 5 EV makes and models by count

# In[471]:


# Bar chart: Top 5 Makes
df['Make'].value_counts().head(5).plot(kind='bar')
plt.title("Top 5 EV Makes")
plt.show()


# 2.Use a heatmap or choropleth map to visualize EV distribution by county

# In[472]:


fig = go.Figure(data=go.Choropleth(
    locations=df['County'], 
     z = df['Electric Range'].astype(float), 
     locationmode = 'USA-states', 
     colorscale = 'Reds', 
     colorbar_title = "EV Count"
 ))
fig.update_layout(title_text = 'EV Distribution by County', geo_scope='usa')
fig.show()


# In[473]:


print(df.columns)


# 3.Create a line graph showing the trend of EV adoption by model year.

# In[474]:


# Line: EV adoption over years
df['Model Year'].value_counts().sort_index().plot(kind='line')
plt.title("EV Adoption by Model Year")
plt.show()


# 4.Generate a scatter plot comparing electric range vs. base MSRP to see pricing trends.

# In[475]:


# Scatter: Range vs MSRP
df.plot.scatter(x='Base MSRP', y='Electric Range', alpha=0.3)
plt.title("Range vs MSRP")
plt.show()


# 5.Plot a pie chart showing the proportion of CAFV-eligible vs. non-eligible EVs.

# In[476]:


# Pie: CAFV Eligibility
df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].apply(lambda x: 'Eligible' if 'Eligible' in str(x) else 'Not Eligible').value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("CAFV Eligibility")
plt.show()


# 6.Use a geospatial map to display EV registrations based on vehicle location.

# In[477]:


plt.figure(figsize=(8,8))
plt.scatter(df['lon'], df['lat'], s=8, alpha=0.5)
plt.title('EV registrations (approx locations)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# # Linear Regression Model

# 1.How can we use Linear Regression to predict the Electric Range of a vehicle?

# In[478]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer


# In[484]:


from sklearn.linear_model import LinearRegression
# Define features and target
X = df[['Model Year', 'Base MSRP']]
y = df['Electric Range']
# Create and fit the model
model = LinearRegression()
model.fit(X, y)


# In[485]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)


# 2.What independent variables (features) can be used to predict Electric Range? (e.g., Model Year, Base MSRP, Make)

# In[480]:


# Features like Model Year, Base MSRP, and Make can be used
X = df[['Model Year', 'Base MSRP', 'Make']]


# 3.How do we handle categorical variables like Make and Model in regression analysis?

# In[481]:


# Use One-Hot Encoding for categorical variables
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded_make = encoder.fit_transform(df[['Make']])


# 4.What is the R² score of the model, and what does it indicate about prediction accuracy?

# In[486]:


from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")


# 5.How does the Base MSRP influence the Electric Range according to the regression model?

# In[488]:


# Check the coefficient of Base MSRP
coefficient = model.coef_[1]  # assuming Base MSRP is the second feature
print(f"Coefficient of Base MSRP: {coefficient:.2f}")


# 6.What steps are needed to improve the accuracy of the Linear Regression model?

# In[ ]:


# Steps to improve accuracy include feature engineering, handling outliers, and checking for multicollinearity
# Feature engineering example: adding new features like Make-encoded
# Outlier handling example: using RobustScaler


# 7.Can we use this model to predict the range of new EV models based on their specifications?

# In[487]:


# Yes, we can use the model to predict Electric Range for new data
new_data = pd.DataFrame({'Model Year': [2024], 'Base MSRP': [50000]})
predicted_range = model.predict(new_data)
print(f"Predicted Electric Range: {predicted_range[0]:.2f} miles")

