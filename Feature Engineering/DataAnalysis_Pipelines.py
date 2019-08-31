#!/usr/bin/env python
# coding: utf-8

# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)



# load dataset
data = pd.read_csv('houseprice.csv')

# rows and columns of the data
print(data.shape)

# visualise the dataset
data.head()

'''
data = pd.read_csv('houseprice.csv')
print(data.shape)
data.head()
'''


# The house price dataset contains 1460 rows, i.e., houses, and 81 columns, i.e., variables. 
# 
# **We will analyse the dataset to identify:**
# 
# 1. Missing values
# 2. Numerical variables
# 3. Distribution of the numerical variables
# 4. Outliers
# 5. Categorical variables
# 6. Cardinality of the categorical variables
# 7. Potential relationship between the variables and the target: SalePrice

# ### Missing values
# 
# Let's go ahead and find out which variables of the dataset contain missing values

# In[8]:


# make a list of the variables that contain missing values
vars_with_na = [var for var in data.columns if data[var].isnull().sum()>1]

# print the variable name and the percentage of missing values
for var in vars_with_na:
#     print(data[var].isnull())

    print('the mean of data[var] is:',data[var].isnull().mean())
    print(var, np.round(data[var].isnull().mean(), 3),  ' % missing values')

    
'''
vars_with_na = [var for var in data.columns if data[var].isnull().sum()>1]
for var in vars_with_na:
    print(var, np.round(data[var].isnull().mean(), 3), '% missing values')
'''


# Our dataset contains a few variables with missing values. We need to account for this in our following notebook / video, where we will engineer the variables for use in Machine Learning Models.

# #### Relationship between values being missing and Sale Price
# 
# Let's evaluate the price of the house for those cases where the information is missing, for each variable.

# In[4]:


def analyse_na_value(df, var):
    # make a copy to a new one, in case the original one will be changed
    df = df.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    df[var] = np.where(df[var].isnull(), 1, 0)
    
    # let's calculate the mean SalePrice where the information is missing or present
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.show()
    
for var in vars_with_na:
    analyse_na_value(data, var)
    
'''
def analyse_na_value(df, var):
    df = df.copy()
    # make a variable that indicates 1 if the observation was missing or zero otherwise
    df['var'] = np.where(df['var'].isnull(),1,0)
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.show()

for var in vars_with_na:
    analyse_na_value(data, var)
'''


# In[20]:


a = np.array([[10, 7, 4], [3, 2, 1]])
np.median(a,axis=0)
# np.mean(a,axis=0)
a[0].dtypes # 'numpy.ndarray' object has no attribute 'dtypes'


# We see that the fact that the information is missing for those variables, is important. We will capture this information when we engineer the variables in our next lecture / video.

# ### Numerical variables
# 
# Let's go ahead and find out what numerical variables we have in the dataset

# In[30]:


# list of numerical variables
num_vars = [var for var in data.columns if data[var].dtypes != 'O']

print('Number of numerical variables: ', len(num_vars))

# visualise the numerical variables
data[num_vars].head()
# data['PoolQC'].dtypes!='O'

'''
num_vars = [var for var in data.columns if data[var].dtypes!='O']

print('Number of numerical variables:', len(num_vars))

# visualize the numerical variables
data[num_vars].head()
'''


# From the above view of the dataset, we notice the variable Id, which is an indicator of the house. We will not use this variable to make our predictions, as there is one different value of the variable per each row, i.e., each house in the dataset. See below:

# In[6]:


print('Number of House Id labels: ', len(data.Id.unique()))
print('Number of Houses in the Dataset: ', len(data))


# #### Temporal variables
# 
# From the above view we also notice that we have 4 year variables. Typically, we will not use date variables as is, rather we extract information from them. For example, the difference in years between the year the house was built and the year the house was sold. We need to take this into consideration in our next video / notebook, where we will engineer our features.

# In[7]:


# list of variables that contain year information
year_vars = [var for var in num_vars if 'Yr' in var or 'Year' in var]

year_vars


# In[8]:


# let's explore the content of these year variables
for var in year_vars:
    print(var, data[var].unique())
    print()


# As you can see, it refers to years.
# 
# We can also explore the evolution of the sale price with the years in which the house was sold:

# In[9]:


data.groupby('YrSold')['SalePrice'].median().plot()
plt.ylabel('Median House Price')
plt.title('Change in House price with the years')


# There has been a drop in the value of the houses. That is unusual, in real life, house prices typically go up as years go by.
# 
# 
# Let's go ahead and explore whether there is a relationship between the year variables and SalePrice. For this, we will capture the elapsed years between the Year variables and the year in which the house was sold:

# In[10]:


# let's explore the relationship between the year variables and the house price in a bit of more details
def analyse_year_vars(df, var):
    df = df.copy()
    
    # capture difference between year variable and year the house was sold
    df[var] = df['YrSold'] - df[var]
    
    plt.scatter(df[var], df['SalePrice'])
    plt.ylabel('SalePrice')
    plt.xlabel(var)
    plt.show()
    
for var in year_vars:
    if var !='YrSold':
        analyse_year_vars(data, var)
    


# We see that there is a tendency to a decrease in price, with older features.

# #### Discrete variables
# 
# Let's go ahead and find which variables are discrete, i.e., show a finite number of values

# In[11]:


#  list of discrete variables
discrete_vars = [var for var in num_vars if len(data[var].unique())<20 and var not in year_vars+['Id']]

print('Number of discrete variables: ', len(discrete_vars))


# In[12]:


# let's visualise the discrete variables
data[discrete_vars].head()


# We can see that these variables tend to be Qualifications or grading scales, or refer to the number of rooms, or units. Let's go ahead and analyse their contribution to the house price.

# In[13]:


def analyse_discrete(df, var):
    df = df.copy()
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.ylabel('SalePrice')
    plt.show()
    
for var in discrete_vars:
    analyse_discrete(data, var)


# We see that there is a relationship between the variable numbers and the SalePrice, but this relationship is not always monotonic. 
# 
# For example, for OverallQual, there is a monotonic relationship: the higher the quality, the higher the SalePrice.  
# 
# However, for OverallCond, the relationship is not monotonic. Clearly, some Condition grades, like 5, favour better selling prices, but higher values do not necessarily do so. We need to be careful on how we engineer these variables to extract the most for a linear model. 
# 
# There are ways to re-arrange the order of the discrete values of a variable, to create a monotonic relationship between the variable and the target. However, for the purpose of this course, we will not do that, to keep feature engineering simple. If you want to learn more about how to engineer features, visit our course [Feature Engineering for Machine Learning](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=UDEMY2018) in Udemy.com

# #### Continuous variables
# 
# Let's go ahead and find the distribution of the continuous variables. We will consider continuous all those that are not temporal or discrete variables in our dataset.

# In[14]:


# list of continuous variables
cont_vars = [var for var in num_vars if var not in discrete_vars+year_vars+['Id']]

print('Number of continuous variables: ', len(cont_vars))


# In[15]:


# let's visualise the continuous variables
data[cont_vars].head()


# In[16]:


# Let's go ahead and analyse the distributions of these variables
def analyse_continous(df, var):
    df = df.copy()
    df[var].hist(bins=20)
    plt.ylabel('Number of houses')
    plt.xlabel(var)
    plt.title(var)
    plt.show()
    
for var in cont_vars:
    analyse_continous(data, var)


# We see that all of the above variables, are not normally distributed, including the target variable 'SalePrice'. For linear models to perform best, we need to account for non-Gaussian distributions. We will transform our variables in the next lecture / video, during our feature engineering section.
# 
# Let's also evaluate here if a log transformation renders the variables more Gaussian looking:

# In[17]:


# Let's go ahead and analyse the distributions of these variables
def analyse_transformed_continous(df, var):
    df = df.copy()
    
    # log does not take negative values, so let's be careful and skip those variables
    if 0 in data[var].unique():
        pass
    else:
        # log transform the variable
        df[var] = np.log(df[var])
        df[var].hist(bins=20)
        plt.ylabel('Number of houses')
        plt.xlabel(var)
        plt.title(var)
        plt.show()
    
for var in cont_vars:
    analyse_transformed_continous(data, var)


# We get a better spread of values for most variables when we use the logarithmic transformation. This engineering step will most likely add performance value to our final model.

# In[18]:


# let's explore the relationship between the house price and the transformed variables
# with more detail
def transform_analyse_continous(df, var):
    df = df.copy()
    
    # log does not take negative values, so let's be careful and skip those variables
    if 0 in data[var].unique():
        pass
    else:
        # log transform
        df[var] = np.log(df[var])
        df['SalePrice'] = np.log(df['SalePrice'])
        plt.scatter(df[var], df['SalePrice'])
        plt.ylabel('SalePrice')
        plt.xlabel(var)
        plt.show()
    
for var in cont_vars:
    if var !='SalePrice':
        transform_analyse_continous(data, var)


# From the previous plots, we observe some monotonic associations between SalePrice and the variables to which we applied the log transformation, for example 'GrLivArea'.

# #### Outliers

# In[19]:


# let's make boxplots to visualise outliers in the continuous variables 

def find_outliers(df, var):
    df = df.copy()
    
    # log does not take negative values, so let's be careful and skip those variables
    if 0 in data[var].unique():
        pass
    else:
        df[var] = np.log(df[var])
        df.boxplot(column=var)
        plt.title(var)
        plt.ylabel(var)
        plt.show()
    
for var in cont_vars:
    find_outliers(data, var)


# The majority of the continuous variables seem to contain outliers. Outliers tend to affect the performance of linear model. So it is worth spending some time understanding if removing outliers will add performance value to our  final machine learning model.
# 
# The purpose of this course is however to teach you how to put your models in production. Therefore, we will not spend more time looking at how best to remove outliers, and we will rather deploy a simpler model.
# 
# However, if you want to learn more about the value of removing outliers, visit our course [Feature Engineering for Machine Learning](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=UDEMY2018).
# 
# The same is true for variable transformation. There are multiple ways to improve the spread of the variable over a wider range of values. You can learn more about it in our course [Feature Engineering for Machine Learning](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=UDEMY2018).

# ### Categorical variables
# 
# Let's go ahead and analyse the categorical variables present in the dataset.

# In[20]:


### Categorical variables

cat_vars = [var for var in data.columns if data[var].dtypes=='O']

print('Number of categorical variables: ', len(cat_vars))


# In[21]:


# let's visualise the values of the categorical variables
data[cat_vars].head()


# #### Number of labels: cardinality
# 
# Let's evaluate how many different categories are present in each of the variables.

# In[22]:


for var in cat_vars:
    print(var, len(data[var].unique()), ' categories')


# All the categorical variables show low cardinality, this means that they have only few different labels. That is good as we won't need to tackle cardinality during our feature engineering lecture.
# 
# #### Rare labels:
# 
# Let's go ahead and investigate now if there are labels that are present only in a small number of houses:

# In[23]:


def analyse_rare_labels(df, var, rare_perc):
    df = df.copy()
    tmp = df.groupby(var)['SalePrice'].count() / len(df)
    return tmp[tmp<rare_perc]

for var in cat_vars:
    print(analyse_rare_labels(data, var, 0.01))
    print()


# Some of the categorical variables show multiple labels that are present in less than 1% of the houses. We will engineer these variables in our next video. Labels that are under-represented in the dataset tend to cause over-fitting of machine learning models. That is why we want to remove them.
# 
# Finally, we want to explore the relationship between the categories of the different variables and the house price:

# In[24]:


for var in cat_vars:
    analyse_discrete(data, var)


# Clearly, the categories give information on the SalePrice. In the next video, we will transform these strings / labels into numbers, so that we capture this information and transform it into a monotonic relationship between the category and the house price.

# **Disclaimer:**
# 
# This is by no means an exhaustive data exploration. There is certainly more to be done to understand the nature of this data and the relationship of these variables with the target, SalePrice.
# 
# However, we hope that through this notebook we gave you both a flavour of what data analysis should look like, and set the bases for the coming steps in the machine learning model building pipeline.

# That is all for this lecture / notebook. I hope you enjoyed it, and see you in the next one!
