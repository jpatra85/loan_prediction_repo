# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import stats

from loan_prediction_repo.src.data import preprocess_utils as pr



df_raw_in = pd.read_csv('./data/raw/trainingData.csv')

# # ===========================================================================
# Assess the data to check the quality and quantity  
# =============================================================================

#Check over all data

print(df_raw_in.head())
print(df_raw_in.shape)
print(df_raw_in.describe().T)



# Split into numeric and non numeric columns into different dataframe

df_temp_numneric = df_raw_in.select_dtypes(exclude = 'object').copy()
df_temp_cat      = df_raw_in.select_dtypes(include = 'object').copy()

print("numebr of numeric columns     : " ,df_temp_numneric.shape[1])
print("number of non numeric columns : " , df_temp_cat.shape[1])



# =============================================================================
# # describe numeric columns
# =============================================================================

df_data_summary = df_temp_numneric.describe().T
print(df_data_summary)


# anlyze the box plot to sense the spread the data and outliers

#overall
df_temp_numneric.plot(kind = 'box')

#individual 
for column in df_temp_numneric:
    plt.figure()
    df_temp_numneric.boxplot([column])

del(df_temp_numneric)


# =============================================================================
# # describe categorical columns
# =============================================================================

# analyze the categorical columns ploting levels by frequency 
print(df_temp_cat.columns)

for column in df_temp_cat:
    plt.figure()

    print("Cat olumns most frequent levels " , column)
    print(df_temp_cat[column].value_counts().head(5))
    print("-" * 30)
    
    df_temp_cat[column].value_counts().head(5).plot(kind = 'bar',title = "most frequent " + column)
    plt.show()

    print("Cat olumns least frequent levels " , column)
    print(df_temp_cat[column].value_counts().tail(5))
    print("*" * 30)
    print("*" * 30)
    
    df_temp_cat[column].value_counts().tail(5).plot(kind = 'bar',title = "least frequent " + column)
    plt.show()
    
del(df_temp_cat)    
    

# =============================================================================
# Check for missing data and outliers
# =============================================================================

# intialize clanup data class for utility fucntions
pdc = pr.Preprocess_Data_Cleanup()

# generate summary table for missing & zero values
df_missing_data_summary = pdc.missing_zero_values_table(df_raw_in)
print(df_missing_data_summary)
df_missing_data_summary.plot(kind = 'bar')
  


# =============================================================================
# Preprocessing by columns 
# =============================================================================

# 26 observations dont have loan purpose and primary business.
df_raw_in = df_raw_in.dropna(subset = ['primary_business'],
                             how = 'any', axis = 0)

# check for outliers by removing data beyond 3 sigma
pdc.detect_outlier(df_raw_in.age,name = "age", cutoff = 3)

# Age below 18 and above 100 are removed . 2s can be data entry error (20-29)
df_raw_in["age"][(df_raw_in.age < 18) | (df_raw_in.age > 100)] = np.mean(df_raw_in[(df_raw_in.age >= 18) & (df_raw_in.age < 100)]['age'])
df_raw_in["age"].describe()


# drop rows by unusual expense by students
df_raw_in.drop(df_raw_in[((df_raw_in['primary_business'] == "School") & 
                       (df_raw_in['monthly_expenses'] > 20000))].index, inplace = True)


# Students without expenditure perhaps stays in hostel or sponsored and 
# all of them opted for education loan and dont residense details
df_raw_in.monthly_expenses[((df_raw_in['primary_business'] == "School") & 
                       df_raw_in['monthly_expenses'].isnull())] = 0


# monthly expenses: missings are replaced by mean from similar professional groups
df_raw_in["avg_expense_by_profession"] = df_raw_in.groupby(['primary_business'])                    ['monthly_expenses'].transform(np.mean).round()

df_raw_in.loc[df_raw_in['monthly_expenses'].isnull()].monthly_expenses = df_raw_in[df_raw_in['monthly_expenses'].isnull()].avg_expense_by_profession

df_raw_in['monthly_expenses']  = np.where(df_raw_in['monthly_expenses'].isnull(), 
                                 df_raw_in["avg_expense_by_profession"],      
                                 df_raw_in['monthly_expenses'])
# verify nulls
df_raw_in['monthly_expenses'].isnull().sum()


# replace none as 'Not Present' for secondary business
bool_secondary_business = ((df_raw_in['secondary_business'].isnull()) | (df_raw_in['secondary_business'] == 'none'))

df_raw_in.secondary_business[bool_secondary_business] = "Not Present"


# recipients who are NOT STUDENT and dont own home have lot of missing data
df_missing = df_raw_in[df_raw_in["home_ownership"].isnull() & 
               (df_raw_in["primary_business"] != 'School')]

print("missing % ", 100*(df_missing.isnull().sum().sum()/(df_missing.shape[0]*df_missing.shape[1])))

df_raw_in.drop(df_missing.index, axis = 0, inplace = True)


# Here mostly students  with lot of missing details are dropped 
df_raw_in.drop(df_raw_in[df_raw_in.sanitary_availability.isnull()].index, axis = 0, inplace = True)


# nulls are removed by default with previous operation
df_raw_in["home_ownership"].isnull().sum()

# where water is no sanitary usually water is not available. 
df_raw_in[df_raw_in["water_availabity"].isnull()].sanitary_availability.value_counts()
df_raw_in["water_availabity"].fillna(0, inplace = True)
df_raw_in["water_availabity"][df_raw_in["water_availabity"] == -1] = 1

#There is fare share of 1 ad 0.5 which looks like a possible correct label
df_raw_in["water_availabity"].value_counts()


# House occuopants 100 and more is unrealistic and it is observed that the data
# is replicated from house area. So replaced by mean occupants 

df_raw_in.occupants_count[df_raw_in.occupants_count >= 100] = df_raw_in.occupants_count[df_raw_in.occupants_count < 100].mean()


# Type of House : drop the rows where house is present but not house type, 6 rows 
# for the rest there is no house, so house type should be zero

df_raw_in.drop(df_raw_in[df_raw_in.type_of_house.isnull() & df_raw_in.house_area > 0].index, axis = 0, inplace = True)
df_raw_in['type_of_house'].fillna(0, inplace = True)
df_raw_in.type_of_house.isnull().sum()


# Where loan amount is less than 1000 looks like a data error as the resipient had
# strong income and expense compared to loan amount
df_raw_in.drop(df_raw_in[df_raw_in.loan_amount < 1000].index, inplace = True)
df_raw_in["loan_purpose"].isnull().sum() 


# Lets check the state of missing values

df_raw_in.isnull().sum().sum()
df_missing_data_summary = pdc.missing_zero_values_table(df_raw_in)
print(df_missing_data_summary)
df_missing_data_summary.plot(kind = 'bar')

# We will drop social class & city
df_raw_in.drop(['social_class', 'city'], axis = 1 , inplace = True)



# =============================================================================
# take copy of the preprocessed data and process further 
# =============================================================================

df_temp_numneric = df_raw_in.select_dtypes(exclude = 'object').copy()

df_temp_numneric.drop("Id", axis = 1, inplace = True)
df_temp_numneric['savings'] = df_temp_numneric.annual_income / 12 - df_temp_numneric.monthly_expenses
cols = ['age', 'annual_income', 'monthly_expenses','house_area', 'loan_amount']


# Check overall correlation
df_corr = df_temp_numneric.corr()
print(df_corr)

# Scatter plots with loan_amount to see any linear trend
df = df_temp_numneric
col_choice = df.columns
for pos, axis1 in enumerate(col_choice):   # Pick a first col
    plt.scatter(df.loc[:, axis1], df.loc[:, 'loan_amount'])
    plt.title(axis1 + " vs loan amount")
    plt.show()



# Boxplots for overall data comparision (onlynumerics)
x = df_temp_numneric.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_temp_numneric_scaled = pd.DataFrame(x_scaled)
df_temp_numneric_scaled.plot(kind = 'box')
print(list(enumerate(df_temp_numneric.columns)))

     

# individual box plots
for col in cols:
    df_temp_numneric[col].plot(kind = 'box')
    plt.title(col)
    plt.show()            
   

#  Check the statistical outliers based on a cutoff

pdc.detect_outlier(df_temp_numneric.age,name = "age")
print(df_temp_numneric.age.describe())
   
pdc.detect_outlier(df_temp_numneric.annual_income,name = "annual_income", cutoff = 3)        
print(df_temp_numneric.annual_income.describe())

pdc.detect_outlier(df_temp_numneric.house_area,name = "house_area", cutoff = 3)        
print(df_temp_numneric.house_area.describe())

pdc.detect_outlier(df_temp_numneric.monthly_expenses,name = "monthly_expenses")
print(df_temp_numneric.monthly_expenses.describe())

pdc.detect_outlier(df_temp_numneric.loan_amount,name = "loan_amount")
print(df_temp_numneric.loan_amount.describe())

pdc.detect_outlier(df_temp_numneric.savings,name = "savings")
print(df_temp_numneric.savings.describe())

plt.scatter(df_temp_numneric.loan_amount, df_temp_numneric.savings)



# =============================================================================
#  Remove outlier statistically as we are not not loosing many observations
#  and do not see much relation with the reponse for those outliers
# =============================================================================

# remove observation based on 3 sigma cutoff in Xs
df_temp = df_temp_numneric.drop(['loan_amount','old_dependents' ,
                            'young_dependents', 'home_ownership' ,      
                            'occupants_count', 'loan_tenure', 
                            'loan_installments' ], axis = 1).copy()

arr_z = np.abs(stats.zscore(df_temp))
idx_z = df[(np.abs(stats.zscore(df_temp)) < 3).all(axis = 1)].index
df.loc[idx_z].plot(kind = 'box')

# scale and check the plots

x = df_temp.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_temp_numneric_scaled = pd.DataFrame(x_scaled)
df_temp_numneric_scaled.plot(kind = 'box')
print(list(enumerate(df_temp_numneric.columns)))

del(df_temp)

print("observations in preprocessed data : " , len(idx_z))
print(" % data loss during preprocessing : ", np.round(100 * (1 - len(idx_z)/40000)))

# save the preprocessed data for next step
df_raw_in.loc[idx_z].to_csv("./data/interim/df_preprocessed_out.csv")




# =============================================================================
# Similarity measure to group wrongly types levels in social class
# =============================================================================

# pr = preprocess_str()
# processed_social_classes = pr.preprocess_list(unique_social_classes)


# match_words_main = {}
# match_words_other = {}
# for social_calss_target in processed_social_classes:

#     macthed_words = {}
#     for social_class_general in processed_social_classes:
#         len_matched_char = len(set(social_calss_target).intersection(social_class_general))
    
#         if len_matched_char > 1:
#             macthed_words["".join(social_class_general.split())] = len_matched_char
    
#     macthed_words = {key:value for key,value 
#                      in sorted(macthed_words.items(), key=lambda x: x[1],reverse = True)}
#     n = 10
#     if len(macthed_words) >= n:
#         match_words_main["".join(social_calss_target.split())]  =   list(macthed_words.keys())[0:n]    
#     else:
#         n = len(macthed_words)
#         match_words_other["".join(social_calss_target.split())]  =   list(macthed_words.keys())[0:n]
     
# ["".join(i.split()) for i in processed_social_classes]


# df_social_groups = pd.read_csv("C:/Users/jpatr_000/loan_prediction_repo/data/external/social_class.csv")

# df_social_groups['social_class'] = [i[2:] for i in df_social_groups.social_class.apply(lambda x: x.strip(",").strip("'"))]


# len(set(processed_social_classes))

# df_class = pd.DataFrame(unique_social_classes_counts.index,columns=["actual_class"])
# df_class['social_class'] = ["".join(i.split()) for i in processed_social_classes]

# df_class_merged = pd.merge(df_class , df_social_groups , how = 'left', on = [ 'social_class'] )

# df_class_merged['social_class'] = df_class_merged.actual_class
# df_class_merged.drop(['actual_class'],axis = 1,inplace = True) 
# df_class_merged.head()

# df = pd.merge(df_raw_in,df_class_merged,how = 'left', on = 'social_class') 
# df.groupby(['social_bucket'])['annual_income'].mean()
# df.groupby(['social_bucket'])['loan_amount'].max().plot(kind='bar')
# df.groupby(['social_bucket'])['age'].mean().plot(kind='bar')
# df.groupby(['social_bucket'])['monthly_expenses'].mean().plot(kind='bar')
# df.groupby(['social_bucket'])['house_area'].mean().plot(kind='bar')


# df['amount_per_unit_tenure'] = df['loan_amount'] / df['loan_installments']

# df.groupby(['loan_installments'])['loan_amount'].max().plot(kind='bar')

# df.groupby(['occupants_count'])['loan_amount'].max().plot(kind='bar')

# df.groupby(['old_dependents'])['loan_amount'].mean().plot(kind='bar')





