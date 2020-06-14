# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:05:57 2020

@author: jpatr_000
"""
# =============================================================================
# Create fetures from the preprocessed data
# =============================================================================

import numpy as np 
import pandas as pd

# read the preprocessed data

df_preprocessed = pd.read_csv("./data/interim/df_preprocessed_out.csv")
df_preprocessed.drop(['Unnamed: 0', 'Id'] , axis = 1 , inplace = True)


df_loan_purpose_groups = pd.read_csv("./data/external/loan_purpose_groups.csv")

df_primary_busness_groups = pd.read_csv("./data/external/primary_business_groups.csv")

df_preprocessed = pd.merge(df_preprocessed,df_loan_purpose_groups).drop('loan_count', axis =1)

df_preprocessed = pd.merge(df_preprocessed,df_primary_busness_groups).drop('loan_count', axis =1)

df_preprocessed.columns

#df_preprocessed.select_dtypes(include ='object').columns
#['sex', 'primary_business', 'secondary_business', 'type_of_house',
#       'loan_purpose']


# =============================================================================
# Regroup the levels of the categorical variables using average loan_amount
# =============================================================================

# Primary business
df_preprocessed.age.hist()

bins   = [0, 25, 35, 45, 100]
labels = ['age' + str(i) for i in range(4)]
df_preprocessed['age_bins'] = pd.cut(df_preprocessed.age, bins,labels=labels)
df_preprocessed['age_bins'].value_counts()


# Primary business
primary_business_avg_loan = df_preprocessed.groupby('primary_business')['loan_amount'].transform(lambda x : np.log(np.mean(x)) ).round(2)
primary_business_avg_loan.hist()

bins   = [-np.inf,8.5, 9, 9.5, 10, np.inf]
labels = ['primary_business' + str(i) for i in range(5)]
df_preprocessed['primary_business_bins'] = pd.cut(primary_business_avg_loan, bins,labels=labels)
df_preprocessed['primary_business_bins'].value_counts()



#loan_purpose
loan_purpose_avg_loan = df_preprocessed.groupby('loan_purpose')['loan_amount'].transform(lambda x : np.log(np.mean(x))).round(2)
loan_purpose_avg_loan.hist()

bins = [-np.inf , 8.9, 9,  9.2 , np.inf ]
labels = ['loan_purpose' + str(i) for i in range(4)]
df_preprocessed['loan_purpose_bins'] = pd.cut(loan_purpose_avg_loan, bins, labels=labels)
df_preprocessed['loan_purpose_bins'].value_counts()



# income group
df_preprocessed.annual_income.hist()
bins = [-np.inf, 20000 , 45000 ,60000 ,  75000 , np.inf]
labels = ['income_group' + str(i) for i in range(5)]
df_preprocessed['annual_income_bins'] = pd.cut(df_preprocessed.annual_income, bins,labels=labels)
df_preprocessed['annual_income_bins'].value_counts()


#monthly expense
df_preprocessed.monthly_expenses.hist()
bins = [-np.inf ,4000 ,10000 ,20000 ,50000 , np.inf]
labels = ['monthly_expense' + str(i) for i in range(5)]
df_preprocessed['monthly_expense_bins'] = pd.cut(df_preprocessed.annual_income, bins,labels=labels)
df_preprocessed['monthly_expense_bins'].value_counts()



# old dependents 
df_preprocessed.old_dependents = np.where(df_preprocessed.old_dependents > 0 , 1 , 0)


# young dependents
df_preprocessed.young_dependents.value_counts()
df_preprocessed.young_dependents.hist()
bins = [-np.inf ,0.1 ,2.1 ,5.1 , np.inf]
labels = ['young_dependents' + str(i) for i in range(4)]
df_preprocessed['young_dependents_bins'] = pd.cut(df_preprocessed.young_dependents, bins,labels=labels)
df_preprocessed['young_dependents_bins'].value_counts()



# occumpant binning
df_preprocessed.occupants_count.value_counts()
bins = [-np.inf , 1 , 3, 7 , np.inf]
labels = ['occupants_grounps' + str(i) for i in range(4)]
df_preprocessed['occupants_count_bins'] = pd.cut((df_preprocessed.occupants_count + 1), bins,labels=labels)
df_preprocessed['occupants_count_bins'].value_counts()
df_preprocessed['occupants_count_bins'].value_counts().sum()


# house_area binning
df_preprocessed.house_area.hist()
bins = [-np.inf , 400 , 1200, 2500 , np.inf]
labels = ['house_area_group' + str(i) for i in range(4)]
df_preprocessed['house_area_bins'] = pd.cut(df_preprocessed.house_area , bins,labels=labels)
df_preprocessed['house_area_bins'].value_counts()
df_preprocessed['house_area_bins'].value_counts().sum()


# binning the installments
df_preprocessed.loan_installments.value_counts()
df_preprocessed.loan_installments[df_preprocessed.loan_installments == 0] = df_preprocessed.loan_tenure[df_preprocessed.loan_installments == 0]
bins = [-np.inf , 15 , 50, np.inf]
labels = ['loan_installment_group' + str(i) for i in range(3)]
df_preprocessed['loan_installment_bins'] = pd.cut(df_preprocessed.loan_installments , bins,labels=labels)
df_preprocessed['loan_installment_bins'].value_counts()
df_preprocessed['loan_installment_bins'].value_counts().sum()


# binning the loan tenures
df_preprocessed.loan_tenure.value_counts()
bins = [-np.inf , 6 , 15, 48 , np.inf]
labels = ['loan_tenure_group' + str(i) for i in range(4)]
df_preprocessed['loan_tenure_bins'] = pd.cut(df_preprocessed.loan_installments , bins,labels=labels)
df_preprocessed['loan_tenure_bins'].value_counts()    


# savings by individual anually
df_preprocessed['savings'] = df_preprocessed.annual_income - 12 * df_preprocessed.monthly_expenses
df_preprocessed['savings'].describe()
df_preprocessed['savings'].hist()



# savings by individual anually
df_preprocessed['savings'] = df_preprocessed.annual_income - 12 * df_preprocessed.monthly_expenses
df_preprocessed['savings'].describe()
df_preprocessed['savings'].hist()



# append column names to the levels for encoding
df_preprocessed.sanitary_availability.value_counts()
df_preprocessed.water_availabity.value_counts()
df_preprocessed.type_of_house.value_counts()

df_preprocessed.type_of_house = ["type_of_house" + str(i) for i in  df_preprocessed.type_of_house]

df_preprocessed.water_availabity = ["water_availabity" + str(i) for i in  df_preprocessed.water_availabity]

df_preprocessed.secondary_business.value_counts()

df_preprocessed.to_csv("./data/processed/df_binned_out_toaday.csv")

# =============================================================================
# Encode the levels and save feature dataframe
# =============================================================================

ex_ft_cols = ['age','primary_business', 'secondary_business', 
              'annual_income', 'monthly_expenses', 
              'old_dependents', 'young_dependents',
              'occupants_count', 'house_area',
              'loan_purpose','loan_tenure', 'loan_installments',
              'avg_expense_by_profession', 
              #'loan_amount'
              ]

df_ft_out = pd.get_dummies(df_preprocessed.drop(ex_ft_cols, axis = 1))

print("missing values in ft out :", df_ft_out.isnull().sum().sum())
print("Row Col num of ft out ", df_ft_out.shape)
print(df_ft_out.columns)


df_ft_out.to_csv("./data/processed/df_ft_out_today.csv")





# =============================================================================
# 
# =============================================================================

df = pd.read_csv("./data/processed/df_ft_out_today.csv").drop('Unnamed: 0' , axis = 1)

df.columns
