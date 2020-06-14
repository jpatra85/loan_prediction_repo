# -*- coding: utf-8 -*-

import re
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

snowball_stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()

class Preprocess_Text_Cleanup:
    def __int__(self):
        pass
    

    def word_cleaner(self,word):
        """
        remove noises e.g. !#$% etc. from the word
        """
        rules = [
            {r'[-()\"_#^/@%&*!;:<>{}=~|.?,]': u''}, # remove special characters
            {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
            {r'\s+': u' '},  # replace consecutive spaces
            {r'^\s+': u''}  # remove spaces at the beginning
            ]
        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                word = regex.sub(v, word)
            word = word.strip()
        return word.strip(" ")


    def to_lower(self,text):
        """
        :param text:
        :return:
            Converted text to lower case as in, converting "Hello" to "hello" or "HELLO" to "hello".
        """
        return text.lower()

    def remove_numbers(self,text):
        """
        take string input and return a clean text without numbers.
        Use regex to discard the numbers.
        """
        output = ''.join(c for c in text if not c.isdigit())
        return output

    def remove_punct(self,word):
        """
        take string input and clean string without punctuations.
        use regex to remove the punctuations.
        """
        return ''.join(c for c in word if c not in punctuation)


    def replace_space(self,word):
        """
        :param text:
        :return: list of words
        """
        if word == " " or word == "" : 
            word = np.nan
        return  word

    def preprocess_list(self,word_list):
        char_list   = []
        for word in word_list:
                clean_word  = self.word_cleaner(word)
                lower_word  = self.to_lower(clean_word)
                clean_word  = self.remove_numbers(lower_word)
                clean_word  = self.remove_punct(clean_word)
                clean_word = self.replace_space(clean_word)
                char_list.append(clean_word)
        return char_list

    def preprocess_str(self,word):
        char_list   = []
        clean_word  = self.word_cleaner(word)
        lower_word  = self.to_lower(clean_word)
        clean_word  = self.remove_numbers(lower_word)
        clean_word  = self.remove_punct(clean_word)
        clean_word  = self.replace_space(clean_word)
        char_list.append(clean_word)
        return char_list


# =============================================================================
# 
# =============================================================================

class Preprocess_Data_Cleanup:

        def __int__(self):
            pass

# =============================================================================
#  This method removes outlier based on z-score cutoff        
# =============================================================================
                    
        def detect_outlier(self, col, name = " " , pahses = range(2),cutoff = 3):
            
            col.plot(kind = 'box',title = "Actual : " + name )
            plt.show()
            col_in = col
            dict_out_liers = {}
            
            for i in pahses:
                z = (col_in - np.mean(col_in))/np.std(col_in)
                dict_out_liers[i] = set(col_in[abs(z) > cutoff])
                col_in = col_in[abs(z) <= cutoff]
                col_in.plot(kind = 'box',title = "Phase :" + str(i +1) + " " + name)
                plt.show()
            return dict_out_liers 


# =============================================================================
#  This method summarises missing values        
# =============================================================================


        def missing_zero_values_table(self,df):
            zero_val = (df == 0.00).astype(int).sum(axis=0)
            mis_val = df.isnull().sum() + (df == 'nan').sum()
            mis_val_percent = 100 * df.isnull().sum() / len(df)
            mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
            mz_table = mz_table.rename(
            columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
            mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
            mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
            mz_table['Data Type'] = df.dtypes
            mz_table = mz_table[
                mz_table.iloc[:,1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
            print ("Your selected dataframe has " + str(df.shape[1]) + 
                   " columns and " + str(df.shape[0]) + 
                   " Rows.\n" "There are " + 
                   str(mz_table.shape[0]) +
                  " columns that have missing values.")
        #         mz_table.to_excel('~ missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
            return mz_table
            

        
