##============================================================================
## PREDICT 452 - Summer Quarter - Section 55
## Web and Network Data Science
##
## Project: Innovation of the Apple iPhone
## 
##   
##
## Program:cleanup_tweets.py
## Program Goal:Cleans up the raw tweets 
##
## Program Description: This program cleans up the raw tweets
##============================================================================
# -*- coding: utf-8 -*-
# Import the pandas package, then use the "read_csv" function to read csv file of the tweet
import pandas as pd   
from bs4 import BeautifulSoup  
import re  
from nltk.corpus import stopwords
import sys

train = pd.read_csv("/Users/ngaonkar/raw_tweet.csv", header=0, \
                    delimiter="\t", quoting=3)
csvFile = open('/Users/ngaonkar/test.txt', 'w')


def main():
    n=0
    for index, row in train.iterrows():
        clean_tweet = review_to_words( train["Tweet"][n] )
        n=n+1
        print clean_tweet
        #clean_tweet=(clean_tweet)
        sys.stdout = csvFile
    

def review_to_words(raw_tweet):
    # Function to convert a raw tweet to a string of words
    # The input is a single string (a raw tweet), and 
    # the output is a single string (a preprocessed tweet)
    #
    # 1. Remove HTML
    tweet_text = BeautifulSoup(raw_tweet).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", tweet_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  
    
        
if __name__ == '__main__':
     main()
