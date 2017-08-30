import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sets import Set
from bs4 import BeautifulSoup  
import re  
from nltk.corpus import stopwords
import sys
import numpy as np


train = pd.read_csv("/Users/ngaonkar/Desktop/Predict542/Gaonkar_assigment4/train_clean_tweets.csv", header=0, \
                 delimiter=",", quoting=3)

print (train.shape)
print(train.columns.values)


def main():
    #train.columns=["Sentiment"]["Tweet"]
    num_tweets = train['Tweet'].size
    clean_train_tweets = []

    for Tweet in range(0, num_tweets):
        # If the index is evenly divisible by 100, print a message
        if (Tweet+1) % 1000 == 0:
            print('Tweet {} of {}'.format(Tweet+1, num_tweets))
            # Call our function for each one, and add the result to the list of clean reviews
        clean_train_tweets.append(review_to_words(train['Tweet'][Tweet]))
        
    
    print('Creating the bag of words...')
   
    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    vectorizer = CountVectorizer(analyzer = 'word',
                            tokenizer = None,
                            preprocessor = None,
                            stop_words = None,
                            max_features = 5000)
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabaulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of strings.
    train_data_features = vectorizer.fit_transform(clean_train_tweets)

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()
    print('Bag of words completed')
    print(train_data_features.shape)
    vocab = vectorizer.get_feature_names()
    print(vocab)
    

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    for tag, count in zip(vocab, dist):
            print(count, tag)
            
            
    print('Training the random forest...')
    from sklearn.ensemble import RandomForestClassifier

        # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    
    forest = forest.fit(train_data_features, train['Sentiment'])
    print('Training Complete!')
    
    
    
    # Read the test data
    test = pd.read_csv('/Users/ngaonkar/Desktop/Predict542/Gaonkar_assigment4/test_clean_data.csv', header=0, delimiter=',', quoting=3)

    # Verifying that there are 1490 rows and 2 columns
    print(test.shape)

    # Create an empty list and append the clean tweet one by one
    num_tweets = len(test['Tweet'])
    clean_test_tweets = []

    print('Cleaning and parsing the test set tweet data...')
    for Tweet in range(0, num_tweets):
        if (Tweet + 1) % 1000 == 0:
            print('Tweet {} of {}'.format(Tweet + 1, num_tweets))
        clean_tweet = review_to_words(test['Tweet'][Tweet])
        clean_test_tweets.append(clean_tweet)

    # Get a bag of words for the test set, and convert to a numpy 
    print('Transforming to array...')
    test_data_features = vectorizer.transform(clean_test_tweets)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    print('Predicting...')
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with the "id" column and 
    # a sentiment column
    output = pd.DataFrame(data={'id': test['id'], 'sentiment': result})

    # Use pandas to write the comma-seperated output file
    print('Writing to CSV')
    output.to_csv('/Users/ngaonkar/Desktop/Predict542/Gaonkar_assigment4/Bag_of_words_model.csv', index=False, quoting=3)
    
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
    # 4 stops = set(stopwords.words("english"))    
    stops = ['js','md','mcj','the','of','to','and','in','it','luctthfibk','its',\
    'they','their','we','us','our','you','me','mine','my',\
    'for','by','with','within','about','between','from',\
    'as','for','an','what','who','how','when','where',\
    'whereas','is','are','were','this','that','if','or','oqcjdjfmrz',\
    'not','nor','ltnyq','lrjbrs','lpkupvkl','lozrzsnbf','luzopxg','momknwsshopping','at','why','your','on','yvqwcy','off',\
    'url','png','jpg','jpeg','gif','zkerphs','zechqf','hover','em','px','pdf','orazewuonp',\
    'header','footer','zz','zydd','zwobusl','zrwnnvgrut','zx','padding','before','after','ie','tm','zyqpeknhjq','zyadragab','zvwaudptxl','zurodb','zniavkfsy']     
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  

    
        
if __name__ == '__main__':
     main()
