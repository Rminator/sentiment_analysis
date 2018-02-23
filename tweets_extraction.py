##============================================================================
## PREDICT 452 - Summer Quarter - Section 55
## Web and Network Data Science
##
## Project: Innovation of the Apple iPhone
## Program: tweets_extraction.py
## Program Goal:Extracts the tweets from twitter API
##
## Program Description: This program extracts the tweets from the twitter API baesd on the 
## search field and the date filters and creates a csv file on the defined path
##============================================================================
#!/usr/bin/python
import tweepy
import csv #Import csv
auth = tweepy.auth.OAuthHandler('xxxxxx', 'xxxxxxx')
auth.set_access_token('xxxxxx', 'xxxxxxx')

api = tweepy.API(auth)

# Open/create a file to append data to
csvFile = open('/Users/ngaonkar/raw_tweet.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,
                           q = "iPhone",
                           since = "2017-01-01",
                           until = "2017-08-18",
                           lang = "en").items():

    # Write a row to the CSV file. I use encode UTF-8
    csvWriter.writerow([tweet.created_at,tweet.text.encode('utf-8')])
    print tweet.created_at, tweet.text
csvFile.close()
