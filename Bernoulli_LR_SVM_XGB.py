######### Bernoulli, SVM, Logistic Regression XGboost ##########
# utilities
import re
import numpy as np
import pandas as pd
# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk
import nltk
from nltk.stem import PorterStemmer

# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

import xgboost as xgb



# Importing the dataset
DATASET_COLUMNS=['target','ids','date','flag','user','text']
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv (r'Dataset_2.csv' , encoding=DATASET_ENCODING, names=DATASET_COLUMNS)


########### Data Preprocessing ############

#select text and target for analysis
data=df[['text','target']]

#Replacing the values to ease understanding. (Assigning 1 to Positive sentiment 4)
data['target'] = data['target'].replace(4,1)


#Separating positive and negative tweets
data_pos = data[data['target'] == 1]
data_neg = data[data['target'] == 0]


#taking one fourth data so we can run on our machine easily
data_pos = data_pos.iloc[:int(200000)]
data_neg = data_neg.iloc[:int(200000)]

#Combining positive and negative tweets
dataset = pd.concat([data_pos, data_neg])


#Making statement text in lower case
dataset['text']=dataset['text'].str.lower()


#Defining set containing all stopwords in English.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

#Cleaning and removing the above stop words list from the tweet text
STOPWORDS = set(stopwordlist)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
dataset['text'] = dataset['text'].apply(lambda text: cleaning_stopwords(text))


#Cleaning and removing punctuations
import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
dataset['text']= dataset['text'].apply(lambda x: cleaning_punctuations(x))



#Cleaning and removing repeating characters
def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_repeating_char(x))



#Cleaning and removing URL’s
def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_URLs(x))


#Cleaning and removing numbers
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
st=PorterStemmer()

#Stemming tweets
def stemming_sentence(sentence):   
    words = nltk.word_tokenize(sentence)
    res_words = []
    for word in words:
        res_words.append(st.stem(word))
    return " ".join(res_words)

dataset['text'] = dataset['text'].apply(lambda x: stemming_sentence(x)) 

#Separating input feature and label
X=dataset.text
y=dataset.target


# Separating the 95% data for training data and 5% for testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state =0)

#TF-IDF Vectorizer
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print(X_train)

#Transform the data using TF-IDF Vectorizer
X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

#Model Bernoulli Naive Bayes
BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
y_pred1 = BNBmodel.predict(X_test)


print(classification_report(y_test, y_pred1))
#Model SVM(Support Vector Machine)
SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
y_pred2 = SVCmodel.predict(X_test)


print(classification_report(y_test, y_pred2))

# Model Logistic Regression
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
y_pred3 = LRmodel.predict(X_test)


print(classification_report(y_test, y_pred3))

#Model XGBoost
xg = xgb.XGBClassifier(use_label_encoder=False, eval_metric='error')
xg.fit(X_train, y_train)
y_pred4 = xg.predict(X_test)

print(classification_report(y_test, y_pred4))

########## Classify real tweets fetching them with twitter API ###########

#Setting twitter API 
import tweepy

consumer_key = "z4eXbyALxvkOODCNafLiywMmr"
consumer_secret_key = "Z1DmX40VkH6HsXfOlW2yObUQeBeoxmW1wIlS9MVwkUwpa3N5dS"

access_token = '1372256857050517509-USUehtp3cqFAFUEjLdEm9nuQ4No8SX'
access_token_secret = 'o0cKXbjgaF49adIZATEqUNnWYvGwsVW1OCpyi59LsPBxk'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Fetching tweets ordered by a topic

keyword = input('Enter a field: ')
count = 500
#TWEETS
tweets = tweepy.Cursor(api.search_tweets,q=keyword,count = 100,tweet_mode = 'extended',lang='en').items(count)
#tweets = tweepy.Cursor(api.user_timeline,screen_name='JoeBiden',count = 100,tweet_mode = 'extended').items(count)


#DATAFRAME OF TWEETS
columns_t = ['User','Tweet']
data_t = []
for tweet in tweets:
    data_t.append([tweet.user.screen_name, tweet.full_text])
df_tweets = pd.DataFrame(data_t,columns=columns_t)


#Clean tweets the same way with our data
df_tweets['Tweet'] = df_tweets['Tweet'].str.lower()    
df_tweets['Tweet'] = df_tweets['Tweet'].apply(lambda x: cleaning_stopwords(x))
df_tweets['Tweet'] = df_tweets['Tweet'].apply(lambda x: cleaning_punctuations(x))
df_tweets['Tweet'] = df_tweets['Tweet'].apply(lambda x: cleaning_repeating_char(x))
df_tweets['Tweet'] = df_tweets['Tweet'].apply(lambda x: cleaning_URLs(x))
df_tweets['Tweet'] = df_tweets['Tweet'].apply(lambda x: cleaning_numbers(x))
df_tweets['Tweet'] = df_tweets['Tweet'].apply(lambda x: stemming_sentence(x))


#Sentiment analysis with the already trained model 
#Tweet to vector
X_tweet = df_tweets.Tweet
tweet_to_vector = vectoriser.transform(X_tweet)

#Bernoulli
tweet_predicted_BNB = BNBmodel.predict(tweet_to_vector)

#Plot the results
tweet_predicted_BNB = pd.DataFrame(tweet_predicted_BNB,columns=['category'])
plotted_1 = tweet_predicted_BNB.value_counts().plot(kind='pie', x='Sentiment', autopct='%1.1f%%',figsize=(7,7))

#SVC
tweet_predicted_SVC = SVCmodel.predict(tweet_to_vector)

#Plot the results
tweet_predicted_SVC = pd.DataFrame(tweet_predicted_SVC,columns=['category'])
plotted_2 = tweet_predicted_SVC.value_counts().plot(kind='pie',y='Sentiment', autopct='%1.1f%%',figsize=(7,7))

#Logistic Regression
tweet_predicted_LR = LRmodel.predict(tweet_to_vector)

tweet_predicted_LR = pd.DataFrame(tweet_predicted_LR,columns=['category'])
plotted_3 = tweet_predicted_LR.value_counts().plot(kind='pie',y='Sentiment', autopct='%1.1f%%',figsize=(7,7))

#XGBoost
tweet_predicted_XGB = xg.predict(tweet_to_vector)

#Plot the results
tweet_predicted_XGB = pd.DataFrame(tweet_predicted_XGB,columns=['category'])
plotted_4 = tweet_predicted_XGB.value_counts().plot(kind='pie',y='Sentiment', autopct='%1.1f%%',figsize=(7,7))

