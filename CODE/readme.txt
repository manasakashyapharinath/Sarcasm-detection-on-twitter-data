The aim of this project is to detect the sarcasm in the twitter data. Data files used here is Mature.train and all the tweets are extracted using 
twitter API. To obtain access to the twitter data, twitter API requires a user to create a twitter application and get 4 important keys, which are,
access_key, access_secret, consumer_key and consumer_secret. Please refer the below link and follow the steps to create the twitter APP.

https://iag.me/socialmedia/how-to-create-a-twitter-app-in-8-easy-steps/

Files enclosed and instructions to execute them:

preprocess.py - used to preprocess the tweets before classification.
word2vecTest.py - used to train and save word2vec vectors for the data file.
project.py - used to implement the MVME algorithm and hence helps in designing the kernel matrix.
main.py - the main file, used to execute the project. This file has all the classifiers and calculation of performance metrics
twittext.py - used to download the tweets using the tweet ID and save all the tweets into a file. The file generated is the Mature.train 
              which has the target word, sentiment label and text  of the tweet.

Here, 0 represents non-sarcastic tweets and 1 represents the sarcastic tweets.

Steps:
1) Execute the twittext.py to obtain Mature.train file.
2) Execute word2vecTest.py to obtain the feature matrix in csv file.
3) Finally, execute main.py to classify the train examples and test them against test samples.

Packages to be installed:
1) gensim for word2vec--- Using, pip install gensim
2) NLTK package for tokenizer --- using, pip install NLTK
3) To convert the feature matrix into dataframes, install pandas package--- Using, pip install pandas
4) Numpy --- Using, pip install numpy
5) sklearn --- using, pip install sklearn (This is used for using SVM, random forrest and Logistic regression classifiers)

Environment:
-->linux (Ubuntu 14.0) on VMWare Workstation 

Miscillaneous:
1) Mature.train is obtained from twittext.py
2) text.txt and modelWord2Vec_mature is obtained from word2vecTest.py 
3) Matrix.csv is created from main.py.



