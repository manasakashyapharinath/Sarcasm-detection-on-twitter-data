'''
@Authors: Manasa Kashyap Harinath
	  Sravanthi Avasarala
          Siddhi Khandge

Description:
1)1000 Tweets with the target word,'Mature' is extracted from the Twitter using Twitter API
2)To do this, twitter application has to be created. If we create twitter Application, consumer_key,consumer_secret,access_key and          access_secret are generated
3)This gives us access to the twitter account and can search for all the tweets using search API. For our application.
4)Of all the 100 tweets, all the tweets with #sarcasm are extracted.
5).train file is created which is named as 'Mature.train'
'''
import tweepy

consumer_key = "Or9K5ClUiaawEXH4qXEErtnTn"
consumer_secret = "EJ01MxNxjI2GVe3jB5pkrnbn9wo78e9DxzrCMRSlozZjdGWDw0"
access_key = "857019015482486785-J7vx8XXfN8AnpZEfKPgEpKk8VQeQxZ0"
access_secret = "qhuTMrMRXvYNhr2kYuQbwfZP9oBqa6BoaKL1lEAygBYzT"

OAUTH_KEYS = {'consumer_key':consumer_key, 'consumer_secret':consumer_secret,
 'access_token_key':access_key, 'access_token_secret':access_secret}
auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])
api = tweepy.API(auth)

tweetWithMature = tweepy.Cursor(api.search, q='mature').items(1000)
file=open("tweet","w")
i=0
for tweet in tweetWithMature:
   #if '#sarcasm' in tweet.text:
	i+=1
   	print tweet.text
	
	if(i%2==0):
	 file.write(tweet.text.encode("utf-8")+"#sarcasm"+"\n")
	else:
	 file.write(tweet.text.encode("utf-8")+"\n")

file.close()
file=open("tweet","r")
fileFinal=open("Mature.train","w")
fileLines=file.readlines()
#print(fileLines)

for word in fileLines:
	if 'mature'in word:
	 if '#sarcasm' in word:
	     fileFinal.write('mature'+"\t"+'1'+"\t"+word)
	     print(word)
	 else:
	     fileFinal.write('mature'+"\t"+'0'+"\t"+word)
	     print(word)
	else:
	 continue    

fileFinal.close()
file.close()
	
