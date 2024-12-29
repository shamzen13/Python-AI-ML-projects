from textblob import TextBlob
from newspaper import Article




url = 'https://en.wikipedia.org/wiki/Germany'


article = Article(url)

#article object is for getting the text
article.download()
article.parse()
article.nlp()


#the actual text
text = article.summary

print(text)

blob = TextBlob(text)

sentiment = blob.sentiment.polarity

print(sentiment)



