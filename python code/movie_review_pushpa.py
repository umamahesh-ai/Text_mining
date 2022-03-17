 # request module --> To extract the url
import requests
import re # re--> regural expression
# import BeautifulSoup from bs4 --> it is used for webscrapping 
from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt # It is used for Data Visulization
#wordcloud
#it is a data visualization technique
#It is used for the representing the text data
# In which the size of each word indicates its frequency or importance
from wordcloud import wordcloud
movie_review=[]
for i in range(1,31): # It will take the 1 to 30 pages of reviews
    ip=[]
    
    
url="https://www.imdb.com/title/tt9389998/reviews?ref_=tt_urv"
response=requests.get(url) # It requests the url fro google to here
soup=bs(response.content,"html.parser")
# From that url we can get the content of in html format and it assing to soup
# Extracting  the content under the specific tag 
reviews=soup.find_all("div",attrs={"class","text show-more__control"})
for i in range(len(reviews)):
     ip.append(reviews[i].text)
movie_review=movie_review+ip
# create a file with movie_review in the  local system 
# "w" means write operator,this method is used to write thw data to the local file 
with open("movie_review.txt","w",encoding='utf8') as output:
    output.write(str(movie_review))
# Joinining all the reviews into single paragraph in the local system  of movie_review file 
ip_rev_string = " ".join(movie_review)
# Text summerization 
import nltk
from nltk.corpus import stopwords # import the stopwords fron nltk.corpus
# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower() # it can convert is any upper(capital)words that can convert into lower(small)words
ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)

# words that contained in movie reviews
ip_reviews_words = ip_rev_string.split(" ")
#TF-IDF vectorizers
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ip_reviews_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(ip_reviews_words)
with open("C:/Users/HAI/movie_review.txt","r",encoding='UTF-8') as sw:
    stop_words = sw.read()
#"r" means read operator,this method is used to read whole data from the local file as a single string 
    
stop_words = stop_words.split("\n")
ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words] 
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)
# WordCloud can be performed on the string inputs.
# Corpus level word cloud
# if there is no install wordcloud , just install the word cloud
#pip install wordcloud 
from wordcloud import WordCloud

WordCloud_ip = WordCloud(background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)
plt.title("Generalized wordcloud_Pushpa Raj")

plt.imshow(WordCloud_ip) # this is the generalized the wordcloud 

#Now we can create the +ve and -ve wordcloud:
    
# positive words # Choose the path for +ve words stored in system
with open("C:/Users/HAI/Desktop/360DigitMG/Text mining_sentment analysis/Datasets NLP/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n") # importing the positive words 
# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.title("Positive words_WordCloud of Pushpa Raj movie")
plt.imshow(wordcloud_pos_in_pos)
# Negative  wordcloud
# negative words Choose path for -ve words stored in system
with open("C:/Users/HAI/Desktop/360DigitMG/Text mining_sentment analysis/Datasets NLP/negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.title("Negative words_WordCloud of Pushpa Raj move")
plt.imshow(wordcloud_neg_in_neg)

# we are displaying the wordcloud of one gram (Single word)only...
# Now, we can create the Bigram (Two words and it is meaningful words)
# wordcloud with bigram
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS # we are import the WordCloud and the stopwords
# lemmatizer is written the base form of words in the that can be found in the dictionary
WNL = nltk.WordNetLemmatizer() 
# Lowercase and tokenize
text = ip_rev_string.lower() #ip_rev_string--> it can convert the all words in lowercase

# Remove single quote early since it causes problems with the tokenizer.
# example don't---> it can remove the ' words and it written dont
text = text.replace("'", "")
# sentences is converted in to (words)tokens and it will display in list
tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens) # tokens is conver into text format
# Remove the stopwords and Special characters in the text and join all the words
# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS) 
customised_words = ['price', 'great'] # If you want to remove any particular word form text which does not contribute much in meaning

new_stopwords = stopwords_wc.union(customised_words) 

# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

nltk_tokens = nltk.word_tokenize(text)   # text convert into tokens
bigrams_list = list(nltk.bigrams(text_content)) # 2 words are combining
print(bigrams_list) # display the bigram words(2 words)

dictionary2 = [' '.join(tup) for tup in bigrams_list] # Bigram is convert into the tuple format
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)
wordCloud.generate_from_frequencies(words_dict)

plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()
   
    

