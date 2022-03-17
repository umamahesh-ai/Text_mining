
import requests # requests module is the exctact the content from the url
from bs4 import BeautifulSoup as bs # bs is used for webscrapping used to scrap specific content
import re # re--> Regural Expression 
import matplotlib.pyplot as plt
# pip install wordcloud 
from wordcloud import wordcloud
oneplus_review=[]
for i in range(1,26):
    ip=[]
    
url="https://www.amazon.in/product-reviews/B092ZJVB6Z/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&filterByStar=five_star&reviewerType=all_reviews&pageNumber="+str(i)
response=requests.get(url) # requesting the url from the google to here 
soup=bs(response.content,"html.parser") # creating soup object to iterate over the extracted content
# Extracting the content under specific tags 
reviews=soup.find_all("span",attrs={"class","a-size-base review-text review-text-content"})
for i in range(len(reviews)):
    ip.append(reviews[i].text)
oneplus_review=oneplus_review+ip
# writng reviews in a text file 
with open("oneplus_bluettoth.txt","w",encoding='utf8') as output:
    output.write(str(oneplus_review))
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(oneplus_review)

# Text summerization 
import nltk
from nltk.corpus import stopwords # import the stopwords fron nltk.corpus
# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ", ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)

# words that contained in one plues earphone reviews
ip_reviews_words = ip_rev_string.split(" ")
#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ip_reviews_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(ip_reviews_words)
with open("C:/Users/HAI/oneplus_bluettoth.txt","r",encoding="utf-8") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")
ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)
# WordCloud can be performed on the string inputs.
# Corpus level word cloud

#pip install wordcloud
from wordcloud import WordCloud

WordCloud_ip = WordCloud(background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)
plt.title("Word Cloud of Oneplus earphones")
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
plt.title("Positive words_WordCloud")
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
plt.title("Negative words_WordCloud")
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




