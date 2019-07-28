from nltk.corpus import movie_reviews 
 
# Total reviews
print (len(movie_reviews.fileids())) 
 
# Review categories
print (movie_reviews.categories()) 
 
# Total positive reviews
print (len(movie_reviews.fileids('pos')))
 
# Total negative reviews
print (len(movie_reviews.fileids('neg'))) 
 
positive_review_file = movie_reviews.fileids('pos')[0] 
print (positive_review_file) 

documents = []
 
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        
        documents.append((movie_reviews.words(fileid), category))
 
print (len(documents)) 
 
 
# print first tuple
print (documents[0])
from random import shuffle 
shuffle(documents)

all_words = [word.lower() for word in movie_reviews.words()]
 
# print first 10 words
print (all_words[:10])

from nltk import FreqDist
 
all_words_frequency = FreqDist(all_words)
 
print (all_words_frequency)
print (all_words_frequency.most_common(10))

from nltk.corpus import stopwords
 
stopwords_english = stopwords.words('english')
print (stopwords_english)
all_words_without_stopwords = [word for word in all_words if word not in stopwords_english]
print (all_words_without_stopwords[:10])

import string
 
print (string.punctuation)
all_words_without_punctuation = [word for word in all_words if word not in string.punctuation]
 
# Let's name the new list as all_words_clean 
# because we clean stopwords and punctuations from the word list
all_words_clean = []
for word in all_words:
    if word not in stopwords_english and word not in string.punctuation:
        all_words_clean.append(word)
 
print (all_words_clean[:10])

all_words_frequency = FreqDist(all_words_clean)
 
print (all_words_frequency)
print (all_words_frequency.most_common(10))

print (len(all_words_frequency)) 
 
# get 2000 frequently occuring words
most_common_words = all_words_frequency.most_common(2000)
print (most_common_words[:10])
print (most_common_words[1990:])
word_features = [item[0] for item in most_common_words]
print (word_features[:10])

#creating function for feature extraction
def document_features(document):
    
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
 
# get the first negative movie review file
movie_review_file = movie_reviews.fileids('neg')[0] 
print (movie_review_file)

print (documents[0])
feature_set = [(document_features(doc), category) for (doc, category) in documents]
print (feature_set[0])

print (len(feature_set)) 
 
test_set = feature_set[:400]
train_set = feature_set[400:]
 
print (len(train_set)) 
print (len(test_set)) 

from nltk import NaiveBayesClassifier
 
classifier = NaiveBayesClassifier.train(train_set)

from nltk import classify 
 
accuracy = classify.accuracy(classifier, test_set)
print (accuracy) 

from nltk.tokenize import word_tokenize
 
custom_review = "I hated the film. It was a disaster. Poor direction, bad acting."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = document_features(custom_review_tokens)
print (classifier.classify(custom_review_set)) 

 
# probability result
prob_result = classifier.prob_classify(custom_review_set)
print (prob_result) 
print (prob_result.max()) 
print (prob_result.prob("neg"))
print (prob_result.prob("pos")) 
 
custom_review = "It was a wonderful and amazing movie. I loved it. Best direction, good acting."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = document_features(custom_review_tokens)
 
print (classifier.classify(custom_review_set)) 

 
# probability result
prob_result = classifier.prob_classify(custom_review_set)
print (prob_result) 
print (prob_result.max()) 
print (prob_result.prob("neg")) 
print (prob_result.prob("pos")) 

print (classifier.show_most_informative_features(10))