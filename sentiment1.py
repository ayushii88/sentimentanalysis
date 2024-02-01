from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

#extracting features from the input
def extract_features(words):
    return dict([word,True] for word in words)

#loading reviews from the corpus
fileids_pos = movie_reviews.fileids('pos')
#print(fileids_pos)
fileids_neg = movie_reviews.fileids('neg')

#extract the features and label accordingly
features_pos= [(extract_features(movie_reviews.words(
        fileids=[f])),'Positive') for f in fileids_pos]
features_neg= [(extract_features(movie_reviews.words(
        fileids=[f])),'Negative') for f in fileids_neg]    

#define training and  testing data, 80% & 20%
threshold = 0.8
num_pos = int(threshold * len(features_pos))
num_neg = int(threshold * len(features_neg))    
 #creating traning and testing data
features_train = features_pos[ :num_pos] + features_neg[ :num_neg]
features_test = features_pos[ num_pos: ] + features_neg[num_neg: ]    

#printing the training and testing data
print('\n Number of training datapoints:',len(features_train))
print('\n Number of testing datapoints:',len(features_test))

#naive bayes classifier
classifier = NaiveBayesClassifier.train(features_train)
print('\n Accuracy of the classifier:' ,nltk_accuracy(classifier, features_test))

#top N informative reviews

N=20
print('\n most informative words:')
for i, item in enumerate(classifier.most_informative_features()):
    print(str(i+1) + '.' + item[0])
    if i == N-1:
        break

input_reviews = [
        'The costumes in this movie were great',
        'I think the story was terrible and the characters were very sick',
        'People say that the director of movie is amazing',
        'I did not understand the plot',
        'I loved the movie',
        ' This is such an idiotic movie. I will not recommend it to anyone',
        'The movie plots could have been better'] 

print('/n Movie review predictions')
for review in input_reviews:
    print("\nReview",review)
    
    #compute probablilities
    probabilities = classifier.prob_classify(extract_features(review.split()))
    #pick the maximum values
    predicted_sentiment = probabilities.max()
    
    #print outputs
    print("Predicted : ",predicted_sentiment)
    print("Probability: ", round(probabilities.prob(predicted_sentiment),2))
    
    
    