from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# load train and test datasets
data_train = fetch_20newsgroups(subset='train')
data_test = fetch_20newsgroups(subset='test')

# create vectorizer, set parameters
tfidf_model = TfidfVectorizer(max_df=0.99, min_df=0.01, stop_words='english', use_idf=True)

# vectorize data
X_train = tfidf_model.fit_transform(data_train.data)
X_test = tfidf_model.transform(data_test.data)

# get y
y_train = data_train.target
y_test = data_test.target
# get y names
y_names = data_train.target_names

# create logistic model, train and fit
logistic_model = LogisticRegression()
logistic_model = logistic_model.fit(X_train,y_train)

# predict results on test set
y_hat = logistic_model.predict(X_test)

# draw confusion matrix
cm = confusion_matrix(y_test, y_hat)
cmp = ConfusionMatrixDisplay(cm, display_labels=y_names)
fig, ax = plt.subplots(figsize=(10,10))
cmp.plot(xticks_rotation='vertical', ax=ax)
plt.show()

# multiclass classification report 
print(classification_report(y_test, y_hat, target_names = y_names))

'''
Highest precision: 0.94, talk.politics.mideast
Lowest precision: 0.58, talk.politics.misc
Probably because for mideast it appears a lot of proper nouns including iconic names and geographic terms,
while for misc politics it includes all kinds of words without few iconic words like those in science and CS.
'''