from sklearn.feature_extraction.text import CountVectorizer

#examplify the target
vectorizer=CountVectorizer(min_df=1)
#process the text with bag-of-words
corpus1=['This is the first document.',
        'This is the second document.',
        'And the third one.',
        'Is this the first document?',]
corpus2=['This is a spam.',
        'We invite you to join in our team.',
        'What are you doing now.',
        'Can we make friends.']
X=vectorizer.fit_transform(corpus1)
Y=vectorizer.fit_transform(corpus2)
print(X)
print(X.toarray())
print(X.sum(),'\n')
print(Y.toarray())
print(Y.sum(),'\n')
#get the name of features
names=vectorizer.get_feature_names()
print(names)