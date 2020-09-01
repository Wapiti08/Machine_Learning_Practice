from sklearn.metrics import pairwise_distances
import numpy as np

# 1.def the Euclidean distance
def Euc_distance(x1, x2):
    return abs(np.linalg.norm(x1)-np.linalg.norm(x2))

# 2.def the Cos distance 1-a*b/(|a||b|)
def Cos_distance(x1, x2):
    return 1 - np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))

dist, metric = Cos_distance, 'cosine'

# 3.def the more intuitive way to calculate the distance
def word_distance(w1, w2, w3, word2vec):
    # find whether the word is in dictionary
    for w in [w1, w2, w3]:
    # get wordvector for these words
        if w not in word2vec:
            print('The word {} does not exist in dictionary'.format(w))
            return
    # king - man = queen - woman
    king = word2vec[w1]
    man = word2vec[w2]
    woman = word2vec[w3]
    v0 = king -man + woman
    # define the min_dist
    min_dist = float('inf')
    best_word = ''
    # travsal the dictionary to find the min_distance between the selected word and king
    for word, vec in word2vec.items():
    # print the min_distance and best_word
        # ensure the calculated word not in three words
        if word not in (w1, w2, w3):
            d = dist(v0, vec)
            # choose any distance solution from Euc or Cos
            if d < min_dist:
                min_dist = d
                best_word = word
                print('{} - {} = {} - {}'.format(best_word, w1, w2, w3))

# 4.faster with pairwise_distances
# pairwise_distances(X, Y=None, metric='euclidean', *, n_jobs=None, force_all_finite=True, **kwds)
def word_distance_fast(w1, w2, w3, embedding, word2vec):
    # find whether the word is in dictionary
    for w in [w1, w2, w3]:
    # get wordvector for these words
        if w not in word2vec:
            print('The word {} does not exist in dictionary'.format(w))
            return
    # king - man = queen - woman
    king = word2vec[w1]
    man = word2vec[w2]
    woman = word2vec[w3]
    v0 = king -man + woman

    distances = pairwise_distances(v0.reshape(1,D), embedding, metric = metric ).reshape(V)
    min_idx = distances.argmin()
    best_word = word2vec[min_idx]
    return best_word


# nearest_neighbor
def nearest_neighbor(w, n=5, embedding = embedding, metric = metric, word2vec = word2vec):
    if w not in word2vec:
        print("The word {} is not in dictionary.".format(w))
        return
    v0 = word2vec[w]
    distances = pairwise_distances(v0.reshape(1,D), embedding, metric = metric).reshapce(V)
    # the first element is not index while word
    idxes = distances.argsort[1:n+1]
    for idx in idxes:
        best_word = distances[idx]
        print("the best word is:",best_word)


# load in pre-trained word vectors

word2vec ={}
embedding = []
idx2word = []

with open("./large_files/glove.6B.50d.txt", encoding='utf8') as f:
    for line in f:
        # get the word
        values = line.split()
        # print(values)
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
        embedding.append(vec)
        idx2word.append(word)
    print("Found %s word vectors."%len(word2vec))
    embedding = np.array(embedding)
    V, D = embedding.shape

word_distance('king', 'man', 'women', word2vec)
