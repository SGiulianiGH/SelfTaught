###
# This is developed using the youtube chanel, sentdex (https://www.youtube.com/user/sentdex)
# These py files are written based on the sentdex playlist, "Machine Learning with Python".
# https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v

# Videos 13 : 19

###

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# K-NN (from scratch, using sklearn is below)

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')
import pandas as pd
import random

# dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
# new_features = [5,7]

# [[plt.scatter(ii[0], ii[1],s=100, color=i) for ii in dataset[i]] for i in dataset]
# """
# # Does the same as the line above
# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)
# """
# plt.scatter(new_features[0],new_features[1], s=100, color = 'g')
# plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total classes!')
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = sqrt( (features[0] - predict[0])**2 + (features[1] - predict[1])**2 )
            # euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict[1])) ** 2))
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / float(k)
    return vote_result, confidence

accuracies = []

for i in range(25):  #itterate a bunch of times and saves accuracy to the accuracies list
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)  # the 99999 is because most algorithms see this as an outlier and ignore it
    df.drop(['id'], 1, inplace=True)  # the id column doesn't really help classify anything
    full_data = df.astype(float).values.tolist()  # converting everything to floats for simpler read
    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]  # train data is the first 80% of full_data
    test_data = full_data[-int(test_size*len(full_data)):]  # test data is the last 20% of full_data

    for i in train_data:
        train_set[i[-1]].append(i[:-1])  # calls the last column (the class)
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1

    #print('Accuracy: ', (correct)/float(total))  # have to add 'float' so that the result isn't just an int of 0
    accuracies.append((correct)/float(total))

print("Scratch: ",sum(accuracies)/len(accuracies))
#print('Accuracy: ', float(float(correct)/float(total)))

# [[plt.scatter(ii[0], ii[1],s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0],new_features[1], s=100, color = 'g')
# plt.show()



# Euclidean Distance example
"""
plot1 = [1,3]
plot2 = [2,5]
euclidean_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)
print(euclidean_distance)
"""



# # K-NN (using Sklearn's classifier)
#
# import numpy as np
# import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors

accuracies_s = []
for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)  # the 99999 is because most algorithms see this as an outlier and ignore it
    df.drop(['id'], 1, inplace=True)  # the id column doesn't really help classify anything

    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    # #print(accuracy)
    #
    # example_measures = np.array([4,2,1,1,1,2,3,2,1])
    # example_measures = example_measures.reshape(len(example_measures), -1)
    # # otherwise you get a warning that you're passing the wrong dimensions -- this is common for sklearn classifiers
    #
    # prediction = clf.predict(example_measures)
    # #print(prediction)
    accuracies_s.append(accuracy)


print("\nSklearn: ",sum(accuracies_s)/len(accuracies_s))

