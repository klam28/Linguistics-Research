import os
os.chdir('/Users/samspevack/Desktop/glove.twitter.27B')

import numpy as np
import re
from scipy.spatial import distance
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt



traits = open('levy.csv')
#traits = open('twords.csv')
terms = []
times = []
curline = traits.readline()
while(True):
	curline = traits.readline()
	if (curline == ''): break
	word = curline.strip('\r\n').split(',')[2]
	word = word.strip(' ')
	terms.append(word.lower())
	time = curline.strip('\r\n').split(',')[4]
	times.append(time)

TraitDict = {}

Times = [float(x) for x in times[1:]]

for term in terms:
	TraitDict[term.lower()] = ''

file = open('glove.6B.50d.txt')

while(True):
	curline = file.readline()
	if (curline == ''): break
	word = curline.split(' ')[0]
	word = re.sub('[^A-Za-z0-9]+', '', word)
	if (TraitDict.has_key(word)):
		vector = curline.strip('\n').split(' ')[1:-1]
		vector = [float(x) for x in vector]
		TraitDict[word] = vector
		print word

Dists = []


for word_num in range(1,len(terms)):
	a = np.array(TraitDict[terms[word_num-1]])
	b = np.array(TraitDict[terms[word_num]])
	if ((TraitDict[terms[word_num-1]] == '') or (TraitDict[terms[word_num]] == '')):
		dist = 6
	else:
		dist = distance.euclidean(a,b)
	Dists.append(dist)


# Create linear regression object
Dists = np.array(Dists).reshape(-1,1)
Times = np.array(Times).reshape(-1,1)
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(Dists, Times)
y_pred = regr.predict(Dists)

plt.scatter(Dists, Times,  color='black')
plt.plot(Dists, y_pred, color='blue', linewidth=3)

plt.show()
