import numpy as np
from sklearn import cross_validation, neighbors
import pandas as pd

df = pd.read_csv('column_3C.dat', sep=None, engine='python')
df.replace('?', -99999, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[08.52,53.92,41.47,44,115.51,30.39]])
example_measures = example_measures.reshape(1, -1)

prediction = clf.predict(example_measures)

print(prediction)
