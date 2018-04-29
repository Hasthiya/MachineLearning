import numpy as np
from sklearn import cross_validation, neighbors
import pandas as pd
import pickle


def get_values_from_keyboard(value_name):
    nb = input(value_name + ' : ')
    return nb


def print_prediction(argument, accuracy):
    switcher = {
        'DH': "Disk Hernia",
        'SL': "Spondylolisthesis",
        'NO': "Normal"
    }
    print('\nPatient Status : ' + switcher.get(argument, "Invalid Prediction")
          + ' with a prediction accuracy of ' + str(accuracy))
    return


# reading the data set and cleansing the data
df = pd.read_csv('column_3C.dat', sep=None, engine='python')
df.replace('?', -99999, inplace=True)

# defining X and y
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# randomly split data set into training and test sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# searching for pickle and use or creating classifier
try:
    pickle_in = open('KNeighborsClassifier.pickle', 'rb')
    clf = pickle.load(pickle_in)
except (OSError, IOError) as e:
    clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)
    with open('KNeighborsClassifier.pickle', 'wb') as f:
        pickle.dump(clf, f)

# measuring accuracy from test set
accuracy = clf.score(X_test, y_test)

# Taking input from user
print("\nPlease Enter the following details to predict patient\'s status \n")

p_incidence = get_values_from_keyboard("Pelvic Incidence")
p_tilt = get_values_from_keyboard("Pelvic Tilt")
lla = get_values_from_keyboard("Lumbar Lordosis Angle")
ss = get_values_from_keyboard("Sacral Slope")
p_radius = get_values_from_keyboard("Pelvic Radius")
gos = get_values_from_keyboard("Grade of Spondylolisthesis")

# creating measures from input
measures = np.array([[p_incidence, p_tilt, lla, ss, p_radius, gos]])
measures = measures.reshape(1, -1)

# making a prediction from the classifier and measures
prediction = clf.predict(measures)

# printing prediction and accuracy
print_prediction(prediction[0], accuracy)

