import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

DATA_DICT = pickle.load(open('D:\Machine Learning Projects\Sign-Language-Detection\dataset_list.pickle', 'rb'))

data = np.asarray(DATA_DICT['dataset_list'])
labels = np.array(DATA_DICT['labels_list'])

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size = 0.2, shuffle = True, stratify = labels)

model = RandomForestClassifier()

model.fit(train_x, train_y)

y_predict = model.predict(test_x)

accuracy = accuracy_score(y_predict, test_y)

print('{}% of samples were classified correctly'.format(accuracy * 100))

model_file = open('model.p', 'wb')
pickle.dump({'model' : model}, model_file)
model_file.close()