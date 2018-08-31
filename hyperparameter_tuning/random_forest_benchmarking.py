import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data_directory = os.path.join(os.environ['VIRTUAL_ENV'], '..', 'data', 'tox21')

data = {}
for pickle_file in os.listdir(data_directory):
    with open(os.path.join(data_directory, pickle_file), 'rb') as reader:
        data[pickle_file.strip('.pkl')] = pickle.load(reader)

train_X, train_y, train_w = data['train_X'], data['train_y'], data['train_w']
valid_X, valid_y, valid_w = data['valid_X'], data['valid_y'], data['valid_w']
test_X, test_y, test_w = data['test_X'], data['test_y'], data['test_w']

# Remove extra tasks
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

model = RandomForestClassifier(class_weight='balanced', n_estimators=50)
model.fit(train_X, train_y)

valid_y_pred = model.predict(valid_X)
test_y_pred = model.predict(test_X)

score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
print('Accuracy score of validation set: {}'.format(score))

score = accuracy_score(test_y, test_y_pred, sample_weight=test_w)
print('Accuracy score of test set: {}'.format(score))
