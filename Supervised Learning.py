import csv
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# ----------- Ερώτημα 1 -----------

# ----------- A -----------

# Import csv into list
file = 'winequality-red.csv'
with open(file, newline='') as csvFile:
    my_List = list(csv.reader(csvFile))

# Remove titles from data
my_List.remove(my_List[0])

# Convert list to array
data = np.array(my_List)
# data = data.astype(np.float64)

# Split array into data and label
result = data[:, -1]
data = data[:, :-1]

# Split data: 75% training anf 25% testing
data_train, data_test, result_train, result_test = train_test_split(data, result, test_size=0.25)

# Creating a SVM model
model = svm.SVC()

# defining parameter range
#param_grid = [
#    {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
#    {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
#]

#grid = GridSearchCV(model, param_grid, refit=True, verbose=3)

# fitting the model for grid search
#grid.fit(data_train, result_train)

# print best parameter after tuning
#print(grid.best_params_)

# print how model looks after hyper-parameter tuning
# print(grid.best_estimator_)


# creating model with optimal parameters
opt_model = svm.SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
                    max_iter=-1, probability=False, random_state=None, shrinking=True,
                    tol=0.001, verbose=False)

opt_model.fit(data_train, result_train)
result_predict = opt_model.predict(data_test)
print(classification_report(result_test, result_predict, zero_division=0))

# ----------- B -----------


# --- 1 ---

# remove pH column
data1 = np.delete(data, 8, 1)

# Split new data: 75% training anf 25% testing
data_train, data_test, result_train, result_test = train_test_split(data1, result, test_size=0.25)

opt_model.fit(data_train, result_train)
result_predict = opt_model.predict(data_test)
print(classification_report(result_test, result_predict, zero_division=0))

# --- 2 ---
data2 = data
i = 0
avg = 0

# find average of 66% of elements in column pH
for x in range(0, len(data2)):
    if x % 3 != 0:
        avg = avg + float(data2[x, 8])
        i = i + 1
avg = avg / i

# set the rest 33% to average
for x in range(0, len(data2)):
    if x % 3 == 0:
        data2[x, 8] = avg

# Split new data: 75% training anf 25% testing
data_train, data_test, result_train, result_test = train_test_split(data2, result, test_size=0.25)

opt_model.fit(data_train, result_train)
result_predict = opt_model.predict(data_test)
print(classification_report(result_test, result_predict, zero_division=0))

# --- 3 ---

# Split array into data and label
pH = data[:, 8]
data_without_pH = data[:, :8]

# Split data: 67% training and 33% testing for regression labeling
reg_train, reg_test, pH_train, pH_test = train_test_split(data_without_pH, pH, test_size=0.33)

pH_clf = LogisticRegression(max_iter=10000)
pH_clf.fit(reg_train, pH_train)
reg_test = reg_test.astype(np.float64)

pH_pred = pH_clf.predict(reg_test)

pH = np.append(pH_train, pH_pred)
pH = pH[:, np.newaxis]

data3 = np.append(data_without_pH, pH, axis=1)

# Split new data: 75% training anf 25% testing
data_train, data_test, result_train, result_test = train_test_split(data3, result, test_size=0.25)

opt_model.fit(data_train, result_train)
result_predict = opt_model.predict(data_test)
print(classification_report(result_test, result_predict, zero_division=0))


# --- 4 ---

data4 = data

for x in range(0, len(data2), 3):
    data4[x, 8] = 0

kmeans = KMeans(n_clusters=10)
kmeans.fit(data4)
labels = kmeans.predict(data4)

avg_pH = np.zeros((10, 3))

for i in range(0, len(data4)):
    if data4[i, 8] != 0:
        avg_pH[labels[i], 0] = avg_pH[labels[i], 0] + float(data4[i, 8])
        avg_pH[labels[i], 1] = avg_pH[labels[i], 1] + 1

for i in range(0, len(avg_pH)):
    avg_pH[i, 2] = avg_pH[i, 0] / avg_pH[i, 1]

for x in range(0, len(data2), 3):
    data4[x, 8] = avg_pH[labels[x], 2]

data_train, data_test, result_train, result_test = train_test_split(data4, result, test_size=0.25)

opt_model.fit(data_train, result_train)
result_predict = opt_model.predict(data_test)
print(classification_report(result_test, result_predict, zero_division=0))
