import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

metricas = ['accuracy', 'f1']

dataPath = r"D:\Pesquisa\ML-Algorithms\final-data.csv"
dataset = pd.read_csv(dataPath)

X = dataset.iloc[:, 1:13]
X = X.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
y = dataset['Smelly'].astype(int)


def runNeuralNetwork10GridSearch(X_train, X_test, y_train, y_test):
    print("Starting: 'MLPClassifier'")

    NN = MLPClassifier()

    hyper = {
        'random_state': [42],
        'hidden_layer_sizes': [(10, 10)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.00001],
        'learning_rate_init': [0.0001],
        'max_iter': [1000000],
        'tol': [0.0001],
        'n_iter_no_change': [1000]
    }

    NN_GRID = GridSearchCV(NN, param_grid=hyper, scoring=metricas, return_train_score=False, refit='accuracy', n_jobs=12)

    NN_GRID.fit(X_train, y_train)

    y_true, y_pred = y_test, NN_GRID.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("Accuracy score: ", acc)
    print("F1-Score: ", f1)

    # print(DT_GRID.best_score_)
    # f.write(DT_GRID.best_estimator_)
    # f.write(DT_GRID.best_score_)
    return acc, f1


def runNeuralNetwork100GridSearch(X_train, X_test, y_train, y_test):
    print("Starting: 'MLPClassifier'")

    NN = MLPClassifier()

    hyper = {
        'random_state': [42],
        'hidden_layer_sizes': [(100, 100)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.00001],
        'learning_rate_init': [0.0001],
        'max_iter': [1000000],
        'tol': [0.0001],
        'n_iter_no_change': [1000]
    }

    NN_GRID = GridSearchCV(NN, param_grid=hyper, scoring=metricas, return_train_score=False, refit='accuracy', n_jobs=12)

    NN_GRID.fit(X_train, y_train)

    y_true, y_pred = y_test, NN_GRID.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("Accuracy score: ", acc)
    print("F1-Score: ", f1)

    # print(DT_GRID.best_score_)
    # f.write(DT_GRID.best_estimator_)
    # f.write(DT_GRID.best_score_)
    return acc, f1


def runSVMLinearGridSearch(X_train, X_test, y_train, y_test):
    print("Starting: 'SVC' Linear")

    SVCLinear = SVC()

    hyper = {
        'random_state': [42],
        'kernel': ['linear'],
        'degree': [2, 3, 4],
        'gamma': ['scale'],
        'shrinking': [True],
        'decision_function_shape': ['ovr'],
        'tol': [0.0001, 0.00001],
        'C': [1, 10, 100]
    }

    SVC_GRID = GridSearchCV(SVCLinear, param_grid=hyper, scoring=metricas, return_train_score=False, refit='accuracy', n_jobs=12)

    SVC_GRID.fit(X_train, y_train)

    y_true, y_pred = y_test, SVC_GRID.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("Accuracy score: ", acc)
    print("F1-Score: ", f1)

    # print(DT_GRID.best_score_)
    # f.write(DT_GRID.best_estimator_)
    # f.write(DT_GRID.best_score_)
    return acc, f1


def runSVMPolyGridSearch(X_train, X_test, y_train, y_test):
    print("Starting: 'SVC' Poly")

    SVCLinear = SVC()

    hyper = {
        'random_state': [42],
        'kernel': ['poly'],
        'degree': [2, 3, 4],
        'gamma': ['scale'],
        'shrinking': [True],
        'decision_function_shape': ['ovr'],
        'tol': [0.0001, 0.00001],
        'C': [1, 10, 100]
    }

    SVC_GRID = GridSearchCV(SVCLinear, param_grid=hyper, scoring=metricas, return_train_score=False, refit='accuracy', n_jobs=12)

    SVC_GRID.fit(X_train, y_train)

    y_true, y_pred = y_test, SVC_GRID.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("Accuracy score: ", acc)
    print("F1-Score: ", f1)

    # print(DT_GRID.best_score_)
    # f.write(DT_GRID.best_estimator_)
    # f.write(DT_GRID.best_score_)
    return acc, f1


def runSVMRBFGridSearch(X_train, X_test, y_train, y_test):
    print("Starting: 'SVC' RBF")

    SVCLinear = SVC()

    hyper = {
        'random_state': [42],
        'kernel': ['rbf'],
        'degree': [2, 3, 4],
        'gamma': ['scale'],
        'shrinking': [True],
        'decision_function_shape': ['ovr'],
        'tol': [0.0001, 0.00001],
        'C': [1, 10, 100]
    }

    SVC_GRID = GridSearchCV(SVCLinear, param_grid=hyper, scoring=metricas, return_train_score=False, refit='accuracy', n_jobs=12)

    SVC_GRID.fit(X_train, y_train)

    y_true, y_pred = y_test, SVC_GRID.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("Accuracy score: ", acc)
    print("F1-Score: ", f1)

    # print(DT_GRID.best_score_)
    # f.write(DT_GRID.best_estimator_)
    # f.write(DT_GRID.best_score_)
    return acc, f1


def runSVMSigmoidGridSearch(X_train, X_test, y_train, y_test):
    print("Starting: 'SVC' Sigmoid")

    SVCLinear = SVC()

    hyper = {
        'random_state': [42],
        'kernel': ['sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale'],
        'shrinking': [True],
        'decision_function_shape': ['ovr'],
        'tol': [0.0001, 0.00001],
        'C': [1, 10, 100]
    }

    SVC_GRID = GridSearchCV(SVCLinear, param_grid=hyper, scoring=metricas, return_train_score=False, refit='accuracy', n_jobs=12)

    SVC_GRID.fit(X_train, y_train)

    y_true, y_pred = y_test, SVC_GRID.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("Accuracy score: ", acc)
    print("F1-Score: ", f1)

    # print(DT_GRID.best_score_)
    # f.write(DT_GRID.best_estimator_)
    # f.write(DT_GRID.best_score_)
    return acc, f1


def runDecisionTreeGiniGridSearch(X_train, X_test, y_train, y_test):
    print("Starting: 'DecisionTree' Gini")

    DecisionTree = DecisionTreeClassifier()

    hyper = {
        'random_state': [42],
        'criterion': ['gini'],
        'splitter': ['best'],
        'max_features': [None, 'sqrt', 'log2'],
    }

    DT_GRID = GridSearchCV(DecisionTree, param_grid=hyper, scoring=metricas, return_train_score=False, refit='accuracy', n_jobs=12)

    DT_GRID.fit(X_train, y_train)

    y_true, y_pred = y_test, DT_GRID.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("Accuracy score: ", acc)
    print("F1-Score: ", f1)

    # print(DT_GRID.best_score_)
    # f.write(DT_GRID.best_estimator_)
    # f.write(DT_GRID.best_score_)
    return acc, f1


def runDecisionTreeEntropyGridSearch(X_train, X_test, y_train, y_test):
    print("Starting: 'DecisionTree' Entropy")

    DecisionTree = DecisionTreeClassifier()

    hyper = {
        'random_state': [42],
        'criterion': ['entropy'],
        'splitter': ['best'],
        'max_features': [None, 'sqrt', 'log2'],
    }

    DT_GRID = GridSearchCV(DecisionTree, param_grid=hyper, scoring=metricas, return_train_score=False, refit='accuracy', n_jobs=12)

    DT_GRID.fit(X_train, y_train)

    y_true, y_pred = y_test, DT_GRID.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("Accuracy score: ", acc)
    print("F1-Score: ", f1)

    # print(DT_GRID.best_score_)
    # f.write(DT_GRID.best_estimator_)
    # f.write(DT_GRID.best_score_)
    return acc, f1


scores = {
    'NN10': {
        'f1': [],
        'acc': []
    },
    'NN100': {
        'f1': [],
        'acc': []
    },
    'SVMLinear': {
        'f1': [],
        'acc': []
    },
    'SVMPoly': {
        'f1': [],
        'acc': []
    },
    'SVMRBF': {
        'f1': [],
        'acc': []
    },
    'SVMSigmoid': {
        'f1': [],
        'acc': []
    },
    'DTGini': {
        'f1': [],
        'acc': []
    },
    'DTEntropy': {
        'f1': [],
        'acc': []
    }
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)


for i in range(5):
    fold = next(kf.split(X), None)
    X_train = X.iloc[fold[0]]
    X_test = X.iloc[fold[1]]
    y_train = y.iloc[fold[0]]
    y_test = y.iloc[fold[1]]

    a, b = runNeuralNetwork10GridSearch(X_train, X_test, y_train, y_test)
    scores['NN10']['acc'].append(a)
    scores['NN10']['f1'].append(b)
    a, b = runNeuralNetwork100GridSearch(X_train, X_test, y_train, y_test)
    scores['NN100']['acc'].append(a)
    scores['NN100']['f1'].append(b)
    a, b = runSVMLinearGridSearch(X_train, X_test, y_train, y_test)
    scores['SVMLinear']['acc'].append(a)
    scores['SVMLinear']['f1'].append(b)
    a, b = runSVMPolyGridSearch(X_train, X_test, y_train, y_test)
    scores['SVMPoly']['acc'].append(a)
    scores['SVMPoly']['f1'].append(b)
    a, b = runSVMRBFGridSearch(X_train, X_test, y_train, y_test)
    scores['SVMRBF']['acc'].append(a)
    scores['SVMRBF']['f1'].append(b)
    a, b = runSVMSigmoidGridSearch(X_train, X_test, y_train, y_test)
    scores['SVMSigmoid']['acc'].append(a)
    scores['SVMSigmoid']['f1'].append(b)
    a, b = runDecisionTreeGiniGridSearch(X_train, X_test, y_train, y_test)
    scores['DTGini']['acc'].append(a)
    scores['DTGini']['f1'].append(b)
    a, b = runDecisionTreeEntropyGridSearch(X_train, X_test, y_train, y_test)
    scores['DTEntropy']['acc'].append(a)
    scores['DTEntropy']['f1'].append(b)

for score in scores:
    print(score + " scores:")
    print("\tAccuracy: ", np.mean(scores[score]['acc']))
    print('\tF1-Score: ', np.mean(scores[score]['f1']))
    print()
