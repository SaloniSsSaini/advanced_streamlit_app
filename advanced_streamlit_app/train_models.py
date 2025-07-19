
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import os

os.makedirs('model', exist_ok=True)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

rf = RandomForestClassifier()
rf.fit(X, y)
pickle.dump(rf, open('model/random_forest_model.pkl', 'wb'))

lr = LogisticRegression(max_iter=200)
lr.fit(X, y)
pickle.dump(lr, open('model/logistic_regression_model.pkl', 'wb'))

svm = SVC(probability=True)
svm.fit(X, y)
pickle.dump(svm, open('model/svm_model.pkl', 'wb'))

print("✅ Models saved in 'model/' folder")
