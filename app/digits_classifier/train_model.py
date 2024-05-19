import pickle

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from app.extensions import ML_MODELS_DIR 

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# 20% test data, 80% train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ml model
clf = RandomForestClassifier(n_jobs=-1)

# fitting data into a model
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

with open(f'{ML_MODELS_DIR}handwritten_digits.pkl', 'wb') as f:
    # saving model into pickle file
    pickle.dump(clf, f)
    

    
    
    