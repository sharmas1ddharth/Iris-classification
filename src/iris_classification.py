import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import pickle


def split_data(data):
    data['species'] = data['species'].astype('category')
    X = data.drop('species', axis=1)
    y = data['species']
    return X, y


def train_model(X, y):
    clf = AdaBoostClassifier(learning_rate=1.0, n_estimators=97, random_state=1)
    clf.fit(X, y)
    return clf


def save_model(model):
    model_name = 'iris_classification.model'
    pickle.dump(model, open(f"../Models/{model_name}", 'wb'))


def main():
    train_data = pd.read_csv('../data/processed/train_data.csv')
    X, y = split_data(train_data)
    model = train_model(X, y)
    save_model(model)


if __name__ == "__main__":
    main()
