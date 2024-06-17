import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

train_data = pd.read_csv("C:\\Users\\tomas\\OneDrive\\Pulpit\\Drzewa_decyzyjne\\train.csv")
test_data = pd.read_csv("C:\\Users\\tomas\\OneDrive\\Pulpit\\Drzewa_decyzyjne\\test.csv")

test_names = test_data[['Name']].copy()
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
test_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

for dataset in [train_data, test_data]:
    if 'Age' in dataset.columns:
        dataset['Age'] = dataset['Age'].fillna(train_data['Age'].mean())
    if 'Embarked' in dataset.columns:
        dataset['Embarked'] = dataset['Embarked'].fillna(train_data['Embarked'].mode()[0])

categorical_vars = ['Sex', 'Embarked']
train_data = pd.get_dummies(train_data, columns=categorical_vars, drop_first=True)
test_data = pd.get_dummies(test_data, columns=categorical_vars, drop_first=True)

if 'SibSp' in train_data.columns and 'Parch' in train_data.columns:
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
    test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
    
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data
parameters = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
clf = DecisionTreeClassifier()
grid_search = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
y_pred = model.predict(X_test)
results = test_names.copy()
results['Predicted_Survival'] = y_pred
results.to_csv("test_predictions_with_names.csv", index=False)
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X_train.columns, class_names=['Died', 'Survived'], fontsize=10)
plt.title("Decision Tree for Predicting Survival on the Titanic with Tuned Parameters")
plt.show()
