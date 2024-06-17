import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
import matplotlib.pyplot as plt


data = pd.read_csv("C:\\Users\\Piotr\\Desktop\\Uczenie_Maszynowe_02-main\\train.csv")

data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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
accuracy = accuracy_score(y_test, y_pred)
print("Improved accuracy of the model: {:.2f}%".format(accuracy * 100))

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: {:.2f}".format(mse))

# Plot MSE over different depths
depths = parameters['max_depth']
mses = []

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    y_pred_depth = clf.predict(X_test)
    mse_depth = mean_squared_error(y_test, y_pred_depth)
    mses.append(mse_depth)

plt.figure()
plt.plot(depths, mses, marker='o')
plt.xlabel('Max Depth of Decision Tree')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Depth of Decision Tree')
plt.show()

# Calculate Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])

# Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Plot Decision Tree
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Died', 'Survived'], fontsize=10)
plt.title("Drzewo decyzyjne")
plt.show()
