import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pydotplus
from PIL import Image as PImage

# Load data
train_data = pd.read_csv("C:\\Users\\tomas\\OneDrive\\Pulpit\\Drzewa_decyzyjne\\train.csv")
test_data = pd.read_csv("C:\\Users\\tomas\\OneDrive\\Pulpit\\Drzewa_decyzyjne\\test.csv")

# Copy test names for final results
test_names = test_data[['Name']].copy()

# Drop unnecessary columns
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
test_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Fill missing values
for dataset in [train_data, test_data]:
    if 'Age' in dataset.columns:
        dataset['Age'] = dataset['Age'].fillna(train_data['Age'].mean())
    if 'Embarked' in dataset.columns:
        dataset['Embarked'] = dataset['Embarked'].fillna(train_data['Embarked'].mode()[0])

# Convert categorical variables to dummy variables
categorical_vars = ['Sex', 'Embarked']
train_data = pd.get_dummies(train_data, columns=categorical_vars, drop_first=True)
test_data = pd.get_dummies(test_data, columns=categorical_vars, drop_first=True)

# Create FamilySize feature
if 'SibSp' in train_data.columns and 'Parch' in train_data.columns:
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
    test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# Split data into features and target
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data

# Define parameter grid for GridSearchCV
parameters = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
clf = DecisionTreeClassifier()
grid_search = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Predict on the test set
y_pred = model.predict(X_test)
results = test_names.copy()
results['Predicted_Survival'] = y_pred
results.to_csv("test_predictions_with_names.csv", index=False)

# Generate decision tree plot using graphviz
dot_data = export_graphviz(model, out_file=None, 
                           feature_names=X_train.columns,  
                           class_names=['Died', 'Survived'],  
                           filled=True, rounded=True,  
                           special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png("decision_tree.png")

# Display the image
im = PImage.open("decision_tree.png")
im.show()