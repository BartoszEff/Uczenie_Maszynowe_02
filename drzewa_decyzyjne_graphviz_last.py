import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import pydotplus
from PIL import Image as PImage
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve

# Wczytywanie danych
train_data = pd.read_csv("C:\\Users\\tomas\\OneDrive\\Pulpit\\Drzewa_decyzyjne\\train.csv")
test_data = pd.read_csv("C:\\Users\\tomas\\OneDrive\\Pulpit\\Drzewa_decyzyjne\\test.csv")

# Przygotowanie danych
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

# Podział danych
X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data

# Podział danych na zestaw uczący i walidacyjny do obliczenia MSE
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Definiowanie siatki parametrów
parameters = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Trenowanie modelu i wybór najlepszego
clf = DecisionTreeClassifier()
grid_search = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Predykcja na zestawie testowym
y_pred = model.predict(X_test)

# Zapis wyników
results = test_names.copy()
results['Predicted_Survival'] = y_pred
results.to_csv("test_predictions_with_names.csv", index=False)

""" # Wizualizacja drzewa decyzyjnego za pomocą plot_tree
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X_train.columns, class_names=['Died', 'Survived'], fontsize=10)
plt.title("Decision Tree for Predicting Survival on the Titanic")
plt.show() """

# Generowanie grafu drzewa decyzyjnego za pomocą Graphviz
dot_data = export_graphviz(model, out_file=None, 
                           feature_names=X_train.columns,  
                           class_names=['Died', 'Survived'],  
                           filled=True, rounded=True,  
                           special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png("decision_tree.png")

# Wyświetlanie obrazu drzewa
im = PImage.open("decision_tree.png")
im.show()

# Obliczanie dokładności modelu
accuracy = accuracy_score(y_train, model.predict(X_train))
print("Accuracy of the model: {:.2f}%".format(accuracy * 100))

# Znaczenie cech
importances = model.feature_importances_
feature_names = X_train.columns

# Tworzenie wykresu znaczenia cech
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='b', align='center')
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Decision Tree")
plt.show()

# Obliczanie MSE dla różnych głębokości drzewa
depths = range(1, 21)
mse_values = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train_split, y_train_split)
    y_val_pred = model.predict(X_val_split)
    mse = mean_squared_error(y_val_split, y_val_pred)
    mse_values.append(mse)

# Tworzenie wykresu MSE w zależności od głębokości drzewa
plt.figure(figsize=(10, 6))
plt.plot(depths, mse_values, marker='o', linestyle='-', color='r')
plt.xlabel("Tree Depth")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs. Tree Depth")
plt.show()

# Obliczanie Precision-Recall
y_scores = model.predict_proba(X_val_split)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val_split, y_scores)

# Tworzenie wykresu Precision-Recall
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Tabela czynników wpływających na przeżycie
factors = [
    ["Czynnik", "Opis"],
    ["Płeć", "Kobiety mają większe szanse na przeżycie."],
    ["Cena biletu", "Osoby z droższymi biletami mają większe szanse na przeżycie."],
    ["Klasa", "Pasażerowie z wyższych klas mają większe szanse na przeżycie."],
    ["Wiek", "Młodsi pasażerowie mają większe szanse na przeżycie."],
    ["Wielkość rodziny", "Pasażerowie z umiarkowaną wielkością rodziny mają większe szanse na przeżycie."]
]

print("\nTabela czynników wpływających na przeżycie:\n")
for factor in factors:
    print(f"{factor[0]:<20} {factor[1]}")
