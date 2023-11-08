import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

iris = datasets.load_iris()
iris1 = pd.read_csv('brief/classer ses iris/iris.csv')

data, labels = iris.data, iris.target

res = train_test_split(data, labels,
                        train_size = 0.8,
                        test_size = 0.2,
                        random_state = 990)

train_data, test_data, train_labels, test_labels = res

#créons et entrainons un KNN

knn = KNeighborsClassifier()
knn.fit(train_data, train_labels)

#on peut tester le modele de 2 façons :
# 1)
test_data_predicted = knn.predict(test_data)
print(accuracy_score(test_data_predicted, test_labels))
# 2)
score_test = knn.score(test_data, test_labels)
print(score_test)

#on test le modèle par rapport a son propre entrainement

learn_data_predicted = knn.predict(train_data)
print(accuracy_score(learn_data_predicted, train_labels))



knn2 = KNeighborsClassifier()

print(knn.predict([[1, 2, 3, 4]]))
# print(train_data)
    # knn.predict([10,15,4,3]))

knn2 = KNeighborsClassifier()
def prédire_espece(sepal_length,sepal_width,petal_length,petal_width):
    a = knn.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if (a == 0):
        print('setosa')
    elif (a == 1):
        print('versicolor')
    elif (a == 2):
        print('virginica')

prédire_espece(1,2,5,6)

# def my_knn(data_train, labels_train, k, valeur_test):
data_train = data
labels_train = labels
for i in data_train:
    print(i)







    
        
