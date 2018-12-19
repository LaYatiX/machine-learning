from keras.datasets import fashion_mnist
import numpy as np
import seaborn as sns;
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import confusion_matrix
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


images_train =  []
for image_train in x_train:
    images_train.append(image_train.flatten())

images_test = []

for image_test in x_test:
    images_test.append(image_test.flatten())


images_train = np.array(images_train)
images_test = np.array(images_test)

classifier = OneVsRestClassifier(LogisticRegression(verbose=3, max_iter=9))
classifier.fit(images_train, y_train);
conf_matrix = confusion_matrix(y_test, classifier.predict(images_test))
print(conf_matrix)
sns.heatmap(conf_matrix)
score1 = classifier.score(images_test, y_test)
print("Wynik 1:")
print(score1)

multi_class_classifier = LogisticRegression(verbose=2, max_iter=8, multi_class="multinomial", solver="sag")
multi_class_classifier.fit(images_train, y_train)

conf_matrix = confusion_matrix(y_test, multi_class_classifier.predict(images_test))
print(conf_matrix)
sns.heatmap(conf_matrix)
score2 = multi_class_classifier.score(images_test, y_test)
print("Wynik 2:")
print(score2)

#zapis
import pickle
pickle.dump(multi_class_classifier, open('multi_class_classifier.model3', 'wb'));