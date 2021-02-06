from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from KNN.preprocessing import Preprocessor
from KNN.datasets import DatasetLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from imutils import paths


dataset_path = ".\\datasets\\TezYemek"
numberofneighbors = 8



print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset_path))


pp = Preprocessor(100, 50)
dl = DatasetLoader(preprocessors=[pp])
(data, labels) = dl.load(imagePaths, verbose=500)
#print(data.shape)
#print(labels.shape)
data = data.reshape((data.shape[0], -1))# bu kodun doğru boyutu bulmasını sağlıyor
#print(data.shape)
le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)#random_state=42(kodu hep aynı yerden bölmek)

#print(trainX.shape)
#print(testX.shape)


print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=numberofneighbors, metric='minkowski')
model.fit(trainX, trainY)

#print(model.score(testX, testY))
predict =model.predict(testX)
print("Accuracy",accuracy_score(predict,testY))
print(confusion_matrix(testY,predict))

#print(le.classes_)
#print(classification_report(testY, model.predict(testX)))