import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

df = pd.read_csv("news.csv")
print(df.head())
print(df.shape)
print(df.isnull().sum())
labels = df.label
print(labels.head())

x_train, x_test, y_train, y_test = train_test_split(df["text"], labels, test_size = 0.2, random_state = 20)
print(x_train.head())

vector = TfidfVectorizer(stop_words='english', max_df=0.7)
tf_train = vector.fit_transform(x_train)
tf_test = vector.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tf_train,y_train)

y_pred = pac.predict(tf_test)
score = accuracy_score(y_test,y_pred)
print(f"Accuracy : {round(score*100,2)}%")

confusion_matrix(y_test,y_pred,labels=('FAKE', 'REAL'))

filename = 'finalized_model.pkl'
pickle.dump(pac, open(filename, 'wb'))

filename = 'vectorizer.pkl'
pickle.dump(vector, open(filename, 'wb'))
