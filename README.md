Program No : 1
Aim : Perform all matrix operations using python (Using numpy)
PROGRAM
import numpy
array1 = []
n=int(input("Enter the array Size"))
for i in range(n):
 array1.append(int(input("Enter thr first array elements : ")))
array1=numpy.array(array1)
print(numpy.floor(array1))
array2=[]
for i in range(n):
 array2.append(int(input("Enter the second array elemensts : ")))
array2=numpy.array(array2)
print(numpy.floor(array2))
print("Array Addition ")
print(numpy.add(array1,array2))
print("Array Substraction ")
print(numpy.subtract(array1,array2))
print("Array Multiplication")
print(numpy.multiply(array1,array2))
print("Array Division")
print(numpy.divide(array1,array2))
print("Array Dot")
print(numpy.dot(array1,array2))
print("Array Squareroot")
print(numpy.sqrt(array1))
print("Array Summation of array1 ")
print(numpy.sum(array1))
print("Array Transpose of array1")
print(array1.T)
OUTPUT
PROGRAM 2:
AIM: Perform SVD (Singular Value Decomposition ) Using python.
PROGRAM:
from numpy import array
from scipy.linalg import svd
A1= array([[2,13,3],[5,9,10],[8,7,3],[26,18,30],[22,31,45]])
print(A1)
a,b,c=svd(A1)
print(a)
print(b)
print(c)






PROGRAM 3:
AIM : Program to implement K-NN classification using any standard dataset available in
public domain and find the accuracy of the algorithm.
PROGRAM:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
irisData=load_iris()
a=irisData.data
b=irisData.target
a_train,a_test,b_train,b_test
=train_test_split(a,b,test_size=0.6,random_state=10)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(a_train,b_train)
print(knn.predict(a_test))
x=knn.predict(a_test)
z=accuracy_score(b_test,x)
print(z)







PROGRAM 4
AIM : Program to implement K-NN classification using any random dataset without using inbuilt
packages.
PROGRAM:
from math import sqrt
def euclidean_distance(row1,row2):
 distance = 0.0
 for i in range(len(row1) - 1):
 distance += (row1[i] - row2[i] )**2
 return sqrt(distance)
def get_neighbors(train,test_row, num_neighbors):
 distances = list()
 for train_row in train:
 dist =euclidean_distance(test_row, train_row)
 distances.append((train_row,dist))
 distances.sort(key=lambda tup:tup[1])
 neighbors = list()
 for i in range(num_neighbors):
 neighbors.append(distances[i][0])
 return neighbors
def predict_classification(train,test_row, num_neighbors):
 neighbors = get_neighbors(train, test_row, num_neighbors)
 output_values = [row[-1] for row in neighbors]
 prediction = max(set(output_values), key=output_values.count)
 return prediction
dataset = [[2.7810836, 2.550537003,0],
 [1.465458936,2.64785645,0],
 [3.56789536,4.568555858,0],
 [1.468956556,3.1464756654,0],
 [5.135663212,2.621254545,0],
 [6.2545449552,5.1436870564,1],
 [8.4365631212,7.56655252636,1],
 [2.146589696,5.66655665555,1],
 [3.4664565252,5.46558866,1],
 [5.895525255,3.46565858,1]]
prediction = predict_classification(dataset,dataset[0], 5)
print('expected %d, Got %d. ' % (dataset[0][-1], prediction))






PROGRAM NO : 5
AIM : Program to implement Naïve Bayes algorithm classification using any standard
dataset available in the public domain and find the accuracy of the algorithm
PROGRAM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[: , -1].values
#Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train ,y_test = train_test_split(x,y,test_size =
0.20,random_state = 20)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
print(x_test)
#training the naive bayes model on the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)
#predicting the test set results
y_pred = classifier.predict(x_test)
print(y_pred)
# making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
print(ac)
print(cm)







PROGRAM NO : 6
AIM : Program to implement linear and multiple regression techniques using any standard
dataset available in the public domain and evaluate its performance (Using Builtin Function)
PROGRAM
import numpy as np
from sklearn.linear_model import LinearRegression
x=np.array([5,15,25,35,45,55]).reshape((-1,1))
y=np.array([5,20,14,32,22,38])
print(x)
print(y)
model=LinearRegression()
model.fit(x,y)
r_sq=model.score(x,y)
print("Coeffient of determination : ",r_sq)
print("Intercept : ",model.intercept_)
print("Slope : ",model.coef_)
y_pred=model.predict(x)
print("Predicting Responce : ",y_pred)






PROGRAM NO : 7
AIM : Program to implement linear and multiple regression techniques using any standard
dataset available in the public domain and evaluate its performance (Using Builtin Function)
and Plot
PROGRAM
import numpy as np
from sklearn.linear_model import LinearRegression
x=np.array([5,15,25,35,45,55]).reshape((-1,1))
y=np.array([5,20,14,32,22,38])
print(x)
print(y)
model=LinearRegression()
model.fit(x,y)
r_sq=model.score(x,y)
print("Coeffient of determination : ",r_sq)
print("Intercept : ",model.intercept_)
print("Slope : ",model.coef_)
y_pred=model.predict(x)
print("Predicting Responce : ",y_pred)
plt.scatter(x,y,color="m",marker="o",s=30)
plt.plot(x,y_pred,color="g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()







PROGRAM NO : 8
AIM : Program to implement linear and multiple regression techniques using any standard
dataset available in the public domain and evaluate its performance (Using Without Builtin
Function)
PROGRAM
import numpy as np
import matplotlib.pyplot as plt
def estimate_coef(x,y):
 n=np.size(x)
 m_x=np.mean(x)
 m_y=np.mean(y)
 SS_xy=np.sum(y*x) - n * m_y * m_x
 SS_xx=np.sum(x*x) - n * m_y * m_x
 b_1 = SS_xx / SS_xx
 b_0 = m_y - b_1 * m_x
 #plot_regression_line()
 return (b_0,b_1)
def plot_regression_line(x, y, b):
 plt.scatter(x, y, color = "m", marker= "o",s=30)
 y_pred = b[0] + b[1] * x
 plt.plot(x, y_pred, color="g")
 plt.xlabel('X')
 plt.ylabel('Y')
 plt.show()
def main():
 x=np.array([0,1,2,3,4,5,6,7,8,9])
 y=np.array([1,3,2,5,7,8,8,9,10,12])
 b=estimate_coef(x,y)
 print("Estimated Coeffiecnts : \n b_0 ={} \n b_1 ={}
".format(b[0],b[1]))
 plot_regression_line(x,y,b)
if __name__ == "__main__":
 main()






PROGRAM NO : 9
AIM : Program to implement multiple regression techniques using any standard dataset
available in the public domain and evaluate its performance (Using Without Builtin
Function)
PROGRAM
import pandas
from sklearn import linear_model
df=pandas.read_csv('cars.csv')
x = df[['Weight','Volume']]
y = df['CO2']
regr=linear_model.LinearRegression()
regr.fit(x.values,y)
pridictedC02=regr.predict([[2300,1300]])
print(pridictedC02)
OUTPUT
PROGRAM NO : 10
AIM : Program to implement multiple regression techniques using any standard dataset
available in the public domain and evaluate its performance (Using Without Builtin
Function)
PROGRAM
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model, metrics
boston=datasets.load_boston(return_X_y=False)
x=boston.data
y=boston.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4)
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print('Prediction',y_pred)
print('Coefficeint : ',reg.coef_)
print('Variance scores : {}'.format(reg.score(x_test,y_test)))
OUTPUT






AIM : Program to implement Decision Tree using any standard dataset available in the
public domain and find the accuracy of the algorithm
PROGRAM
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import plot_tree
df = sns.load_dataset('iris')
print(df.head())
print(df.info())
df.isnull().any()
print(df.shape)
sns.pairplot(data=df,hue='species')
plt.savefig("decison_tree.png")
#correlation matrix
sns.heatmap(df.corr())
plt.savefig("one.png")
target=df['species']
df1=df.copy()
df1=df1.drop('species',axis=1)
print(df1.shape)
print(df1.head())
#defining the attribute
x=df1;
print(target)
#label encoding
le=LabelEncoder()
target=le.fit_transform(target)
print(target)
y=target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print('Training split input- ',x_train.shape)
print('testing split input- ',x_test.shape)
#Defing the Decision tree algorithm
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)
y_pred=dtree.predict(x_test)
print('Classification Report - \n',classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidth=5,annot=True,square=True,cmap="Blues")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
all_sample_title = 'Accuracy Score : {0}'.format(dtree.score(x_test,y_test))
plt.title(all_sample_title,size= 15)
plt.savefig("2.png")
#visualizong the graph without the use of graphics
plt.figure(figsize=(20,20))
dec_tre=plot_tree(decision_tree=dtree,feature_names=df1.columns,class_names=["satosa","vercicolor","v
enginica"],filled=True,precision=4,rounded=True)
plt.savefig("3.png")








PROGRAM NO : 12
AIM : Program to implement K-Means Clustering technique using any standard dataset
available in the public domain
PROGRAM
import matplotlib.pyplot as mtp
import pandas as pd
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values
print(x)
#find elbow
from sklearn.cluster import KMeans
wcss_list=[] #initializing the list for the values of WCSS (sum of squared distance b/w each value)
#using the loop for iteration from 1 to 10
for i in range(1,11):
 kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
 kmeans.fit(x)
 wcss_list.append(kmeans.inertia_)
mtp.plot(range(1,11),wcss_list)
mtp.title("The elbow method Graph")
mtp.xlabel("Number of clusters(k)")
mtp.ylabel("wcss_list")
mtp.show()
#traning thr K-Means model pm a dataset
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_predict=kmeans.fit_predict(x)
print(y_predict)
#Visualizing the Clusters
mtp.scatter(x[y_predict == 0,0],x[y_predict == 0,1],s=100,c='blue',label='Cluster 1')
mtp.scatter(x[y_predict == 1,0],x[y_predict == 1,1],s=100,c='green',label='Cluster 2')
mtp.scatter(x[y_predict == 2,0],x[y_predict == 2,1],s=100,c='red',label='Cluster 3')
mtp.scatter(x[y_predict == 3,0],x[y_predict == 3,1],s=100,c='cyan',label='Cluster 4')
mtp.scatter(x[y_predict == 4,0],x[y_predict == 4,1],s=100,c='magenta',label='Cluster 5')
mtp.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black')
mtp.title("Clusters of Customers")
mtp.xlabel("Annuael Income (K$)")
mtp.ylabel("Spending Score (1-100")
mtp.legend()
mtp.show()






PROGRAM NO : 13
AIM : Programs on convolutional neural network to classify images from any standard
dataset in the public domain
PROGRAM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
np.random.seed(42)
fashion_mnist = keras.datasets.fashion_mnist
(x_train,y_train),(x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape,x_test.shape)
x_train = x_train/255.0
x_test = x_test/255.0
plt.imshow(x_train[1], cmap ='binary')
plt.show()
np.unique(y_test)
class_name = ['Tshirt/Top','Trouser','Pullover','Dress','Cost','Sandal','Shirt','Sneaker','Bag','Ankle
Boot']
n_rows = 5
n_cols = 10
plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))
for row in range(n_rows):
 for col in range(n_cols):
 index = n_cols * row + col
 plt.subplot(n_rows, n_cols,index + 1)
 plt.imshow(x_train[index], cmap='binary', interpolation='nearest')
 plt.axis('off')
 plt.title(class_name[y_train[index]])
plt.show()
model_CNN = keras.models.Sequential()
model_CNN.add(keras.layers.Conv2D(filters=32, kernel_size = 7,padding='same',
activation='relu', input_shape=[28, 28, 1]))
model_CNN.add(keras.layers.MaxPooling2D(pool_size= 2))
model_CNN.add(keras.layers.Conv2D(filters=64, kernel_size = 3,padding='same',
activation='relu'))
model_CNN.add(keras.layers.MaxPooling2D(pool_size= 2))
model_CNN.add(keras.layers.Conv2D(filters=32, kernel_size = 3,padding='same',
activation='relu'))
model_CNN.add(keras.layers.MaxPooling2D(pool_size= 2))
model_CNN.summary()
model_CNN.add(keras.layers.Flatten())
model_CNN.add(keras.layers.Dense(units=128,activation='relu'))
model_CNN.add(keras.layers.Dense(units=64,activation='relu'))
model_CNN.add(keras.layers.Dense(units=10,activation='softmax'))
model_CNN.summary()
model_CNN.compile(loss='sparse_categorical_crossentropy',
optimizer='adam',metrics=['accuracy'])
x_train = x_train[...,np.newaxis]
x_test = x_test[...,np.newaxis]
history_CNN = model_CNN.fit(x_train, y_train, epochs=2,validation_split=0.1)
pd.DataFrame(history_CNN.history).plot()
plt.grid(True)
plt.xlabel('epochs')
plt.ylabel('loss/accuracy')
plt.title('Training and Validation plot')
plt.show()
test_loss, test_accuracy = model_CNN.evaluate(x_test, y_test)
print('Test Loss : {},Test Accuracy : {}'.format(test_loss, test_accuracy))







Program No 14
Program to implement a webcrawler
Code
import requests
from bs4 import BeautifulSoup
url = "https://www.rottentomatoes.com/top/bestofrt/"
headers = {
 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like
Gecko) Chrome/63.0.3239.132 '
 'Safari/537.36 QIHU 360SE '
}
f = requests.get(url, headers=headers)
movies_lst = []
soup = BeautifulSoup(f.content,'lxml')
movies = soup.find('table', {
 'class': 'table'
}).find_all('a')
print(movies)
num = 0
for anchor in movies:
 urls = 'https://www.rottentomatoes.com' + anchor['href']
 movies_lst.append(urls)
print(movies_lst)
num += 1
movie_url = urls
movie_f = requests.get(movie_url, headers=headers)
movie_soup = BeautifulSoup(movie_f.content, 'lxml')
movie_content = movie_soup.find('div', {'class': 'movie_synopsis clamp clamp-6 js-clamp'})
print(num, urls, '\n', 'Movie:' + anchor.string.strip())
print('Movie info:' + movie_content.string.strip()
Output
Program No 15
Code:
from bs4 import BeautifulSoup
import requests
pages_crawled = []
def crawler(url):
 page = requests.get(url)
 soup = BeautifulSoup(page.text, 'html.parser')
 links = soup.find_all('a')
 for link in links:
 if 'href' in link.attrs:
 if link['href'].startswith('/wiki') and ':' not in link['href']:
 if link['href'] not in pages_crawled:
 new_link = f"https:en.wikipedia.org{link['href']}"
 pages_crawled.append(link['href'])
 try:
 with open('data.csv', 'a') as file:
 file.write(f'{soup.title.text};{soup.h1.text};{link["href"]}\n')
 crawler(new_link)
 except:
 continue
crawler('https://en.wikipedia.org')








Program No 16
Implement a program to scrap the webpage of any popular website
Code
import requests
from bs4 import BeautifulSoup
import csv
import lxml
url = "https://www.values.com/inspirational-quotes"
r = requests.get(url)
print(r.content)
soup = BeautifulSoup(r.content, 'lxml')
print(soup.prettify())
quotes = []
table = soup.find('div', attrs={'id': 'all_quotes'})
for row in table.findAll('div',
 attrs={'class': 'col-6 col-lg-3 text-center margin-30px-bottom sm-margin-30pxtop'}):
 quote = {}
 quote['theme'] = row.h5.text
 quote['url'] = row.a['href']
 quote['img'] = row.img['src']
 quote['lines'] = row.img['alt'].split(" #")[0]
 quote['author'] = row.img['alt'].split(" #")[1]
 quotes.append(quote)
filename = 'inspirational_quotes.csv'
with open(filename, 'w', newline=
'') as f:
 w = csv.DictWriter(f, ['theme', 'url', 'img', 'lines', 'author'])
 w.writeheader()
 for quote in quotes:
 w.writerow(quote)
Output
Program No 17
Python program for natural language processing-Ngram(without using in-built functions)
Code
def generate_ngrams(text, WordsToCombine):
 words = text.split()
 output = []
 for i in range(len(words) - WordsToCombine + 1):
 output.append(words[i:1 + WordsToCombine])
 return output
x = generate_ngrams(text="this is a very good book study", WordsToCombine=3)
print(x)







Program No 18
Python program for natural language processing-Ngram(with using in-built functions)
Code
import nltk
nltk.download()
from nltk.util import ngrams
sampletext="This is a very good book to study"
NGRAMS=ngrams(sequence=nltk.word_tokenize(sampletext),n=2)
for grams in NGRAMS:
 print(grams)
Output
Program No 19
Python program for natural language processing- part of speech tagging
Code
from cgitb import text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))
text
txt = "Sukanya, Rajib and Naba are my good friends. " \
 "Sukanya is getting married next year. " \
 "Marriage is a big step in ones's life"\
 "It is both exciting and frightening. " \
 "But friendship is a sacred bond between people." \
 "It is a special kind of love between us. " \
 "Many of you must have tried searching for a friend " \
 "but never found the right one."
tokenized = sent_tokenize(txt)
for i in tokenized:
 wordsList = nltk.word_tokenize(i)
 wordsList = [w for w in wordsList if not w in stop_words]
 tagged = nltk.pos_tag(wordsList)
 print(tagged)
Output
Program no:20
Aim: Program for Natural Language Processing which performs Chunking.
Program
import nltk
nltk.download('punkt')
new = "The big cat ate the little mouse who was after the fresh cheese"
new_tokens = nltk.word_tokenize(new)
print(new_tokens)
new_tag = nltk.pos_tag(new_tokens)
[print(new_tag)]
grammer=r"NP: {<DT>?<JJ>*<NN>}"
chunkParser = nltk.RegexpParser(grammer)
chunked=chunkParser.parse(new_tag)
print(chunked)
chunked.draw()
Output
PROGRAM NO : 21
Aim: Write a python program for natural program language processing with chunking.
Program:
import nltk
#nltk.download('averaged_perception_tagger')
sample_text="""
Rama killed Ravana to save Sita from Lanka.The legend of the Ramayan is the most
popular Indian epic.
A lot of movies and serials have already been shot in several languages here in India
based Ramayana"""
tokenized=nltk.sent_tokenize(sample_text)
for i in tokenized:
 words=nltk.word_tokenize(i)
 tagged_words=nltk.pos_tag(words)
 chunkGram=r"""VB: {}"""
 chunkParser=nltk.RegexpParser(chunkGram)
 chunked=chunkParser.parse(tagged_words)
 print(chunked)
 chunked.draw()

 
