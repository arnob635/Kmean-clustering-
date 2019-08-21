import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
style.use('ggplot')

d = pd.read_csv("Dataset.csv") #reading from csv

X = d.to_numpy() #converting csv to array

#X=np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[3,2],[1.5,5.8],[5,5],[2,8],[1,2.6],[9,1]])

#plt.scatter(X[:,0], X[:,1], s=150)
#plt.show()

colors = 10*["g","r","c","b","k"]

number =int(input("Enter number of clusters: "))

class Kmeans:
    #Implementing Kmeans algorithm.#


    def __init__(self, k=number, tol=0.001, max_iter=100):
        self.k =k
        self.tol = tol
        self.max_iter = max_iter


    def fit (self,data):
        self.centroids={}

        for i in range (self.k):
            self.centroids[i]=data[i]

        for i in range (self.max_iter):
            self.classifications={}

            for i in range (self.k):
                self.classifications[i]= []
               
        
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid])for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis = 0)
            optimized = True

            for c in self.centroids:
                original_cenrtoid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_cenrtoid)/original_cenrtoid*100.00)>self.tol: 
                    print(np.sum((current_centroid-original_cenrtoid)/original_cenrtoid*100.0))
                    optimized = False

            if optimized:
                break


    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid])for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = Kmeans()
clf.fit(X)


for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],marker = "o", color="k",s=200,linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1], marker="x",color=color,s=150,linewidths=5)





plt.show()

