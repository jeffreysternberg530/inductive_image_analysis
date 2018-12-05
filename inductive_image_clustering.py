from PIL import Image
from numpy import *
import numpy as np
from pylab import *
import os
import glob
import io
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage  
from scipy.cluster.hierarchy import fcluster
from image_functions import resize, cluster_lister, image_lister, image_cluster_viewer

#Use the resize function to choose what directory of images you are resizing and are
#are running the image analysis on, followed by the format of the images your are resizing

#The function will automatically re-save them in a folder called "resized_images",
#in the same directory with the original title plus "_resized".jpg  

resize("image_samples_copy","jpeg")

#We make an image list from our resized images folder
data = os.listdir("resized_images")

imlist = []
for photos in data:
    if photos.endswith(".jpg"):
        imlist.append(photos)
      
#Here we flatten the images and convert them to numpy arrays
img = []
for d in imlist:
    img.append(io.imread("resized_images/" +d).mean(axis=2).flatten())
    
img = np.array(img)

#Principle Components Analysis w/ 30 Components 
sklearn_pca = PCA(n_components=30, random_state=4)
Y_sklearn = sklearn_pca.fit_transform(img)

#Make a dataframe to store our image names and their corresponding clusters
df = pd.DataFrame()

df["image"] = imlist

#Here we perform a hierarchical clustering of the PCA components and return a dendrogram 
#with all of these clusters

linked = linkage(Y_sklearn, 'ward')

#dendrogram plotter
labelList = ['' for i in range(len(imlist))]

plt.figure(figsize=(10, 7))  
dendrogram(linked,  
            orientation='right',
            labels = labelList,
            distance_sort='descending',
            show_leaf_counts=True,
            show_contracted=True,
            color_threshold=32000) #this value is the same as the cutoff distance
plt.xlabel('Distance', fontsize=24)
plt.xticks(fontsize = 18)
plt.tight_layout()  # fixes margins

plt.axvline(x=32000) #plot vertical line showing cutoff point assigning clusters

plt.show()

#Here we run the hierarchical clustering, choosing a cutoff point, and making a dataframe
#column assigning which cluster each image falls under at this cutoff point 

max_d = 32000
clusters = fcluster(linked, max_d, criterion='distance')
    
df['cluster'] = clusters


#Show the Dataframe of Images by Cluster
print(df.sort_values(by= ['cluster','image'], ascending=True))

#List of Function Commands Printing Out Images by Cluster
cluster_list = cluster_lister(df, 'cluster')
image_list = image_lister(cluster_list)
image_cluster_viewer(image_list, cluster_list)



