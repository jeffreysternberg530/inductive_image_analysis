from PIL import Image
from numpy import *
import numpy as np
from pylab import *
import os
import glob
import io
import matplotlib.pyplot as plt
from skimage import io

def resize(path,format):

    # get list of images from your file path, specifying the type of file you are pulling such as .jpeg or .png
    imlist = glob.glob(path+"/*."+format)
    #print(imlist)
    
    arraylist = [array(Image.open(im)).astype(float) for im in imlist]
    
    shapelist = [array.data.shape for array in arraylist]
    #print(shapelist) #print the shapelist to see what size of images you are working with
    
    shapelistmean = (int(round(mean([shape[0] for shape in shapelist])))),(int(round(mean([shape[1] for shape in shapelist]))))
   
                                                                  
    os.makedirs("resized_images")
    for infile in imlist:
        file, ext = os.path.splitext(infile)
        folder, name = file.split("/")
        im = Image.open(infile).resize(shapelistmean)
        im.save("resized_images/"+ name + "_resized.jpg", "JPEG")
        
def cluster_lister(df, column):
    
    cluster_list = []
    k = df[column].max()
    x = df[column].min()
    
    k=k+1
    
    for i in range(x,k):
    
        cluster_i= df[df[column]==i].sort_values(by= 'image', ascending=True)
        cluster_i_list = cluster_i['image'].tolist()
        cluster_list.append(cluster_i_list)
        
    return cluster_list
    
def image_lister(cluster_list):
    
    image_list = []
    k = len(cluster_list)
    
    for i in range(k):
        img_i = []
        for image in cluster_list[i]:
            img_i.append(io.imread("resized_images/" +image))
            
        image_list.append(img_i)
            
    return image_list
    
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    
def image_cluster_viewer(image_list, cluster_list):
    
    k = len(image_list)
    
    for i in range(k):
        show_images(image_list[i], cols = 3, titles = cluster_list[i])



