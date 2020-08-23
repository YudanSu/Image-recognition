'''
docstring
'''
# imports 
''' Initial Imports'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from sklearn import decomposition
from sklearn.decomposition import PCA
import matplotlib.image as mpimg
from scipy import interpolate
from sklearn.svm import SVC
import argparse

'''Loading handwritten digit data'''
from sklearn.datasets import load_digits

# functions 
def sample_image(image, num_sample_x, num_sample_y):
    xrange = lambda x: np.linspace(0, 1, x) 
    num_x = image.shape[0]
    num_y = image.shape[1]
    interp_func = interpolate.interp2d(xrange(num_y), xrange(num_x), image, kind='linear')
    return interp_func(xrange(num_sample_y), xrange(num_sample_x))


def interpol_im(im, dim1=8, dim2=8, plot_new_im=False, cmap='binary', grid_off=False):
    original_img = im
    let_im = sample_image(original_img,  dim1, dim2)
    if plot_new_im:
        plt.grid(not grid_off)
        plt.imshow(let_im, interpolation = 'nearest', cmap=cmap)
        plt.show()
    let_im_flat = np.reshape(let_im, (-1,))
    return let_im_flat



def pca_X(X, n_comp=10):
    pca = PCA(n_comp, whiten=True)  
    Xproj = pca.fit_transform(X)
    return pca, Xproj

def rescale_pixel(unseen): 
    rescale_im = []
    min_val = min(unseen)
    max_val = max(unseen)
    for ele in unseen:
        rescale_im.append(int(15 - round(15 * (ele - min_val)/ (max_val - min_val))))
    return np.array(rescale_im).reshape((-1,))

def pca_svm_pred(imfile, md_pca, md_clf, dim1=45, dim2=60):
    imfile_img = mpimg.imread(imfile)[:, :, 0]
    flatten_im = interpol_im(imfile_img, dim1=dim1, dim2=dim2)
    rescale_im = rescale_pixel(flatten_im).reshape((1, -1))
    x_proj = md_pca.transform(rescale_im)
    result = md_clf.predict(x_proj)
    return result

def svm_train(X, y, gamma=0.001, C=100):
    svc = SVC(C=C, gamma=gamma)
    svc.fit(X, y)   
    return svc

def compare(y_pred, y_true, start_idx=60):
    success = 0
    for i in range(len(y_pred)):
        if (y_pred[i] == y_true[i]):
            success += 1
        else:
            print("--------> index, actual digit, svm_prediction: {index} {actual} {pred}".format(
                index=i+start_idx, actual=y_true[i], pred=y_pred[i]))
    
    print("Total number of mid-identifications: {mis}".format(mis=len(y_pred)-success))
    print("Success rate: {rate}".format(rate=float(success)/len(y_pred)))

