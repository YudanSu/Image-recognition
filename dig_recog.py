from pattern_recog_func import *
from sklearn.datasets import load_digits
from sklearn.svm import SVC

##a

digit_data = load_digits()
dig_img = digit_data.images
X = digit_data.data
Y = digit_data.target

print (X.shape, Y.shape)
x_train = X[:60]
y_train = Y[:60]
x_valid = X[60:80]
y_valid = Y[60:80]
model = svm_train(x_train, y_train)
y_pred = model.predict(x_valid)
accuracy = compare(y_pred, y_valid)


##b
unseen = mpimg.imread("./unseen_dig.png")[:, :, 0]
unseen_flat = interpol_im(unseen, plot_new_im=False, grid_off=True)
unseen_flat = rescale_pixel(unseen_flat).reshape((1, -1))
result = model.predict(unseen_flat)
print(result)

##c
md_pca, x_train_proj = pca_X(x_train, n_comp=2)
model_pca = svm_train(x_train_proj, y_train)
imfile = "./unseen_dig.png"
unseen_pred = pca_svm_pred(imfile, md_pca, model_pca, dim1=8, dim2=8)
print(unseen_pred)

