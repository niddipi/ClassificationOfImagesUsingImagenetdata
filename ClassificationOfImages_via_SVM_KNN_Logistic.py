import numpy as np
import urllib.request
from PIL import Image
from matplotlib import pyplot as plt
from scipy.misc import imread,imresize,imsave
from scipy import ndimage
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def raw_images_features(image):
	return image.flatten('F')

def color_histogram_of_images(image):
	image = np.array(image)
	return ndimage.histogram(image,0,1,10 )

def grey_scale_features(image):
	img = imresize(image,(128,128))
	return img.flatten('F')
	
raw_images= []
labels =[]
features =[]
greyscales = []
pic_num = 0
arr = os.listdir("train1")
print(np.array(arr).size)

	
for filename in arr:
	try:
		pic_num =pic_num+1
		print(pic_num)
		image=Image.open('./train1/'+filename)
		greyimage = Image.open('./train1/'+filename).convert('L')
		scaled_image = imresize(image,(128,128))
		pixels= raw_images_features(scaled_image)
		label = filename.split(os.path.sep)[-1].split(".")[0]		
		hist  = color_histogram_of_images(image)
		greyscale = grey_scale_features(greyimage)
		raw_images.append(pixels)
		features.append(hist)
		greyscales.append(greyscale)
		labels.append(label)
	except Exception as e:
			print(str(e))
	

raw_images = np.array(raw_images)
features = np.array(features)
labels = np.array(labels)
print(raw_images.size)
print(features.size)
print(labels.size)

(trainRI, testRI, trainRL, testRL) = train_test_split(
	raw_images, labels, test_size=0.15, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.15, random_state=42)
(traingrey, testgrey, trainGL, testGL) = train_test_split(
	greyscales, labels, test_size=0.15, random_state=42)

	
# k-NN
print("\n")
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=2)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] k-NN classifier: k=%d" % 2)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# k-NN
print("\n")
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=2)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] k-NN classifier: k=%d" % 2)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))


#SVC
print("\n")
print("[INFO] evaluating raw pixel accuracy...")
model = SVC(max_iter=1000,class_weight='balanced')
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] SVM-SVC raw pixel accuracy: {:.2f}%".format(acc * 100))


#SVC
print("\n")
print("[INFO] evaluating histogram accuracy...")
model = SVC(max_iter=1000,class_weight='balanced')
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] SVM-SVC histogram accuracy: {:.2f}%".format(acc * 100))
'''
'''
print("\n")
print("[INFO] evaluating raw pixel accuracy...")
model =  LogisticRegression(C=1e5)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] logistic regression classifier: k=%d" % 2)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

print("\n")
print("[INFO] evaluating histogram accuracy...")
model = LogisticRegression(C=1e5)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] LogisticRegression histogram accuracy: {:.2f}%".format(acc * 100))

print("[INFO] evaluating histogram accuracy...")
model = LogisticRegression(C=1e5)
model.fit(traingrey, trainGL)
acc = model.score(testgrey,testGL)
print("[INFO] LogisticRegression histogram accuracy: {:.2f}%".format(acc * 100))
