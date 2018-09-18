import numpy as np
import urllib.request
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.misc import imread,imresize,imsave
from scipy import ndimage

tf.reset_default_graph()


def raw_images_features(image):
	return image.flatten('F')

def color_histogram_of_images(image):
	image = np.array(image)
	return ndimage.histogram(image,0,1,10 )

def grey_scale_features(image):
	img = imresize(image,(128,128))
	return img.flatten('F')
	
raw_images= []

features =[]
greyscales = []
pic_num = 0
arr = os.listdir("train1")
print(np.array(arr).size)
labels = np.zeros((np.array(arr).size,1),dtype=float, order='F')
	
for filename in arr:
	try:
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
		if label == "cat":
			labels[pic_num] = 0
		else:
			labels[pic_num] = 1
		
		pic_num =pic_num+1
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

img_height = 128
img_width  = 128
value = trainRI[0].size	
print(value)
print('yes')
print(img_height*img_height)
def getModel():

	b_size = 60
	img_height = 128
	img_width  = 128
	classes    = 2
	value = trainRI[0].size

	weights = {
		'wc1': tf.get_variable("wc1",shape=[value,1024],initializer=tf.contrib.layers.xavier_initializer()),
		'wc2': tf.get_variable("wc2",shape=[1024,128],initializer=tf.contrib.layers.xavier_initializer()),
		'wc3': tf.get_variable("wc3",shape=[128,2],initializer=tf.contrib.layers.xavier_initializer()),
	}
  
	biases = {
		'bc1': tf.get_variable("bc1",initializer=tf.zeros([1024])),
		'bc2': tf.get_variable("bc2",initializer=tf.zeros([128])),
		'bc3': tf.get_variable("bc3",initializer=tf.zeros([2])),

	}

	#Input : img_height * img_width
	xi = tf.placeholder(tf.float32,[None,value])
	yi = tf.placeholder(tf.float32,[None,1,2])
	is_training = tf.placeholder(tf.bool,[])
	
	with tf.variable_scope("dense1") as scope:
		dense1 = tf.matmul(xi,weights['wc1'])+biases['bc1']
		act1 = tf.nn.relu(dense1)
	with tf.variable_scope("dense2") as scope:
		dense2 = tf.matmul(act1,weights['wc2'])+biases['bc2']
		act2 = tf.nn.relu(dense2)
	
	with tf.variable_scope("dense3") as scope:
		dense3 = tf.matmul(act2,weights['wc3'])+biases['bc3']
			
	Predict = tf.nn.softmax(dense3)
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense3, labels=yi))
	error = tf.reduce_sum(cross_entropy)
	train_step = tf.train.AdamOptimizer().minimize(error)
	
	return (xi,yi,is_training),train_step,error,Predict
	
(xi,yi,is_training),train_step,cost,Predict = getModel()
	
init = tf. global_variables_initializer()
b_size = 60
img_height = 128
img_width  = 128
classes    = 2
with tf.Session() as sess:
	sess.run(init)
	for allot in range(0,trainRI.shape[0],b_size):
		x_raw = trainRI[allot:allot+b_size]
		trainRL_int = trainRL.astype(int)
		new_labels=np.eye(classes)[trainRL_int]
		y_raw = new_labels[allot:allot+b_size]
		[pred] = sess.run([Predict],feed_dict={xi:x_raw,yi:y_raw,is_training:True})
	#print("Testing")
	c=0;g=0
	for i in range(0,testRI.shape[0]):
		x_raw = testRI[i] # It will just have the proper shape
		y_raw = testRL[i]
		x_raw = x_raw.reshape(1,len(x_raw))
		[pred]=sess.run([Predict],feed_dict={xi: x_raw, is_training: False})
		#print(pred)
		if np.argmax(y_raw)==np.argmax(pred):
			g+=1
		c+=1
	print("Accuracy: "+str(1.0*g/c))