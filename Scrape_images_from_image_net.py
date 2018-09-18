import numpy as np
import urllib.request
from PIL import Image
from scipy.misc import imread,imresize,imsave
import os

def gather_images(url):
	image_urls = urllib.request.urlopen(url).read().decode()
	'''if not os.path.exists('images')
	os.makedirs('images')'''
	pic_num = 0
	for item in image_urls.split('\n'):
		try:
			pic_num =pic_num+1
			print(pic_num)
			path = "images/"+str(pic_num)+'.jpg'
			try:
				urllib.request.urlretrieve(item,path)
			except urllib2.HTTPError as err:
				print(str(err.code))
			print("Ok")
			if(pic_num == 1):
				Image.open(path).show()
			#image = imread(path)
			#scaled_image = imresize(image,(200,200,3))
			#imsave(path,scaled_image)
			if(pic_num == 1):
				Image.open(path).show()

		except Exception as e:
			print(str(e))
			
gather_images("http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00007846")
