
import cv2
import os 
import pickle
import numpy as np
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img
import sys
from tensorflow import keras


class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName

		# if the layer name is None, attempt to automatically find
		# the target output layer
		if self.layerName is None:
			self.layerName = self.find_target_layer()

	def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(self.model.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
				return layer.name

		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

	def compute_heatmap(self, image, eps=1e-8):
		# construct our gradient model by supplying (1) the inputs
		# to our pre-trained model, (2) the output of the (presumably)
		# final 4D layer in the network, and (3) the output of the
		# softmax activations from the model
		gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer('time_distributed').output, 
				self.model.output])

		# record operations for automatic differentiation
		with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			loss = predictions[:, self.classIdx]
			#print("is this loss? "+ str(loss))

		# use automatic differentiation to compute the gradients
		grads = tape.gradient(loss, convOutputs)
		#print("¡Hola!")
		#print("grads" +str(grads.shape))
		# compute the guided gradients
		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads

		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
		#print("A" +str(convOutputs.shape))
		#print("A" +str(guidedGrads.shape))
		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]
		#print("B" +str(convOutputs.shape))
		#print("B" +str(guidedGrads.shape))
		
		heatmap_final = []
		print(convOutputs.shape)
		print(guidedGrads.shape)

		weights = tf.reduce_mean(guidedGrads, axis=(0))
		#cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=1)
		cam = tf.multiply(weights,convOutputs)
		#cam = tf.reduce_sum(weights,axis=-1)
		print(image.shape)
		print(weights.shape)
		print(cam.shape)
		(w, h) = (image.shape[2], image.shape[1])
		print(str(w) + " --- " + str(h))
		heatmap = cv2.resize(cam.numpy(), (w, h))
		print("heatmap"+ str(heatmap.shape))
		#print("heatmap"+ str(heatmap[0]))
		#print("heatmap"+ str(cam[0]))
		#print("heatmap"+ str(cam[1]))
		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom
		heatmap = (heatmap * 255).astype("uint8")

		return heatmap
		'''
		for i in range(0,len(convOutputs)):
			conv = convOutputs[i]
			guided = guidedGrads[i]

			#print("C1" +str(conv.shape))
			#print("C2" +str(guided.shape))			
		# compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
			weights = tf.reduce_mean(guided, axis=(0,1))
			cam = tf.reduce_sum(tf.multiply(weights, conv), axis=-1)

		# grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
		#print(cam.numpy().shape)
		#print("Yikes")
			(w, h) = (image.shape[3], image.shape[2])
			heatmap = cv2.resize(cam.numpy(), (w, h))
			#print("heatmap"+ str(heatmap.shape))
		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer
			numer = heatmap - np.min(heatmap)
			denom = (heatmap.max() - heatmap.min()) + eps
			heatmap = numer / denom
			heatmap = (heatmap * 255).astype("uint8")

			heatmap_final.append(heatmap)
		# return the resulting heatmap to the calling function
		return np.asarray(heatmap_final)
		'''

	def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_VIRIDIS):
		# apply the supplied color map to the heatmap and then
		# overlay the heatmap on the input image
		print("En Overlay Heatmap")
		heatmap = cv2.applyColorMap(heatmap, colormap)
		print(heatmap.shape)
		print(image.shape)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

		# return a 2-tuple of the color mapped heatmap and the output,
		# overlaid image
		return (heatmap, output)


test_labels = np.load('/home/elbisbe/TFM_MUIT/Clasificador-trafico/ModeloCNN_LSTM/test_labels.npy')
test_images = np.load('/home/elbisbe/TFM_MUIT/Clasificador-trafico/ModeloCNN_LSTM/test_images.npy')

modelo = keras.models.load_model('/home/elbisbe/TFM_MUIT/Clasificador-trafico/ModeloCNN_LSTM/CNN_2D.h5')

print(test_labels[0].shape)
print(test_labels[0])
print(test_images[0:1,:,:,:].shape)

pred = modelo.predict(test_images[0:1,:,:,:], verbose=1, batch_size=1)
prediction = pred > 0.5

res = np.where(prediction[0] == True)

print(prediction)
print("Predicho:" + str(res))
print("Real:" + str(test_labels[0]))

cam = GradCAM(modelo, 0)
heatmap = cam.compute_heatmap(test_images[0:1,:,:,:])

A = test_images[0] #(32,32,1)
B = cv2.normalize(src=A, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#h_A = np.expand_dims(heatmap,axis=2)
#print(heatmap.shape)
#print(B.shape)
#print(np.expand_dims(heatmap,axis=2).shape)

np.matmul(B, heatmap)
#np.matmul(np.expand_dims(B,1),np.expand_dims(heatmap,1))

B = np.stack((B,)*3, axis=-1)

(h_B, output) = cam.overlay_heatmap(heatmap, B, alpha=0.5)
array_to_img(test_images[0]).save("OriginalFile.png")
array_to_img(output).save("File.png")