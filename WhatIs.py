from ctypes import *
import os.path

'''
void jpcnn_classify_image(void* networkHandle, void* inputHandle, unsigned int flags, int layerOffset, float** outPredictionsValues, int* outPredictionsLength, char*** outPredictionsNames, int* outPredictionsNamesLength);

networkHandle = jpcnn_create_network(networkFileName);
  if (networkHandle == NULL) {
    fprintf(stderr, "DeepBeliefSDK: Couldn't load network file '%s'\n", networkFileName);
    return 1;
  }

  imageHandle = jpcnn_create_image_buffer_from_file(imageFileName);
  if (imageHandle == NULL) {
    fprintf(stderr, "DeepBeliefSDK: Couldn't load image file '%s'\n", imageFileName);
    return 1;
  }

  jpcnn_classify_image(networkHandle, imageHandle, 0, 0, &predictions, &predictionsLength, &predictionsLabels, &predicti
onsLabelsLength);
  jpcnn_destroy_image_buffer(imageHandle);
'''

class deepBelief():
	def __init__(self,netfile='networks/jetpac.ntwk'):
		self.lib = CDLL("libjpcnn.so")
		self.network = self.lib.jpcnn_create_network(netfile)
		self.image = ""
		self.best = ""
		self.bestLikelihood = 0	
	def classify(self,imagefile):
		if not os.path.isfile(imagefile):
			print "No such file", imagefile
			return "",0
		imageHandle = self.lib.jpcnn_create_image_buffer_from_file(imagefile)
		self.image=imagefile
		z = c_int(0)
		predictionsLength = c_int()
		predictionsLabelsLength = c_int()
		predictions = (POINTER(c_float))()
		predictionsLabels = (POINTER(c_char_p))()

		classify = self.lib.jpcnn_classify_image
		classify.argtypes = (c_void_p, c_void_p, c_int, c_int, POINTER(POINTER(c_float)), POINTER(c_int), POINTER(POINTER(c_char_p)), POINTER(c_int) )
		
		classify(self.network, imageHandle, z, z, byref(predictions), predictionsLength, byref(predictionsLabels), predictionsLabelsLength)
		
		self.predictionResults = predictions
		self.predictionNames = predictionsLabels
		self.lib.jpcnn_destroy_image_buffer(imageHandle)
		
		bestId = 0
		bestValue = 0
		for i in range(predictionsLength.value):
			if predictions[i]>bestValue:
				bestValue = predictions[i]
				bestId=i
		
		self.best = self.predictionNames[bestId]
		self.bestLikelihood = self.predictionResults[bestId]
		
		return self.best, self.bestLikelihood

if __name__ == "__main__":
	print "Testing WhatIs"
	db = deepBelief()
	print "==Testing a file that doesn't exist=="
	#Test a non-existant file
	db.classify("Fake.jpg")
	print db.best, db.bestLikelihood
	print "==Testing a photo of a hippo=="
	#Test the hippo
	db.classify("tests/Hippo.jpg")
	print db.best, db.bestLikelihood
