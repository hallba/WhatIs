import ctypes

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
	def __init__(self,netfile):
		self.lib = ctypes.CDLL("libjpcnn.so")
		self.network = self.lib.jpcnn_create_network(netfile)
		self.image = ""
		
	def classify(self,imagefile):
		imageHandle = self.lib.jpcnn_create_image_buffer_from_file(imagefile)
		self.image=imagefile
		z = ctypes.c_int(0)
		predictionsLength = ctypes.c_int()
		predictionsLabelsLength = ctypes.c_int()
		predictions = (ctypes.c_float*1000 )()
		predictionsLabels = ctypes.POINTER ( ctypes.c_char_p*1000  )()
		classify = self.lib.jpcnn_classify_image
		classify.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)), ctypes.POINTER(ctypes.c_int) )
		
		self.lib.jpcnn_classify_image(self.network, imageHandle, z, z, ctypes.cast(predictions, ctypes.POINTER(ctypes.POINTER(ctypes.c_float))), predictionsLength, ctypes.cast(predictionsLabels,ctypes.POINTER(ctypes.c_char_p)) , predictionsLabelsLength )
		self.predictionResults = predictions
		self.predictionNames = predictionsLabels
		self.lib.jpcnn_destroy_image_buffer(imageHandle)
		
		return predictionsLength, predictionsLabelsLength, predictions, predictionsLabels
