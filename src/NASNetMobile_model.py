from keras.applications.nasnet import NASNetMobile
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model


def model_imagenet(img_width, img_height, num_classes, x_all=None, y_all=None,
				   optimizer=None):
	base_model = NASNetMobile(input_shape=(img_height, img_width, 3),
                  include_top=False,
                  weights='imagenet')
	
	binary = num_classes == 2
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1 if binary else 2, activation='sigmoid' if binary else 'softmax', name='predictions')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=x)
	
	return model, base_model

def NASNetMobile_imagenet():
	return {"model": model_imagenet, "name": "NASNetMobile_imagenet", "shape": (224, 224, 3),
		   "pretrained": True}

def NASNetMobile_transfer():
	return {"model": model_imagenet, "name": "NASNetMobile_transfer", "shape": (224, 224, 3), 
			"transfer": True}

		   
def model_sinpesos(img_width, img_height, num_classes, x_all=None, y_all=None,
				   optimizer=None):
	base_model = NASNetMobile(input_shape=(img_height, img_width, 3),
                  include_top=False,
                  weights=None)
	
	binary = num_classes == 2
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1 if binary else 2, activation='sigmoid' if binary else 'softmax', name='predictions')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=x)

	return model, base_model

def NASNetMobile_sinpesos():
	return {"model": model_sinpesos, "name": "NASNetMobile_sinpesos", "shape": (224, 224, 3)}


