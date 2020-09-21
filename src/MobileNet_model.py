from keras.applications.mobilenet import MobileNet
from keras.layers import Reshape, Conv2D, Activation, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras import backend as K

from keras.applications.mobilenet import preprocess_input

def process(x):
	return preprocess_input(x)

def model_imagenet(img_width, img_height, num_classes, x_all=None, y_all=None,
				   optimizer=None):
	base_model = MobileNet(weights='imagenet', include_top=False,
						   input_shape=(img_height, img_width, 3))
	alpha = 1.0
	dropout = 1e-3

	constant = 1 if num_classes == 2 else num_classes

	if K.image_data_format() == 'channels_first':
		shape = (int(1024 * alpha), 1, 1)
	else:
		shape = (1, 1, int(1024 * alpha))

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Reshape(shape, name='reshape_1')(x)
	x = Dropout(dropout, name='dropout')(x)
	x = Conv2D(constant, (1, 1),
			   padding='same', name='conv_preds')(x)
	x = Activation('sigmoid' if num_classes == 2 else 'softmax', name='activation_layer')(x)
	predictions = Reshape((constant,), name='reshape_2')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def MobileNet_imagenet():
	return {"model": model_imagenet, "name": "MobileNet_imagenet", "shape": (224, 224, 3),
		   "pretrained": True, "preprocessing": process}


def model_sinpesos(img_width, img_height, num_classes, x_all=None, y_all=None,
				   optimizer=None):
	base_model = MobileNet(weights=None, include_top=False,
						   input_shape=(img_height, img_width, 3))
	alpha = 1.0
	dropout = 1e-3

	constant = 1 if num_classes == 2 else num_classes

	if K.image_data_format() == 'channels_first':
		shape = (int(1024 * alpha), 1, 1)
	else:
		shape = (1, 1, int(1024 * alpha))

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Reshape(shape, name='reshape_1')(x)
	x = Dropout(dropout, name='dropout')(x)
	x = Conv2D(constant, (1, 1),
			   padding='same', name='conv_preds')(x)
	x = Activation('sigmoid' if num_classes == 2 else 'softmax', name='activation_layer')(x)
	predictions = Reshape((constant,), name='reshape_2')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def MobileNet_sinpesos():
	return {"model": model_sinpesos, "name": "MobileNet_sinpesos", "shape": (224, 224, 3), "preprocessing": process}


