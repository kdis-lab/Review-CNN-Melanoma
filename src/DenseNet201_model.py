from keras.applications.densenet import DenseNet201
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import time

from keras.applications.densenet import preprocess_input

def process(x):
	return preprocess_input(x)

def model_imagenet(img_width, img_height, num_classes, x_all=None, y_all=None,
				   optimizer=None):
	base_model = DenseNet201(weights='imagenet', include_top=False,
							 input_shape=(img_height, img_width, 3))

	binary = num_classes == 2

	x = base_model.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)
	predictions = Dense(1 if binary else num_classes,
						activation='sigmoid' if binary else 'softmax', name='fc1000')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def DenseNet201_imagenet():
	return {"model": model_imagenet, "name": "DenseNet201_imagenet", "shape": (224, 224, 3),
			"pretrained": True, "preprocessing": process}


def model_sinpesos(img_width, img_height, num_classes, x_all=None, y_all=None,
				   optimizer=None):
	base_model = DenseNet201(weights=None, include_top=False,
							 input_shape=(img_height, img_width, 3))

	binary = num_classes == 2

	x = base_model.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)
	predictions = Dense(1 if binary else num_classes,
						activation='sigmoid' if binary else 'softmax', name='fc1000')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def DenseNet201_sinpesos():
	return {"model": model_sinpesos, "name": "DenseNet201_sinpesos", "preprocessing": process}


def model_imagenet_transfer(img_width, img_height, num_classes, x_all=None, y_all=None,
							optimizer=None):
	base_model = DenseNet201(weights='imagenet', include_top=False,
							 input_shape=(img_height, img_width, 3))

	binary = num_classes == 2

	x = base_model.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)
	predictions = Dense(1 if binary else num_classes,
						activation='sigmoid' if binary else 'softmax', name='fc1000')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def DenseNet201_transfer():
	return {
		"model": model_imagenet_transfer, 
		"name": "DenseNet201_transfer", 
		"shape": (224, 224, 3),
		"transfer": True		
	}
