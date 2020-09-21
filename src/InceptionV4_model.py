from keras.layers.convolutional import AveragePooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.models import Model

from InceptionV4 import InceptionV4

from keras.applications.inception_v3 import preprocess_input

def process(x):
	return preprocess_input(x)

def model_imagenet(img_width,
                   img_height,
                   num_classes,
                   x_all=None,
                   y_all=None,
                   optimizer=None):
	base_model = InceptionV4(
	    weights='imagenet',
	    include_top=False,
	    input_shape=(img_height, img_width, 3))

	x = base_model.output

	# 1 x 1 x 1536
	x = AveragePooling2D((8,8), padding='valid')(x)
	x = Dropout(0.2)(x)
	x = Flatten()(x)
	# 1536
	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
                    else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def InceptionV4_imagenet():
	return {
	    "model": model_imagenet,
	    "name": "InceptionV4_imagenet",
	    "shape": (299, 299, 3),
		"pretrained": True,
		"preprocessing": process
	}


def model_sinpesos(img_width,
                   img_height,
                   num_classes,
                   x_all=None,
                   y_all=None,
                   optimizer=None):
	base_model = InceptionV4(
	    weights=None, include_top=False, input_shape=(img_height, img_width, 3))

	x = base_model.output

	# 1 x 1 x 1536
	x = AveragePooling2D((8,8), padding='valid')(x)
	x = Dropout(0.2)(x)
	x = Flatten()(x)
	# 1536
	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
                    else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def InceptionV4_sinpesos():
	return {
	    "model": model_sinpesos,
	    "name": "InceptionV4_sinpesos",
	    "shape": (299, 299, 3),
		"preprocessing": process
	}


