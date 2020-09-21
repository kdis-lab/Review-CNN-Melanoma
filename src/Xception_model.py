from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


def model_imagenet(img_width, img_height, num_classes, x_all=None, y_all=None,
				   optimizer=None):
	base_model = Xception(weights='imagenet', include_top=False,
						  input_shape=(img_height, img_width, 3))

	x = base_model.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)

	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
		else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def Xception_imagenet(optimizer=None):
	return {"model": model_imagenet, "name": "Xception_imagenet", "shape": (299, 299, 3),
			"pretrained": True, "optimizer": optimizer}


def model_sinpesos(img_width, img_height, num_classes, x_all=None, y_all=None,
				   optimizer=None):
	base_model = Xception(weights=None, include_top=False,
						  input_shape=(img_height, img_width, 3))

	x = base_model.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)

	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
		else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def Xception_sinpesos(optimizer=None):
	return {"model": model_sinpesos, "name": "Xception_sinpesos", "optimizer": optimizer}


def model_imagenet_transfer(img_width, img_height, num_classes, x_all=None, y_all=None,
							optimizer=None):
	base_model = Xception(weights='imagenet', include_top=False,
						  input_shape=(img_height, img_width, 3))

	x = base_model.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)

	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
		else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def Xception_transfer(optimizer=None):
	return {"model": model_imagenet_transfer, "name": "Xception_transfer", "shape": (299, 299, 3),
			"transfer": True, "optimizer": optimizer}
