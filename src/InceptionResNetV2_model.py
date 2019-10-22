from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


def model_imagenet(img_width, img_height, num_classes, x_all=None, y_all=None,
				   optimizer=None):
	base_model = InceptionResNetV2(weights='imagenet', include_top=False,
								   input_shape=(img_height, img_width, 3))

	x = base_model.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)

	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
		else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def InceptionResNetV2_imagenet():
	return {"model": model_imagenet, "name": "InceptionResNetV2_imagenet", "shape": (224, 224, 3),
			"pretrained": True}


def model_sinpesos(img_width, img_height, num_classes, x_all=None, y_all=None,
				   optimizer=None):
	base_model = InceptionResNetV2(weights=None, include_top=False,
								   input_shape=(img_height, img_width, 3))

	x = base_model.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)

	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
		else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def InceptionResNetV2_sinpesos():
	return {"model": model_sinpesos, "name": "InceptionResNetV2_sinpesos"}


def model_imagenet_transfer(img_width, img_height, num_classes, x_all=None, y_all=None,
							optimizer=None):
	base_model = InceptionResNetV2(weights='imagenet', include_top=False,
								   input_shape=(img_height, img_width, 3))

	x = base_model.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)

	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
		else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def InceptionResNetV2_transfer():
	return {"model": model_imagenet_transfer, "name": "InceptionResNetV2_transfer", "shape": (224, 224, 3),
			"transfer": True}
