from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Model


def model_imagenet(img_width, img_height, num_classes, x_all=None, y_all=None,
				   optimizer=None):
	base_model = ResNet50(weights='imagenet', include_top=False,
						  input_shape=(img_height, img_width, 3))

	x = base_model.output
	x = Flatten()(x)

	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
		else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def ResNet50_imagenet():
	return {"model": model_imagenet, "name": "ResNet50_imagenet", "shape": (224, 224, 3),
			"pretrained": True}


def model_sinpesos(img_width, img_height, num_classes, x_all=None, y_all=None,
				   optimizer=None):
	base_model = ResNet50(weights=None, include_top=False,
						  input_shape=(img_height, img_width, 3))

	x = base_model.output
	x = Flatten()(x)

	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
		else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	return model, base_model


def ResNet50_sinpesos():
	return {"model": model_sinpesos, "name": "ResNet50_sinpesos"}


def test_model():
	model1 = ResNet50()
	print('top', len(model1.layers))
	print(model1.summary())

	model2 = ResNet50(include_top=False, input_shape=(224, 224, 3))
	print('sin top', len(model2.layers))


def model_imagenet_transfer(img_width, img_height, num_classes, x_all=None, y_all=None,
							optimizer=None):
	base_model = ResNet50(weights='imagenet', include_top=False,
						  input_shape=(img_height, img_width, 3))

	x = base_model.output
	x = Flatten()(x)

	predictions = Dense(1, activation='sigmoid')(x) if num_classes == 2 \
		else Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	# fine tune
	for layer in base_model.layers:
		layer.trainable = False

	return model, base_model


def ResNet50_transfer():
	return {"model": model_imagenet_transfer, "name": "ResNet50_transfer", "shape": (224, 224, 3),
			"transfer": True}
