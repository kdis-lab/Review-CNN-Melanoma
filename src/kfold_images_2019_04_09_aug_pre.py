from sklearn.model_selection import StratifiedKFold
import numpy
import keras
import json
import os
from os.path import join
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight
import sys
import time
from multiprocessing import Pool
from skimage.transform import resize

from callback_measures import SingleLabel, Binary, AccumulativeTime, EpochsRegister
from callback_measures_aug import SingleLabelAug, BinaryAug
from callback_measures_aug_avg import SingleLabelAugAvg, BinaryAugAvg
from balance import image_aug_balance
import sklearn

from keras.applications.inception_v3 import preprocess_input


def temp_file():
	tempdir = '../tempdatafold'
	if not os.path.exists(tempdir):
		os.makedirs(tempdir)
	tempfile = join(tempdir, '-'.join([str(i) for i in numpy.random.randint(1000000, size=5)] + ['.npy']))
	return tempfile

def resize_images(x_resize, shape):
	# before x_resize = x_resize.astype('float32') / 255
	transform = []
	for current_image in x_resize:
		transform.append(resize(current_image, shape))
	transform = numpy.asarray(transform)
	return transform
	
def remove(data, indices, batch):
	'''
	2018-09-03
	data is the labels, ex: [0,0,0,0,1,1,1,1]
	data.shape[0] >= batch
	return indices to remove of data
	indices of this data or the indices of the data in another numpy
	'''	
	array = data.astype('int64')
	nremove = array.shape[0] % batch
	counts = numpy.bincount(array)
	remove = []
	while nremove > 0:
		max = numpy.argmax(counts)
		for index in range(len(array)):
			if (array[index] == max) & (indices[index] not in remove):
				remove.append(indices[index])
				nremove = nremove - 1
				break
		counts[max] = counts[max] - 1
	return remove



def one_fold(mapitems):
	# ocupa la memoria gpu necesaria
	from keras import backend as k
	import tensorflow as tf
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	try:
		k.set_session(sess)
	except:
		print('No session available')
	
	fold = mapitems['fold']
	epochs = mapitems['epochs']
	epochs_pre = mapitems['epochs_pre']
	batch = mapitems['batch']
	model = mapitems['model']
	img_width = mapitems['img_width']
	img_height = mapitems['img_height']
	NUM_CLASSES = mapitems['NUM_CLASSES']
	optimizer = mapitems['optimizer']
	dirpath = mapitems['dirpath']
	metric = mapitems['metric']
	metric_mode = mapitems['metric_mode']
	type_class_weight = mapitems['type_class_weight']
	gpu = mapitems['gpu']

	pathX = mapitems['X']
	Y = mapitems['Y']
	pathX2 = mapitems['X2']
	Y2 = mapitems['Y2']
	
	X = numpy.load(pathX)
	os.remove(pathX)
	X2 = numpy.load(pathX2)
	os.remove(pathX2)
	
	X,Y,indices = image_aug_balance(X,Y,0)
	indices = sklearn.utils.shuffle(numpy.arange(Y.shape[0]))
	X = X[indices]
	Y = Y[indices]
	
	# se eliminan instancias de la mayor clase para solucionar el bug de keras
	# instancias de la clase mayoritaria							
	nremove = remove(Y, [i for i in range(len(Y))], batch)
	newtrain = [i for i in range(len(Y)) if i not in nremove]
	X = X[newtrain]
	Y = Y[newtrain]
	
	print('train', X.shape)

	# test
	# callbacks
	X2_aug, Y2_aug, indices2 = image_aug_balance(X2,Y2,10)
	print('test', X2.shape, 'aug', X2_aug.shape)
	
	train_model, base_model = model['model'](img_width, img_height, NUM_CLASSES, X, Y, optimizer)
	# emplea todas las gpu disponibles
	from keras.utils import multi_gpu_model
	parallel_model = multi_gpu_model(train_model, gpus=gpu) if gpu > 1 else train_model
	
	countbase = len(base_model.layers)
	if 'transfer' in model:
		if model['transfer']:
			for layer in train_model.layers[:countbase]:
				layer.trainable = False
	if 'pretrained' in model:
		if model['pretrained']:
			# fine tune
			for layer in train_model.layers[:countbase]:
				layer.trainable = False
			parallel_model.compile(optimizer=optimizer, loss='binary_crossentropy',
					  metrics=['accuracy'])
			callbacks = [
				SingleLabelAugAvg(X2, Y2, X2_aug, Y2_aug, indices2), BinaryAugAvg(X2, Y2, X2_aug, Y2_aug, indices2),
				SingleLabelAug(X2, Y2, X2_aug, Y2_aug), BinaryAug(X2, Y2, X2_aug, Y2_aug),
				SingleLabel(X2, Y2), Binary(X2, Y2), AccumulativeTime(),
				# it can be noted that a higher factor increase performance
				ReduceLROnPlateau(monitor=metric, factor=0.2, mode=metric_mode, verbose=1),
				EpochsRegister(join(dirpath, 'epochs.txt'),
									join(dirpath, 'epochs-mean.txt'),
									do_mean=False)
			]
			parallel_model.fit(X, Y, epochs=epochs_pre, batch_size=batch,
						validation_data=(X2_aug, Y2_aug),
						callbacks=callbacks,
						verbose=2)
			for layer in train_model.layers[:countbase]:
				layer.trainable = True
		
	parallel_model.compile(optimizer=optimizer, loss='binary_crossentropy',
						metrics=['accuracy'])


	
	callbacks = [
		SingleLabelAugAvg(X2, Y2, X2_aug, Y2_aug, indices2), BinaryAugAvg(X2, Y2, X2_aug, Y2_aug, indices2),
		SingleLabelAug(X2, Y2, X2_aug, Y2_aug), BinaryAug(X2, Y2, X2_aug, Y2_aug),
		SingleLabel(X2, Y2), Binary(X2, Y2), AccumulativeTime()
	]
	if optimizer == "sgd":
		callbacks.append(
			# it can be noted that a higher factor increase performance
			ReduceLROnPlateau(monitor=metric, factor=0.2, mode=metric_mode, verbose=1))
	callbacks.append(EpochsRegister(join(dirpath, 'epochs.txt'),
									join(dirpath, 'epochs-mean.txt'),
									epoch_start=epochs_pre))
	# end callbacks

	class_data = class_weight.compute_class_weight('balanced',
												   numpy.unique(Y),
												   Y)

	if type_class_weight == "weight":
		print('>> class weight', class_data)
		parallel_model.fit(X, Y, epochs=epochs, batch_size=batch,
						validation_data=(X2_aug, Y2_aug),
						callbacks=callbacks,
						verbose=2, class_weight=class_data)
	else:
		parallel_model.fit(X, Y, epochs=epochs, batch_size=batch,
						validation_data=(X2_aug, Y2_aug),
						callbacks=callbacks,
						verbose=2)


def kfold(config_file, models):
	# code
	with open(config_file) as json_data:
		configuration = json.load(json_data)

	folds = int(configuration['folds'])
	epochs = int(configuration['epochs'])
	epochs_pre = int(configuration['epochs_pre'])
	seed = int(configuration['seed'])
	reportsDir = configuration['reportsDir']
	metric = configuration['metric']
	metric_mode = configuration['metric_mode']
	gpu = configuration['gpu']

	for dataset in configuration['datasets']:
		for batch in dataset['batch']:
			for model in models:
				for optimizer in configuration['optimizers']:
					for type_class_weight in configuration['class_weight']:
						num_batch = int(batch)
						
						if 'optimizer' in model:
							if model['optimizer'] is not None:
								final_optimizer = model['optimizer']
							else:
								final_optimizer = optimizer
						else:
							final_optimizer = optimizer
						
						dirpath = join(reportsDir, dataset['name'], model['name'], "batch_" + str(batch),
									   final_optimizer, type_class_weight)

						try:
							# if this experiment was finished continue
							if os.path.exists(join(dirpath, 'summary.txt')):
								continue
							else:
								# if not, delete the partial results
								if os.path.exists(join(dirpath, 'epochs-mean.txt')):
									os.remove(join(dirpath, 'epochs-mean.txt'))
								if os.path.exists(join(dirpath, 'epochs.txt')):
									os.remove(join(dirpath, 'epochs.txt'))

							if not os.path.exists(dirpath):
								os.makedirs(dirpath)

							# fix random seed for reproducibility
							numpy.random.seed(seed)

							X = numpy.load(dataset['x'])
							X = preprocess_input(X)
							img_width = len(X[0][0])
							img_height = len(X[0])
							# transform image according with model
							if 'shape' in model:
								model_shape = model['shape']
								if img_height != model_shape[0] | img_width != model_shape[1]:
									print('>> resize images', dirpath)
									X = resize_images(X, model_shape)
									img_height = model_shape[0]
									img_width = model_shape[1]
							
							Y = numpy.load(dataset['y'])

							NUM_CLASSES = keras.utils.to_categorical(Y).shape[1]

							# define N-fold cross validation test harness
							kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

							fold = 0
							for train, test in kfold.split(X, Y):
									
								new_x_train = X[train]								
								file_x_train = temp_file()
								numpy.save(file_x_train, new_x_train)
								del new_x_train
								
								new_x_test = X[test]
								file_x_test = temp_file()
								numpy.save(file_x_test, new_x_test)
								del new_x_test
								
								time.sleep(5)
								start_time = time.time()
								with Pool(processes=1) as pool:
									pool.map(one_fold, [{
										'fold': fold,
										'epochs': epochs,
										'epochs_pre': epochs_pre,
										'batch': num_batch,
										'model': model,
										'img_width': img_width,
										'img_height': img_height,
										'NUM_CLASSES': NUM_CLASSES,
										'X': file_x_train,
										'Y': Y[train],
										'X2': file_x_test,
										'Y2': Y[test],
										'optimizer': final_optimizer,
										'dirpath': dirpath,
										'metric': metric,
										'metric_mode': metric_mode,
										'type_class_weight': type_class_weight,
										'gpu': gpu
									}])
									print('>> fold', dirpath)
									print('>> fold', fold, 'completed in', str(time.time() - start_time), 'seconds')
									fold = fold + 1

							final_evaluations = numpy.genfromtxt(join(dirpath, 'epochs-mean.txt'), delimiter=',',
																 dtype=numpy.float64, names=True)
							# evaluaciones de una metrica
							metric_column = final_evaluations[metric]
							# el indice de la fila donde esta la mejor metrica
							row = metric_column.argmax() if metric_mode == 'max' else metric_column.argmin()

							evaluation = final_evaluations[row]
							summary = open(join(dirpath, 'summary.txt'), mode='w')
							# leer las keys, las metricas
							summary.write(','.join(final_evaluations.dtype.names))
							summary.write('\n')
							summary.write(','.join(map(str, evaluation)))
							summary.close()
							
							#delete dataset
							del X
							del Y
							time.sleep(3)
						except Exception as exception:
							print('error >>', dirpath)
							print('reason >>', exception)



if __name__ == '__main__':
	config_file = str(sys.argv[1])
	from import_all_models import DenseNet121_imagenet, DenseNet121_sinpesos
	from import_all_models import DenseNet169_imagenet, DenseNet169_sinpesos
	from import_all_models import DenseNet201_imagenet, DenseNet201_sinpesos
	from import_all_models import InceptionResNetV2_imagenet, InceptionResNetV2_sinpesos
	from import_all_models import InceptionV3_imagenet, InceptionV3_sinpesos
	from import_all_models import MobileNet_imagenet, MobileNet_sinpesos
	from import_all_models import ResNet50_sinpesos, ResNet50_imagenet
	from import_all_models import VGG16_imagenet, VGG16_sinpesos
	from import_all_models import VGG19_imagenet, VGG19_sinpesos
	from import_all_models import Xception_imagenet, Xception_sinpesos
	from import_all_models import InceptionV4_imagenet, InceptionV4_sinpesos
	
	from import_all_models import NASNetMobile_imagenet, NASNetMobile_sinpesos

	
	kfold(config_file, [
		# 224x224
		MobileNet_imagenet(),
		#DenseNet121_imagenet(),
		#DenseNet169_imagenet(),
		#DenseNet201_imagenet(),
		# 299x299
		#Xception_imagenet(),
		#InceptionV3_imagenet(),
		# 224x224
		#ResNet50_imagenet(),
		# 299x299
		#InceptionResNetV2_imagenet(),
		# 224x224
		#VGG16_imagenet(),
		#VGG19_imagenet(),
		#299x299
		#InceptionV4_imagenet(),
		#224
		#NASNetMobile_imagenet()
	])
