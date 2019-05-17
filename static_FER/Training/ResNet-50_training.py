import numpy as np
import pandas as pd
from tensorflow.python.lib.io import file_io
from skimage.transform import resize
from keras import backend as K
from keras.utils import to_categorical
from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Flatten, Dense 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, Callback
import h5py # For saving the model



# Folder where logs and models are stored
folder = 'logs/ResNet-50'

# Size of the images
img_height, img_width = 197, 197

# Parameters
num_classes         = 7     # ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
epochs_top_layers   = 5
epochs_all_layers   = 100
batch_size          = 128


# Folder where logs and models are stored
folder = '/gdrive/My Drive/Colab Notebooks/FER/data/'

# Data paths
train_dataset	= '/gdrive/My Drive/Colab Notebooks/FER/data/fer2013_train.csv'
eval_dataset 	= '/gdrive/My Drive/Colab Notebooks/FER/data/fer2013_eval.csv'


base_model = VGGFace(
    model       = 'resnet50',
    include_top = False,
    weights     = 'vggface',
    input_shape = (img_height, img_width, 3))

# Places x as the output of the pre-trained model
x = base_model.output

# Flattens the input. Does not affect the batch size
x = Flatten()(x)

x = Dense(1024, activation = 'relu')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

# The model we will train
model = Model(inputs = base_model.input, outputs = predictions)
# model.summary()


def preprocess_input(x):
    x -= 128.8006 # np.mean(train_dataset)
    return x

# Function that reads the data from the csv file, increases the size of the images and returns the images and their labels
    # dataset: Data path
def get_data(dataset):
    file_stream = file_io.FileIO(dataset, mode='r')
    data = pd.read_csv(file_stream)
    pixels = data['pixels'].tolist()
    images = np.empty((len(data), img_height, img_width, 3))
    i = 0

    for pixel_sequence in pixels:
        single_image = [float(pixel) for pixel in pixel_sequence.split(' ')]  # Extraction of each single
        single_image = np.asarray(single_image).reshape(48, 48) # Dimension: 48x48
        single_image = resize(single_image, (img_height, img_width), order = 3, mode = 'constant') # Dimension: 139x139x3 (Bicubic)
        ret = np.empty((img_height, img_width, 3))  
        ret[:, :, 0] = single_image
        ret[:, :, 1] = single_image
        ret[:, :, 2] = single_image
        images[i, :, :, :] = ret
        i += 1
    
    images = preprocess_input(images)
    labels = to_categorical(data['emotion'])

    return images, labels    

# Data preparation
train_data_x, train_data_y  = get_data(train_dataset)
val_data  = get_data(eval_dataset)

train_datagen = ImageDataGenerator(
    rotation_range  = 10,
    shear_range     = 10, # 10 degrees
    zoom_range      = 0.1,
    fill_mode       = 'reflect',
    horizontal_flip = True)

train_generator = train_datagen.flow(
    train_data_x,
    train_data_y,
    batch_size  = batch_size)


# First: train only the top layers (which were randomly initialized) freezing all convolutional ResNet-50 layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer   = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0), 
    loss        = 'categorical_crossentropy', 
    metrics     = ['accuracy'])


tensorboard_top_layers = TensorBoard(
	log_dir         = folder + '/logs_top_layers',
	histogram_freq  = 0,
	write_graph     = True,
	write_grads     = False,
	write_images    = True)


model.fit_generator(
    generator           = train_generator,
    steps_per_epoch     = len(train_data_x) // batch_size,  # samples_per_epoch / batch_size
    epochs              = epochs_top_layers,                            
    validation_data     = val_data,
    callbacks           = [tensorboard_top_layers])



model.compile(
    optimizer   = SGD(lr = 1e-4, momentum = 0.9, decay = 0.0, nesterov = True),
    loss        = 'categorical_crossentropy', 
    metrics     = ['accuracy'])

# This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of your training and test metrics, 
tensorboard_all_layers = TensorBoard(
    log_dir         = folder + '/logs_all_layers',
    histogram_freq  = 0,
    write_graph     = True,
    write_grads     = False,
    write_images    = True)

def scheduler(epoch):
    updated_lr = K.get_value(model.optimizer.lr) * 0.5
    if (epoch % 3 == 0) and (epoch != 0):
        K.set_value(model.optimizer.lr, updated_lr)
        print(K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)

# Learning rate scheduler
    # schedule: a function that takes an epoch index as input (integer, indexed from 0) and current learning
    #           rate and returns a new learning rate as output (float)
reduce_lr = LearningRateScheduler(scheduler)


reduce_lr_plateau = ReduceLROnPlateau(
	monitor 	= 'val_loss',
	factor		= 0.5,
	patience	= 3,
	mode 		= 'auto',
	min_lr		= 1e-8)

early_stop = EarlyStopping(
	monitor 	= 'val_loss',
	patience 	= 10,
	mode 		= 'auto')

class ModelCheckpoint(Callback):

	def __init__(self, filepath, folder, monitor = 'val_loss', verbose = 0, save_best_only = False, save_weights_only = False, mode = 'auto', period = 1):
		super(ModelCheckpoint, self).__init__()
		self.monitor 				= monitor
		self.verbose		 		= verbose
		self.filepath 				= filepath
		self.folder 				= folder
		self.save_best_only 		= save_best_only
		self.save_weights_only		= save_weights_only
		self.period 				= period
		self.epochs_since_last_save	= 0
		
		if mode not in ['auto', 'min', 'max']:
			warnings.warn('ModelCheckpoint mode %s is unknown, ' 'fallback to auto mode.' % (mode), RuntimeWarning)
			mode = 'auto'

		if mode == 'min':
			self.monitor_op = np.less
			self.best = np.Inf
		elif mode == 'max':
			self.monitor_op = np.greater
			self.best = -np.Inf
		else:
			if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
			    self.monitor_op = np.greater
			    self.best = -np.Inf
			else:
			    self.monitor_op = np.less
			    self.best = np.Inf
	
	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		self.epochs_since_last_save += 1
		if self.epochs_since_last_save >= self.period:
			self.epochs_since_last_save = 0
			filepath = self.filepath.format(epoch = epoch + 1, **logs)
			if self.save_best_only:
				current = logs.get(self.monitor)
				if current is None:
				    warnings.warn('Can save best model only with %s available, ' 'skipping.' % (self.monitor), RuntimeWarning)
				else:
					if self.monitor_op(current, self.best):
					    if self.verbose > 0:
					        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,' ' saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath))
					    self.best = current
					    if self.save_weights_only:
					        self.model.save_weights(filepath, overwrite=True)
					    else:
							self.model.save(filepath, overwrite=True)
							# Save model.h5 on to google storage
							with file_io.FileIO(filepath, mode='r') as input_f:
								with file_io.FileIO(self.folder + '/checkpoints/' + filepath, mode='w+') as output_f:	# w+ : writing and reading
									output_f.write(input_f.read())
					else:
						if self.verbose > 0:
						    print('\nEpoch %05d: %s did not improve' %
						          (epoch + 1, self.monitor))
			else:
				if self.verbose > 0:
				    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
				if self.save_weights_only:
				    self.model.save_weights(filepath, overwrite=True)
				else:
					self.model.save(filepath, overwrite=True)
					# Save model.h5 on to google storage
					with file_io.FileIO(filepath, mode='r') as input_f:
						with file_io.FileIO(self.folder + '/checkpoints/' + filepath, mode='w+') as output_f:	# w+ : writing and reading
							output_f.write(input_f.read())


check_point = ModelCheckpoint(
	filepath		= 'ResNet-50_{epoch:02d}_{val_loss:.2f}.h5',
	folder 			= folder,
	monitor 		= 'val_loss', # Accuracy is not always a good indicator because of its yes or no nature
	save_best_only	= True,
	mode 			= 'auto',
	period			= 1)

# We train our model again (this time fine-tuning all the resnet blocks)
model.fit_generator(
    generator           = train_generator,
    steps_per_epoch     = len(train_data_x) // batch_size,  # samples_per_epoch / batch_size 
    epochs              = epochs_all_layers,                        
    validation_data     = val_data,
    callbacks           = [tensorboard_all_layers, reduce_lr, reduce_lr_plateau, early_stop, check_point])

model.save(folder + '/ResNet-50.h5')
# Save model.h5 on to google storage
with file_io.FileIO('ResNet-50.h5', mode='r') as input_f:
    with file_io.FileIO(folder + '/ResNet-50.h5', mode='w+') as output_f:  # w+ : writing and reading
        output_f.write(input_f.read())

