import inspect, os, sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import shutil
from distutils.dir_util import copy_tree

import numpy as np
from keras.applications import *
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation
from keras.models import Model
import keras.backend as K
import densenet_vis

# pyqt5 class for main window
class App(QMainWindow):

	#constructor
	def __init__(self):
		super().__init__()
		self.title = 'ConvNet Zoo'
		self.left = 10
		self.top = 10
		self.width = 1400
		self.height = 750
		self.filename = ''
		self.initUI()

	#initial UI
	def initUI(self):
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

		self.model_label = QLabel(self)
		self.model_label.move(20, 50)
		self.model_label.setText("Choose Model: ")		
		self.model_label.resize(200, 20)

		self.model_combo = QComboBox(self)
		self.model_combo.move(150, 48)
		self.model_combo.resize(250,25)
		self.model_combo.activated[str].connect(self.setModel)
		self.model_combo.addItem('Xception')
		self.model_combo.addItem('VGG16')
		self.model_combo.addItem('VGG19')
		self.model_combo.addItem('ResNet50')
		self.model_combo.addItem('InceptionV3')
		self.model_combo.addItem('InceptionResNetV2')
		self.model_combo.addItem('MobileNet')
		self.model_combo.addItem('MobileNetV2')
		self.model_combo.addItem('DenseNet121')
		self.model_combo.addItem('DenseNet169')
		self.model_combo.addItem('DenseNet201')
		self.model_combo.addItem('NASNetMobile')
		self.model_combo.addItem('NASNetLarge')

		self.model_loading_label = QLabel(self)
		self.model_loading_label.move(20, 80)
		self.model_loading_label.setText("Note: For first use, it will download the weights. \n            This may take a while to switch the model.")		
		self.model_loading_label.resize(400, 50)

		self.browse_button1_label = QLabel(self)
		self.browse_button1_label.move(210, 170)
		self.browse_button1_label.setText("Input Image")		
		self.browse_button1_label.resize(200, 20)

		self.button1 = QPushButton('Browse', self)
		self.button1.setToolTip('Only Image')
		self.button1.move(200,200) 
		self.button1.clicked.connect(self.browse_button)

		self.button2 = QPushButton('Predict', self)
		self.button2.setToolTip('Only Image')
		self.button2.move(200,600) 
		self.button2.clicked.connect(self.predict)

		self.label1 = QLabel(self)
		self.label1.move(50, 250)

		self.font = QFont()
		self.font.setPointSize(20)
		self.font.setBold(True)
		self.font.setWeight(75)

		self.output = QLabel(self)
		self.output.move(50, 650)
		self.output.resize(500, 50)
		self.output.setFont(self.font)

		self.button2.setEnabled(False)

		self.model_name = 'Xception'
		self.model = None
		self.target_size = None
		self.preprocess_fun = None
		self.decode_fun = None
		self.get_model()

		self.featuremap_label = QLabel(self)
		self.featuremap_label.move(500, 20)
		self.featuremap_label.setText("Generate features learned in each layer of NN for the image you've selected\nIt takes time, so press Initialize layers button and check in comboBox to select layer to see features.")
		self.featuremap_label.resize(800, 50)
		self.featuremap_label.setEnabled(False)

		self.generate = QPushButton('Generate', self)
		self.generate.setToolTip('Only Image')
		self.generate.move(1200,30) 
		self.generate.clicked.connect(self.generate_activations)
		self.generate.setEnabled(False)

		self.label2 = QLabel(self)
		self.label2.move(500, 150)

		self.layer_initialize_button = QPushButton('Initialize layers', self)
		self.layer_initialize_button.setToolTip('Only Image')
		self.layer_initialize_button.move(750,70) 
		self.layer_initialize_button.resize(200, 30)
		self.layer_initialize_button.clicked.connect(self.initialize_layers)
		self.layer_initialize_button.setEnabled(False)

		self.layer_selection_label = QLabel(self)
		self.layer_selection_label.move(650, 110)
		self.layer_selection_label.setText("Select layer to see features")		
		self.layer_selection_label.resize(200, 20)
		self.layer_selection_label.setEnabled(False)

		self.layer_selection_combo = QComboBox(self)
		self.layer_selection_combo.move(850, 110)
		self.layer_selection_combo.resize(220,20)
		self.layer_selection_combo.setEnabled(False)
		self.layer_selection_combo.activated[str].connect(self.showActivation)

		self.save_button = QPushButton('Save Results', self)
		self.save_button.setToolTip('saves log and activations')
		self.save_button.move(750,670)
		self.save_button.resize(200, 30)
		self.save_button.clicked.connect(self.saveFileDialog)
		self.save_button.setEnabled(False)

		self.show()

	#Event method for button
	def showActivation(self, text):
		#self.model_label.setText(text)
		#self.model_label.adjustSize()
		self.show_image('./vis/activation/output/' + text + '/activations/grid_activation.png')
		#self.show_image('.\\vis\\activation\output\\' + text + '\\activations\\grid_activation.png')

	def setModel(self, text):
		if self.model_name != text:
			self.model_name = text
			self.get_model()
		self.model_name = text

	def get_model(self):
		if self.model_name == 'Xception':
			K.clear_session()
			self.model = xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.xception import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (299, 299)

		if self.model_name == 'VGG16':
			K.clear_session()
			self.model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.vgg16 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_name == 'VGG19':
			K.clear_session()
			self.model = vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.vgg19 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_name == 'ResNet50':
			K.clear_session()
			self.model = resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.resnet50 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_name == 'InceptionV3':
			K.clear_session()
			self.model = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.inception_v3 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (299, 299)

		if self.model_name == 'InceptionResNetV2':
			K.clear_session()
			self.model = inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (299, 299)

		if self.model_name == 'MobileNet':
			K.clear_session()
			self.model = mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
			from keras.applications.mobilenet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_name == 'MobileNetV2':
			K.clear_session()
			self.model = mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000) 
			from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_name == 'DenseNet121':
			K.clear_session()
			self.model = densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.densenet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_name == 'DenseNet169':
			K.clear_session()
			self.model = densenet.DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.densenet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_name == 'DenseNet201':
			K.clear_session()
			self.model = densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
			from keras.applications.densenet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_name == 'NASNetMobile':
			K.clear_session()
			self.model = nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
			from keras.applications.nasnet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (224, 224)

		if self.model_name == 'NASNetLarge':
			K.clear_session()
			self.model = nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
			from keras.applications.nasnet import preprocess_input, decode_predictions
			self.preprocess_fun = preprocess_input
			self.decode_fun = decode_predictions
			self.target_size = (331, 331)

	#function for displaying image in GUI window
	def show_image(self, filepath):
		file = filepath.split('/')[-1]
		#file = filepath.split('\\')[-1]
		shutil.copyfile(filepath, './' + file)
		#shutil.copyfile(filepath, '.\\' + file)
		pixmap = QPixmap(file)
		pixmap = pixmap.scaled(700, 500)
		self.label2.setPixmap(pixmap)
		self.label2.resize(700, 500)
		os.remove("./" + file)
		#os.remove(".\\" + file)

	@pyqtSlot()
	def initialize_layers(self):
		self.layer_selection_combo.clear()
		if os.path.exists('./vis/activation/output/'):
		#if os.path.exists('.\\vis\\activation\\output\\'):
			layers_list = os.listdir('./vis/activation/output/')
			#layers_list = os.listdir('.\\vis\\activation\\output\\')
			for layer in layers_list:
				self.layer_selection_combo.addItem(layer)
		else:
			print('No folder found')

	@pyqtSlot()
	def browse_button(self):
		self.openFileNameDialog1()
		print('browse button click')

	@pyqtSlot()
	def predict(self):
		if self.filename == '':
			print('Select image')
		else:
			img = image.load_img(self.filename, target_size=self.target_size)
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = self.preprocess_fun(x)
			preds = self.model.predict(x)
			self.output.setText('Label: ' + self.decode_fun(preds, top=3)[0][0][1])

	#handle/delete copied images in current directory  		
	def delete_history(self):
		folder = './vis/activation/output/'
		#folder = '.\\vis\\activation\\output\\'
		for file in os.listdir(folder):
			shutil.rmtree(folder + file)

	def visualization(self, model_retrain, im):
		sess = K.get_session()
		layers = ['r', 'p', 'c']

		self.delete_history()
		with sess.as_default():
		# with sess_graph_path = None, the default Session will be used for visualization.
			is_success = densenet_vis.activation_visualization(sess_graph_path = None,
								value_feed_dict = {model_retrain.get_layer('input_1').input : im}, 
								layers=layers, path_logdir='./vis/activation/log/', 
								path_outdir='./vis/activation/output/')
			print('Activation: ', is_success)

	@pyqtSlot()
	def generate_activations(self):
		img = image.load_img(self.filename, target_size=self.target_size)
		x = image.img_to_array(img)
		x = self.preprocess_fun(x)
		x = np.expand_dims(x, axis=0)

		self.layer_selection_combo.setEnabled(True)
		self.layer_selection_label.setEnabled(True)
		self.layer_initialize_button.setEnabled(True)
		self.save_button.setEnabled(True)

		self.visualization(self.model, x)

	def saveFileDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		foldername = dir_name = QFileDialog.getExistingDirectory(self, 'Select Directory')
		if foldername:
			copy_tree('./vis/activation', foldername)

	def openFileNameDialog1(self):    
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "All files(*);;Image Files (*.jpeg);;Python Files (*.jpg)", 
			options=options)
		if fileName:
			print(fileName)
			file = fileName.split('/')[-1]
			#file = fileName.split('\\')[-1]
			if self.check_file(fileName, file):
				shutil.copyfile(fileName, './' + file)
				#shutil.copyfile(fileName, '.\\' + file)
				pixmap = QPixmap(file)
				pixmap = pixmap.scaled(400, 300)
				self.label1.setPixmap(pixmap)
				self.label1.resize(400, 300)
				os.remove("./" + file)
				#os.remove(".\\" + file)
			else:
				pixmap = QPixmap(file)
				pixmap = pixmap.scaled(400, 300)
				self.label1.setPixmap(pixmap)
				self.label1.resize(400, 300)
			self.filename = fileName
			self.generate.setEnabled(True)
			self.featuremap_label.setEnabled(True)
			self.button2.setEnabled(True)

	def check_file(self, fileName, file):
		currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
		temp = str(currentdir + '/' + file).strip()
		#temp = str(currentdir + '\\' + file).strip()

		if temp == fileName.strip():
			return False
		return True

if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = App()
	sys.exit(app.exec_())