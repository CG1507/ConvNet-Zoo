# ConvNet-Zoo

ConvNet-Zoo is playground for benchmark architectures of Convolutional Neural Networks (CNN). It is implemented in Keras with TensorFlow backend to compare the different architectures and understand the **How ConvNet's layers see the images?**.

## PlayGround

<img src="https://github.com/CG1507/ConvNet-Zoo/blob/master/images/demo.gif" width="900" height="500" />

## Requirements:
* PyQt5
* Tensorflow
* Keras
* numpy
* scipy
* h5py
* wget
* Pillow
* six
* scikit-image

## Run:
Very first time it will download the weights of the model you pick, so it requires an internet connection.
```
python gui.py
```

## For Tensorboard:

TensorBoard gives you flexibility to visualize all the test image on same model with brightness and contrast adustment.

```
tensorboard --logdir=<LOG-PATH (layerwise)>
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Author

[Ghanshyam Chodavadiya](https://cg1507.github.io)
