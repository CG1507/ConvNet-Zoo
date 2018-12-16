<h1 align="center">
  :deciduous_tree: ConvNet-Zoo :deciduous_tree:
  <h3 align="center">
  :video_game: PlayGround
  </h3>
  <br />
  <img src="https://github.com/CG1507/ConvNet-Zoo/blob/master/images/demo.gif" width="900" height="500" />
</h1>

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
python3 gui.py
```

## For Tensorboard:

TensorBoard gives you flexibility to visualize all the test image on same model with brightness and contrast adustment.

```
tensorboard --logdir=<LOG-PATH (layerwise)>
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Author

[<img src="https://avatars3.githubusercontent.com/u/24426731?s=460&v=4" width="70" height="70" alt="Ghanshyam_Chodavadiya">](https://github.com/CG1507)

## Acknowledgement

:green_heart: [tfcnn_vis](https://github.com/InFoCusp/tf_cnnvis)
