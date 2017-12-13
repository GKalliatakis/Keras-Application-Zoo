# Keras | Application Zoo - DIY Deep Learning for Vision

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/GKalliatakis/Keras-Application-Zoo/blob/master/LICENSE)


## You have just found Keras Application Zoo

Keras Application Zoo is a public clearinghouse for [`keras.applications`](https://keras.io/applications/)-compatible models to promote progress and reproduce research.

Lots of researchers and engineers have made their deep learning models public in various frameworks for different tasks with all kinds of architectures and data. These models are learned and applied for problems ranging from simple regression, to large-scale visual classification. 

However, [Keras](https://github.com/fchollet/keras) does not contain the degree of pre-trained models that come complete with [Caffe](http://caffe.berkeleyvision.org/). 


To lower the friction of sharing these models, we introduce the Keras Application Zoo:

- A **central GitHub repo** for sharing popular deep learning models with **Keras code & weights files**
- Contains **ONLY additional** deep learning models which are **not yet available within `keras.applications` module itself**
- Tools to upload/download model info to/from Github, and to download trained Keras Applications-compatible binaries
- Models can be used for prediction, feature extraction, and fine-tuning just like the default `keras.applications` models.
- No separate models configuration files in a declarative format. Models are described in Python code, which is compact, easier to debug, and allows for ease of extensibility.

**Benefit from networks that you could not practically train yourself by taking Keras to the Zoo !**

Read the official documentation at [Keras.io](https://keras.io).

------------------

## Usage: 
All architectures are compatible with both TensorFlow and Theano, and upon instantiation the models will be built according to the image dimension ordering set in your Keras configuration file at ~/.keras/keras.json. For instance, if you have set image_dim_ordering=tf, then any model loaded from this repository will get built according to the TensorFlow dimension ordering convention, "Width-Height-Depth".

Pre-trained weights can be automatically loaded upon instantiation (weights='places' argument in model constructor for all scene-centric models). Weights are automatically downloaded.

------------------

## Examples

### Classify images

```python
from vgg16_places_365 import VGG16_Places365
from keras.preprocessing import image

model = VGG16_Places365(weights='places')

img_path = 'restaurant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', preds)
```

### Extract features from images

```python
from vgg16_places_365 import VGG16_Places365
from keras.preprocessing import image

model = VGG16_Places365(weights='places', include_top=False)

img_path = 'restaurant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```
------------------
### Licensing 
- All code in this repository is under the MIT license as specified by the LICENSE file.
- The VGG16-places365 and VGG16-hybrid1365 weights are ported from the ones [released by CSAILVision](https://github.com/CSAILVision/places365) under the [MIT license](https://github.com/CSAILVision/places365/blob/master/LICENSE).
- The ResNet-152 weights are ported from ones [released by adamcasson](https://github.com/adamcasson/resnet152).

We are always interested in how these models are being used, so if you found them useful or plan to make a release of code based on or using this package, it would be great to hear from you. 

Additionally, don't forget to cite this repo if you use these models:

    @misc{GKalliatakis_Keras_Application_Zoo,
    title={Keras-Application-Zoo},
    author={Grigorios Kalliatakis},
    year={2017},
    publisher={GitHub},
    howpublished={\url{https://github.com/GKalliatakis/Keras-Application-Zoo}},
    }

### Other Models 
This is going to be an evolving repository and I will keep updating it with Keras Applications-compatible models which are not included in the original [Keras Applications](https://keras.io/applications/), so make sure you have starred :star2: and forked this repository before moving on !

------------------
### Questions and Comments
If you have any suggestions or bugs to report you can pull a request or start a discussion.
