# Air-writting
This project is built on [hand-gesture-recognition-using-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe).<br> This is a sample 
program that recognizes hand signs, finger gestures and handwritten characters with a simple MLP using the detected key points and simple CNN.

![demo](https://github.com/Tkag0001/Air-writing/assets/107709392/41fc1c69-6c2e-45e8-aa30-4848267898ac)<br>
This repository contains the following contents.
* Sample program
* Hand sign recognition model(TFLite)
* Finger gesture recognition model(TFLite)
* Handwritten character recognition(TFLite)
* Learning data for hand sign recognition and notebook for learning
* Learning data for finger gesture recognition and notebook for learning
* Learning data for handwritten character recognition and notebook for learning

# Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
* scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix) 
* matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)

# Demo
Here's how to run the demo using your webcam.
```bash
python app.py
```

The following options can be specified when running the demo.
* --device<br>Specifying the camera device number (Default：0)
* --width<br>Width at the time of camera capture (Default：640)
* --height<br>Height at the time of camera capture (Default：480)
* --use_static_image_mode<br>Whether to use static_image_mode option for MediaPipe inference (Default：Unspecified)
* --min_detection_confidence<br>
Detection confidence threshold (Default：0.5)
* --min_tracking_confidence<br>
Tracking confidence threshold (Default：0.5)

# Directory

<pre>
│   .gitignore
│   app.py
│   demo.gif
│   check_data_train.ipynb
│   agument_image_for_keypoint_classification.py
│   create_train_data_for_keypoint_classification_from_image.py
│   keypoint_classification.ipynb
│   keypoint_classification_EN.ipynb
│   character-recognition.ipynb
│   point_history_classification.ipynb
│   LICENSE
│   README.md
├─data
│
├─model
│   ├─handwritten_classifier
│   │   │   handwritten_classifier_v2.hdf5
│   │   │   handwritten_classifier_v2.keras
│   │   │   handwritten_classifier.py
│   │   │   handwritten_classifier_v2.tflite
│   │   └─   handwritten_classifier_label.csv
│   │           
│   ├─keypoint_classifier
│   │   │   keypoint.csv
│   │   │   keypoint_classifier.py
│   │   │   keypoint_classifier_label.csv
│   │   │   keypoint_classifier_v4.hdf5
│   │   │   keypoint_classifier_v4.tflite
│   │   └─   keypoint_v4.csv
│   │           
│   └─point_history_classifier
│       │   point_history.csv
│       │   point_history_classifier.hdf5
│       │   point_history_classifier.py
│       │   point_history_classifier.tflite
│       └─  point_history_classifier_label.csv
│           
└─utils
    └─cvfpscalc.py
</pre>
### app.py
This is a sample program for inference.<br>
In addition, learning data (key points) for hand sign recognition,<br>
You can also collect training data (index finger coordinate history) for finger gesture recognition.

### handwritten_classifier.ipynb
This is model training script for handwritten character recognition.

### keypoint_classification.ipynb
This is a model training script for hand sign recognition.

### point_history_classification.ipynb
This is a model training script for finger gesture recognition.

### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### model/point_history_classifier
This directory stores files related to finger gesture recognition.<br>
The following files are stored.
* Training data(point_history.csv)
* Trained model(point_history_classifier.tflite)
* Label data(point_history_classifier_label.csv)
* Inference module(point_history_classifier.py)

### utils/cvfpscalc.py
This is a module for FPS measurement.

# Training
Hand sign recognition and finger gesture recognition can add and change training data and retrain the model.

### Hand sign recognition training
#### 1.Learning data collection
#### 1.1. Manual data collection
Press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」）<br>
<img src="https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg" width="60%"><br><br>
If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Key point coordinates<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>
In the initial state, five types of learning data are included: stop (class ID: 0), cut (class ID: 1), space (class ID: 2), delete (class ID: 3) and write (class ID: 4).<br>
If necessary, add 3 or later, or delete the existing data of csv to prepare the training data.<br>
<img src="https://github.com/Tkag0001/Air-writing/assets/107709392/0fdc027a-fa18-4e75-8103-ce51a542aced" width="15%">　
<img src="https://github.com/Tkag0001/Air-writing/assets/107709392/6c786f30-cd7d-4da0-b1e6-b2725bdeaf37" width="15%">　
<img src="https://github.com/Tkag0001/Air-writing/assets/107709392/95132117-4314-4f45-a38e-402b94461f3d" width="15%">
<img src="https://github.com/Tkag0001/Air-writing/assets/107709392/33370421-2929-436d-9476-16f824aaa5f1" width="15%">
<img src="https://github.com/Tkag0001/Air-writing/assets/107709392/7348607c-c453-44d5-b6af-d42d2c90cc3b" width="15%">


#### 1.2. Auto data collection
We use [asl_alphabet_train](https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train) to generate keypoint.
After downloading dataset, you can run [agument_image_for_keypoint_classification.py](agument_image_for_keypoint_classification.py) to generate rotated image (*Note: change list_data to the types of pose hand).
Then run [create_train_data_for_keypoint_classification_from_image.py](create_train_data_for_keypoint_classification_from_image.py) to generate keypoint data to recognize pose hand.
#### 2.Model training
Open "[keypoint_classification.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 5" <br>and modify the label of "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate.<br><br>

#### X.Model structure
The image of the model prepared in "[keypoint_classification.ipynb](keypoint_classification.ipynb)" is as follows.
<img src="https://user-images.githubusercontent.com/37477845/102246723-69c76a00-3f42-11eb-8a4b-7c6b032b7e71.png" width="50%"><br><br>

### Finger gesture recognition training
#### 1.Learning data collection
Press "h" to enter the mode to save the history of fingertip coordinates (displayed as "MODE:Logging Point History").<br>
<img src="https://user-images.githubusercontent.com/37477845/102249074-4d78fc80-3f45-11eb-9c1b-3eb975798871.jpg" width="60%"><br><br>
If you press "0" to "9", the key points will be added to "model/point_history_classifier/point_history.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Coordinate history<br>
<img src="https://user-images.githubusercontent.com/37477845/102345850-54ede380-3fe1-11eb-8d04-88e351445898.png" width="80%"><br><br>
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102244148-49e27700-3f3f-11eb-82e2-fc7de42b30fc.png" width="80%"><br><br>
In the initial state, 4 types of learning data are included: stationary (class ID: 0), clockwise (class ID: 1), counterclockwise (class ID: 2), and moving (class ID: 4). <br>
If necessary, add 5 or later, or delete the existing data of csv to prepare the training data.<br>
<img src="https://user-images.githubusercontent.com/37477845/102350939-02b0c080-3fe9-11eb-94d8-54a3decdeebc.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350945-05131a80-3fe9-11eb-904c-a1ec573a5c7d.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350951-06444780-3fe9-11eb-98cc-91e352edc23c.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350942-047a8400-3fe9-11eb-9103-dbf383e67bf5.jpg" width="20%">

#### 2.Model training
Open "[point_history_classification.ipynb](point_history_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 4" and <br>modify the label of "model/point_history_classifier/point_history_classifier_label.csv" as appropriate. <br><br>

#### X.Model structure
The image of the model prepared in "[point_history_classification.ipynb](point_history_classification.ipynb)" is as follows.
<img src="https://user-images.githubusercontent.com/37477845/102246771-7481ff00-3f42-11eb-8ddf-9e3cc30c5816.png" width="50%"><br>
The model using "LSTM" is as follows. <br>Please change "use_lstm = False" to "True" when using (tf-nightly required (as of 2020/12/16))<br>
<img src="https://user-images.githubusercontent.com/37477845/102246817-8368b180-3f42-11eb-9851-23a7b12467aa.png" width="60%">

# Reference
* [MediaPipe](https://mediapipe.dev/)

