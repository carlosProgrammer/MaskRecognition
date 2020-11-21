from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize the starting learning rate, epochs to train and batch size
init_lr = 1e-14
epochs = 20
batchSize = 32

# Dataset directory and categories
dir = r"C:\MASKRECOGNITION\dataset"
categories = ["with_mask", "without_mask"]

# Fetch and select the list of images in the dataset, initialize the data list and images classes
print("[INFO] loading images...")

data = []
labels = []

for category in categories:
    path = os.path.join(dir, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# Label encoding
label_bin = LabelBinarizer()
labels = label_bin.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(ata, dtype="float32")
labels = np.array(labels)

(trainX, tastX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Training image generator for data augmentation
augmentation = ImageDataGenerator(
    rotation_range = 20,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = "nearest")

# Loading MobileNetV2 network to ensure the FC layer sets are left off
baseModel = MobileNetV2(
    weights = "imagenet", 
    include_top = False,
    input_tensor = Input (shape = (224, 224, 3)))

# Head of the model that will be placed on top of the baseModel
headModel = baseModel.output
headModel = AveragePooling2D(pool_size = (7, 7))(headModel)
headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(128, activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation = "softmax")(headModel)

# Placing the FC model on top of baseModel, this is the actual training model
model = Model(
    inputs = baseModel.input,
    outputs = headModel)

# Looping over every layer in the base model and freeze them so they won't be updated during the first training
for layer in baseModel.layers:
    layer.trainable = False

# Compiling the model
print("[INFO] compiling model...")
opt = Adam(lr = init_lr, decay = init_lr / epochs)
model.compile(
    loss = "binary_crossentropy", 
    optimizer = opt, 
    metrics = ["accuracy"])

# Training the head of the network
print("[INFO] training head of network...")
head = model.fit(
    augmentation.flow(trainX, trainY, batch_size = batchSize),
    steps_per_epoch = len(trainX) // batchSize,
    validation_data = (testX, testY),
    validation_steps = len(textX) // batchSize
    epochs = epochs)

# Predictions on the testing proccess
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

