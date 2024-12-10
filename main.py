import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
import re
import tensorflow_hub as hub
import tensorflow_text
from official.nlp import optimization 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import keras
from keras.layers import Dense
from tensorflow.keras.models import model_from_json
import math
from sklearn.metrics import classification_report,confusion_matrix
import os

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train = pd.read_csv('/Users/florian/Documents/GitHub/SarcasmClassBERT/train.csv')
test = pd.read_csv('/Users/florian/Documents/GitHub/SarcasmClassBERT/test.csv')
val = pd.read_csv('/Users/florian/Documents/GitHub/SarcasmClassBERT/validate.csv')
X_train = train[["text"]].copy()
X_test = test[["text"]].copy()
X_val = val[["text"]].copy()
Y_train = train[["labels"]].copy()
Y_test = test[["labels"]].copy()
Y_val = val[["labels"]].copy()
X_train.head()

def preprocess(data):
  data = data.astype(str)
  data = data.apply(lambda x: x.lower())
  data = data.apply((lambda x: re.sub('[^a-zA-Z0-9\s]','',x)))

  for idx,row in enumerate(data):
      row = row.replace('rt',' ')

  return data

X_train["text"] = preprocess(X_train["text"])
X_val["text"] = preprocess(X_val["text"])
X_test["text"] = preprocess(X_test["text"])
X_train.head()


keras.backend.clear_session()

encoder_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1"
preprocessor_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(preprocessor_url, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(encoder_url, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.2)(net)
  net = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(net)
  return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()


epochs =  4
batch_size = 64
steps_per_epoch = math.floor(X_train.shape[0]/batch_size)
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)
init_lr = 3e-5

loss = tf.keras.losses.BinaryCrossentropy()

metrics = tf.metrics.BinaryAccuracy()

optimizer = optimization.create_optimizer(init_lr=init_lr,
            num_train_steps=num_train_steps, 
            num_warmup_steps=num_warmup_steps,
            optimizer_type='adamw')


json_file = open('usercode/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

classifier_model = model_from_json(loaded_model_json, custom_objects={'KerasLayer':hub.KerasLayer})

classifier_model.load_weights("usercode/model.h5")
print("Loaded model from disk")

classifier_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)

classifier_model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

history = classifier_model.fit(X_train, 
pd.get_dummies(Y_train["labels"]),
     validation_data=(X_val, pd.get_dummies(Y_val["labels"])),
     epochs=epochs, batch_size=batch_size, verbose=1)

plt.rcParams["figure.figsize"] = (12,8)
N = np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_loss"], label="val_loss")
plt.plot(N, history.history['binary_accuracy'], label="binary_accuracy")
plt.plot(N, history.history["val_binary_accuracy"], label="val_binary_accuracy")
plt.title("Training Loss and Accuracies (Finetuning BERT for Sarcasm Classification)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

loss, accuracy = classifier_model.evaluate(X_test, pd.get_dummies(Y_test["labels"]))

print('Loss: ', loss)
print('Accuracy: ', accuracy)

actuals = Y_test["labels"]
Y_predicted = classifier_model.predict(X_test)
predictions= np.argmax(Y_predicted,axis=1)

cm = confusion_matrix(actuals, predictions)

plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')

target_names = ['Non-Sarcastic', "Sarcastic"]
print(classification_report(actuals, predictions, target_names=target_names))


