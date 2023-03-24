""""
A bunch of function that I used in Dog Breed Identification problem
"""



""""
a function that takes an image filepath and turns it into a tensor
"""
def process_image(imagepath):
  #read img file
  image= tf.io.read_file(imagepath)
  #turn img into num tensor with 3 color channels (red, green, blue)
  image= tf.image.decode_jpeg(image, channels=3) #notice that our images are in jpeg format
  #convert colour channel values from 0-225 values to 0-1 values (for ccomputational efficiency purposes)
  image= tf.image.convert_image_dtype(image, tf.float32)
  #resize the img
  image= tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image


""""
function that turns data into batches
"""
def data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  #case 1: test data
  if test_data:
    print("creating test data batches in process..")
    data= tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data_batch= data.map(process_image).batch(BATCH_SIZE)
    return data_batch
  #case2: validation data
  elif valid_data:
    print("creating validation data batches in process..")
    data= tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
    data_batch= data.map(img_label).batch(BATCH_SIZE)
    return data_batch
  #case 3: training data
  else :
    print("creating training data batches in process..")
    data= tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y)))

    #shuffling pathnames and labels before mapping image processor function is faster
    data= data.shuffle(buffer_size=len(x))
    #create image/label tuple
    data= data.map(img_label)
    #turn data into batches
    data_batch= data.batch(BATCH_SIZE)
  return data_batch



"""
functionn to create TensorBoard callback
"""
def create_tensorboard_callback():
  #log dir for storing tensorboard logs
  logdir= os.path.join( "drive/My Drive/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  return tf.keras.callbacks.TensorBoard(logdir)


"""
function trains a given model and returns a trained one
"""
def train_model():
  #create the model
  model= create_model()
  #create tensorboard session everytime we train a model
  tensorboard= create_tensorboard_callback()
  #fit the model
  model.fit(x= train_data,
            epochs=NUM_EPOCHS,
            validation_data=valid_data,
            validation_freq=1,
            callbacks= [tensorboard, early_stopping])
  #return the fitted version
  return model



"""
function turns prediction probabilities into prediction labels 
"""
def turn_pred_label(predictions_probabilities):
  """
  Turns array of prediction probabilities into label
  """
  return unique_breeds[np.argmax(predictions_probabilities)]



"""
function takes a batched dataset of (image, label) tensors and returns separate arrays of images and labels 
"""
def data_unbatch(data):

  images=[]
  labels=[]
  for image, label in data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(unique_breeds[np.argmax(label)])
  return images, labels

"""
function that :
Take an array of prediction probabilities, an array of truth labels, an array of images and an integer.
Convert the prediction probabilities to a predicted label.
Plot the predicted label, its predicted probability, the truth label and target image on a single plot.
"""
def plot_pred(prediction_probabilities, labels, images, n=1):

  pred_proba, true_label, image= prediction_probabilities[n], labels[n], images[n]

  pred_label= turn_pred_label(pred_proba)

  #plot image & remove ticks

  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])

  #change the color of the title depending of the prediction if it is right or wrong

  if pred_label== true_label:
    plt.title("{} {:2.0f}% ({})".format(pred_label,
                                        np.max(pred_proba)*100,
                                        true_label),
                                        color="green")
  else :
    plt.title("{} {:2.0f}% ({})".format(pred_label,
                                        np.max(pred_proba)*100,
                                        true_label),
                                        color="red")


"""
function that plots the top 10 highest prediction confidences along with the truth
label for sample n
"""
def plot_pred_confidence(prediction_probabilities, labels, n=1):

pred_proba, true_label = prediction_probabilities[n], labels[n]
# get the predicted label
pred_label = turn_pred_label(pred_proba)
# the top 10 prediction probabilities indexes
top10_pred_index = pred_proba.argsort()[-10:][::-1]
# the top 10 prediction probabilities values
top10_pred_values = pred_proba[top10_pred_index]
# the top 10 prediction labels
top10_pred_label = unique_breeds[top10_pred_index]

# setting up the plot
top10_plot = plt.bar(np.arange(len(top10_pred_label)),
                     top10_pred_values,
                     color="grey")
plt.xticks(np.arange(len(top10_pred_label)),
           labels=top10_pred_label,
           rotation="vertical")

# setting up the color of the label
if np.isin(true_label, top10_pred_label):
    top10_plot[np.argmax(top10_pred_label == true_label)].set_color("green")
else:
    pass


"""
functions to save and load the model
"""


def save_model(model, suffix=None):
    # first cretae a dir with current time
    model_dir = os.path.join("drive/My Drive/models", datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
    model_path = model_dir + "-" + suffix + ".h5"
    print(f"saving the model to : {model_path}.... ")
    model.save(model_path)
    return model_path


def load_model(model_path):
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
    return model

