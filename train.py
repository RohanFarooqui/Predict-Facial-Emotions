#############################################
### Author      : M.ROHAN FAROOQUI          #
##  Application : Predict Facial Emotions   #
##  File        : train.py                  #
############################################# 

### Machine Learning Libraries for Training Data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam 
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
#> Logs epoches in CSV
from keras.callbacks import CSVLogger

##> Variables Define 

#> Train Folder Path
trainingDirectory   = 'data/train'
#> Test Folder Path
validationDirectory = 'data/test'
#> Modal Weights / CSV log file name 
file_name = "epochs_1000_model"
epoch_val = 1000

##> Initialize the training and validation generators
trainDataGenerator      = ImageDataGenerator(rescale=1./255)
validationDataGenerator = ImageDataGenerator(rescale=1./255)

trainGenerator = trainDataGenerator.flow_from_directory(
        trainingDirectory,target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode='categorical')

validation_generator = validationDataGenerator.flow_from_directory(
        validationDirectory,target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode='categorical')

##> Convolution Network Architecture

#> Sequential : allows you to create models
emotionsModal = Sequential()
#> Conv2D : Creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.
#> Kernal Size : An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
#> activation relu : Use to neglact values less than or equal to ZERO 
emotionsModal.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotionsModal.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#> MaxPooling2D : Take the maximum value over an input window size
emotionsModal.add(MaxPooling2D(pool_size=(2, 2)))
#> Dropout : To Prevent Neural Networks from Overfitting
# There are two techniques for from prevent modal from unerfitting and overfitting we need some regulization technique
#tech 1 : dropout.
#tech 2 : batch normalization.
emotionsModal.add(Dropout(0.25))

emotionsModal.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotionsModal.add(MaxPooling2D(pool_size=(2, 2)))
emotionsModal.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotionsModal.add(MaxPooling2D(pool_size=(2, 2)))
emotionsModal.add(Dropout(0.25))
#> Flatten take array of elements and convert into 1D
emotionsModal.add(Flatten())
#> Dense : It feeds all outputs from the previous layer to all its neurons, each neuron providing one output to the next layer.
emotionsModal.add(Dense(1024, activation='relu'))
emotionsModal.add(Dropout(0.5))
emotionsModal.add(Dense(7, activation='softmax'))


##> Compile and train the model
# Optimizer : update weight in back progration using in loss function.
emotionsModal.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
#> Epochs : The number times that the learning algorithm will work through the entire training dataset.

#> Save Logs in CSV FILE
csv_logger = CSVLogger("train_modal_logs/"+file_name+'.log', append=True, separator=',')
#> fit_generator : Used to train our machine learning and deep learning models
#> Epoch : indicates the number of passes of the entire training dataset
emotionsModal_info = emotionsModal.fit_generator(trainGenerator,steps_per_epoch=28709 // 64,epochs=epoch_val,validation_data=validation_generator,validation_steps=7178 // 64, callbacks=[csv_logger])

##> Save Modal Weights 
emotionsModal.save_weights("train_modal/"+file_name+'.h5')
