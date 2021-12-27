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


##> Variables Define 

#> Train Folder Path
trainingDirectory   = 'data/train'
#> Test Folder Path
validationDirectory = 'data/test'

##> Initialize the training and validation generators
trainDataGenerator      = ImageDataGenerator(rescale=1./255)
validationDataGenerator = ImageDataGenerator(rescale=1./255)

trainGenerator = trainDataGenerator.flow_from_directory(
        trainingDirectory,target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode='categorical')

validation_generator = validationDataGenerator.flow_from_directory(
        validationDirectory,target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode='categorical')

##> Convolution Network Architecture
emotionsModal = Sequential()

emotionsModal.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotionsModal.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotionsModal.add(MaxPooling2D(pool_size=(2, 2)))
emotionsModal.add(Dropout(0.25))

emotionsModal.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotionsModal.add(MaxPooling2D(pool_size=(2, 2)))
emotionsModal.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotionsModal.add(MaxPooling2D(pool_size=(2, 2)))
emotionsModal.add(Dropout(0.25))

emotionsModal.add(Flatten())
emotionsModal.add(Dense(1024, activation='relu'))
emotionsModal.add(Dropout(0.5))
emotionsModal.add(Dense(7, activation='softmax'))


##> Compile and train the model
emotionsModal.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
#> Epoches : The number times that the learning algorithm will work through the entire training dataset.



emotionsModal_info = emotionsModal.fit_generator(trainGenerator,steps_per_epoch=28709 // 64,epochs=50,validation_data=validation_generator,validation_steps=7178 // 64)

##> Save Modal Weights 
emotionsModal.save_weights('model.h5')