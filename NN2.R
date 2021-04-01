###############################################################
# Neural Networks (DNN) of MNIST Zalando Data - Otto Dahlin 
###############################################################

library(keras)
library(keras)
library(caret)
library(lattice)
library(ggplot2)
library(rlang)
library(tidyverse)
library(rlang)


data <- dataset_fashion_mnist()
str(data)

# The labels are as follows:
#  0: T-shirt/tops
# 1: Trouser
# 2: Pullover
# 3: Dress
# 4: Coat
# 5: Sandal
# 6: Shirt
# 7: Sneaker
# 8: Bag
# 9: Ankle Boot


x_train <- data$train$x
y_train <- data$train$y

x_test <- data$test$x
y_test <- data$test$y


# reshape and rescale

x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Now values will range between 0 and 1
x_train <- x_train / 255
x_test <- x_test / 255

hist(x_train)
hist(x_train)


# ONE-hot encoding. We have 10 different categories
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


# 2 hidden layers:
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 512,  activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate=0.4) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units = 10, activation = "softmax") 


summary(model)
model %>%
  compile(loss = "categorical_crossentropy",
          optimizer  = optimizer_rmsprop(),
          metrics = "accuracy")


# fit adjusted model
history <- model %>%
  fit(x_train,
      y_train,
      epochs = 30,
      batch_size = 32,
      validation_split = 0.2)


model %>% evaluate(x_test, y_test, verbose = 0)
# test data accuracy: 0.8192000
#########################################################################


# specify a 2-layer NN with relu activation in thidden layer and softmax in last layer
model <- keras_model_sequential()
model %>%
  layer_dense(units = 50, activation = "relu", input_shape = c(784)) %>%
  layer_dense(units = 10, activation = "softmax")

summary(model)
model %>%
  compile(loss = "categorical_crossentropy",
          optimizer  = optimizer_rmsprop(),
          metrics = "accuracy")


# fit adjusted model
history <- model %>%
  fit(x_train,
      y_train,
      epochs = 50,
      batch_size = 32,
      validation_split = 0.2)

model %>% evaluate(x_test, y_test, verbose = 0)
# test accuracy: 0.8683000

#############################################################################################
############################################################################################

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

# loss function:
model %>%
  compile(loss = "categorical_crossentropy",
          optimizer  = optimizer_rmsprop(),
          metrics = "accuracy")


# fit adjusted model
history <- model %>%
  fit(x_train,
      y_train,
      epochs = 70,
      batch_size = 128,
      validation_split = 0.2)

# acc: 0.9121
# val acc: 0.8885

model %>% evaluate(x_test, y_test, verbose = 0)
# test acc: 0.88
#########################################################################

# Bästa arkitektur so far: ingen markant overfitting alls som i vanliga fall ovan

model <- keras_model_sequential()
model %>%  
  layer_dense(units = 784, activation = "relu", input_shape = c(784)) %>%  
  layer_dense(units = 512, activation = "relu") %>%                            
  layer_dense(units = 10,  activation = "softmax")  

summary(model)

# loss function:
model %>%
  compile(loss = "categorical_crossentropy",
          optimizer  = optimizer_rmsprop(),
          metrics = "accuracy")


# fit adjusted model
history <- model %>%
  fit(x_train,
      y_train,
      epochs = 50,
      batch_size = 1000,
      validation_split = 0.2)

# acc: 0.92
# val acc: 0.89
# train loss: 0.2041


model %>% evaluate(x_test, y_test, verbose = 0)
# test acc: 0.88
# test loss: 0.347

# since the test loss is greater than implies overfitting..


# we will not try another architecture based on the above results....
# we will specify the model addina second hidden layer with 512 neurons
# we'll also use a technique called "dropout" which randomly sets some fraction of the neurons in each layer
# to zero during each training step, which can help to avoid overfitting.


# drop out regularization below to avoid overfitting.
# with probability dropout rate
model1 <- keras_model_sequential()
model1 %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 512, activation = "relu") %>%  # new layer
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = "softmax")

model1 %>% compile(
  optimizer = "rmsprop", 
  loss = "categorical_crossentropy", 
  metrics = c("accuracy"))


# fit adjusted model
history <- model1 %>%
  fit(x_train,
      y_train,
      epochs = 20,
      batch_size = 1000,
      validation_split = 0.2)

# loss train: 0.2767
# train acc: 0.89

# val loss: 0.34
# val acc: 0.87


model1 %>% evaluate(x_test, y_test, verbose = 0)
# test loss: 0.37
# test acc: 0.86



# * fortsättning:


# MODEL 2 - Arkitektur nr 3
model2 <- keras_model_sequential()
model2 %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 128, activation = "relu") %>%  # new layer
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = "softmax")

model2 %>% compile(
  optimizer = "rmsprop", 
  loss = "categorical_crossentropy", 
  metrics = c("accuracy"))



# loss function:
model2 %>%
  compile(loss = "categorical_crossentropy",
          optimizer  = optimizer_rmsprop(),
          metrics = "accuracy")

# fit adjusted model
history <- model2 %>%
  fit(x_train,
      y_train,
      epochs = 30,
      batch_size = 128,
      validation_split = 0.2)
# train loss: 0.29
# train acc: 0.89

# val loss: 0.33
# val acc: 0.88


model2 %>% evaluate(x_test, y_test, verbose = 0)
# test loss: 0.35
# test acc: 0.87


# Last try:


# MORE modifications: Deeper Network, Arkitektur nr 4

model3 <- keras_model_sequential()
model3 %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>% # input and hidden layer
  layer_dropout(rate=0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units = 10, activation = "softmax") # output layer (10 different digits,)


summary(model3)
model3 %>%
  compile(loss = "categorical_crossentropy",
          optimizer  = optimizer_rmsprop(),
          metrics = "accuracy")


# fit adjusted model
history <- model3 %>%
  fit(x_train,
      y_train,
      epochs = 50,
      batch_size = 1000,
      validation_split = 0.2)

# loss train: 0.2393
# train acc: 0.9095

# val loss: 0.30
# val acc: 0.89


model3 %>% evaluate(x_test, y_test, verbose = 0)
# test loss: 0.33
# test acc: 0.88


##############################################################


x_train <- data$train$x
y_train <- data$train$y

x_test <- data$test$x
y_test <- data$test$y


# reshape and rescale
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Now values will range between 0 and 1
x_train <- x_train / 255
x_test <- x_test / 255

hist(x_train)
hist(x_train)


# ONE-hot encoding. We have 10 different categories
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


########################################################

model3 <- keras_model_sequential()
model3 %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>% # input and hidden layer
  layer_dropout(rate=0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate=0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate=0.2) %>%
  layer_dense(units = 10, activation = "softmax") # output layer (10 different digits,)


summary(model3)
model3 %>%
  compile(loss = "categorical_crossentropy",
          optimizer  = optimizer_rmsprop(),
          metrics = "accuracy")


# fit adjusted model
history <- model3 %>%
  fit(x_train,
      y_train,
      epochs = 50,
      batch_size = 1000,
      validation_split = 0.2)



######################################################################


# feature extraction
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1)) %>% # första lager
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% #subsampling lager
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")


# nedan lager sköter klassificering, fully conncected layers
model <- model %>%
  layer_flatten() %>% # 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

# 
train_datagen <- image_data_generator(
  rescale = 1/255,
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  horizontal_flip = TRUE,
  vertical_flip = TRUE
)

val_datagen = image_data_generator(
  rescale = 1/255,
)


batch_size = 64 # 64 bilder i taget kommer att köras. Taget från luften. Vanligt ta potens av 2.


train_generator <- flow_images_from_data(
  x = x_train, 
  y = y_train,
  generator = train_datagen,
  batch_size = batch_size,
  seed = 1234
)

# bilder för validering (dras från valideringssetet)
val_generator <- flow_images_from_data(
  x = x_val, 
  y = y_val,
  generator = val_datagen
)



# förlustfunktionen, en binomial likelihood funktion med 10 klasser därför categorical
# optimeringsalgoritm (sättet att välja learning rate, kan köra RMSprop)
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)


# kör träningsloopen
# 100 epocher, 1 epoch är att gå igenom alla bilder ett varv.
# Om epoch är 100 så tränar vi alla bilder 100 gånger.

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = as.integer(n_samples_train/batch_size),
  epochs = 20,
  validation_data = val_generator,
  validation_steps = as.integer(n_samples_val/batch_size)
)


img <- x_test[5,,, ,drop = FALSE]
dim(img)
class_pred <- model %>% predict(img)
as.integer(k_argmax(class_pred))



pred <- model %>% predict(x_test) 
pred_class <- as.factor(as.numeric(k_argmax(pred)))
actual_class <-  as.factor(as.numeric(k_argmax(y_test)))
caret::confusionMatrix( pred_class, actual_class) # confusion matrix


model %>% evaluate(x_test, y_test,verbose = 0)

########################################################
# END
########################################################
