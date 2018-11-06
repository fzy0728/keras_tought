import bilstm_attention
import keras.utils as utils

EPOCHS = 10
model, (train_x, train_y), (test_x, test_y) = bilstm_attention.create_model()
train_y = utils.to_categorical(train_y)
test_y = utils.to_categorical(test_y)
print(train_y[0])
# train model
model.fit(train_x, train_y,batch_size=64,epochs=EPOCHS, validation_data=[test_x, test_y])
model.save('model/bilstm.h5')