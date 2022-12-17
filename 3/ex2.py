
# Translate the previous exercise with TensorFlow to Keras's sequential model.
# Print the model summary to screen.
# Verify that predictions between both models are in agreement.
# Print the weights from the model object.

# ==============================================
# INITIALIZE Neural Network (Sequential) model
# ==============================================

# compose the NN model
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects

tf.random.set_seed(0)

model = tf.keras.Sequential()
model.add(Dense(1, input_shape=(1,)))   # nota: qui non c'è una funzione di attivazione (non lineare), 
                                        # mentre è necessario aggiungerla per i livelli successivi
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(2, activation='linear'))
model.add(Dense(1))

# compile the model choosing optimizer, loss and metrics objects
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

# ==============================================
# Get a summary of our composed model
# ==============================================
model.summary()

print(model.get_weights())

# Create random weights and biases of the same shape as the model (non va)

weights = [tf.Variable([[1.1111]]), # <-- perchè tanto il primo non serve a niente.. giusto?
           tf.Variable(tf.random.normal([1, 5])),
           tf.Variable(tf.random.normal([5, 2])),
           tf.Variable(tf.random.normal([2, 1]))]

biases = [ tf.Variable([2.22222]), # <-- idem..?
           tf.Variable(tf.random.normal([5])),
           tf.Variable(tf.random.normal([2])),
           tf.Variable(tf.random.normal([1]))]

print('\n\n Weights \n')
print(weights)
print('\n\n Biases \n')
print(biases)

for i in range(len(weights)):
    model.layers[i].set_weights([weights[i], biases[i]])

print(model.get_weights())


# ==============================================
# Verify model
# ==============================================

