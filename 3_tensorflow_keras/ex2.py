
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

model.add(Dense(5, activation='sigmoid', input_dim=1))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='linear'))

# compile the model choosing optimizer, loss and metrics objects
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

# ==============================================
# Get a summary of our composed model
# ==============================================
model.summary()

#print(model.get_weights())

# Create random weights and biases of the same shape as the model (non va)

weights = [tf.Variable(tf.random.normal([1, 5])),
           tf.Variable(tf.random.normal([5, 2])),
           tf.Variable(tf.random.normal([2, 1]))]

biases = [ tf.Variable(tf.random.normal([5])),
           tf.Variable(tf.random.normal([2])),
           tf.Variable(tf.random.normal([1]))]

for i in range(len(weights)):
    model.layers[i].set_weights([weights[i], biases[i]])

#print('\n\n\n')
#print(model.get_weights())

# ==============================================
# Verify model
# ==============================================
    
x = tf.linspace([-1.0], [1.0], 10)
y = model.predict(x)

print(y)