#%%
# from keras.src import ops
# from keras.src.layers import RNN
import tensorflow as tf
import keras

#%%
# First, let's define a RNN Cell, as a layer subclass.
class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = tf.matmul(inputs, self.kernel)
        output = h + tf.matmul(prev_output, self.recurrent_kernel)
        return output, [output]

# Let's use this cell in a RNN layer:

#%% simple rnn 1 cell
# cell = MinimalRNNCell(32)
# x = keras.Input((None, 5))
# layer = keras.layers.RNN(cell)
# y = layer(x)

#%% my rnn
cell = MinimalRNNCell(32)
cells = [MinimalRNNCell(32), MinimalRNNCell(64), MinimalRNNCell(64)]
# mymodel = keras.models.Sequential()
# mymodel.add(keras.layers.RNN(cell))
# mymodel.build((1, 10, 1))
# print(mymodel.summary())
inputlayer = keras.Input((10, 1))
rnnlayer = keras.layers.RNN(cells) (inputlayer)
denselayer = keras.layers.Dense(4) (rnnlayer)
outputlayer =  denselayer

mymodel = keras.Model(inputlayer , outputlayer )
print(mymodel.summary())

#%% fitting


#%% stacked
# Here's how to use the cell to build a stacked RNN:

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = keras.layers.RNN(cells)
y = layer(x)
