import numpy as np
import tensorflow as tf
import keras

#%% Generate synthetic sequential data
seq_length = 10
input_dim = 1
num_samples = 1000

def generate_data(seq_length, num_samples):
    X = np.random.randn(num_samples, seq_length, input_dim)
    y = np.sum(X, axis=1)  # Sum across the sequence dimension
    return X, y

X_train, y_train = generate_data(seq_length, num_samples)
X_val, y_val = generate_data(seq_length, 100)

#%% class definition
# Define CRU cell
class CRUCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CRUCell, self).__init__(**kwargs)
        self.units = units
        
        ##should fix it
        self.state_size = units  # Specify the state size as the number of units
        
        # Define CRU components here (for simplicity, we omit actual components)

    # def build(self, input_shape):
    def build(self, input_shape):
        # Define the weights for input connections
        
        ##should fix it
        input_dim = input_shape[-1]
        ##old nothing
        
        
        
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer='glorot_uniform',
                                      name='kernel')
        
        # Define the weights for recurrent connections
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                 initializer='orthogonal',
                                                 name='recurrent_kernel')
        
        # Define the biases
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='zeros',
                                    name='bias')
        
        self.built = True
        
        
    def call(self, inputs, states):
        
        ##NOTE:
        # def cru_step(inputs, states):
        #     h_tm1 = states[0]  # Previous state
        #     W = states[1]      # Weight matrix
        #     b = states[2]      # Bias vector

        #     # Continuous-time recurrent unit operation
        #     h_t = K.dot(inputs, W) + K.dot(h_tm1, W) + b
        #     h_t = K.tanh(h_t)

        #     return h_t, [h_t]
        
        # Implement the forward pass of the CRU cell
        ##should fix it
        prev_output = states  # Previous output
        ##old
        # prev_output = states[0]  # Previous output
        
        # Compute the weighted sum of inputs and previous output
        z = tf.matmul(inputs, self.kernel) + tf.matmul(prev_output, self.recurrent_kernel) + self.bias
        output = tf.nn.relu(z)  # Apply ReLU activation function
        
        ##should fix it
        return output, output  # Return both output and new state
        ## old
        # return output, [output]
        
        
#%% build
seq_length = 10
input_dim = 1
num_samples = 1000
hidden_units = 64

mymodel = keras.models.Sequential()
# mymodel.add( 
#     keras.layers.Dense(units=input_dim,input_shape=(seq_length,input_dim)) 
#     )
mymodel.add( tf.keras.layers.RNN(CRUCell(hidden_units), return_sequences=True)  )
# mymodel.add( tf.keras.layers.RNN()  )
mymodel.add( keras.layers.Dense(1) )
mymodel.build(input_shape=(None,seq_length, input_dim))
mymodel.compile(optimizer='rmsprop', loss=keras.losses.MeanSquaredError() )
print(mymodel.summary())

#%% their build
# # Build CRU model
# hidden_units = 64
# inputs = tf.keras.Input(shape=(seq_length, input_dim))
# # print(inputs.output_shape)
# rnn_layer = tf.keras.layers.RNN(CRUCell(hidden_units), return_sequences=True) 
# outputs = rnn_layer(inputs)

# # hidden_units = 64
# # inputs = tf.keras.Input(shape=(seq_length, input_dim))
# # rnn_layer = tf.keras.layers.RNN(CRUCell(hidden_units), return_sequences=True)
# # outputs = rnn_layer(inputs)

# # Add a final dense layer for prediction
# predictions = tf.keras.layers.Dense(1)(outputs)

# model = tf.keras.Model(inputs=inputs, outputs=predictions)

# # Compile the model
# model.compile(optimizer='adam', loss='mse')

# # Train the model
# model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# # Evaluate the model
# loss = model.evaluate(X_val, y_val)
# print("Validation Loss:", loss)
