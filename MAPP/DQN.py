import tensorflow as tf


class DQN(tf.keras.layers.Layer):
    def __init__(self, num_actions):
        super().__init__()

        self.num_actions = num_actions

        self.loss = tf.keras.losses.MeanSquaredError() 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

        self.metrics_list = [tf.keras.metrics.Mean(name="loss")]
        
        self.q_net = [tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu'), 
                      tf.keras.layers.MaxPool2D(),
                      tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu'),
                      tf.keras.layers.GlobalMaxPool2D(),
                      tf.keras.layers.Dense(self.num_actions,
                                            activation=None,
                                            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
                                            bias_initializer=tf.keras.initializers.Constant(-0.2))] #last is q_values layer

    """# Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.

    def _dense_layer(self, num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))"""
    
    def call(self, x, training = False): # x is observation/ state of the environment
        x = tf.cast(tf.expand_dims(x, 0), tf.float32) / 256.
        for layer in self.q_net:
            x = layer(x) 
        q_values = x
        return q_values 
    

    def reset_metrics(self):

        for metric in self.metrics:
            metric.reset_states()

    @tf.function
    def train(self, observation, target): 

        with tf.GradientTape() as tape:
            predictions = self(observation, training=True)
            loss = self.loss(target, predictions)
      
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
      
        # update loss metric
        self.metrics[0].update_state(loss)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def test(self, observation, target): 

        predictions = self(observation, training=True)
        loss = self.loss(target, predictions)
      
        # update loss metric
        self.metrics[0].update_state(loss)

        # Return a dictionary mapping metric names to current value
        return {"val_" + m.name : m.result() for m in self.metrics}