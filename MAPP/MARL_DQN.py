import tensorflow as tf


class MARL_DQN(tf.keras.layers.Layer):
    def __init__(self, num_actions):
        super().__init__()

        self.num_actions = num_actions

        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

        self.metrics_list = [tf.keras.metrics.Mean(name="loss")]

        self.q_net = [tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides = 4, padding='same', activation='relu'),
                      tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()),
                      tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides = 2, padding='same', activation='relu'),
                      tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()),
                      tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
                      tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalMaxPool2D()),
                      tf.keras.layers.Flatten(),
                      tf.keras.layers.Dense(512),
                      tf.keras.layers.Dense(self.num_actions,
                                            activation=None,
                                            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
                                            bias_initializer=tf.keras.initializers.Constant(-0.2))] #last is q_values layer

        '''self.q_net = [tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides = 4, padding='same', activation='relu'), 
                      tf.keras.layers.MaxPool2D(),
                      tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides = 2, padding='same', activation='relu'),
                      tf.keras.layers.MaxPool2D(),
                      tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
                      tf.keras.layers.GlobalMaxPool2D(),
                      tf.keras.layers.Dense(512),
                      tf.keras.layers.Dense(self.num_actions,
                                            activation=None,
                                            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
                                            bias_initializer=tf.keras.initializers.Constant(-0.2))] #last is q_values layer'''

    """# Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.

    def _dense_layer(self, num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))"""

    def __call__(self, x, training = False):
        """
        Predicts Q-values (expected rewards) for each action.

        Parameters:
            x (ndarray): observation
            training (boole): enables to train the network

        Returns:
            x (ndarray): Q-values
        """
        for layer in self.q_net:
            x = layer(x)
        return x

    @tf.function
    def train(self, observation, target):
        """
        Trains the network via backpropagation.

        Parameters:
            observation (ndarray): the state of the environment
            target (float): expected reward from "optimal" action
        """

        with tf.GradientTape() as tape:
            predictions = self(observation, training=True) # type predictions:  <class 'tensorflow.python.framework.ops.Tensor'>
            loss = self.loss(target, predictions) # type loss  <class 'tensorflow.python.framework.ops.Tensor'>


        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update loss metric
        self.metrics[0].update_state(loss)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    # do we really need a test function?
    # Isn't the reward of the model our test and the only metric we log so far?
    @tf.function
    def test(self, observation, target):

        predictions = self(observation, training=True)
        loss = self.loss(target, predictions)

        # update loss metric
        self.metrics[0].update_state(loss)

        # Return a dictionary mapping metric names to current value
        return {"val_" + m.name : m.result() for m in self.metrics}