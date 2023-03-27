import tensorflow as tf


class MARL_DQN(tf.keras.Model):
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

    def call(self, x, training = False):
        """
        Predicts Q-values (expected rewards) for each action.

        Parameters:
            x (ndarray): observation
            training (boole): enables to train the network

        Returns:
            x (ndarray): Q-values
        """
        for layer in self.q_net:
            x = layer(x, training = training)
        return x


    def reset_metrics(self):
        """
        Resets the metrics.
        """
        for metric in self.metrics:
            metric.reset_states()


    @tf.function
    def train(self, observation, action, target):
        """
        Trains the network via backpropagation.

        Parameters:
            observation (ndarray): the state of the environment
            action (int): the action the agent chose when faced with the observation
            target (float): expected reward from "optimal" action
        """

        with tf.GradientTape() as tape:
            predictions = self(observation, training=True)
            predictions = tf.gather(predictions, action, axis = 1, batch_dims=1)
            loss = self.loss(target, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(loss)

        return {m.name: m.result() for m in self.metrics}
