import os
#import setGPU
import numpy as np
import tensorflow as tf
from collections import namedtuple
import h5py
import vande.vae.layers as layers

@tf.function
def kl_loss(z_mean, z_log_var):
    kl = 1. + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    return -0.5 * tf.reduce_mean(kl, axis=-1) # multiplying mse by N -> using sum (instead of mean) in kl loss (todo: try with averages)
    
class Sampling(tf.keras.layers.Layer):
    """ 
    Custom sampling layer for latent space
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        self.add_loss(tf.reduce_mean(kl_loss(z_mean, z_log_var))) # adding kl-loss to layer
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super(Sampling, self).get_config()


class VAEClassifier(tf.keras.Model):
    def __init__(self, params,mean,stdev, **kwargs):
        super(VAEClassifier, self).__init__(name='VAEClassifier',**kwargs)
        self.params = params
        self.norm_mean=mean
        self.norm_stdev=stdev
        self.kernel_n=self.params.kernel_ini_n
        self.loss_fn=tf.keras.losses.BinaryCrossentropy()
        # Custom Normalization layer (using your custom layers module)
        self.loss_tracker = tf.keras.metrics.BinaryCrossentropy(name='bce')
        self.auc_metric = tf.keras.metrics.AUC(name='auc')
        self.acc_metric=tf.keras.metrics.BinaryAccuracy(name='accuracy')
        self.normalization = layers.StdNormalization(mean_x=self.norm_mean, std_x=self.norm_stdev)
        #self.normalization = tf.keras.layers.BatchNormalization()
        # Add channel dimension
        self.expand_dims = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=3))

        # Conv2D layer
        self.conv2d = tf.keras.layers.Conv2D(filters=self.kernel_n, kernel_size=self.params.kernel_sz, activation=self.params.activation, kernel_initializer=self.params.initializer)

        # Squeeze
        self.squeeze = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))

        # Conv1D layers
        self.kernel_n+=4
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=self.kernel_n, kernel_size=self.params.kernel_1D_sz, activation=self.params.activation, kernel_initializer=self.params.initializer)
        self.kernel_n+=4
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=self.kernel_n, kernel_size=self.params.kernel_1D_sz, activation=self.params.activation, kernel_initializer=self.params.initializer)

        # Pooling
        self.pooling = tf.keras.layers.AveragePooling1D()
        self.flatten = tf.keras.layers.Flatten()
        # Dense layers
        self.dense1 = tf.keras.layers.Dense(int(self.params.z_sz*params.dense_factor1), activation=self.params.activation, kernel_initializer=self.params.initializer)
        self.dense2 = tf.keras.layers.Dense(int(self.params.z_sz*params.dense_factor2), activation=self.params.activation, kernel_initializer=self.params.initializer)

        # Latent space
        self.z_mean = tf.keras.layers.Dense(self.params.z_sz, name='z_mean')
        self.z_log_var = tf.keras.layers.Dense(self.params.z_sz, name='z_log_var')
        self.z = Sampling()

        # Probability output
        self.prob_output = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=self.params.initializer)
        self.build(input_shape=(None,)+params.input_shape)
        self.summary()
        #self.summary()
    def call(self, x):
        x = self.normalization(x)
        x = self.expand_dims(x)
        x = self.conv2d(x)
        x = self.squeeze(x)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        #z_mean = self.z_mean(x)
        #z_log_var = self.z_log_var(x)
        #z = self.z((z_mean, z_log_var))
        y_pred = self.prob_output(x)
        
        return y_pred  # Only return the probability output for training
    #def build_model(self):
    #    inputs=tf.keras.layers.Input(shape=self.params.input_shape)
    #    outputs=self.call(inputs)
    #    self.model=tf.keras.Model(inputs=inputs,outputs=outputs)
    #    print(self.model.summary())
    #    return self.model
    @property
    def metrics(self):
        return [self.loss_tracker, self.auc_metric,self.acc_metric]
    def save_model(self, filepath):
        self.save_weights(os.path.join(filepath,'weights.h5'))
        with h5py.File(os.path.join(filepath,'model_params.h5'),'w') as f:
            ds = f.create_group('params')
            for name, value in self.params._asdict().items():
                ds.attrs[name] = value
    
    def load_model(self, filepath):
        #custom_objects = {'Sampling': layers.Sampling, 'StdNormalization': layers.StdNormalization}
        
        self.load_weights(os.path.join(filepath,'weights.h5'))
        print('weights loaded from file')
        return 0
    
    # def summary(self):
    #     x = tf.keras.layers.Input(shape=(None,)+self.params.input_shape)
    #     model = tf.keras.Model(x, outputs=self.call(x))
    #     return model.summary()

    @tf.function
    def train_step(self,data):
        #tf.print("Using custom train step")
        x_batch, y_batch = data
        with tf.GradientTape() as tape:
            y_pred = self(x_batch, training=True)
            bce_loss=self.loss_fn(y_batch, y_pred)
            #kl_loss=self.losses[0]*self.params.beta
            total_loss=bce_loss#+kl_loss
        #tf.print('min=',tf.math.reduce_min(y_pred))
        #tf.print('max=',tf.math.reduce_max(y_pred))
        gradients=tape.gradient(total_loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
        self.compiled_metrics.update_state(y_batch,y_pred)
        self.loss_tracker.update_state(y_batch,y_pred)
        self.auc_metric.update_state(y_batch,y_pred)
        self.acc_metric.update_state(y_batch,y_pred)

        return {'total_loss':total_loss,'bce loss':self.loss_tracker.result(),'auc':self.auc_metric.result(),'accuracy':self.acc_metric.result()}

    def test_step(self, data):
        # Unpack the data
        x_batch, y_batch = data
        # Compute predictions
        y_pred = self(x_batch, training=False)
        # Updates the metrics tracking the loss
        #self.loss_fn(y_batch, y_pred)
        # Update the metrics.
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y_batch, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
        
    #def summary_model(self):
    #    inputs = tf.keras.Input(shape=(100, 3))
    #    outputs = self.call(inputs)
    #    tf.keras.Model(inputs=inputs, outputs=outputs, name="thing").summary()





class VAELargeClassifier(tf.keras.Model):
    def __init__(self, params,mean,stdev, **kwargs):
        super(VAELargeClassifier, self).__init__(**kwargs)
        self.params = params
        self.mean=mean
        self.stdev=stdev
        self.kernel_n=self.params.kernel_ini_n
        self.loss_fn=tf.keras.losses.BinaryCrossentropy()
        # Custom Normalization layer (using your custom layers module)
        self.loss_tracker = tf.keras.metrics.BinaryCrossentropy(name='bce')
        self.auc_metric = tf.keras.metrics.AUC(name='auc')
        self.acc_metric=tf.keras.metrics.BinaryAccuracy(name='accuracy')
        
        #self.encoder = self.build_encoder(mean,stdev)
        #self.decoder = self.build_decoder()
        
    def build_encoder(self):
        inputs = tf.keras.layers.Input(shape=self.params.input_shape, dtype=tf.float32, name='encoder_input')
        # Assuming normalization is done outside, as there's no layer `StdNormalization` by default
        normalized = layers.StdNormalization(mean_x=self.mean, std_x=self.stdev)(inputs)
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=3))(normalized)
        x = tf.keras.layers.Conv2D(filters=self.kernel_n, kernel_size=self.params.kernel_sz, 
                                   activation=self.params.activation, kernel_initializer=self.params.initializer)(x)
        x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=2))(x)
        x = tf.keras.layers.Conv1D(filters=self.kernel_n + 4, kernel_size=self.params.kernel_1D_sz, 
                                   activation=self.params.activation, kernel_initializer=self.params.initializer)(x)
        x = tf.keras.layers.Conv1D(filters=self.kernel_n + 8, kernel_size=self.params.kernel_1D_sz, 
                                   activation=self.params.activation, kernel_initializer=self.params.initializer)(x)
        x = tf.keras.layers.AveragePooling1D()(x)
        self.shape_convolved = x.get_shape().as_list()
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(int(self.params.z_sz*17), activation=self.params.activation, 
                                  kernel_initializer=self.params.initializer)(x)
        x = tf.keras.layers.Dense(int(self.params.z_sz*4), activation=self.params.activation, 
                                  kernel_initializer=self.params.initializer)(x)
        z_mean = tf.keras.layers.Dense(self.params.z_sz, name='z_mean')(x)
        z_log_var = tf.keras.layers.Dense(self.params.z_sz, name='z_log_var')(x)
        z = layers.Sampling()((z_mean, z_log_var))
        encoder= tf.keras.Model(inputs, [z, z_mean, z_log_var], name='encoder')
        #encoder.build()
        encoder.summary()
        return encoder

    def build_decoder(self):
        latent_inputs = tf.keras.layers.Input(shape=(self.params.z_sz,), name='z_sampling')
        x = tf.keras.layers.Dense(int(self.params.z_sz*4), activation=self.params.activation, kernel_initializer=self.params.initializer)(latent_inputs)
        x = tf.keras.layers.Dense(int(self.params.z_sz*17), activation=self.params.activation, kernel_initializer=self.params.initializer)(x)
        x = tf.keras.layers.Dense(np.prod(self.shape_convolved[1:]), activation=self.params.activation, kernel_initializer=self.params.initializer)(x)
        x = tf.keras.layers.Reshape(self.shape_convolved[1:])(x)
        x = tf.keras.layers.UpSampling1D()(x)
        x = tf.keras.layers.Conv1DTranspose(filters=self.kernel_n + 4, kernel_size=self.params.kernel_1D_sz, 
                                            activation=self.params.activation, kernel_initializer=self.params.initializer)(x)
        x = tf.keras.layers.Conv1DTranspose(filters=self.kernel_n, kernel_size=self.params.kernel_1D_sz, 
                                            activation=self.params.activation, kernel_initializer=self.params.initializer)(x)
        x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))(x)
        x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=self.params.kernel_sz, name='conv2d_transpose')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=self.params.initializer)(x)
        decoder= tf.keras.Model(latent_inputs, x, name='decoder')
        #decoder.build()
        decoder.summary()
        return decoder
    def summary(self):
        # build encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        # link encoder and decoder to full vae model
        inputs = tf.keras.layers.Input(shape=self.params.input_shape, dtype=tf.float32, name='model_input')
        self.z, self.z_mean, self.z_log_var = self.encoder(inputs)
        outputs = self.decoder(self.z)  # link encoder output to decoder
        # instantiate VAE model
        self.model = tf.keras.Model(inputs, outputs, name='vae')
        self.model.summary()
        return self.model
    
    def call(self, inputs):
        z, z_mean, z_log_var = self.encoder(inputs)
        y_pred = self.decoder(z)
        return y_pred

    #def build_model(self):
    #    inputs=tf.keras.layers.Input(shape=self.params.input_shape)
    #    outputs=self.call(inputs)
    #    self.model=tf.keras.Model(inputs=inputs,outputs=outputs)
    #    print(self.model.summary())
    #    return self.model
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.auc_metric,self.acc_metric]
    def save_model(self, filepath):
        self.save_weights(os.path.join(filepath,'weights.h5'))
        with h5py.File(os.path.join(filepath,'model_params.h5'),'w') as f:
            ds = f.create_group('params')
            for name, value in self.params._asdict().items():
                ds.attrs[name] = value
    
    def load_model(self, filepath):
        #custom_objects = {'Sampling': layers.Sampling, 'StdNormalization': layers.StdNormalization}
        
        self.load_weights(os.path.join(filepath,'weights.h5'))
        print('weights loaded from file')
        return 0
    
    # def summary(self):
    #     x = tf.keras.layers.Input(shape=(None,)+self.params.input_shape)
    #     model = tf.keras.Model(x, outputs=self.call(x))
    #     return model.summary()

    @tf.function
    def train_step(self,data):
        #tf.print("Using custom train step")
        x_batch, y_batch = data
        with tf.GradientTape() as tape:
            y_pred = self(x_batch, training=True)
            bce_loss=self.loss_fn(y_batch, y_pred)
            #kl_loss=self.losses[0]*self.params.beta
            total_loss=bce_loss#+kl_loss
        #tf.print('min=',tf.math.reduce_min(y_pred))
        #tf.print('max=',tf.math.reduce_max(y_pred))
        gradients=tape.gradient(total_loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
        self.compiled_metrics.update_state(y_batch,y_pred)
        self.loss_tracker.update_state(y_batch,y_pred)
        self.auc_metric.update_state(y_batch,y_pred)
        self.acc_metric.update_state(y_batch,y_pred)

        return {'total_loss':total_loss,'bce loss':self.loss_tracker.result(),'auc':self.auc_metric.result(),'accuracy':self.acc_metric.result()}

    def test_step(self, data):
        # Unpack the data
        x_batch, y_batch = data
        # Compute predictions
        y_pred = self(x_batch, training=False)
        # Updates the metrics tracking the loss
        #self.loss_fn(y_batch, y_pred)
        # Update the metrics.
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y_batch, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
        