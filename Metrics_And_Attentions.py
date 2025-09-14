
class Sensitivity(Metric):
    def __init__(self, name='sensitivity', **kwargs):
        super(Sensitivity, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.possible_positives = self.add_weight(name='pp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.reshape(y_pred, [-1])
        y_pred = K.round(y_pred)

        # Calculate true positives
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        # Calculate possible positives
        pp = K.sum(K.round(K.clip(y_true, 0, 1)))

        # Update accumulated values
        self.true_positives.assign_add(tp)
        self.possible_positives.assign_add(pp)

    def result(self):
        sensitivity_value = self.true_positives / (self.possible_positives + K.epsilon())
        return sensitivity_value

    def reset_state(self):
        self.true_positives.assign(0)
        self.possible_positives.assign(0)


class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.reshape(y_pred, [-1])
        y_pred = K.round(y_pred)

        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
        fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)



class MCC(Metric):
    def __init__(self, name='mcc', **kwargs):
        super(MCC, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.reshape(y_pred, [-1])
        y_pred = K.round(K.clip(y_pred, 0, 1))

        tp = K.sum(y_true * y_pred)
        tn = K.sum((1 - y_true) * (1 - y_pred))
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))

        self.true_positives.assign_add(tp)
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        numerator = (self.true_positives * self.true_negatives) - (self.false_positives * self.false_negatives)
        denominator = K.sqrt(
            (self.true_positives + self.false_positives) *
            (self.true_positives + self.false_negatives) *
            (self.true_negatives + self.false_positives) *
            (self.true_negatives + self.false_negatives)
        )
        return numerator / (denominator + K.epsilon())

    def reset_state(self):
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)



class Specificity(Metric):
    def __init__(self, name='specificity', **kwargs):
        super(Specificity, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.reshape(y_pred, [-1])
        y_pred = K.round(K.clip(y_pred, 0, 1))

        tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
        fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))

        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_positives + K.epsilon())

    def reset_state(self):
        self.true_negatives.assign(0)
        self.false_positives.assign(0)



class SpatialAttention(Layer):
    def __init__(self, activation='sigmoid'):
        super(SpatialAttention, self).__init__()
        self.activation = activation

    def call(self, inputs):
        # Reduce across temporal and channel dimensions
        # inputs: [batch_size, num_frames, height, width, channels]
        spatial_features = tf.reduce_mean(inputs, axis=[1, 4])  # Shape: [batch_size, height, width]

        # Apply the chosen activation function to generate attention weights
        if self.activation == 'sigmoid':
            attention_weights = tf.nn.sigmoid(spatial_features)  # Shape: [batch_size, height, width]
        elif self.activation == 'softmax':
            attention_weights = tf.nn.softmax(spatial_features, axis=-1)  # Shape: [batch_size, height, width]
        else:
            raise ValueError("Activation must be 'sigmoid' or 'softmax'")

        # Expand dimensions to match the input shape for broadcasting
        attention_weights = tf.expand_dims(tf.expand_dims(attention_weights, axis=1), axis=-1)  # Shape: [batch_size, 1, height, width, 1]

        # Apply attention weights to the input
        weighted_inputs = inputs * attention_weights  # Broadcasting happens here

        return weighted_inputs



# Define Temporal Attention Layer
class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def call(self, inputs):
        # Inputs shape: (batch, num_frames, height, width, channels)
        # Apply temporal mean across spatial dimensions
        temporal_features = tf.reduce_mean(inputs, axis=[2, 3])  # Shape: (batch, num_frames, channels)
        attention_weights = tf.nn.softmax(temporal_features, axis=1)  # Shape: (batch, num_frames, channels)
        # Expand dimensions to match inputs
        attention_weights = tf.expand_dims(tf.expand_dims(attention_weights, axis=2), axis=3)  # Shape: (batch, num_frames, 1, 1, channels)
        # Apply attention
        weighted_inputs = inputs * attention_weights
        return weighted_inputs

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape
        return input_shape

# Feature-Based Attention (Squeeze-and-Excitation style)
class FeatureBasedAttention(Layer):
    def __init__(self, reduction_ratio=16):
        super(FeatureBasedAttention, self).__init__()
        self.reduction_ratio = reduction_ratio
        # Define Dense layers during initialization
        self.reduce_dense = Dense(
            units=None,  # Set dynamically based on input channels
            activation='relu',
            use_bias=True
        )
        self.restore_dense = Dense(
            units=None,  # Set dynamically based on input channels
            activation='sigmoid',
            use_bias=True
        )

    def build(self, input_shape):
        # Infer channels dynamically and configure Dense layers
        channels = input_shape[-1]
        self.reduce_dense.units = channels // self.reduction_ratio
        self.restore_dense.units = channels

    def call(self, inputs):
        # Squeeze: Compute global channel descriptors
        channel_weights = tf.reduce_mean(inputs, axis=[1, 2, 3])  # Shape: (batch, channels)

        # Reduce dimensionality
        reduced = self.reduce_dense(channel_weights)

        # Restore dimensionality
        restored = self.restore_dense(reduced)

        # Apply attention
        attention_weights = tf.expand_dims(tf.expand_dims(tf.expand_dims(restored, axis=1), axis=1), axis=1)
        return inputs * attention_weights

# Self-Attention
class SelfAttention(Layer):
    def __init__(self, num_heads=4, key_dim=64):
        super(SelfAttention, self).__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def build(self, input_shape):
        # Define any additional layers or variables here if needed
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        # Get the input shape dynamically
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        num_frames = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        channels = input_shape[4]

        # Reshape input for multi-head attention (combine spatial and temporal dimensions)
        reshaped_inputs = tf.reshape(inputs, (batch_size, num_frames * height * width, channels))
        attended = self.attention(reshaped_inputs, reshaped_inputs)
        
        # Reshape back to original 5D shape
        output = tf.reshape(attended, (batch_size, num_frames, height, width, channels))
        return output



# Add Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(1,),
                                 initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.activations.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)









