
# Model definition
inputs = Input(shape=(numFrames, target_size[0], target_size[1], 1))

# Add first conv block
x = Conv3D(32, kernel_size=(3, 3, 3), padding="same")(inputs)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(1, 2, 2))(x)
x = Dropout(0.25)(x)

# Apply Spatial Attention
x = SpatialAttention(activation='sigmoid')(x)

# Add second conv block
x = Conv3D(64, kernel_size=(3, 3, 3), padding="same")(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(1, 2, 2))(x)
x = Dropout(0.25)(x)

# Apply Temporal Attention
x = TemporalAttention()(x)

# Add third conv block
x = Conv3D(128, kernel_size=(3, 3, 3), padding="same")(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling3D(pool_size=(1, 2, 2))(x)
x = Dropout(0.25)(x)

# Apply Feature-Based Attention
x = FeatureBasedAttention(reduction_ratio=16)(x)

# Reshape for Bi-ConvLSTM
x = Reshape((numFrames, target_size[0] // 8, target_size[1] // 8, 128))(x)

# Forward ConvLSTM
forward_lstm = ConvLSTM2D(64, kernel_size=(3, 3), padding="same", return_sequences=True)(x)

# Backward ConvLSTM
backward_lstm = ConvLSTM2D(64, kernel_size=(3, 3), padding="same", return_sequences=True, go_backwards=True)(x)

# Concatenate forward and backward outputs
x = Concatenate()([forward_lstm, backward_lstm])

# Reshape for Attention
x = Reshape((-1, x.shape[-1]))(x)  # Flatten temporal dimension

# Add Attention
x = Attention()(x)

# Fully connected layers
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer
outputs = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs, outputs)
model.summary()


learning_rate = 0.0001  
optimizer = Adam(learning_rate=learning_rate)


model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=[
                  'accuracy',
                  AUC(name='auc'),
                  AUC(name='auc_pr', curve='PR'),
                  Recall(name='recall'),
                  Precision(name='precision'),
                  Sensitivity(),
                  Specificity(),
                  F1Score(),
                  MCC()])