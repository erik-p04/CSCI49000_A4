import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# load dataset
max_features = 10000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen = max_len)

# Implementing 1D CNN model
model = Sequential([
    Embedding(max_features, 128, input_length = max_len),
    Conv1D(128, 5, activation = 'relu'),
    GlobalMaxPooling1D(),
    Dense(10, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs = 5, batch_size = 128, validation_split = 0.2)
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f} | Test Loss: {loss:.4f}")

# Trying different filter sizes
def build_cnn_model(filter_size):
    model = Sequential([
        Embedding(max_features, 128, input_length = max_len),
        Conv1D(128, filter_size, activation = 'relu'),
        GlobalMaxPooling1D(),
        Dense(10, activation = 'relu'),
        Dense(1, activation = 'sigmoid')
    ])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

for fs in [3, 5, 7]:
    print(f"Training with filter size: {fs}")
    cnn = build_cnn_model(fs)
    cnn.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.2)
