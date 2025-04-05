import tensorflow as tf
import numpy as np

#preprocess text dataset
path_to_file = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

#training sequence
seq_length = 100
examples_per_epoch = len(text) // seq_length
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder = True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)

#Vanilla RNN model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
EPOCHS = 5
LOSS_FN = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

def build_vanilla_rnn_model(vocab_size, embedding_dim, rnn_units):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=(None,)),
        tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
        tf.keras.layers.Dense(vocab_size)
    ])

model_vanilla = build_vanilla_rnn_model(vocab_size, embedding_dim, rnn_units)
model_vanilla.compile(optimizer = 'adam', loss=LOSS_FN)
model_vanilla.fit(dataset, epochs=EPOCHS)

#Stacked RNN
def build_stacked_rnn_model(vocab_size, embedding_dim, rnn_units):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=(None,)),
        tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
        tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
        tf.keras.layers.Dense(vocab_size)
    ])

model_stacked = build_stacked_rnn_model(vocab_size, embedding_dim, rnn_units)
model_stacked.compile(optimizer='adam', loss=LOSS_FN)
model_stacked.fit(dataset, epochs=EPOCHS)

#Bidirectional RNN
def build_bidirectional_rnn_model(vocab_size, embedding_dim, rnn_units):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=(None,)),
        tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True)),
        tf.keras.layers.Dense(vocab_size)
    ])

model_bi = build_bidirectional_rnn_model(vocab_size, embedding_dim, rnn_units)
model_bi.compile(optimizer='adam', loss=LOSS_FN)
model_bi.fit(dataset, epochs=EPOCHS)

#Generating text
def generate_text(model, start_string, num_generate = 300, temperature = 1.0):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :] / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

print("\n--- VANILLA RNN ---")
print(generate_text(model_vanilla, start_string="To be or not to be"))

print("\n--- STACKED RNN ---")
print(generate_text(model_stacked, start_string="To be or not to be"))

print("\n--- BIDIRECTIONAL RNN ---")
print(generate_text(model_bi, start_string="To be or not to be"))
