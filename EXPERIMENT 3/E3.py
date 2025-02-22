import tensorflow as tf
import numpy as np

# Load MNIST dataset from TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape data
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape([-1, 784])  # Flatten images (28x28 -> 784)
x_test = x_test.reshape([-1, 784])

# One-hot encode labels
y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

# Parameters
input_size = 784
hidden1_size = 128
hidden2_size = 64
output_size = 10
learning_rate = 0.01
batch_size = 128
epochs = 5

# Initialize weights and biases
initializer = tf.initializers.GlorotUniform()
W1 = tf.Variable(initializer([input_size, hidden1_size]))
b1 = tf.Variable(tf.zeros([hidden1_size]))
W2 = tf.Variable(initializer([hidden1_size, hidden2_size]))
b2 = tf.Variable(tf.zeros([hidden2_size]))
W3 = tf.Variable(initializer([hidden2_size, output_size]))
b3 = tf.Variable(tf.zeros([output_size]))

# Feed-forward function
def forward_pass(x):
    z1 = tf.add(tf.matmul(x, W1), b1)
    a1 = tf.nn.relu(z1)
    z2 = tf.add(tf.matmul(a1, W2), b2)
    a2 = tf.nn.relu(z2)
    z3 = tf.add(tf.matmul(a2, W3), b3)
    return z3  # Logits

# Loss function (cross-entropy)
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# Accuracy calculation
def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    true_labels = tf.argmax(labels, axis=1)
    correct = tf.equal(predictions, true_labels)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

# Training step with backpropagation
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = forward_pass(x_batch)
        loss = compute_loss(logits, y_batch)
    gradients = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])
    for var, grad in zip([W1, b1, W2, b2, W3, b3], gradients):
        var.assign_sub(learning_rate * grad)
    return loss

# Training loop
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(batch_size)

for epoch in range(epochs):
    epoch_loss = 0
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss_value = train_step(x_batch, y_batch)
        epoch_loss += loss_value
    
    # Evaluate on test set
    test_logits = forward_pass(x_test)
    test_acc = compute_accuracy(test_logits, y_test)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / (step + 1):.4f}, Test Accuracy: {test_acc:.4f}")

print("\nTraining completed.")

# Final evaluation
test_logits = forward_pass(x_test)
final_accuracy = compute_accuracy(test_logits, y_test)
print(f"Final Test Accuracy: {final_accuracy:.4f}")
