#model_train
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# 2. Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Create a Deep Learning model
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 4. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Train the model
model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)

# 6. Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", accuracy)

# 7. Save the trained model
model.save("digit_model.keras")

print("Model saved successfully!")
