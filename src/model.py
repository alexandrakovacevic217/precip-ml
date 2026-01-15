import tensorflow as tf


def build_model(input_shape, output_dim: int, activation: str = "relu") -> tf.keras.Model:
    """Baseline MLP used for regression."""
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape, dtype="float32"),
        tf.keras.layers.Dense(2048, activation=activation),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation=activation),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation=activation),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_dim, activation="linear", dtype="float32"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_model_deep(input_shape, output_dim: int, activation="relu") -> tf.keras.Model:
    """Deeper variant with flexible activation (supports 'leaky_relu')."""

    def get_activation_layer(act):
        if act == "leaky_relu":
            return tf.keras.layers.LeakyReLU(alpha=0.01)
        if isinstance(act, str):
            return tf.keras.layers.Activation(act)
        return act

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape, dtype="float32"),

        tf.keras.layers.Dense(2048),
        get_activation_layer(activation),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1024),
        get_activation_layer(activation),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Dense(512),
        get_activation_layer(activation),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(256),
        get_activation_layer(activation),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Dense(output_dim, activation="linear", dtype="float32"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model
