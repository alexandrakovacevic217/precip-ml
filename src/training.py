import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping

from .model import build_model


def train_model(
    model: tf.keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 60,
    batch_size: int = 512,
    callbacks=None,
):
    """Fit a model and return training history."""
    model.summary()
    if callbacks is None:
        callbacks = []
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=2,
        callbacks=callbacks,
    )
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    print(f"Val Loss (MSE): {val_loss}")
    return history


def build_and_train(
    X_train,
    y_train,
    X_val,
    y_val,
    activation: str = "relu",
    batch_size: int = 512,
    callbacks=None,
):
    """
    Convenience wrapper used in the thesis code.
    Note: mixed precision is set here (as in your original code).
    """
    mixed_precision.set_global_policy("mixed_float16")

    # defined here, but you usually pass your own callback from the runner
    _early_stopping = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=False)

    input_shape = (X_train.shape[1],)
    output_dim = y_train.shape[1]

    model = build_model(input_shape, output_dim, activation=activation)

    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    return model, history
