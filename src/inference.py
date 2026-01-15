import numpy as np


def predict_and_reshape(model, X, imask, output_shape, multi_time: bool = False):
    """
    Predict flattened valid-grid outputs and reshape to (time, lat, lon) using imask.
    Note: invalid cells are currently filled with 0 (same as your original code).
    """
    preds = model.predict(X)
    t_steps = output_shape[0]
    flat_preds = np.zeros((t_steps, output_shape[1] * output_shape[2]))
    flat_preds[:, imask] = preds
    return flat_preds.reshape(output_shape)

