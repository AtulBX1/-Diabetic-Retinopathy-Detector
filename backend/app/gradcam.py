import cv2
import numpy as np
import tensorflow as tf
import keras


def make_gradcam(model, image, pred_index=None):
    nested = next(layer for layer in model.layers if isinstance(layer, keras.Model))

    grad_model = keras.Model(
        inputs=model.input,
        outputs=[nested.output, model.output],
    )

    image = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        conv_outputs, predictions = grad_model(image, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    denominator = tf.reduce_max(heatmap)
    heatmap = tf.where(
        denominator > 0,
        heatmap / denominator,
        tf.zeros_like(heatmap),
    )
    return heatmap.numpy().astype(np.float32)


def overlay(image, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
