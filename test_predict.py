import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
TARGET_SIZE = (224, 224)


def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)


def main(image_path, out_path):
    model = tf.keras.models.load_model("./src/models/model_Xception_ft.hdf5")

    grad_model = tf.keras.models.clone_model(model)
    grad_model.set_weights(model.get_weights())
    grad_model.layers[-1].activation = None
    grad_model = tf.keras.models.Model(
        inputs=[grad_model.inputs],
        outputs=[
            grad_model.get_layer("global_average_pooling2d_1").input,
            grad_model.output,
        ],
    )

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=TARGET_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img_aux = img.copy()

    img_array = np.expand_dims(img_aux, axis=0)
    img_array = np.float32(img_array)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)

    y_pred = 100 * model.predict(img_array, verbose=0)[0]
    grade_idx = int(np.argmax(y_pred))
    print(f"input        : {image_path}")
    print(f"prediction   : {CLASS_NAMES[grade_idx]} - {y_pred[grade_idx]:.2f}%")
    print("probabilities: " + ", ".join(
        f"{n}={p:.2f}%" for n, p in zip(CLASS_NAMES, y_pred)
    ))

    heatmap = make_gradcam_heatmap(grad_model, img_array)
    overlay = save_and_display_gradcam(img, heatmap)
    overlay.save(out_path)
    print(f"gradcam      : {out_path}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1], sys.argv[2])
