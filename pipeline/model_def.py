import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

# Keep this function identical to build_multimodal_model() in
# kaggle/dualsight_kaggle_training.ipynb. Weights trained on Kaggle are moved
# into this architecture layer-by-layer *by name*, so the explicit layer names
# below must match exactly.


def build_multimodal_model(num_tab_features: int = 0, image_size: int = 128, pretrained: bool = True) -> tf.keras.Model:
    """Satellite image (+ optional numeric features) -> log(price) model."""

    # Image branch
    image_input = layers.Input(shape=(image_size, image_size, 3), name="image_input")
    resnet = ResNet50(include_top=False, weights="imagenet" if pretrained else None,
                      input_shape=(image_size, image_size, 3))
    try:
        resnet._name = "resnet_backbone"
    except Exception:
        pass
    resnet.trainable = False
    # training=False keeps BatchNorm statistics frozen even while fine-tuning
    x_img = resnet(image_input, training=False)
    x_img = layers.Flatten(name="img_flatten")(x_img)
    x_img = layers.Dense(256, activation="relu", name="img_dense")(x_img)
    x_img = layers.Dropout(0.3, name="img_dropout")(x_img)
    x_img = layers.BatchNormalization(name="img_bn")(x_img)

    # Numeric branch (optional)
    if num_tab_features > 0:
        tabular_input = layers.Input(shape=(num_tab_features,), name="tabular_input")
        x_num = layers.Dense(128, activation="relu", name="tab_dense")(tabular_input)
        x_num = layers.Dropout(0.2, name="tab_dropout")(x_num)
        x_num = layers.BatchNormalization(name="tab_bn")(x_num)
        fused = layers.Concatenate(name="fusion")([x_img, x_num])
        inputs = [image_input, tabular_input]
    else:
        fused = x_img
        inputs = image_input

    fused = layers.Dense(128, activation="relu", name="fused_dense")(fused)
    fused = layers.Dropout(0.3, name="fused_dropout")(fused)
    output = layers.Dense(1, activation="linear", name="price")(fused)

    model = Model(inputs=inputs, outputs=output, name="satellite_price_model")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse", metrics=["mae"])
    return model
