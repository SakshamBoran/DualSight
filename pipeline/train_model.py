import tensorflow as tf
from tf_pipeline_from_master import load_and_prepare, make_datasets
from model_def import build_multimodal_model

INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 35
LEARNING_RATE_FINE_TUNE = 1e-5
BATCH_SIZE = 32

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

    print("--- Loading dataset ---")
    train_tuple, val_tuple = load_and_prepare()
    train_ds, val_ds = make_datasets(train_tuple, val_tuple, batch_size=BATCH_SIZE)

    print("--- Building model ---")
    model = build_multimodal_model(num_tab_features=0)
    print(model.summary())

    print("--- Training stage 1 ---")
    model.fit(train_ds, epochs=INITIAL_EPOCHS, validation_data=val_ds)

    print("--- Fine-tuning stage ---")
    base_model = model.get_layer("resnet_backbone")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss="mse",
        metrics=["mae"]
    )

    early = tf.keras.callbacks.EarlyStopping(patience=8, monitor="val_loss", restore_best_weights=True)
    model.fit(train_ds, epochs=FINE_TUNE_EPOCHS, validation_data=val_ds, callbacks=[early])

    print("--- Saving model ---")
    model.save("models/final_model_satellite.h5")
    print("✅ Model saved as models/final_model_satellite.h5")

if __name__ == "__main__":
    main()
