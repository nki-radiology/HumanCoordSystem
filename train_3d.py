"""Three-coordinate UCS training"""

from pathlib import Path

import numpy as np
from tensorflow.keras.activations import linear
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Input, Subtract
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

from preprocessing import extract_random_patch, preprocess_cached_slice


class CachedDataGenerator(Sequence):
    """Generate patch pairs and axial/row/column comparison labels."""

    def __init__(self, dataset, image_size=256, patch_size=128, batch_size=32):
        self.files = sorted(Path(dataset).glob("*.npy"))
        self.indices = np.arange(len(self.files))
        self.image_size = image_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.random_seed = 0
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.indices.size / self.batch_size))

    def __getitem__(self, index):
        selected = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        shape = (self.batch_size, self.patch_size, self.patch_size, 1)
        first = np.zeros(shape, dtype=np.float32)
        second = np.zeros(shape, dtype=np.float32)
        labels = np.zeros((self.batch_size, 3), dtype=np.float32)

        for batch_index, dataset_index in enumerate(selected):
            sample = np.load(self.files[dataset_index])
            image_1, image_2, axial_label = np.rollaxis(sample, 2, 0)
            image_1 = preprocess_cached_slice(image_1, self.image_size)
            image_2 = preprocess_cached_slice(image_2, self.image_size)

            image_1, position_1 = extract_random_patch(image_1, self.patch_size)
            image_2, position_2 = extract_random_patch(image_2, self.patch_size)
            first[batch_index] = image_1
            second[batch_index] = image_2
            labels[batch_index] = [
                axial_label.max(),
                position_1[0] > position_2[0],
                position_1[1] > position_2[1],
            ]

        return [first, second], labels

    def on_epoch_end(self):
        np.random.seed(self.random_seed)
        np.random.shuffle(self.indices)
        self.random_seed += 1


def build_3d_models(patch_size=128, axial_weights=None):
    """Build the three-score ResNet50 and its shared Siamese wrapper."""
    scorer = ResNet50(
        include_top=True,
        weights=None,
        input_shape=(patch_size, patch_size, 1),
        pooling="avg",
        classes=3,
    )
    scorer.layers[-1].activation = linear

    if axial_weights is not None:
        scorer.load_weights(axial_weights, by_name=True, skip_mismatch=True)

    input_1 = Input((patch_size, patch_size, 1), name="input_1")
    input_2 = Input((patch_size, patch_size, 1), name="input_2")
    difference = Subtract(name="subtract")([scorer(input_1), scorer(input_2)])
    output = Activation("sigmoid")(difference)
    siamese = Model([input_1, input_2], output)
    return siamese, scorer


def train_3d(
    train_dir,
    validation_dir,
    epochs=100,
    axial_weights=None,
    output_model=None,
    checkpoint="ucs_siamese_checkpoint.keras",
):
    """Train three-axis ordering and return the single-patch scorer."""
    train = CachedDataGenerator(train_dir)
    validation = CachedDataGenerator(validation_dir)
    siamese, scorer = build_3d_models(axial_weights=axial_weights)

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_binary_accuracy",
            factor=0.5,
            patience=10,
            min_delta=0.02,
            mode="max",
        ),
        ModelCheckpoint(
            checkpoint,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
    ]
    siamese.compile(
        Adam(1e-4), binary_crossentropy, metrics=[binary_accuracy]
    )
    siamese.fit(
        train,
        validation_data=validation,
        epochs=epochs,
        callbacks=callbacks,
    )

    if output_model is not None:
        scorer.save(output_model)
    return scorer
