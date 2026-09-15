"""Axial UCS training"""

from pathlib import Path

import numpy as np
from tensorflow.keras.activations import linear
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Input, Subtract
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

from preprocessing import preprocess_cached_slice


class CachedDataGenerator(Sequence):
    """Generate pairs of complete axial slices and their relative order."""

    def __init__(
        self, dataset, image_size=256, batch_size=16, augment=True, shuffle=True
    ):
        self.files = sorted(Path(dataset).glob("*.npy"))
        self.indices = np.arange(len(self.files))
        self.image_size = image_size
        self.batch_size = batch_size
        self.use_augmentation = augment
        self.shuffle = shuffle
        self.random_seed = 0
        self.augment = ImageDataGenerator(
            rotation_range=30,
            height_shift_range=0.2,
            width_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.5,
            zoom_range=0.5,
        ).random_transform
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.indices.size / self.batch_size))

    def __getitem__(self, index):
        selected = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        first = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 1), dtype=np.float32
        )
        second = np.zeros_like(first)
        labels = np.zeros((self.batch_size, 1), dtype=np.float32)

        for batch_index, dataset_index in enumerate(selected):
            sample = np.load(self.files[dataset_index])
            image_1, image_2, axial_label = np.rollaxis(sample, 2, 0)
            image_1 = preprocess_cached_slice(image_1, self.image_size)
            image_2 = preprocess_cached_slice(image_2, self.image_size)

            if self.use_augmentation:
                image_1 = self.augment(image_1)
                image_2 = self.augment(image_2)
            first[batch_index] = image_1
            second[batch_index] = image_2
            labels[batch_index] = axial_label.max()

        return [first, second], labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.indices)
        self.random_seed += 1


def build_axial_models(image_size=256):
    """Build the one-score ResNet50 and its shared Siamese wrapper."""
    scorer = ResNet50(
        include_top=True,
        weights=None,
        input_shape=(image_size, image_size, 1),
        pooling="avg",
        classes=1,
    )
    scorer.layers[-1].activation = linear

    input_1 = Input((image_size, image_size, 1), name="input_1")
    input_2 = Input((image_size, image_size, 1), name="input_2")
    difference = Subtract(name="subtract")([scorer(input_1), scorer(input_2)])
    output = Activation("sigmoid")(difference)
    siamese = Model([input_1, input_2], output)
    return siamese, scorer


def train_axial(train_dir, validation_dir, epochs=100, output_model=None):
    """Train relative axial ordering and return the single-slice scorer."""
    train = CachedDataGenerator(train_dir)
    validation = CachedDataGenerator(validation_dir, augment=False, shuffle=False)
    siamese, scorer = build_axial_models()

    reduce_learning_rate = ReduceLROnPlateau(
        monitor="val_binary_accuracy",
        factor=0.5,
        patience=10,
        min_delta=0.02,
        mode="max",
    )
    siamese.compile(
        Adam(1e-4), binary_crossentropy, metrics=[binary_accuracy]
    )
    siamese.fit(
        train,
        validation_data=validation,
        epochs=epochs,
        callbacks=[reduce_learning_rate],
    )

    if output_model is not None:
        scorer.save(output_model)
    return scorer
