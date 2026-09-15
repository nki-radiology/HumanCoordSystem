"""Fit scan-specific UCS mappings from positions and 
predicted patch scores."""

import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from preprocessing import extract_random_patch, preprocess_hu_volume


def fit_ransac(inputs, targets, polynomial=False):
    """Fit the robust linear (or quadratic) mapping used by UCS."""
    regressor = RANSACRegressor(random_state=42)
    if polynomial:
        regressor = make_pipeline(
            PolynomialFeatures(2, include_bias=False), regressor
        )
    return regressor.fit(inputs, targets)


def generate_ucs(
    raw_hu_volume,
    scorer,
    image_size=256,
    patch_size=128,
    number_of_patches=1000,
    polynomial=False,
):
    """Predict patch scores, then fit position->score and score->position."""
    volume = preprocess_hu_volume(raw_hu_volume, image_size)
    coordinates = []
    patches = []

    for _ in range(number_of_patches):
        z = np.random.randint(0, volume.shape[0])
        patch, (row, column) = extract_random_patch(volume[z], patch_size)
        coordinates.append([z, row, column])
        patches.append(patch)

    coordinates = np.asarray(coordinates)
    predicted_scores = scorer.predict(np.asarray(patches), batch_size=64)

    forward = fit_ransac(coordinates, predicted_scores, polynomial)
    inverse = fit_ransac(predicted_scores, coordinates, polynomial)
    return forward, inverse


def generate_axial_ucs(raw_hu_volume, scorer, image_size=256, polynomial=False):
    """Fit the equivalent one-dimensional mapping for an axial scorer."""
    volume = preprocess_hu_volume(raw_hu_volume, image_size)
    z_coordinates = np.arange(volume.shape[0])[:, None]
    predicted_scores = scorer.predict(volume, batch_size=64)
    forward = fit_ransac(z_coordinates, predicted_scores, polynomial)
    inverse = fit_ransac(predicted_scores, z_coordinates, polynomial)
    return forward, inverse


# forward, inverse = generate_ucs(volume, scorer)
# score = forward.predict([[z, row, column]])
# predicted_location = inverse.predict(score)

