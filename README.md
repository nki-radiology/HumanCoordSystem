# Universal Coordinate System (UCS)

Simplified reference implementation for training the axial and three-coordinate UCS scorers and fitting a scan-specific coordinate system with RANSAC. The implementation is intended to illustrate the method.

## Contents

- `train_axial.py`: learns a single superior/inferior slice score.
- `train_3d.py`: learns axial, row and column scores from random axial patches.
- `inference.py`: samples a new CT scan and fits forward/inverse RANSAC mappings.
- `preprocessing.py`: preprocessing shared by training and inference.

Dataset creation, evaluation, plotting, private paths and experimental variants
are intentionally omitted.

## Cached training pairs

Each `.npy` file has shape `(H, W, 3)`:

1. first CT slice;
2. second CT slice;
3. constant binary label indicating whether the first slice has a larger axial
   position than the second.

The cached slices are expected to have been prepared as
`clip(HU, -120, 300) + 120`, matching the original training pipeline.

The single-image `scorer` returned by each training function is the model used
during inference. The Siamese model is only the training wrapper.

## Citation
> A Radiological-Based Coordinate System for the Human Body: A Proof-of-Concept.
> In *Medical Image Computing and Computer Assisted Intervention – MICCAI 2024
> Workshops*, Lecture Notes in Computer Science, vol. 15401, pp. 101–110.
> https://doi.org/10.1007/978-3-031-84525-3_9

```bibtex
@inproceedings{bucho2025radiological,
  title     = {A Radiological-Based Coordinate System for the Human Body:
               A Proof-of-Concept},
  author    = {Bucho, Teresa M. T. and Boellaard, Thierry N. and
               Taveira, Mateus and Bodalal, Zuhir and
               Nguyen-Kim, Thi D. L. and Beets-Tan, Regina and
               Trebeschi, Stefano},
  booktitle = {Medical Image Computing and Computer Assisted Intervention --
               MICCAI 2024 Workshops},
  series    = {Lecture Notes in Computer Science},
  volume    = {15401},
  pages     = {101--110},
  year      = {2025},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-031-84525-3_9}
}
```
