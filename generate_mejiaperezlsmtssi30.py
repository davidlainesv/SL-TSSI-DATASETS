"""load mejiaperezlsmtssi30 dataset."""

import tensorflow_datasets as tfds
# pylint: disable=unused-import
from src.mejiaperezlsm import mejiaperezlsmtssi30

ds, info = tfds.load('mejia_perez_lsm_tssi30', data_dir="./datasets", with_info=True)

print(ds["train"].cardinality())
print(ds["validation"].cardinality())
print(ds["test"].cardinality())
print(info.features["data"].shape)
print(info.features["label"].num_classes)
