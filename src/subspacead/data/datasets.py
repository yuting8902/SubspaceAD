import os
from pathlib import Path

import numpy as np
from PIL import Image


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_IGNORED_DIR_NAMES = {".ipynb_checkpoints"}
_NORMAL_FOLDER_NAMES = {"good", "normal"}


def _is_ignored_path(path: Path) -> bool:
    """Return True when any path component should be ignored."""
    return any(part in _IGNORED_DIR_NAMES for part in path.parts)


def _list_images(folder: Path, recursive: bool = False):
    """Return image files with common extensions, ignoring notebook checkpoints."""
    if not folder.exists():
        return []
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(
        str(p)
        for p in iterator
        if p.is_file()
        and p.suffix.lower() in _IMAGE_EXTENSIONS
        and not _is_ignored_path(p)
    )


class BaseDatasetHandler:
    """Common interface expected by SubspaceAD main.py."""

    def __init__(self, root_path, category):
        self.root_path = Path(root_path)
        self.category = category
        self.category_path = self.root_path / category

    def get_train_paths(self):
        raise NotImplementedError

    def get_validation_paths(self):
        return []

    def get_test_paths(self):
        raise NotImplementedError

    def get_ground_truth_path(self, test_path: str):
        raise NotImplementedError

    def get_ground_truth_mask(self, test_path: str, res: tuple):
        gt_path = self.get_ground_truth_path(test_path)
        if not gt_path or not os.path.exists(gt_path):
            return np.zeros((res[1], res[0]), dtype=np.uint8)

        mask = Image.open(gt_path).convert("L").resize(res, Image.Resampling.NEAREST)
        return (np.asarray(mask) > 0).astype(np.uint8)

    def get_defect_type(self, image_path: str) -> str:
        """Use the immediate parent folder as the defect type."""
        return Path(image_path).parent.name

    def get_image_label(self, image_path: str) -> int:
        """0=normal, 1=anomaly using the immediate parent folder name."""
        defect_type = self.get_defect_type(image_path).lower()
        return 0 if defect_type in _NORMAL_FOLDER_NAMES else 1


class MVTecADDataset(BaseDatasetHandler):
    def get_train_paths(self):
        return _list_images(self.category_path / "train" / "good")

    def get_test_paths(self):
        return _list_images(self.category_path / "test", recursive=True)

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        return str(
            self.category_path / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"
        )


class MVTecLOCODataset(BaseDatasetHandler):
    def get_train_paths(self):
        return _list_images(self.category_path / "train" / "good")

    def get_validation_paths(self):
        return _list_images(self.category_path / "validation" / "good")

    def get_test_paths(self):
        return _list_images(self.category_path / "test", recursive=True)

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        anomaly_type = p.parent.name
        if anomaly_type.lower() == "good":
            return None
        candidates = [
            self.category_path / "ground_truth" / anomaly_type / f"{p.stem}_mask.png",
            self.category_path / "ground_truth" / anomaly_type / p.stem / "000.png",
            self.category_path / "ground_truth" / anomaly_type / p.stem / p.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None


class MVTecAD2Dataset(BaseDatasetHandler):
    def get_train_paths(self):
        return _list_images(self.category_path / "train" / "good")

    def get_validation_paths(self):
        return _list_images(self.category_path / "validation" / "good")

    def get_test_paths(self):
        return _list_images(self.category_path / "test_public", recursive=True)

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        return str(
            self.category_path
            / "test_public"
            / "ground_truth"
            / p.parent.name
            / f"{p.stem}_mask.png"
        )


class VisADataset(BaseDatasetHandler):
    def get_train_paths(self):
        return _list_images(self.category_path / "train" / "good")

    def get_test_paths(self):
        return _list_images(self.category_path / "test", recursive=True)

    def get_ground_truth_path(self, test_path: str):
        p = Path(test_path)
        if p.parent.name.lower() != "bad":
            return None
        candidate = self.category_path / "ground_truth" / "bad" / f"{p.stem}.png"
        return str(candidate) if candidate.exists() else None


class WaferDataset(BaseDatasetHandler):
    """
    Custom wafer dataset with independent roots for train / validation / test.

    Expected structure:
        train_root/<category>/train/good/*

        val_root/<category>/val/good/*
        val_root/<category>/val/<anomaly_type>/*

        test_root/<category>/test/good/*
        test_root/<category>/test/<anomaly_type>/*

    Any '.ipynb_checkpoints' directory is ignored recursively.
    Ground-truth segmentation masks are intentionally not required.

    For validation/test image-level labels:
        parent folder 'good' or 'normal' -> 0 (normal)
        every other parent folder         -> 1 (anomaly)
    """

    def __init__(self, root_path, category):
        # root_path is train_root because config.py maps args.dataset_path to it.
        super().__init__(root_path, category)

        train_root = os.environ.get("SUBSPACEAD_WAFER_TRAIN_ROOT", str(root_path))
        val_root = os.environ.get("SUBSPACEAD_WAFER_VAL_ROOT")
        test_root = os.environ.get("SUBSPACEAD_WAFER_TEST_ROOT")
        if not test_root:
            raise ValueError(
                "Wafer test root is missing. Run with --dataset_name wafer "
                "--train_root ... --test_root ..."
            )

        self.train_category_path = Path(train_root) / category
        self.val_category_path = Path(val_root) / category if val_root else None
        self.test_category_path = Path(test_root) / category

    def get_train_paths(self):
        return _list_images(self.train_category_path / "train" / "good")

    def get_validation_paths(self):
        if self.val_category_path is None:
            return []
        # Mixed validation is supported: good/normal plus any anomaly subfolders.
        return _list_images(self.val_category_path / "val", recursive=True)

    def get_test_paths(self):
        # Supports normal-only, anomaly-only, or mixed test sets.
        return _list_images(self.test_category_path / "test", recursive=True)

    def get_ground_truth_path(self, test_path: str):
        return None


def get_dataset_handler(name: str, root_path: str, category: str) -> BaseDatasetHandler:
    handlers = {
        "mvtec_ad": MVTecADDataset,
        "mvtec_loco": MVTecLOCODataset,
        "mvtec_ad2": MVTecAD2Dataset,
        "visa": VisADataset,
        "wafer": WaferDataset,
    }
    try:
        return handlers[name](root_path, category)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc
