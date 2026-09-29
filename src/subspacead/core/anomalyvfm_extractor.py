import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AnomalyVFMFeatureExtractor:
    """
    SubspaceAD adapter for the AnomalyVFM / Meta-DINOv2 ablation.

    Variants
    --------
    a0_original_final
        Meta DINOv2 ViT-L/14-Reg with the original local pretrained .pth.
        No DoRA and no AnomalyVFM checkpoint.
        Feature = final normalized patch tokens from forward_features().

    a1_adapted_final
        Current H1.
        Meta DINOv2 ViT-L/14-Reg + AnomalyVFM DoRA rank 64 +
        anomalyvfm_dinov2.pkl.
        Feature = final normalized x_norm_patchtokens.

    a2_adapted_middle
        Same adapted backbone as A1, but features are taken from the selected
        intermediate transformer blocks instead of the final block.
        The selected intermediate outputs are normalized with the model's final
        LayerNorm (Meta DINOv2 get_intermediate_layers(..., norm=True)), patch
        tokens are kept, and the selected layers are mean-aggregated.

    All variants use the same AnomalyVFM DINOv2 preprocessing and return the
    same SubspaceAD interface:
        tokens:   [B, H_patch, W_patch, 1024]
        grid:     (H_patch, W_patch)
        saliency: zero placeholder, because SubspaceAD DINO-attention saliency
                  is intentionally not implemented for this extractor.
    """

    EXPECTED_IMAGE_SIZE = 672
    PEFT_RANK = 64
    VALID_VARIANTS = {
        "a0_original_final",
        "a1_adapted_final",
        "a2_adapted_middle",
    }

    def __init__(
        self,
        anomalyvfm_root: str,
        anomalyvfm_ckpt: str,
        dino_repo_path: str,
        dino_weight_path: str,
        variant: str = "a1_adapted_final",
    ):
        self.variant = str(variant)
        if self.variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"Unknown anomalyvfm variant: {self.variant}. "
                f"Choices: {sorted(self.VALID_VARIANTS)}"
            )

        self.anomalyvfm_root = Path(anomalyvfm_root).expanduser().resolve()
        self.anomalyvfm_ckpt = Path(anomalyvfm_ckpt).expanduser().resolve()
        self.dino_repo_path = Path(dino_repo_path).expanduser().resolve()
        self.dino_weight_path = Path(dino_weight_path).expanduser().resolve()

        self._validate_paths()

        anomalyvfm_root_str = str(self.anomalyvfm_root)
        if anomalyvfm_root_str not in sys.path:
            sys.path.insert(0, anomalyvfm_root_str)

        try:
            from models.dinov2_offline import OfflineAnomalyDINOv2
        except Exception as exc:
            raise ImportError(
                "Failed to import OfflineAnomalyDINOv2 from the local "
                f"AnomalyVFM_ project: {self.anomalyvfm_root}\n"
                "Expected file: models/dinov2_offline.py\n"
                "Expected package: peft_local/\n"
                f"Original error: {exc}"
            ) from exc

        logging.info(
            "Loading Meta/AnomalyVFM DINOv2 extractor variant: %s",
            self.variant,
        )
        logging.info("AnomalyVFM root: %s", self.anomalyvfm_root)
        logging.info("Local DINOv2 repo: %s", self.dino_repo_path)
        logging.info("Local DINOv2 weight: %s", self.dino_weight_path)

        self.model = OfflineAnomalyDINOv2(
            dino_repo_path=self.dino_repo_path,
            dino_weight_path=self.dino_weight_path,
            image_size=self.EXPECTED_IMAGE_SIZE,
            peft_rank=self.PEFT_RANK,
            load_base_weights=True,
        )

        if self.variant in {"a1_adapted_final", "a2_adapted_middle"}:
            logging.info("Adding AnomalyVFM DoRA, rank=%d.", self.PEFT_RANK)
            self.model.add_anomalyvfm_dora()
            logging.info(
                "Loading AnomalyVFM adapted checkpoint: %s",
                self.anomalyvfm_ckpt,
            )
            self.model.load_anomalyvfm_checkpoint(
                self.anomalyvfm_ckpt,
                strict=True,
            )
        else:
            logging.info(
                "A0 selected: keeping original Meta DINOv2 weights; "
                "DoRA and anomalyvfm_dinov2.pkl are not applied."
            )

        self.model = self.model.eval().to(DEVICE)

        self.transform = self.model.get_img_transform()

        self.feature_dim = int(self.model.feature_dim)
        self.patch_size = int(self.model.patch_size)
        self.h_p = self.EXPECTED_IMAGE_SIZE // self.patch_size
        self.w_p = self.EXPECTED_IMAGE_SIZE // self.patch_size
        self.expected_patch_count = self.h_p * self.w_p

        logging.info(
            "Extractor ready: variant=%s, image_size=%d, patch_grid=%dx%d, "
            "patch_count=%d, feature_dim=%d.",
            self.variant,
            self.EXPECTED_IMAGE_SIZE,
            self.h_p,
            self.w_p,
            self.expected_patch_count,
            self.feature_dim,
        )

    def _validate_paths(self):
        if not self.anomalyvfm_root.is_dir():
            raise FileNotFoundError(
                f"AnomalyVFM_ root not found: {self.anomalyvfm_root}"
            )

        offline_wrapper = self.anomalyvfm_root / "models" / "dinov2_offline.py"
        if not offline_wrapper.is_file():
            raise FileNotFoundError(
                f"Required offline AnomalyVFM wrapper not found: {offline_wrapper}"
            )

        peft_dir = self.anomalyvfm_root / "peft_local"
        if not peft_dir.is_dir():
            raise FileNotFoundError(
                f"Required AnomalyVFM peft_local directory not found: {peft_dir}"
            )

        if self.variant in {"a1_adapted_final", "a2_adapted_middle"}:
            if not self.anomalyvfm_ckpt.is_file():
                raise FileNotFoundError(
                    f"AnomalyVFM DINOv2 checkpoint not found: "
                    f"{self.anomalyvfm_ckpt}"
                )

        if not self.dino_repo_path.is_dir():
            raise FileNotFoundError(
                f"Local official DINOv2 repo not found: {self.dino_repo_path}"
            )

        if not (self.dino_repo_path / "hubconf.py").is_file():
            raise FileNotFoundError(
                "Local DINOv2 repo does not contain hubconf.py: "
                f"{self.dino_repo_path}"
            )

        if not self.dino_weight_path.is_file():
            raise FileNotFoundError(
                f"Local DINOv2 ViT-L/14-register weight not found: "
                f"{self.dino_weight_path}"
            )

    def _apply_clahe(self, pil_imgs: list) -> list:
        """Preserve the existing SubspaceAD optional CLAHE flag."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = []
        for img in pil_imgs:
            rgb = np.asarray(img)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = clahe.apply(l_channel)
            merged = cv2.merge((l_channel, a_channel, b_channel))
            processed.append(
                Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB))
            )
        return processed

    def _subspace_layers_to_meta_blocks(self, layers: list) -> list:
        """
        Convert SubspaceAD/Hugging-Face hidden-state indices to Meta block indices.

        H0 uses Hugging Face outputs.hidden_states, whose conceptual layout is:
            hidden_states[0]  = embedding output
            hidden_states[1]  = output after transformer block 0
            ...
            hidden_states[24] = output after transformer block 23

        Meta DINOv2 get_intermediate_layers() expects transformer block indices
        0..23.

        Therefore:
            HF hidden-state index h -> Meta block index h - 1

        Example with 24 transformer blocks:
            -18 -> hidden index 7  -> Meta block 6
            -12 -> hidden index 13 -> Meta block 12

        Thus the existing H0 default
            -12,-13,-14,-15,-16,-17,-18
        maps to Meta blocks 12,11,10,9,8,7,6 (same set = blocks 6..12).
        """
        if not layers:
            raise ValueError(
                "A2 requires --layers because it uses adapted middle features."
            )

        num_blocks = len(self.model.net.blocks)
        num_hidden_states = num_blocks + 1

        block_indices = []
        for raw_idx in layers:
            raw_idx = int(raw_idx)
            hidden_idx = (
                raw_idx
                if raw_idx >= 0
                else num_hidden_states + raw_idx
            )

            if hidden_idx <= 0 or hidden_idx >= num_hidden_states:
                raise ValueError(
                    f"A2 layer index {raw_idx} resolves to hidden-state index "
                    f"{hidden_idx}. Valid SubspaceAD transformer-output indices "
                    f"are 1..{num_blocks} or their negative equivalents."
                )

            block_idx = hidden_idx - 1
            block_indices.append(block_idx)

        block_indices = sorted(set(block_indices))

        logging.info(
            "A2 layer mapping: SubspaceAD layers=%s -> Meta DINOv2 blocks=%s",
            layers,
            block_indices,
        )
        return block_indices

    def _extract_final_patch_tokens(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        A0/A1 final representation.
        """
        _, patch_tokens = self.model(input_tensor)
        return patch_tokens

    def _extract_adapted_middle_patch_tokens(
        self,
        input_tensor: torch.Tensor,
        layers: list,
        agg_method: str,
    ) -> torch.Tensor:
        """
        A2 representation.

        Each selected intermediate output is normalized with the model's final
        LayerNorm by get_intermediate_layers(..., norm=True), then selected
        layers are mean-aggregated.
        """
        if agg_method != "mean":
            raise ValueError(
                "A2 is defined as adapted middle layers with mean aggregation. "
                f"Received --agg_method {agg_method!r}; use --agg_method mean."
            )

        block_indices = self._subspace_layers_to_meta_blocks(layers)

        outputs = self.model.net.get_intermediate_layers(
            input_tensor,
            n=block_indices,
            reshape=False,
            return_class_token=False,
            norm=True,
        )

        if not outputs:
            raise RuntimeError(
                "Meta DINOv2 returned no intermediate features for A2."
            )

        patch_tokens = torch.stack(list(outputs), dim=0).mean(dim=0)
        return patch_tokens

    @torch.no_grad()
    def extract_tokens(
        self,
        pil_imgs: list,
        res: int,
        layers: list,
        agg_method: str,
        grouped_layers: list = None,
        docrop: bool = False,
        use_clahe: bool = False,
        dino_saliency_layer: int = 0,
    ):
        """
        Return A0/A1/A2 features using the existing SubspaceAD extractor API.
        """
        del grouped_layers, dino_saliency_layer

        if int(res) != self.EXPECTED_IMAGE_SIZE:
            raise ValueError(
                "Meta/AnomalyVFM DINOv2 ablation requires res=672, "
                f"but received res={res}."
            )

        if docrop:
            raise ValueError(
                "Meta/AnomalyVFM DINOv2 ablation uses the native "
                "Resize(672,672) preprocessing and does not support --docrop."
            )

        if not pil_imgs:
            raise ValueError("extract_tokens() received an empty image batch.")

        if use_clahe:
            pil_imgs = self._apply_clahe(pil_imgs)

        input_tensor = torch.stack(
            [self.transform(img.convert("RGB")) for img in pil_imgs],
            dim=0,
        ).to(DEVICE)

        if self.variant in {"a0_original_final", "a1_adapted_final"}:
            patch_tokens = self._extract_final_patch_tokens(input_tensor)
        elif self.variant == "a2_adapted_middle":
            patch_tokens = self._extract_adapted_middle_patch_tokens(
                input_tensor,
                layers=layers,
                agg_method=agg_method,
            )
        else:
            raise RuntimeError(f"Unhandled variant: {self.variant}")

        if patch_tokens.ndim != 3:
            raise ValueError(
                "Expected patch tokens with shape [B, N, D], "
                f"got {tuple(patch_tokens.shape)}."
            )

        batch_size, patch_count, feature_dim = patch_tokens.shape

        if patch_count != self.expected_patch_count:
            raise ValueError(
                f"Expected {self.expected_patch_count} patch tokens "
                f"({self.h_p}x{self.w_p}), got {patch_count}."
            )

        if feature_dim != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {feature_dim}."
            )

        if not torch.isfinite(patch_tokens).all():
            raise FloatingPointError(
                f"NaN or Inf detected in {self.variant} patch features."
            )

        tokens = (
            patch_tokens.detach()
            .float()
            .reshape(
                batch_size,
                self.h_p,
                self.w_p,
                self.feature_dim,
            )
            .cpu()
            .numpy()
        )

        saliency_placeholder = np.zeros(
            (batch_size, self.h_p, self.w_p),
            dtype=np.float32,
        )

        return tokens, (self.h_p, self.w_p), saliency_placeholder
