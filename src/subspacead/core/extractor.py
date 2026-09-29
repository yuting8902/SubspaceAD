import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# Force Hugging Face / Transformers into offline mode. The user supplies a local
# directory containing config.json, model.safetensors and preprocessor_config.json.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor:
    """DINOv2 feature extraction interface used by SubspaceAD."""

    def __init__(self, model_ckpt: str):
        if not os.path.isdir(model_ckpt):
            raise FileNotFoundError(
                f"Offline model directory not found: {model_ckpt}\n"
                "Expected a local DINOv2 folder containing config.json, "
                "model.safetensors and preprocessor_config.json."
            )

        logging.info("Loading local feature extraction model: %s", model_ckpt)
        self.processor = AutoImageProcessor.from_pretrained(
            model_ckpt,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            model_ckpt,
            local_files_only=True,
        ).eval().to(DEVICE)

        try:
            self.model.set_attn_implementation("eager")
            logging.info("Using eager attention so attention maps are available.")
        except AttributeError:
            logging.warning(
                "This Transformers model does not expose set_attn_implementation(). "
                "DINO saliency may be unavailable."
            )

        cfg = self.model.config
        logging.info(
            "Local model loaded: hidden_size=%s, patch_size=%s, num_hidden_layers=%s, num_register_tokens=%s",
            getattr(cfg, "hidden_size", "unknown"),
            getattr(cfg, "patch_size", "unknown"),
            getattr(cfg, "num_hidden_layers", "unknown"),
            getattr(cfg, "num_register_tokens", 0),
        )

    def _apply_clahe(self, pil_imgs: list) -> list:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = []
        for img in pil_imgs:
            rgb = np.asarray(img)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = clahe.apply(l_channel)
            merged = cv2.merge((l_channel, a_channel, b_channel))
            processed.append(Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)))
        return processed

    @staticmethod
    def _spatial_from_seq(
        seq_tokens: torch.Tensor,
        drop_front: int,
        n_expected: int,
        h_p: int,
        w_p: int,
    ) -> torch.Tensor:
        patch_tokens = seq_tokens[:, drop_front : drop_front + n_expected, :]
        if patch_tokens.shape[1] != n_expected:
            raise ValueError(
                f"Expected {n_expected} patch tokens but got {patch_tokens.shape[1]}."
            )
        return patch_tokens.reshape(seq_tokens.shape[0], h_p, w_p, seq_tokens.shape[-1])

    def _get_saliency_mask(
        self,
        attentions: tuple,
        dino_saliency_layer: int,
        num_reg: int,
        drop_front: int,
        n_expected: int,
        batch_size: int,
        h_p: int,
        w_p: int,
    ) -> np.ndarray:
        if attentions is None:
            raise ValueError(
                "Attention weights were not returned. Check Transformers compatibility "
                "or use an eager attention implementation."
            )

        layer = dino_saliency_layer
        if layer < 0:
            layer += len(attentions)
        if layer < 0 or layer >= len(attentions):
            logging.warning(
                "DINO saliency layer %s is outside [0, %s]; using layer 0.",
                dino_saliency_layer,
                len(attentions) - 1,
            )
            layer = 0

        attn = attentions[layer]
        patch_slice = slice(drop_front, drop_front + n_expected)

        if num_reg > 0:
            # Average attention from all register tokens to all patch tokens.
            saliency = attn[:, :, 1:drop_front, patch_slice].mean(dim=(1, 2))
        else:
            # Fallback for DINO-style checkpoints without registers.
            saliency = attn[:, :, 0, patch_slice].mean(dim=1)

        return saliency.reshape(batch_size, h_p, w_p).detach().cpu().numpy()

    def _aggregate_layers(
        self,
        hidden_states: tuple,
        layers: list,
        grouped_layers: list,
        agg_method: str,
        drop_front: int,
        n_expected: int,
        h_p: int,
        w_p: int,
    ) -> np.ndarray:
        def convert(index):
            return self._spatial_from_seq(
                hidden_states[index], drop_front, n_expected, h_p, w_p
            )

        if agg_method == "group":
            if not grouped_layers:
                raise ValueError("--grouped_layers is required when --agg_method group.")
            groups = [
                torch.stack([convert(index) for index in group], dim=0).mean(dim=0)
                for group in grouped_layers
            ]
            fused = torch.cat(groups, dim=-1)
        else:
            features = [convert(index) for index in layers]
            if agg_method == "mean":
                fused = torch.stack(features, dim=0).mean(dim=0)
            elif agg_method == "concat":
                fused = torch.cat(features, dim=-1)
            else:
                raise ValueError(f"Unknown aggregation method: {agg_method}")

        return fused.detach().cpu().numpy()

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
        grouped_layers = grouped_layers or []

        if use_clahe:
            pil_imgs = self._apply_clahe(pil_imgs)

        if docrop:
            resize_res = int(res / 0.875)
            size = {"height": resize_res, "width": resize_res}
        else:
            size = {"height": res, "width": res}

        crop_size = {"height": res, "width": res}
        inputs = self.processor(
            images=pil_imgs,
            return_tensors="pt",
            do_resize=True,
            size=size,
            do_center_crop=docrop,
            crop_size=crop_size,
        ).to(DEVICE)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
        )

        cfg = self.model.config
        patch_size = int(cfg.patch_size)
        num_reg = int(getattr(cfg, "num_register_tokens", 0))
        drop_front = 1 + num_reg
        h_p = res // patch_size
        w_p = res // patch_size
        n_expected = h_p * w_p
        batch_size = inputs.pixel_values.shape[0]

        saliency_mask = self._get_saliency_mask(
            outputs.attentions,
            dino_saliency_layer,
            num_reg,
            drop_front,
            n_expected,
            batch_size,
            h_p,
            w_p,
        )

        fused_tokens = self._aggregate_layers(
            outputs.hidden_states,
            layers,
            grouped_layers,
            agg_method,
            drop_front,
            n_expected,
            h_p,
            w_p,
        )

        return fused_tokens, (h_p, w_p), saliency_mask