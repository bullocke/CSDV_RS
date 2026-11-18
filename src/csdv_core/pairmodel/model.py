"""csdv_core.pairmodel.model: a Siamese pair network on a ResNet encoder.

The same encoder reads the earlier and the later image. At each level the
two feature maps and their signed difference (later minus earlier) are
joined, and a short top-down decoder brings them to the stride of the output
cells. With 0.6 m pixels a stride of 16 is one 9.6 m cell, so the network
predicts one logit per cell and never upsamples to pixels.

Built on torchvision's ResNet only. Pretrained weights are read from a
file, never downloaded, because compute nodes have no internet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import rasterio  # noqa: F401  (GDAL has to load before torch, see __init__)
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

#: Channels of the four encoder levels (strides 4, 8, 16, 32).
_CHANNELS = {"resnet18": (64, 128, 256, 512), "resnet34": (64, 128, 256, 512)}
_STRIDES = (4, 8, 16, 32)

__all__ = ["SiamesePairNet", "adapt_stem", "count_parameters"]


def adapt_stem(state: dict[str, Any], in_channels: int) -> dict[str, Any]:
    """A ResNet state dict whose first convolution takes ``in_channels`` bands.

    The filters of the first three bands are kept (red, green, blue, the
    order of the chips). Every further band starts as their mean. The
    classification layer is dropped.
    """
    out = {k: v for k, v in state.items() if not k.startswith("fc.")}
    weight = out["conv1.weight"]
    if weight.shape[1] != in_channels:
        extra = weight.mean(dim=1, keepdim=True).repeat(
            1, max(in_channels - weight.shape[1], 0), 1, 1
        )
        out["conv1.weight"] = torch.cat([weight, extra], dim=1)[:, :in_channels]
    return out


def _block(channels_in: int, channels_out: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    )


class SiamesePairNet(nn.Module):
    """Loss logits on cells from two four-band images, or from three.

    The loss is always the one between the first two images. A model built
    with ``n_dates=3`` also reads the image after them, when there is one:
    a canopy that is leafless on the second date and closed again on the
    third was not lost. The third image is optional per sample, so that the
    newest interval, which has none, is mapped by the same weights.

    Args:
        encoder: ``resnet34`` or ``resnet18``.
        in_channels: Bands per image.
        cell_px: Pixels per output cell side: 16, 8 or 4.
        width: Channels of the decoder.
        weights: A torchvision ResNet state dict (``.pth``) for the encoder.
            None: random start.
        n_agents: Number of agent classes of a second head that names the
            agent of a loss (harvest, wind, fire and so on) for every cell.
            0: no such head.
        n_dates: 2 for a pair model. 3 adds the optional later image.
    """

    def __init__(
        self,
        encoder: str = "resnet34",
        in_channels: int = 4,
        cell_px: int = 16,
        width: int = 128,
        weights: Path | None = None,
        n_agents: int = 0,
        n_dates: int = 2,
    ) -> None:
        super().__init__()
        import torchvision

        if encoder not in _CHANNELS:
            raise ValueError(f"Unknown encoder {encoder!r}")
        if cell_px not in (4, 8, 16):
            raise ValueError("cell_px must be 4, 8 or 16")
        if n_dates not in (2, 3):
            raise ValueError("n_dates must be 2 or 3")
        self.n_dates = n_dates
        self.encoder_name = encoder
        self.cell_px = cell_px
        self.n_agents = n_agents
        net = getattr(torchvision.models, encoder)(weights=None)
        net.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        if weights is not None:
            state = torch.load(Path(weights), map_location="cpu", weights_only=True)
            missing, unexpected = net.load_state_dict(
                adapt_stem(state, in_channels), strict=False
            )
            left = [k for k in missing if not k.startswith("fc.")]
            if left or unexpected:
                raise ValueError(
                    f"{weights} does not fit {encoder}: missing {left}, "
                    f"unexpected {list(unexpected)}"
                )
            logger.info("Encoder weights from %s", weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.levels = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        # Levels the decoder uses: from the stride of the cells to stride 32.
        self.first = _STRIDES.index(cell_px)
        channels = _CHANNELS[encoder]
        # Per level: both encodings and their difference, and with a third
        # date its encoding and its difference from the first.
        parts = 3 if n_dates == 2 else 5
        self.fuse = nn.ModuleList(
            [_block(parts * channels[i], width, 1) for i in range(self.first, 4)]
        )
        self.smooth = nn.ModuleList(
            [_block(width, width, 3) for _ in range(self.first, 3)]
        )
        self.head = nn.Sequential(_block(width, width, 3), nn.Conv2d(width, 1, 1))
        self.agent_head = (
            nn.Sequential(_block(width, width, 3), nn.Conv2d(width, n_agents, 1))
            if n_agents
            else None
        )

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Feature maps of one image at strides 4, 8, 16 and 32."""
        out = []
        x = self.stem(x)
        for level in self.levels:
            x = level(x)
            out.append(x)
        return out

    def _joined(
        self,
        fa: list[torch.Tensor],
        fb: list[torch.Tensor],
        fc: list[torch.Tensor] | None = None,
        has_c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decoder features at the stride of the cells, from the encodings."""
        joined = []
        for fuse, i in zip(self.fuse, range(self.first, 4), strict=True):
            parts = [fa[i], fb[i], fb[i] - fa[i]]
            if self.n_dates == 3:
                parts += self._third(fa[i], None if fc is None else fc[i], has_c)
            joined.append(fuse(torch.cat(parts, dim=1)))
        x = joined[-1]
        for lateral, smooth in zip(
            reversed(joined[:-1]), reversed(self.smooth), strict=True
        ):
            x = F.interpolate(
                x, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            x = smooth(x + lateral)
        return x

    @staticmethod
    def _third(
        fa: torch.Tensor, fc: torch.Tensor | None, has_c: torch.Tensor | None
    ) -> list[torch.Tensor]:
        """The third date's part of the fusion input, zero where it is absent."""
        if fc is None:
            zero = torch.zeros_like(fa)
            return [zero, zero]
        if has_c is None:
            return [fc, fc - fa]
        present = has_c.to(fa.dtype).view(-1, 1, 1, 1)
        return [present * fc, present * (fc - fa)]

    def decode(
        self,
        fa: list[torch.Tensor],
        fb: list[torch.Tensor],
        fc: list[torch.Tensor] | None = None,
        has_c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Loss logits ``(batch, 1, rows / cell_px, cols / cell_px)``.

        Args:
            fa: Encoding of the earlier image.
            fb: Encoding of the later image.
            fc: Encoding of the image after that (models with three dates).
                None: no sample has one.
            has_c: ``(batch,)``, 1 where the sample has a third image. None:
                every sample has one.
        """
        return self.head(self._joined(fa, fb, fc, has_c))

    def decode_agents(
        self,
        fa: list[torch.Tensor],
        fb: list[torch.Tensor],
        fc: list[torch.Tensor] | None = None,
        has_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Loss logits, and agent logits ``(batch, n_agents, rows, cols)`` or None."""
        x = self._joined(fa, fb, fc, has_c)
        agents = self.agent_head(x) if self.agent_head is not None else None
        return self.head(x), agents

    def encode_third(
        self, c: torch.Tensor | None, has_c: torch.Tensor | None
    ) -> list[torch.Tensor] | None:
        """Encoding of the third images of a batch, zero where there is none.

        Only the images that exist pass through the encoder, so that empty
        ones do not enter its batch statistics.
        """
        if self.n_dates != 3 or c is None:
            return None
        if has_c is None:
            return self.encode(c)
        present = has_c.bool()
        if not bool(present.any()):
            return None
        if bool(present.all()):
            return self.encode(c)
        out = []
        for feature in self.encode(c[present]):
            full = feature.new_zeros((c.shape[0], *feature.shape[1:]))
            full[present] = feature
            out.append(full)
        return out

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor | None = None,
        has_c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decode(
            self.encode(a), self.encode(b), self.encode_third(c, has_c), has_c
        )

    def forward_agents(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor | None = None,
        has_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Loss logits and agent logits of a batch of pairs or triples."""
        return self.decode_agents(
            self.encode(a), self.encode(b), self.encode_third(c, has_c), has_c
        )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
