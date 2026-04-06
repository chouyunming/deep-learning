import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv(x)
        for _ in range(1, self.t):
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super().__init__()
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):

    def __init__(self, n_class=1):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


class TransUNet(UNet):
    """UNet with a Transformer encoder inserted at the bottleneck.

    The CNN encoder/decoder is inherited from UNet unchanged.
    After dconv_down4 the spatial feature map is flattened into a token
    sequence, processed by a standard Transformer encoder (with learned
    positional embeddings), then reshaped back before being passed to the
    decoder.

    Args:
        n_class   : number of output channels (default 1, binary segmentation)
        img_size  : input spatial resolution; used to pre-compute seq_len
        embed_dim : channel depth at the bottleneck (must match UNet's 512)
        num_heads : attention heads in each Transformer layer
        num_layers: number of stacked Transformer encoder layers
        dropout   : dropout applied inside the Transformer layers
    """

    def __init__(self, n_class=1, img_size=512,
                 embed_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__(n_class=n_class)

        # After 3 max-pools the spatial size is img_size // 8
        bottleneck_size = img_size // 8
        seq_len = bottleneck_size * bottleneck_size   # e.g. 64×64 = 4096

        self._bottleneck_hw = bottleneck_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

    def forward(self, x):
        # ---- Encoder (identical to UNet) ----
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)             # (B, 512, H/8, W/8)

        # ---- Transformer bottleneck ----
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)    # (B, H*W, C)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # ---- Decoder (identical to UNet) ----
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        return self.conv_last(x)


class AttnUNet(UNet):
    """Attention U-Net: UNet with attention gates on every skip connection.

    Attention gates recalibrate encoder features using the decoder signal
    (gating vector) before concatenation, suppressing irrelevant activations.

    Args:
        n_class: number of output channels (default 1)
    """

    def __init__(self, n_class=1):
        super().__init__(n_class=n_class)
        # F_g  : channels of the gating signal (from the decoder path)
        # F_l  : channels of the skip-connection feature map (from the encoder)
        # F_int: intermediate channel dimension (typically F_l // 2)
        self.attn3 = Attention_block(F_g=512, F_l=256, F_int=128)
        self.attn2 = Attention_block(F_g=256, F_l=128, F_int=64)
        self.attn1 = Attention_block(F_g=128, F_l=64,  F_int=32)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        conv3 = self.attn3(g=x, x=conv3)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        conv2 = self.attn2(g=x, x=conv2)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        conv1 = self.attn1(g=x, x=conv1)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        return self.conv_last(x)


class R2UNet(UNet):
    """Recurrent Residual U-Net: replaces every double-conv block with an
    RRCNN block (1×1 projection + two stacked Recurrent_blocks + residual).

    Args:
        n_class: number of output channels (default 1)
        t      : number of recurrent iterations inside each Recurrent_block
    """

    def __init__(self, n_class=1, t=2):
        super().__init__(n_class=n_class)
        # Replace all double_conv layers with RRCNN blocks.
        # Channel layout is identical to UNet so forward() is inherited as-is.
        self.dconv_down1 = RRCNN_block(3,         64,  t=t)
        self.dconv_down2 = RRCNN_block(64,        128, t=t)
        self.dconv_down3 = RRCNN_block(128,       256, t=t)
        self.dconv_down4 = RRCNN_block(256,       512, t=t)
        self.dconv_up3   = RRCNN_block(512 + 256, 256, t=t)
        self.dconv_up2   = RRCNN_block(256 + 128, 128, t=t)
        self.dconv_up1   = RRCNN_block(128 + 64,  64,  t=t)
        # forward() is fully inherited from UNet — no override needed.


if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))

    unet = UNet(n_class=1)
    print("UNet output:     ", unet(x).shape)   # (2, 1, 512, 512)

    transunet = TransUNet(n_class=1, img_size=512)
    print("TransUNet output:", transunet(x).shape)  # (2, 1, 512, 512)

    attnunet = AttnUNet(n_class=1)
    print("AttnUNet output: ", attnunet(x).shape)   # (2, 1, 512, 512)

    r2unet = R2UNet(n_class=1, t=2)
    print("R2UNet output:   ", r2unet(x).shape)     # (2, 1, 512, 512)
