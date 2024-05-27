import torch
from torch import nn
from typing import List, Union, Sequence, Optional
from typing_extensions import Literal
import torch.nn.functional as F


__all__ = ["get_mlp"]


class LassoLayer(torch.nn.Module):
    def __init__(self, in_size, out_size) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_size, in_size))

    def forward(self, input):
        pass


def get_mlp(
    n_in: int,
    n_out: int,
    layers: List[int],
    layer_normalization: Union[None, Literal["bn"], Literal["gn"]] = None,
    output_normalization: Union[None, Literal["bn"], Literal["gn"]] = None,
    output_normalization_kwargs=None,
    act_inf_param=0.2,
):
    """
    Creates an MLP.

    Args:
        n_in: Dimensionality of the input data
        n_out: Dimensionality of the output data
        layers: Number of neurons for each hidden layer
        layer_normalization: Normalization for each hidden layer.
            Possible values: bn (batch norm), gn (group norm), None
        output_normalization: Normalization applied to output of network.
        output_normalization_kwargs: Arguments passed to the output normalization, e.g., the radius for the sphere.
    """
    modules: List[nn.Module] = []

    def add_module(n_layer_in: int, n_layer_out: int, last_layer: bool = False):
        modules.append(nn.Linear(n_layer_in, n_layer_out))
        # perform normalization & activation not in last layer
        if not last_layer:
            if layer_normalization == "bn":
                modules.append(nn.BatchNorm1d(n_layer_out))
            elif layer_normalization == "gn":
                modules.append(nn.GroupNorm(1, n_layer_out))
            modules.append(nn.LeakyReLU(negative_slope=act_inf_param))
        else:
            if output_normalization == "bn":
                modules.append(nn.BatchNorm1d(n_layer_out))
            elif output_normalization == "gn":
                modules.append(nn.GroupNorm(1, n_layer_out))

        return n_layer_out

    if len(layers) > 0:
        n_out_last_layer = n_in
    else:
        assert n_in == n_out, "Network with no layers must have matching n_in and n_out"
        modules.append(layers.Lambda(lambda x: x))

    layers.append(n_out)

    for i, l in enumerate(layers):
        n_out_last_layer = add_module(n_out_last_layer, l, i == len(layers) - 1)

    if output_normalization_kwargs is None:
        output_normalization_kwargs = {}
    return nn.Sequential(*modules)


class CNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        kernel_size: Sequence[int],
        strides: Sequence[int],
        paddings: Sequence[int],
        hidden_dims: Sequence[int],
        layer_transpose: Sequence[bool],
        out_channels: Optional[Sequence[int]],
        output_paddings: Optional[Sequence[int]],
        activation_fn: Optional[torch.nn.Module] = torch.nn.ReLU,
    ):
        super().__init__()
        conv_module = {False: torch.nn.Conv2d, True: torch.nn.ConvTranspose2d}
        # build CNN backbone
        modules = []
        layer_sizes = [in_channels]
        for i, h_dim in enumerate(hidden_dims):
            if not layer_transpose[i]:
                cnn_layer = conv_module[layer_transpose[i]](
                    in_channels=layer_sizes[-1],
                    out_channels=h_dim,
                    kernel_size=kernel_size[i],
                    stride=strides[i],
                    padding=paddings[i],
                )
            else:
                cnn_layer = conv_module[layer_transpose[i]](
                    in_channels=layer_sizes[-1],
                    out_channels=h_dim,
                    kernel_size=kernel_size[i],
                    stride=strides[i],
                    padding=paddings[i],
                    output_padding=output_paddings[i],
                )
            modules.append(
                torch.nn.Sequential(cnn_layer, activation_fn(inplace=True))
            )  
            layer_sizes += [h_dim]
        if out_channels is not None:
            modules.append(
                torch.nn.ConvTranspose2d(
                    in_channels=layer_sizes[-1],
                    out_channels=out_channels,
                    kernel_size=kernel_size[-1],
                    stride=strides[-1],
                    padding=paddings[-1],
                    output_padding=output_paddings[-1],
                ),
            )
        self.model = torch.nn.Sequential(*modules)

    def forward(self, inputs):
        x = inputs  # inputs shape: [batch_size, in_channels=3, H, W]
        for layer in self.model:
            x = layer(x)
        return x


class ConvEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size=3,
        output_size=18,
        output_normalization: Union[None, Literal["bn"], Literal["gn"]] = "bn",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_size, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        self.readout = nn.Linear(4 * 16 * 16, output_size)
        if output_normalization == "bn":
            self.norm = nn.BatchNorm1d(output_size)
        elif output_normalization == "gn":
            self.norm = nn.GroupNorm(1, output_size)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        return self.norm(self.readout(x.reshape(x.shape[0], -1)))


class ConvDecoder(nn.Module):
    def __init__(self, input_size=18, output_size=3) -> None:
        super().__init__()
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.to2d = nn.Linear(input_size, 4 * 16 * 16)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, inputs):
        ## decode ##
        x = self.to2d(inputs).reshape(inputs.shape[0], 4, 16, 16)
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv2(x))
        return x


from torchvision.models import resnet18


class ResizeConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, scale_factor, mode="nearest"
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class ResNetEnc(torch.nn.Module):
    def __init__(
        self, backbone, hidden_size=100, encoding_size=7, output_normalization="bn"
    ) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            backbone(num_classes=hidden_size),
            nn.BatchNorm1d(hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, encoding_size),
        )

    def forward(self, x):
        return self.model(x)


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(
                in_planes, planes, kernel_size=3, scale_factor=stride
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNetDec(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 64, 64)
        return x