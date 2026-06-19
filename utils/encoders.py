import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out


def _resnet_block(x, out_channels, stride=1, name=None):
    """Inline ResNet block — avoids flax submodule int-param issues."""
    residual = x
    x = nn.Conv(out_channels, kernel_size=(3, 3), strides=stride, padding='SAME',
                use_bias=True, name=None if name is None else f'{name}_Conv_0')(x)
    x = nn.LayerNorm(name=None if name is None else f'{name}_LayerNorm_0')(x)
    x = nn.relu(x)
    x = nn.Conv(out_channels, kernel_size=(3, 3), strides=1, padding='SAME',
                use_bias=True, name=None if name is None else f'{name}_Conv_1')(x)
    x = nn.LayerNorm(name=None if name is None else f'{name}_LayerNorm_1')(x)
    if residual.shape != x.shape:
        residual = nn.Conv(out_channels, kernel_size=(1, 1), strides=stride,
                          use_bias=True, name=None if name is None else f'{name}_Conv_2')(residual)
    return nn.relu(x + residual)


class ResNet10Encoder(nn.Module):
    """ResNet-10 encoder with optional torchvision pretrained conv weights.

    64×64 RGB → stem(conv7×7,s2 + maxpool,s2) → 4 res_blocks(1 each)
    → global avgpool → MLP(512) → 512.
    """
    mlp_hidden_dims: Sequence[int] = (512,)
    pretrained: bool = False

    @staticmethod
    def load_pretrained_weights():
        """Load torchvision ResNet-18 conv weights, convert to Flax format."""
        try:
            from torchvision.models import resnet18, ResNet18_Weights
        except ImportError:
            return None
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        state = model.state_dict()
        flax_params = {}
        # Stem: (64,3,7,7) → (7,7,3,64)
        flax_params['conv_init'] = {
            'kernel': state['conv1.weight'].numpy().transpose(2, 3, 1, 0),
            'bias': state.get('conv1.bias', None) and jnp.array(state['conv1.bias'].numpy()),
        }
        for i, layer in enumerate(['layer1.0', 'layer2.0', 'layer3.0', 'layer4.0']):
            block = {}
            for j, conv in enumerate(['conv1', 'conv2']):
                k = f'{layer}.{conv}'
                block[f'Conv_{j}'] = {
                    'kernel': state[f'{k}.weight'].numpy().transpose(2, 3, 1, 0),
                    'bias': jnp.array(state[f'{k}.bias'].numpy()) if f'{k}.bias' in state else None,
                }
            if f'{layer}.downsample.0.weight' in state:
                block['Conv_2'] = {
                    'kernel': state[f'{layer}.downsample.0.weight'].numpy().transpose(2, 3, 1, 0),
                }
            flax_params[f'ResNetBlock_{i}'] = block
        return flax_params

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.0

        # Stem: 64×64 → 16×16
        x = nn.Conv(64, kernel_size=(7, 7), strides=2, padding='SAME', use_bias=True, name='conv_init')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # 4 res blocks
        x = _resnet_block(x, 64, name='ResNetBlock_0')
        x = _resnet_block(x, 128, stride=2, name='ResNetBlock_1')
        x = _resnet_block(x, 256, stride=2, name='ResNetBlock_2')
        x = _resnet_block(x, 512, stride=2, name='ResNetBlock_3')

        # Head
        x = nn.relu(x)
        x = x.mean(axis=(1, 2))
        x = MLP(self.mlp_hidden_dims, activate_final=True)(x)
        return x


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'resnet10': ResNet10Encoder,
    'resnet10_pretrained': functools.partial(ResNet10Encoder, pretrained=True),
}
