# TorchRec Ruby

Deep learning recommendation systems for Ruby

[![Build Status](https://github.com/ankane/torchrec-ruby/workflows/build/badge.svg?branch=master)](https://github.com/ankane/torchrec-ruby/actions)

## Installation

Add this line to your application’s Gemfile:

```ruby
gem "torchrec"
```

## Getting Started

This library follows the [Python API](https://pytorch.org/torchrec/). Many methods and options are missing at the moment. PRs welcome!

## Models

DeepFM

```ruby
TorchRec::Models::DeepFM::DenseArch.new(in_features, hidden_layer_size, embedding_dim)
TorchRec::Models::DeepFM::OverArch.new(in_features)
```

DLRM

```ruby
TorchRec::Models::DLRM::DenseArch.new(in_features, layer_sizes, device: nil)
```

## Modules

```ruby
TorchRec::Modules::Activation::SwishLayerNorm.new(input_dims, device: nil)
TorchRec::Modules::MLP::MLP.new(in_size, layer_sizes, bias: true, activation: :relu, device: nil)
TorchRec::Modules::MLP::Perceptron.new(in_size, out_size, bias: true, activation: Torch.method(:relu), device: nil)
```

## History

View the [changelog](https://github.com/ankane/torchrec-ruby/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/torchrec-ruby/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/torchrec-ruby/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/torchrec-ruby.git
cd torchrec-ruby
bundle install
bundle exec rake test
```
