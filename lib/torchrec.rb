# dependencies
require "torch"

# models
require "torchrec/models/deepfm/dense_arch"
require "torchrec/models/deepfm/over_arch"
require "torchrec/models/dlrm/dense_arch"

# modules
require "torchrec/modules/activation/swish_layer_norm"
require "torchrec/modules/cross_net/cross_net"
require "torchrec/modules/deepfm/deepfm"
require "torchrec/modules/mlp/mlp"
require "torchrec/modules/mlp/perceptron"
require "torchrec/modules/utils"

# other
require "torchrec/version"

module TorchRec
  class Error < StandardError; end
end
