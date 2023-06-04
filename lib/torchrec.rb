# dependencies
require "torch-rb"

# models
require_relative "torchrec/models/deepfm/dense_arch"
require_relative "torchrec/models/deepfm/over_arch"
require_relative "torchrec/models/dlrm/dense_arch"

# modules
require_relative "torchrec/modules/activation/swish_layer_norm"
require_relative "torchrec/modules/cross_net/cross_net"
require_relative "torchrec/modules/deepfm/deepfm"
require_relative "torchrec/modules/deepfm/factorization_machine"
require_relative "torchrec/modules/mlp/mlp"
require_relative "torchrec/modules/mlp/perceptron"
require_relative "torchrec/modules/utils"

# sparse
require_relative "torchrec/sparse/jagged_tensor"

# other
require_relative "torchrec/version"

module TorchRec
  class Error < StandardError; end
end
