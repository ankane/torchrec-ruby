module TorchRec
  module Models
    module DLRM
      class DenseArch < Torch::NN::Module
        def initialize(in_features, layer_sizes, device: nil)
          super()
          @model = Modules::MLP::MLP.new(
            in_features, layer_sizes, bias: true, activation: :relu, device: device
          )
        end

        def forward(features)
          @model.call(features)
        end
      end
    end
  end
end
