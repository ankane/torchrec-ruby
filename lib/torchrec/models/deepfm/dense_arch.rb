module TorchRec
  module Models
    module DeepFM
      class DenseArch < Torch::NN::Module
        def initialize(in_features, hidden_layer_size, embedding_dim)
          super()
          @model = Torch::NN::Sequential.new(
            Torch::NN::Linear.new(in_features, hidden_layer_size),
            Torch::NN::ReLU.new,
            Torch::NN::Linear.new(hidden_layer_size, embedding_dim),
            Torch::NN::ReLU.new
          )
        end

        def forward(features)
          @model.call(features)
        end
      end
    end
  end
end
