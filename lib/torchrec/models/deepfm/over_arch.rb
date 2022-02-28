module TorchRec
  module Models
    module DeepFM
      class OverArch < Torch::NN::Module
        def initialize(in_features)
          super()
          @model = Torch::NN::Sequential.new(
            Torch::NN::Linear.new(in_features, 1),
            Torch::NN::Sigmoid.new
          )
        end

        def forward(features)
          @model.call(features)
        end
      end
    end
  end
end
