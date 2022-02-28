module TorchRec
  module Modules
    module Activation
      class SwishLayerNorm < Torch::NN::Module
        def initialize(input_dims, device: nil)
          super()
          @norm = Torch::NN::Sequential.new(
            # TODO add device
            Torch::NN::LayerNorm.new(input_dims), #, device: device),
            Torch::NN::Sigmoid.new
          )
        end

        def forward(input)
          input * @norm.call(input)
        end
      end
    end
  end
end
