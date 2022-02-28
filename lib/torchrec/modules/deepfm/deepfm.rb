module TorchRec
  module Modules
    module DeepFM
      class DeepFM < Torch::NN::Module
        def initialize(dense_module)
          super()
          @dense_module = dense_module
        end

        def forward(embeddings)
          deepfm_input = flatten_input(embeddings)
          @dense_module.call(deepfm_input)
        end

        private

        def flatten_input(inputs)
          Torch.cat(inputs.map { |input| input.flatten(1) }, dim: 1)
        end
      end
    end
  end
end
