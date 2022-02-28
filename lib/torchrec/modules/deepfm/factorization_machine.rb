module TorchRec
  module Modules
    module DeepFM
      class FactorizationMachine < Torch::NN::Module
        def initialize
          super()
        end

        def forward(embeddings)
          fm_input = flatten_input(embeddings)
          sum_of_input = Torch.sum(fm_input, dim: 1, keepdim: true)
          sum_of_square = Torch.sum(fm_input * fm_input, dim: 1, keepdim: true)
          square_of_sum = sum_of_input * sum_of_input
          cross_term = square_of_sum - sum_of_square
          cross_term = Torch.sum(cross_term, dim: 1, keepdim: true) * 0.5  # [B, 1]
          cross_term
        end

        private

        def flatten_input(inputs)
          Torch.cat(inputs.map { |input| input.flatten(1) }, dim: 1)
        end
      end
    end
  end
end
