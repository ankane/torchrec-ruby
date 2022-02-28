module TorchRec
  module Modules
    module CrossNet
      class CrossNet < Torch::NN::Module
        def initialize(in_features, num_layers)
          super()
          @num_layers = num_layers
          @kernels = Torch::NN::ParameterList.new(
            @num_layers.times.map do |i|
              Torch::NN::Parameter.new(
                Torch::NN::Init.xavier_normal!(Torch.empty(in_features, in_features))
              )
            end
          )
          @bias = Torch::NN::ParameterList.new(
            @num_layers.times.map do |i|
              Torch::NN::Parameter.new(Torch::NN::Init.zeros!(Torch.empty(in_features, 1)))
            end
          )
        end

        def forward(input)
          x_0 = input.unsqueeze(2)  # (B, N, 1)
          x_l = x_0

          @num_layers.times do |layer|
            xl_w = Torch.matmul(@kernels[layer], x_l)  # (B, N, 1)
            x_l = x_0 * (xl_w + @bias[layer]) + x_l  # (B, N, 1)
          end

          Torch.squeeze(x_l, dim: 2)
        end
      end
    end
  end
end
