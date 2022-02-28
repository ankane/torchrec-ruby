module TorchRec
  module Modules
    module MLP
      class Perceptron < Torch::NN::Module
        def initialize(in_size, out_size, bias: true, activation: Torch.method(:relu), device: nil)
          super()
          @out_size = out_size
          @in_size = in_size
          @linear = Torch::NN::Linear.new(
            # TODO add device
            @in_size, @out_size, bias: bias #, device: device
          )
          @activation_fn = activation
        end

        def forward(input)
          @activation_fn.call(@linear.call(input))
        end
      end
    end
  end
end
