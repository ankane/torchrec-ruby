module TorchRec
  module Modules
    module MLP
      class MLP < Torch::NN::Module
        def initialize(in_size, layer_sizes, bias: true, activation: :relu, device: nil)
          super()

          if activation == :relu
            activation = Torch.method(:relu)
          elsif activation == :sigmoid
            activation = Torch.method(:sigmoid)
          end

          if !activation.is_a?(Symbol)
            @mlp = Torch::NN::Sequential.new(
              *layer_sizes.length.times.map do |i|
                Perceptron.new(
                  i > 0 ? layer_sizes[i - 1] : in_size,
                  layer_sizes[i],
                  bias: bias,
                  activation: Utils.extract_module_or_tensor_callable(activation),
                  device: device
                )
              end
            )
          else
            if activation == :swish_layernorm
              @mlp = Torch::NN::Sequential.new(
                *layer_sizes.length.times.map do |i|
                  Perceptron.new(
                    i > 0 ? layer_sizes[i - 1] : in_size,
                    layer_sizes[i],
                    bias: bias,
                    activation: Activation::SwishLayerNorm.new(layer_sizes[i], device: device),
                    device: device
                  )
                end
              )
            else
              raise ArgumentError, "This MLP only supports activation function of :relu, :sigmoid, and :swish_layernorm"
            end
          end
        end

        def forward(input)
          @mlp.call(input)
        end
      end
    end
  end
end
