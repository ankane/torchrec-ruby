require_relative "test_helper"

class ModulesTest < Minitest::Test
  def test_swish_layer_norm
    sln = TorchRec::Modules::Activation::SwishLayerNorm.new(100)
    assert sln
  end

  def test_mlp
    batch_size = 3
    in_size = 40
    input = Torch.randn(batch_size, in_size)

    layer_sizes = [16, 8, 4]
    mlp_module = TorchRec::Modules::MLP::MLP.new(in_size, layer_sizes, bias: true)
    output = mlp_module.call(input)
    assert_equal output.shape, [batch_size, layer_sizes[-1]]
  end

  def test_perceptron
    batch_size = 3
    in_size = 40
    input = Torch.randn(batch_size, in_size)

    out_size = 16
    perceptron = TorchRec::Modules::MLP::Perceptron.new(in_size, out_size, bias: true)

    output = perceptron.call(input)
    assert_equal output.shape, [batch_size, out_size]
  end
end
