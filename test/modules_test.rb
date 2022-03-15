require_relative "test_helper"

class ModulesTest < Minitest::Test
  def test_swish_layer_norm
    sln = TorchRec::Modules::Activation::SwishLayerNorm.new(100)
    assert sln
  end

  def test_cross_net
    batch_size = 3
    num_layers = 2
    in_features = 10
    input = Torch.randn(batch_size, in_features)
    dcn = TorchRec::Modules::CrossNet::CrossNet.new(in_features, num_layers)
    output = dcn.call(input)
    assert output
  end

  def test_deepfm
    batch_size = 3
    output_dim = 30
    input_embeddings = [
      Torch.randn(batch_size, 2, 64),
      Torch.randn(batch_size, 2, 32)
    ]
    dense_module = Torch::NN::Linear.new(192, output_dim)
    deepfm = TorchRec::Modules::DeepFM::DeepFM.new(dense_module)
    deep_fm_output = deepfm.call(input_embeddings)
    assert deep_fm_output
  end

  def test_factorization_machine
    batch_size = 3
    input_embeddings = [
      Torch.randn(batch_size, 2, 64),
      Torch.randn(batch_size, 2, 32)
    ]
    fm = TorchRec::Modules::DeepFM::FactorizationMachine.new
    output = fm.call(input_embeddings)
    assert output
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
