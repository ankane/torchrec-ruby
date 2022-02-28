require_relative "test_helper"

class ModulesTest < Minitest::Test
  def test_swish_layer_norm
    sln = TorchRec::Modules::Activation::SwishLayerNorm.new(100)
    assert sln
  end
end
