require_relative "test_helper"

class ModelsTest < Minitest::Test
  def test_deepfm_dense_arch
    dense_arch = TorchRec::Models::DeepFM::DenseArch.new(10, 10, 3)
    dense_embedded = dense_arch.call(Torch.rand([20, 10]))
    assert_equal [20, 3], dense_embedded.shape
  end

  def test_deepfm_over_arch
    over_arch = TorchRec::Models::DeepFM::OverArch.new(10)
    logits = over_arch.call(Torch.rand([20, 10]))
    assert_equal [20, 1], logits.shape
  end

  def test_dlrm_dense_arch
    dense_arch = TorchRec::Models::DLRM::DenseArch.new(10, [15, 3])
    dense_embedded = dense_arch.call(Torch.rand([20, 10]))
    assert_equal [20, 3], dense_embedded.shape
  end
end
