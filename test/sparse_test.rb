require_relative "test_helper"

class SparseTest < Minitest::Test
  def test_jagged_tensor
    assert TorchRec::Sparse::JaggedTensor.new(Torch.tensor([]), offsets: Torch.tensor([]))
  end
end
