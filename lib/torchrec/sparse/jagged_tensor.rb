module TorchRec
  module Sparse
    class JaggedTensor
      def initialize(values, weights: nil, lengths: nil, offsets: nil)
        @values = values
        @weights = weights
        assert_offsets_or_lengths_is_provided(offsets, lengths)
        if !offsets.nil?
          assert_tensor_has_no_elements_or_has_integers(offsets, "offsets")
        end
        if !lengths.nil?
          assert_tensor_has_no_elements_or_has_integers(lengths, "lengths")
        end
        @lengths = lengths
        @offsets = offsets
      end

      private

      def assert_offsets_or_lengths_is_provided(offsets, lengths)
        if offsets.nil? && lengths.nil?
          raise ArgumentError, "Must provide lengths or offsets"
        end
      end

      def assert_tensor_has_no_elements_or_has_integers(tensor, tensor_name)
        if tensor.numel != 0 && ![:int64, :int32, :int16, :int8, :uint8].include?(tensor.dtype)
          raise ArgumentError, "#{tensor_name} must be of integer type, but got #{tensor.dtype}"
        end
      end
    end
  end
end
