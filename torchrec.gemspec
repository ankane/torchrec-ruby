require_relative "lib/torchrec/version"

Gem::Specification.new do |spec|
  spec.name          = "torchrec"
  spec.version       = TorchRec::VERSION
  spec.summary       = "Deep learning recommendation systems for Ruby"
  spec.homepage      = "https://github.com/ankane/torchrec-ruby"
  spec.license       = "BSD-3-Clause"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@ankane.org"

  spec.files         = Dir["*.{md,txt}", "{lib}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.6"

  spec.add_dependency "torch-rb", ">= 0.10"
end
