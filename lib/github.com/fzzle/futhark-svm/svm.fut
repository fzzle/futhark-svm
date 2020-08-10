import "svc"
import "kernel"

-- | Aggregation module for futhark-svm.
module fsvm (R: float) = {
  module svc     = svc R
  module kernels = kernels R
}