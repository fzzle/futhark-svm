import "svc"
import "kernel"
import "types"

-- | Aggregation module for futhark-svm.
module svm (R: float) = {

  -- | Float type
  type t = R.t

  module svc     = svc R
  module kernels = kernels R

  -- | Default training settings.
  let default_fit: training_settings t = {
    n_ws = 1024,       -- # of threads.
    max_t = 100000000, -- 10x libsvm's default.
    max_t_in = 102400, -- Same as tsvm (max 100000 (n_ws*100)).
    max_t_out = -1,    -- Same as tsvm, infinite.
    eps = R.f32 0.001  -- Same as sklearn, tsvm.
  }

  -- | Default prediction settings.
  let default_predict: prediction_settings t = {
    n_ws = 64
  }
}