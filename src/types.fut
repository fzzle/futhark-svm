
type Parameters = {

}

-- Solver taking SVM parameters, xs, and ys. Returns alphas and b.
type Solver [n] [m] = Parameters -> [n][m]f64 -> [n]i8 -> ([n]f64, f64)

type Kernel [m] = Parameters -> [m]f64 -> [m]f64 -> f64
