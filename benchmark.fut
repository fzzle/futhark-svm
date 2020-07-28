module fsvm = import "lib/github.com/fzzle/futhark-svm/main"

-- ==
-- entry: fit
-- compiled input @data/adult_fit_linearC1.in
-- compiled input @data/mnist10000_fit_polynomialC10d3.in
-- compiled input @data/mnist10000_fit_linearC10.in
entry fit = fsvm.fit

-- ==
-- entry: predict
-- compiled input @data/adult_predict_linearC1.in
-- compiled input @data/mnist10000_predict_polynomialC10d3.in
-- compiled input @data/mnist10000_predict_linearC10.in
entry predict = fsvm.predict
