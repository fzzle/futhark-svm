Training MNIST 6 numbers,
C=10, gamma=0.1, coef0=0, degree=3

|kernel    |libsvm|thundersvm|futhark-svm|
|:---------|:-----|:---------|:----------|
|linear    |>2hr  |52sec     |122sec     |
|polynomial|89sec |9.5sec    |9.6sec     |
