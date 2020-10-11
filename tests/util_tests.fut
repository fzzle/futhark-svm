import "../lib/github.com/fzzle/futhark-svm/util"

-- ==
-- entry: test_triu
-- input { 3i64 } output {
--   [0i64, 0i64, 1i64] [1i64, 2i64, 2i64] }
-- input { 5i64 } output {
--   [0i64, 0i64, 0i64, 0i64, 1i64, 1i64, 1i64, 2i64, 2i64, 3i64]
--   [1i64, 2i64, 3i64, 4i64, 2i64, 3i64, 4i64, 3i64, 4i64, 4i64] }
-- input { 10i64 } output {
--   [0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 1i64,
--    1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 1i64, 2i64, 2i64, 2i64,
--    2i64, 2i64, 2i64, 2i64, 3i64, 3i64, 3i64, 3i64, 3i64, 3i64,
--    4i64, 4i64, 4i64, 4i64, 4i64, 5i64, 5i64, 5i64, 5i64, 6i64,
--    6i64, 6i64, 7i64, 7i64, 8i64]
--   [1i64, 2i64, 3i64, 4i64, 5i64, 6i64, 7i64, 8i64, 9i64, 2i64,
--    3i64, 4i64, 5i64, 6i64, 7i64, 8i64, 9i64, 3i64, 4i64, 5i64,
--    6i64, 7i64, 8i64, 9i64, 4i64, 5i64, 6i64, 7i64, 8i64, 9i64,
--    5i64, 6i64, 7i64, 8i64, 9i64, 6i64, 7i64, 8i64, 9i64, 7i64,
--    8i64, 9i64, 8i64, 9i64, 9i64] }
entry test_triu (n: i64) =
  unzip (triu n)

-- ==
-- entry: test_segmented_replicate
-- input { [3i64] [0i64] } output { [0i64, 0i64, 0i64] }
-- input { [1i64, 2i64, 4i64] [0i64, 1i64, 2i64] } output {
--   [0i64, 1i64, 1i64, 2i64, 2i64, 2i64, 2i64] }
entry test_segmented_replicate [n] (ns: [n]i64) (vs: [n]i64) =
  segmented_replicate ns vs

-- ==
-- entry: test_exclusive_scan
-- input { [1] } output { [0] }
-- input { [3, 2] } output { [0, 3] }
-- input { [1, 2, 3, 4] } output { [0, 1, 3, 6] }
entry test_exclusive_scan [n] (vs: [n]i32) =
  exclusive_scan (+) 0 vs

-- ==
-- entry: test_bincount
-- input { 5i64 [0i64] } output { [1i64, 0i64, 0i64, 0i64, 0i64] }
-- input { 3i64 [0i64, 0i64, 1i64, 1i64, 2i64, 2i64] } output {
--   [2i64, 2i64, 2i64] }
-- input { 5i64 [0i64, 3i64, 4i64, 1i64, 3i64] } output {
--   [1i64, 1i64, 0i64, 2i64, 1i64] }
entry test_bincount [n] (k: i64) (vs: [n]i64) =
  bincount k vs

-- ==
-- entry: test_find_unique
-- input { 5i64 [1i64, 2i64, 3i64] } output { -1i64 }
-- input { 2i64 [1i64, 2i64, 3i64] } output { 1i64 }
entry test_find_unique [n] (v: i64) (vs: [n]i64) =
  find_unique v vs