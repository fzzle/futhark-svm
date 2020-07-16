import "../lib/github.com/fzzle/futhark-svm/util"

-- ==
-- entry: test_triu
-- input { 3 } output { [0, 0, 1] [1, 2, 2] }
-- input { 5 } output {
--   [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
--   [1, 2, 3, 4, 2, 3, 4, 3, 4, 4] }
-- input { 10 } output {
--   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
--    1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
--    2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
--    4, 4, 4, 4, 4, 5, 5, 5, 5, 6,
--    6, 6, 7, 7, 8]
--   [1, 2, 3, 4, 5, 6, 7, 8, 9, 2,
--    3, 4, 5, 6, 7, 8, 9, 3, 4, 5,
--    6, 7, 8, 9, 4, 5, 6, 7, 8, 9,
--    5, 6, 7, 8, 9, 6, 7, 8, 9, 7,
--    8, 9, 8, 9, 9] }
entry test_triu (n: i32) =
  unzip (triu n)

-- TODO: Test triu up against loop.

-- ==
-- entry: test_segmented_replicate
-- input { [3] [0] } output { [0, 0, 0] }
-- input { [1, 2, 4] [0, 1, 2] } output { [0, 1, 1, 2, 2, 2, 2] }
entry test_segmented_replicate [n] (ns: [n]i32) (vs: [n]i32) =
  segmented_replicate ns vs