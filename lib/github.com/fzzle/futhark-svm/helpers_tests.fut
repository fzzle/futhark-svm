import "helpers"

-- ==
-- entry: test_distribute
-- input { [true, false, false, false, true, false] [1, 2] }
-- output { [1, 1, 1, 1, 2, 2] }
entry test_distribute (fs: []bool) (vs: []i32): []i32 =
  i32_distribute fs vs
