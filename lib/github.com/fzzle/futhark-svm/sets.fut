module sets (R: float) = {
  type t = R.t

  -- Checks if sample is in the upper set.
  let is_upper (Cp: t) (y: t) (a: t): bool =
    R.((y > i32 0 && a < Cp) || (y < i32 0 && a > i32 0))

  -- Checks if samples is in the lower set.
  let is_lower (Cn: t) (y: t) (a: t): bool =
    R.((y > i32 0 && a > i32 0) || (y < i32 0 && a < Cn))

  -- Checks if sample is free (both in upper and lower).
  let is_free (Cp: t) (Cn: t) (y: t) (a: t): bool =
    R.(a > i32 0 && ((y > i32 0 && a < Cp) || (y < i32 0 && a < Cn)))
}
