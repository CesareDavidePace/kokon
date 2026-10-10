// The Kokon mark (assets/brand/mark/kokon-mark.svg), drawable in any colour and in parts.
export const MARK_PATHS = {
  stem: "M0.00 242.81 C0.00 112.17 70.23 25.08 175.59 25.08 C280.94 25.08 351.17 112.17 351.17 242.81 L351.17 781.19 C351.17 911.83 280.94 998.92 175.59 998.92 C70.23 998.92 0.00 911.83 0.00 781.19 Z",
  upper: "M389.53 548.95 C389.53 250.35 662.35 25.08 1024.00 25.08 C1024.00 323.68 751.18 548.95 389.53 548.95 Z",
  lower: "M389.53 587.31 C389.53 821.93 618.71 998.92 922.49 998.92 C922.49 764.30 693.31 587.31 389.53 587.31 Z",
};

export function Mark({ size = 40, color = "currentColor" }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 1024 1024" aria-hidden>
      <path fill={color} d={`${MARK_PATHS.stem} ${MARK_PATHS.upper} ${MARK_PATHS.lower}`} />
    </svg>
  );
}
