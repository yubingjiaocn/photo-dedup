/**
 * The one contract both zoom viewers implement, written in the *production*
 * stage's vocabulary rather than either library's:
 *
 *   - `scale = 1` is contain-fit (the whole image visible, centred).
 *   - `x = 0, y = 0` is the fitted, centred position; pan is an offset from it.
 *   - `100%` is one image pixel per CSS pixel, clamped to bounds.
 *
 * A viewer that cannot answer these cheaply is a finding, not something to paper
 * over inside the adapter. OpenSeadragon's raw open state is *not* normalised to
 * this contract (its native unit is viewport zoom), so its adapter converts both
 * ways and calls `goHome` on open; Panzoom keeps the `<img>` model and only has
 * to subtract its flex-centred contain position. See FRONTEND_DECISION.md.
 */
export interface ZoomState {
  scale: number;
  x: number;
  y: number;
}

export interface ZoomAdapter {
  readonly kind: 'panzoom' | 'osd';
  /** Swap the displayed original. Must not read any other file. `null` clears. */
  setImage(url: string | null): void;
  /** Multiply scale, keeping the pixel under (fx,fy) client px put (centre if omitted). */
  zoomBy(factor: number, fx?: number, fy?: number): void;
  zoomToScale(scale: number): void;
  /** Back to contain-fit, centred. */
  reset(): void;
  /** Scale at which one image pixel covers one CSS pixel, clamped. */
  actualPixelScale(): number;
  state(): ZoomState;
  /** Mirror another adapter's state onto this one (dual-pane compare). */
  applyState(state: ZoomState): void;
  onChange(listener: (state: ZoomState) => void): void;
  destroy(): void;
}

export const MIN_SCALE = 1;
export const MAX_SCALE = 8;

export const clampScale = (scale: number): number =>
  Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale));

/** Same wording the production zoombar shows. */
export function zoomStateText(scale: number): string {
  return Number(scale) <= MIN_SCALE + 0.001 ? '适应窗口' : `${Math.round(Number(scale) * 100)}%`;
}

/** Same curve as the hand-written stage: bounded exponential per wheel event. */
export function wheelFactor(deltaY: number): number {
  const step = Math.exp(-(Number(deltaY) || 0) * 0.0015);
  return Math.max(1 / 3, Math.min(3, step));
}

export const isZoomed = (scale: number): boolean => scale > MIN_SCALE + 0.001;
