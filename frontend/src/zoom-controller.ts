/**
 * Imperative bridge between the Preact stage and the chosen zoom viewer.
 *
 * Preact owns the pane *elements* (stable ids, never re-created); this controller
 * owns the *adapters* mounted into them and the dual-pane sync/blink glue that
 * must live outside the library. Keeping it here — not inside a component — is
 * what lets the viewer be swapped (Panzoom in production, OSD for the comparison
 * harness) behind the one `ZoomAdapter` contract without touching the UI tree.
 */
import type { ZoomAdapter, ZoomState } from './viewer/zoom-adapter';
import { isZoomed, zoomStateText } from './viewer/zoom-adapter';

export type AdapterFactory = (host: HTMLElement, pane: 'A' | 'B') => ZoomAdapter;

export class ZoomController {
  private a: ZoomAdapter | null = null;
  private b: ZoomAdapter | null = null;
  private readonly make: AdapterFactory;
  private onZoomText: (text: string) => void;
  private compare = false;
  private echoing = false;

  constructor(make: AdapterFactory, onZoomText: (text: string) => void) {
    this.make = make;
    this.onZoomText = onZoomText;
  }

  /** (Re)mount adapters when the host elements appear. Idempotent per element. */
  attach(hostA: HTMLElement | null, hostB: HTMLElement | null): void {
    if (hostA && !this.a) {
      this.a = this.make(hostA, 'A');
      this.a.onChange((state) => this.mirror(state, 'A'));
    }
    if (hostB && !this.b) {
      this.b = this.make(hostB, 'B');
      this.b.onChange((state) => this.mirror(state, 'B'));
    }
  }

  private mirror(state: ZoomState, from: 'A' | 'B'): void {
    if (from === 'A') this.onZoomText(zoomStateText(state.scale));
    this.reflectZoomClass(state);
    if (!this.compare || this.echoing) return;
    const target = from === 'A' ? this.b : this.a;
    this.echoing = true;
    try {
      target?.applyState(state);
    } finally {
      this.echoing = false;
    }
  }

  private reflectZoomClass(state: ZoomState): void {
    document.querySelectorAll('.frame, .osd').forEach((el) => {
      el.classList.toggle('zoomed', isZoomed(state.scale));
    });
  }

  setCompare(on: boolean): void {
    this.compare = on;
  }

  setPaneImage(which: 'A' | 'B', url: string | null): void {
    (which === 'A' ? this.a : this.b)?.setImage(url);
  }

  reset(): void {
    this.a?.reset();
    this.b?.reset();
    this.onZoomText(zoomStateText(this.a?.state().scale ?? 1));
  }

  hundred(): void {
    const scale = this.a?.actualPixelScale() ?? 2;
    this.a?.zoomToScale(scale);
    if (this.compare) this.b?.zoomToScale(scale);
    this.onZoomText(zoomStateText(this.a?.state().scale ?? scale));
  }

  zoomText(): string {
    return zoomStateText(this.a?.state().scale ?? 1);
  }

  states(): { a: ZoomState | null; b: ZoomState | null } {
    return { a: this.a?.state() ?? null, b: this.b?.state() ?? null };
  }

  kind(): string | null {
    return this.a?.kind ?? null;
  }

  destroy(): void {
    this.a?.destroy();
    this.b?.destroy();
    this.a = null;
    this.b = null;
  }
}
