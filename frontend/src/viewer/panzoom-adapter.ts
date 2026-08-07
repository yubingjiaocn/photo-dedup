/**
 * Production zoom viewer: `@panzoom/panzoom` driving a plain `<img>`.
 *
 * Why this and not OpenSeadragon (see FRONTEND_DECISION.md): it keeps the stage's
 * existing `<img>` + CSS-transform rendering model, and one live decoded `<img>`
 * per active-group URL is exactly the HDD policy this project needs — an original
 * is served `no-store`, so it must be fetched *once* while it belongs to the
 * group on screen, moved (not re-`src`-ed) between panes, and dropped when the
 * group changes. That pool is the only "extra" glue Panzoom needs, and it is
 * application HDD policy we would have to own under any viewer.
 */
import Panzoom from '@panzoom/panzoom';
import type { PanzoomObject } from '@panzoom/panzoom';
import { MAX_SCALE, MIN_SCALE, clampScale, type ZoomAdapter, type ZoomState } from './zoom-adapter';

interface StoredImage {
  img: HTMLImageElement;
  url: string;
}

/**
 * A URL owns one live `<img>` across both panes. Moving that decoded element
 * between the hidden store / pane A / pane B never assigns `src` again, so a
 * `no-store` original is fetched once while it belongs to the active group.
 */
class ImageStore {
  private readonly parked: HTMLElement;
  private readonly entries = new Map<string, StoredImage>();

  constructor() {
    this.parked = document.createElement('div');
    this.parked.hidden = true;
    this.parked.id = 'panzoomImageStore';
    document.body.appendChild(this.parked);
  }

  acquire(url: string): HTMLImageElement {
    let entry = this.entries.get(url);
    if (!entry) {
      const img = new Image();
      img.alt = '';
      img.draggable = false;
      img.src = url; // the one and only network read for this URL
      entry = { img, url };
      this.entries.set(url, entry);
    }
    return entry.img;
  }

  park(img: HTMLImageElement): void {
    this.parked.appendChild(img);
  }

  /** Release every decoded original whose URL is not in `allowed`. */
  prune(allowed: Set<string>): void {
    for (const [url, entry] of this.entries) {
      if (allowed.has(url)) continue;
      entry.img.removeAttribute('src');
      entry.img.remove();
      this.entries.delete(url);
    }
  }

  /** Test/lifecycle helper: how many originals are currently held decoded. */
  size(): number {
    return this.entries.size;
  }
}

let store: ImageStore | null = null;
const imageStore = (): ImageStore => (store ??= new ImageStore());

/** Release decoded originals when the app leaves a group (drives HDD policy). */
export function retainImages(urls: string[]): void {
  imageStore().prune(new Set(urls));
}

/** Diagnostic only: distinct decoded originals held right now. */
export function retainedImageCount(): number {
  return store ? store.size() : 0;
}

export class PanzoomZoom implements ZoomAdapter {
  readonly kind = 'panzoom' as const;
  private readonly frame: HTMLElement;
  private img: HTMLImageElement | null = null;
  private pz: PanzoomObject | null = null;
  private listeners: Array<(state: ZoomState) => void> = [];
  private echo = false;
  private natural = { w: 0, h: 0 };
  private fitted = { w: 0, h: 0 };
  /** Panzoom reports its flex-centred contain position as pan; normalise it so
   * the contract still exposes fit as (0,0). */
  private fitPan = { x: 0, y: 0 };

  constructor(frame: HTMLElement) {
    this.frame = frame;
    frame.querySelector('img')?.remove();
    this.frame.addEventListener('wheel', this.onWheel, { passive: false });
    this.frame.addEventListener('dblclick', this.onDblClick);
    window.addEventListener('resize', this.onResize);
  }

  private mount(img: HTMLImageElement): void {
    this.destroyPanzoom();
    this.img = img;
    this.frame.appendChild(img);
    const ready = (): void => {
      if (this.img !== img) return;
      this.natural = { w: img.naturalWidth, h: img.naturalHeight };
      this.layout();
      this.createPanzoom();
      this.emit();
    };
    if (img.complete && img.naturalWidth) ready();
    else img.addEventListener('load', ready, { once: true });
  }

  private createPanzoom(): void {
    if (!this.img) return;
    this.pz = Panzoom(this.img, {
      minScale: MIN_SCALE,
      maxScale: MAX_SCALE,
      startScale: 1,
      animate: false,
      duration: 0,
      cursor: 'grab',
      panOnlyWhenZoomed: true,
      touchAction: 'none',
      canvas: true,
    });
    this.img.addEventListener('panzoomchange', this.onPanzoomChange as EventListener);
    this.pz.reset({ animate: false });
    this.fitPan = this.pz.getPan();
  }

  private destroyPanzoom(): void {
    if (this.img) this.img.removeEventListener('panzoomchange', this.onPanzoomChange as EventListener);
    this.pz?.destroy();
    this.pz = null;
  }

  private layout(): void {
    if (!this.img) return;
    const box = this.frame.getBoundingClientRect();
    if (!this.natural.w || !this.natural.h || !box.width || !box.height) return;
    const ratio = Math.min(box.width / this.natural.w, box.height / this.natural.h);
    this.fitted = { w: Math.round(this.natural.w * ratio), h: Math.round(this.natural.h * ratio) };
    this.img.style.width = `${this.fitted.w}px`;
    this.img.style.height = `${this.fitted.h}px`;
  }

  private onResize = (): void => {
    if (!this.img) return;
    this.layout();
    this.destroyPanzoom();
    this.createPanzoom();
    this.emit();
  };
  private onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    this.pz?.zoomWithWheel(event);
  };
  private onDblClick = (event: MouseEvent): void => {
    event.preventDefault();
    this.reset();
  };
  private onPanzoomChange = (): void => {
    if (!this.echo) this.emit();
  };
  private emit(): void {
    const value = this.state();
    for (const listener of this.listeners) listener(value);
  }

  setImage(url: string | null): void {
    if (url == null) {
      this.destroyPanzoom();
      if (this.img) imageStore().park(this.img);
      this.img = null;
      return;
    }
    if (this.img?.getAttribute('src') === url) return;
    this.destroyPanzoom();
    if (this.img) imageStore().park(this.img);
    this.mount(imageStore().acquire(url));
  }

  zoomBy(factor: number, fx?: number, fy?: number): void {
    if (!this.pz) return;
    const next = clampScale(this.pz.getScale() * factor);
    if (fx == null || fy == null) this.pz.zoom(next, { animate: false });
    else this.pz.zoomToPoint(next, { clientX: fx, clientY: fy }, { animate: false });
  }
  zoomToScale(scale: number): void {
    this.pz?.zoom(clampScale(scale), { animate: false });
  }
  reset(): void {
    if (!this.pz) return;
    this.pz.zoom(1, { animate: false, force: true });
    this.pz.pan(this.fitPan.x, this.fitPan.y, { animate: false, force: true });
  }
  actualPixelScale(): number {
    if (!this.natural.w || !this.fitted.w) return 2;
    return clampScale(this.natural.w / this.fitted.w);
  }
  state(): ZoomState {
    const pan = this.pz?.getPan() ?? { x: 0, y: 0 };
    return { scale: this.pz?.getScale() ?? 1, x: pan.x - this.fitPan.x, y: pan.y - this.fitPan.y };
  }
  applyState(value: ZoomState): void {
    if (!this.pz) return;
    const current = this.state();
    if (
      Math.abs(current.scale - value.scale) < 1e-4 &&
      Math.abs(current.x - value.x) < 0.5 &&
      Math.abs(current.y - value.y) < 0.5
    ) {
      return;
    }
    this.echo = true;
    try {
      this.pz.zoom(value.scale, { animate: false, force: true });
      this.pz.pan(value.x + this.fitPan.x, value.y + this.fitPan.y, { animate: false, force: true });
    } finally {
      this.echo = false;
    }
  }
  onChange(listener: (value: ZoomState) => void): void {
    this.listeners.push(listener);
  }
  destroy(): void {
    this.frame.removeEventListener('wheel', this.onWheel);
    this.frame.removeEventListener('dblclick', this.onDblClick);
    window.removeEventListener('resize', this.onResize);
    this.destroyPanzoom();
    if (this.img) imageStore().park(this.img);
    this.img = null;
    this.listeners = [];
  }
}
