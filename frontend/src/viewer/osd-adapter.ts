/**
 * OpenSeadragon co-finalist, driving a plain (non-pyramidal) local original.
 *
 * This exists for the viewer-decision harness, not for production: the app only
 * constructs it when the page is opened with `?viewer=osd`, and it is
 * lazy-imported so OSD never enters the production bundle graph.
 *
 * OSD's native unit is *viewport zoom* (1 = image width fills the container),
 * which is not the stage's "fit". The spike left OSD's raw open state
 * un-normalised; the migration brief requires the shared `fit=1, pan=0` contract
 * before the decision, so this adapter now:
 *
 *   - calls `goHome` on open (fit) and reports `scale = zoom / homeZoom`, so a
 *     freshly opened image reads exactly `scale = 1`;
 *   - reports pan as an offset from the home centre in *screen px* (0,0 at fit),
 *     matching Panzoom's contract instead of OSD's viewport-coordinate centre.
 *
 * With that normalisation OSD passes the identical centred-fit assertions. It is
 * still the heavier choice — canvas world lifecycle, zoom-unit translation, no
 * `<img>` reuse for the HDD pool — which is why production ships Panzoom.
 */
import OpenSeadragon from 'openseadragon';
import { MAX_SCALE, MIN_SCALE, clampScale, type ZoomAdapter, type ZoomState } from './zoom-adapter';

export class OsdZoom implements ZoomAdapter {
  readonly kind = 'osd' as const;

  private readonly viewer: OpenSeadragon.Viewer;
  private listeners: Array<(state: ZoomState) => void> = [];
  private echo = false;
  private currentUrl: string | null = null;

  constructor(container: HTMLElement) {
    this.viewer = OpenSeadragon({
      element: container,
      prefixUrl: '',
      showNavigationControl: false,
      showNavigator: false,
      showSequenceControl: false,
      maxZoomPixelRatio: MAX_SCALE,
      defaultZoomLevel: 0,
      visibilityRatio: 1,
      constrainDuringPan: true,
      homeFillsViewer: false,
      animationTime: 0,
      springStiffness: 100,
      immediateRender: true,
      preserveViewport: false,
      preserveImageSizeOnResize: false,
      imageLoaderLimit: 1,
      gestureSettingsMouse: {
        scrollToZoom: false,
        clickToZoom: false,
        dblClickToZoom: true,
        pinchToZoom: false,
        flickEnabled: false,
        dragToPan: true,
      },
    });

    this.viewer.addHandler('zoom', this.onViewportChange);
    this.viewer.addHandler('pan', this.onViewportChange);
    this.viewer.addHandler('open', this.onOpen);

    const canvas = this.viewer.element;
    canvas.addEventListener('wheel', this.onWheel, { passive: false, capture: true });
    canvas.addEventListener('dblclick', this.onDblClick);
  }

  private onOpen = (): void => {
    this.viewer.viewport.goHome(true);
    this.emit();
  };

  /** The fitted (home) centre in viewport coords, derived from the home bounds
   * rather than a captured snapshot, so normalisation does not depend on when
   * `open` fired relative to layout. */
  private homeCentre(): OpenSeadragon.Point {
    return this.viewer.viewport.getHomeBounds().getCenter();
  }

  private onViewportChange = (): void => {
    if (this.echo) return;
    this.emit();
  };

  private onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    if (!this.hasImage()) return;
    const step = Math.exp(-(event.deltaY || 0) * 0.0015);
    const factor = Math.max(1 / 3, Math.min(3, step));
    this.zoomBy(factor, event.clientX, event.clientY);
  };

  private onDblClick = (event: MouseEvent): void => {
    event.preventDefault();
    this.reset();
  };

  private hasImage(): boolean {
    return this.viewer.world.getItemCount() > 0;
  }

  private homeZoom(): number {
    return this.hasImage() ? this.viewer.viewport.getHomeZoom() : 1;
  }

  private emit(): void {
    const state = this.state();
    for (const listener of this.listeners) listener(state);
  }

  setImage(url: string | null): void {
    if (url === this.currentUrl) return;
    this.currentUrl = url;
    this.viewer.close();
    if (url == null) return;
    this.viewer.addSimpleImage({ url });
  }

  zoomBy(factor: number, fx?: number, fy?: number): void {
    if (!this.hasImage()) return;
    const viewport = this.viewer.viewport;
    const home = this.homeZoom();
    const nextScale = clampScale((viewport.getZoom(true) / home) * factor);
    const target = nextScale * home;
    if (fx == null || fy == null) {
      viewport.zoomTo(target, undefined, true);
    } else {
      const rect = this.viewer.element.getBoundingClientRect();
      const point = viewport.pointFromPixel(new OpenSeadragon.Point(fx - rect.left, fy - rect.top), true);
      viewport.zoomTo(target, point, true);
    }
    viewport.applyConstraints(true);
  }

  zoomToScale(scale: number): void {
    if (!this.hasImage()) return;
    this.viewer.viewport.zoomTo(clampScale(scale) * this.homeZoom(), undefined, true);
    this.viewer.viewport.applyConstraints(true);
  }

  reset(): void {
    if (!this.hasImage()) return;
    this.viewer.viewport.goHome(true);
    this.viewer.viewport.applyConstraints(true);
  }

  actualPixelScale(): number {
    if (!this.hasImage()) return 2;
    const viewport = this.viewer.viewport;
    const imageZoom = viewport.viewportToImageZoom(viewport.getZoom(true));
    if (!imageZoom) return 2;
    const scaleNow = viewport.getZoom(true) / this.homeZoom();
    return clampScale(scaleNow / imageZoom);
  }

  state(): ZoomState {
    if (!this.hasImage()) return { scale: MIN_SCALE, x: 0, y: 0 };
    const viewport = this.viewer.viewport;
    const scale = viewport.getZoom(true) / this.homeZoom();
    // Report pan as a screen-px delta from the fitted centre, so fit is (0,0)
    // and both viewers speak the same contract.
    const centre = viewport.getCenter(true);
    const homePx = viewport.pixelFromPoint(this.homeCentre(), true);
    const nowPx = viewport.pixelFromPoint(centre, true);
    return { scale, x: homePx.x - nowPx.x, y: homePx.y - nowPx.y };
  }

  applyState(state: ZoomState): void {
    if (!this.hasImage()) return;
    const current = this.state();
    if (
      Math.abs(current.scale - state.scale) < 1e-4 &&
      Math.abs(current.x - state.x) < 0.5 &&
      Math.abs(current.y - state.y) < 0.5
    ) {
      return;
    }
    this.echo = true;
    try {
      const viewport = this.viewer.viewport;
      viewport.zoomTo(state.scale * this.homeZoom(), undefined, true);
      // Convert the screen-px offset back to a viewport centre.
      const homePx = viewport.pixelFromPoint(this.homeCentre(), true);
      const target = viewport.pointFromPixel(
        new OpenSeadragon.Point(homePx.x - state.x, homePx.y - state.y),
        true,
      );
      viewport.panTo(target, true);
      viewport.applyConstraints(true);
    } finally {
      this.echo = false;
    }
  }

  onChange(listener: (state: ZoomState) => void): void {
    this.listeners.push(listener);
  }

  destroy(): void {
    const canvas = this.viewer.element;
    canvas.removeEventListener('wheel', this.onWheel, true);
    canvas.removeEventListener('dblclick', this.onDblClick);
    this.viewer.destroy();
    this.listeners = [];
  }
}
