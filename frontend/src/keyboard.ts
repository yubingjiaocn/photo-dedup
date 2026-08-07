/**
 * Keyboard map as a pure function: a key event plus the current mode/help state
 * becomes one intent. The app layer executes the intent (some are async — a
 * decision posts and reloads); this file only decides *which* intent fires, so
 * the whole key map — including the auto-repeat and modifier rules that were
 * bug sources in the hand-written page — is unit-testable without a DOM.
 */
import type { Mode } from './store';

export type Intent =
  | { kind: 'help'; open: boolean }
  | { kind: 'escape' }
  | { kind: 'evidence' }
  | { kind: 'density' }
  | { kind: 'fit' }
  | { kind: 'hundred' }
  | { kind: 'compare' }
  | { kind: 'blink'; on: boolean }
  | { kind: 'photo'; delta: number }
  | { kind: 'group'; delta: number }
  | { kind: 'decide'; action: 'accept' | 'pick' | 'mark' }
  | { kind: 'undo' }
  | { kind: 'none' };

export interface KeyContext {
  mode: Mode;
  helpOpen: boolean;
  browseOpen: boolean;
  /** keydown vs keyup — blink is press-and-hold. */
  phase: 'down' | 'up';
  repeat: boolean;
  shift: boolean;
}

const NONE: Intent = { kind: 'none' };

export function keyIntent(key: string, ctx: KeyContext): Intent {
  const k = String(key || '');

  if (ctx.phase === 'up') {
    // Only blink cares about key release.
    return k === 'c' || k === 'C' ? { kind: 'blink', on: false } : NONE;
  }

  // Help modal swallows everything but its own close keys.
  if (ctx.helpOpen) {
    if (k === 'Escape' || k === '?' || k === 'F1') return { kind: 'help', open: false };
    return NONE;
  }
  if (k === '?' || k === 'F1') return { kind: 'help', open: true };
  if (k === 'Escape') {
    return ctx.mode !== 'queue' && ctx.browseOpen ? { kind: 'escape' } : NONE;
  }

  if (k === 'e' || k === 'E') return { kind: 'evidence' };
  if (k === 'd' || k === 'D') return { kind: 'density' };
  if (k === 'f' || k === 'F') return { kind: 'fit' };
  if (k === '1') return { kind: 'hundred' };
  if (k === 'c' || k === 'C') {
    // Shift+C is the persistent split; a bare hold-C only blinks.
    if (ctx.shift) return { kind: 'compare' };
    return ctx.repeat ? NONE : { kind: 'blink', on: true };
  }
  if (k === 'ArrowLeft' || k === 'h' || k === 'H') return { kind: 'photo', delta: -1 };
  if (k === 'ArrowRight' || k === 'l' || k === 'L') return { kind: 'photo', delta: 1 };

  // Group navigation and decisions only exist in the queue.
  if (ctx.mode !== 'queue') return NONE;
  if (k === 'ArrowUp' || k === 'k' || k === 'K') return { kind: 'group', delta: -1 };
  if (k === 'ArrowDown' || k === 'j' || k === 'J') return { kind: 'group', delta: 1 };

  const lower = k.toLowerCase();
  const action = lower === 'a' ? 'accept' : lower === 'p' ? 'pick' : lower === 'm' ? 'mark' : null;
  if (action) {
    // A held decision key must not fire once per repeat event.
    return ctx.repeat ? NONE : { kind: 'decide', action };
  }
  if (lower === 'u') return ctx.repeat ? NONE : { kind: 'undo' };
  return NONE;
}

/** Keys the intent map understands, so listeners can preventDefault correctly. */
export function isHandledKey(key: string): boolean {
  const k = String(key || '');
  if (['?', 'F1', 'Escape', '1'].includes(k)) return true;
  if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(k)) return true;
  return /^[aApPmMuUeEdDfFcChHlLjJkK]$/.test(k);
}
