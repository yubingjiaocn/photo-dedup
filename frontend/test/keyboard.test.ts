import { describe, expect, it } from 'vitest';
import { isHandledKey, keyIntent, type KeyContext } from '../src/keyboard';

const ctx = (over: Partial<KeyContext> = {}): KeyContext => ({
  mode: 'queue',
  helpOpen: false,
  browseOpen: false,
  phase: 'down',
  repeat: false,
  shift: false,
  ...over,
});

describe('decisions', () => {
  it('maps A/P/M/U to decide/undo', () => {
    expect(keyIntent('a', ctx())).toEqual({ kind: 'decide', action: 'accept' });
    expect(keyIntent('P', ctx())).toEqual({ kind: 'decide', action: 'pick' });
    expect(keyIntent('m', ctx())).toEqual({ kind: 'decide', action: 'mark' });
    expect(keyIntent('u', ctx())).toEqual({ kind: 'undo' });
  });

  it('a held (auto-repeat) decision key fires nothing', () => {
    expect(keyIntent('a', ctx({ repeat: true }))).toEqual({ kind: 'none' });
    expect(keyIntent('u', ctx({ repeat: true }))).toEqual({ kind: 'none' });
  });

  it('decisions only exist in the queue', () => {
    expect(keyIntent('a', ctx({ mode: 'browse' }))).toEqual({ kind: 'none' });
    expect(keyIntent('m', ctx({ mode: 'browse' }))).toEqual({ kind: 'none' });
  });
});

describe('blink vs compare (the C key)', () => {
  it('bare C blinks on keydown and off on keyup', () => {
    expect(keyIntent('c', ctx())).toEqual({ kind: 'blink', on: true });
    expect(keyIntent('c', ctx({ phase: 'up' }))).toEqual({ kind: 'blink', on: false });
    expect(keyIntent('C', ctx({ phase: 'up' }))).toEqual({ kind: 'blink', on: false });
  });

  it('held C does not re-trigger blink-on', () => {
    expect(keyIntent('c', ctx({ repeat: true }))).toEqual({ kind: 'none' });
  });

  it('Shift+C toggles the persistent split, not blink', () => {
    expect(keyIntent('C', ctx({ shift: true }))).toEqual({ kind: 'compare' });
  });

  it('C works in browse mode too (view-only key)', () => {
    expect(keyIntent('c', ctx({ mode: 'browse' }))).toEqual({ kind: 'blink', on: true });
  });
});

describe('navigation and view keys', () => {
  it('H/L/arrows step photos in both modes', () => {
    expect(keyIntent('l', ctx())).toEqual({ kind: 'photo', delta: 1 });
    expect(keyIntent('ArrowLeft', ctx())).toEqual({ kind: 'photo', delta: -1 });
    expect(keyIntent('h', ctx({ mode: 'browse' }))).toEqual({ kind: 'photo', delta: -1 });
  });

  it('J/K/arrows step groups only in the queue', () => {
    expect(keyIntent('j', ctx())).toEqual({ kind: 'group', delta: 1 });
    expect(keyIntent('ArrowUp', ctx())).toEqual({ kind: 'group', delta: -1 });
    expect(keyIntent('j', ctx({ mode: 'browse' }))).toEqual({ kind: 'none' });
  });

  it('F/1/E/D map to view + chrome intents', () => {
    expect(keyIntent('f', ctx())).toEqual({ kind: 'fit' });
    expect(keyIntent('1', ctx())).toEqual({ kind: 'hundred' });
    expect(keyIntent('e', ctx())).toEqual({ kind: 'evidence' });
    expect(keyIntent('d', ctx())).toEqual({ kind: 'density' });
  });
});

describe('help + escape', () => {
  it('? and F1 open help, and help swallows other keys', () => {
    expect(keyIntent('?', ctx())).toEqual({ kind: 'help', open: true });
    expect(keyIntent('a', ctx({ helpOpen: true }))).toEqual({ kind: 'none' });
    expect(keyIntent('Escape', ctx({ helpOpen: true }))).toEqual({ kind: 'help', open: false });
  });

  it('Escape closes the browse viewer only when one is open', () => {
    expect(keyIntent('Escape', ctx({ mode: 'browse', browseOpen: true }))).toEqual({ kind: 'escape' });
    expect(keyIntent('Escape', ctx({ mode: 'queue' }))).toEqual({ kind: 'none' });
  });
});

describe('isHandledKey', () => {
  it('recognises every mapped key and rejects others', () => {
    for (const k of ['a', 'P', 'm', 'U', 'c', 'C', 'h', 'l', 'j', 'k', 'e', 'd', 'f', '1', '?', 'F1', 'Escape', 'ArrowUp']) {
      expect(isHandledKey(k)).toBe(true);
    }
    for (const k of ['z', 'Enter', '5', 'Tab']) {
      expect(isHandledKey(k)).toBe(false);
    }
  });
});
