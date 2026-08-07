/**
 * The image pool is the HDD policy in code: one live decoded <img> per URL,
 * fetched once, released when its group leaves. Panzoom itself is not exercised
 * here (no layout in happy-dom); we drive the exported pool functions directly,
 * which is exactly what the app's retain step calls.
 */
import { beforeEach, describe, expect, it } from 'vitest';
import { retainImages, retainedImageCount } from '../src/viewer/panzoom-adapter';

// The pool's acquire() is internal to PanzoomZoom.setImage; to test retention
// deterministically we simulate the app loop: it computes the retained URL set
// per group and calls retainImages(). A URL only becomes "held" once a pane has
// displayed it, so we assert the pruning contract: retainImages drops anything
// outside the allowed set. We seed the pool by importing the module-level store
// through a tiny displayed-image stand-in.
import { PanzoomZoom } from '../src/viewer/panzoom-adapter';

function displayInFreshPane(url: string): PanzoomZoom {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const z = new PanzoomZoom(host);
  z.setImage(url); // acquires one <img> for this URL
  return z;
}

describe('image pool retention', () => {
  beforeEach(() => {
    // Start each test from an empty pool.
    retainImages([]);
  });

  it('holds one decoded original per displayed URL', () => {
    displayInFreshPane('/api/original/1');
    displayInFreshPane('/api/original/2');
    expect(retainedImageCount()).toBe(2);
  });

  it('displaying the same URL again does not add a second decode', () => {
    displayInFreshPane('/api/original/1');
    const a = document.createElement('div');
    document.body.appendChild(a);
    const z = new PanzoomZoom(a);
    z.setImage('/api/original/1'); // same URL -> reuse
    expect(retainedImageCount()).toBe(1);
  });

  it('retainImages releases originals outside the allowed group set', () => {
    displayInFreshPane('/api/original/1');
    displayInFreshPane('/api/original/2');
    displayInFreshPane('/api/original/3');
    expect(retainedImageCount()).toBe(3);
    // Move to a group that only contains 4/5/6: everything else is released.
    retainImages(['/api/original/4', '/api/original/5', '/api/original/6']);
    expect(retainedImageCount()).toBe(0);
  });

  it('keeping the current group in the allowed set retains its members', () => {
    displayInFreshPane('/api/original/1');
    displayInFreshPane('/api/original/2');
    retainImages(['/api/original/1', '/api/original/2', '/api/original/3']);
    // 1 and 2 were displayed; still held. 3 was never fetched; not created.
    expect(retainedImageCount()).toBe(2);
  });
});
