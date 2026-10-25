import { expect, it } from 'vitest';
import { render } from 'preact';
import { ReviewText, ReviewObservation } from '../src/components/chrome';

it('collapses long observations without changing or dropping their full text', () => {
  const host = document.createElement('div');
  const text = 'Observable background detail. '.repeat(12);
  render(<ReviewText text={text} />, host);
  const details = host.querySelector('details');
  expect(details).not.toBeNull();
  expect(details?.open).toBe(false);
  expect(host.querySelector('p')?.textContent).toBe(text);
  expect(host.querySelector('summary')?.textContent).toContain('展开完整观察');
  details!.open = true;
  expect(host.querySelector('p')?.textContent).toBe(text);
});

it('counts Unicode code points without splitting surrogate pairs', () => {
  const host = document.createElement('div');
  render(<ReviewText text={'🔎'.repeat(240)} />, host);
  expect(host.querySelector('details')).toBeNull();
  render(<ReviewText text={'🔎'.repeat(241)} />, host);
  expect(host.querySelector('summary')?.textContent).toBe('🔎'.repeat(180) + '…（展开完整观察）');
  expect(host.querySelector('p')?.textContent).toBe('🔎'.repeat(241));
});

it('marks low-confidence observations without dropping mixed-confidence siblings', () => {
  const host = document.createElement('div');
  render(<div><ReviewObservation text="Strong decision support" confidence="high" /><ReviewObservation text="Weak background observation" confidence="low" /></div>, host);
  expect(host.textContent).toContain('Strong decision support');
  expect(host.textContent).toContain('弱观察（不支持推荐）：Weak background observation');
  expect(host.querySelectorAll('[data-confidence]')).toHaveLength(2);
});
