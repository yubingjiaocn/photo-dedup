#!/usr/bin/env node
'use strict';

// Read-only Google Photos smoke sampler. It only navigates date searches,
// scrolls the result grid, records unique media ids, and saves screenshots.
const { chromium } = require('/home/ubuntu/.nvm/versions/node/v24.18.0/lib/node_modules/openclaw/node_modules/playwright-core');
const fs = require('fs/promises');
const path = require('path');

const CDP = 'http://127.0.0.1:18800';
const OUT = '/home/ubuntu/photo-dedup-eval/multiscene-smoke';
const DATES = ['2026-07-31', '2026-08-02', '2026-04-04', '2026-06-14', '2025-11-23'];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function mediaId(href) {
  const match = href.match(/\/photo\/(AF1Qip[A-Za-z0-9_-]+)/);
  return match ? match[1] : null;
}

async function sampleDate(page, date) {
  await page.goto(`https://photos.google.com/search/${date}`, {waitUntil: 'domcontentloaded', timeout: 120000});
  await sleep(2500);
  const ids = new Set();
  const samples = [];
  let lastTop = -1;
  let stable = 0;
  for (let round = 0; round < 500; round += 1) {
    const state = await page.evaluate(() => {
      const scrollers = [...document.querySelectorAll('*')]
        .filter(e => e.scrollHeight > e.clientHeight + 500 && e.clientHeight > 300)
        .sort((a, b) => (b.scrollHeight - b.clientHeight) - (a.scrollHeight - a.clientHeight));
      const scroller = scrollers[0];
      const links = [...document.querySelectorAll('a[href*="/photo/"]')].map(a => a.href);
      if (!scroller) return {links, top: 0, max: 0, next: 0};
      const top = scroller.scrollTop;
      const max = scroller.scrollHeight - scroller.clientHeight;
      const next = Math.min(max, top + Math.max(650, Math.floor(scroller.clientHeight * 0.9)));
      scroller.scrollTop = next;
      return {links, top, max, next};
    });
    for (const href of state.links) {
      const id = mediaId(href);
      if (id) ids.add(id);
    }
    const fraction = state.max ? state.top / state.max : 0;
    if (samples.length === 0 || (samples.length === 1 && fraction >= 0.45) || (samples.length === 2 && fraction >= 0.9)) {
      const file = path.join(OUT, `${date}-${['start','middle','end'][samples.length]}.jpg`);
      await page.screenshot({path: file, type: 'jpeg', quality: 82});
      samples.push({position: fraction, file});
    }
    stable = (state.next === state.top || state.top === lastTop) ? stable + 1 : 0;
    lastTop = state.top;
    if (stable >= 4) break;
    await sleep(150);
  }
  await sleep(800);
  const finalLinks = await page.evaluate(() => [...document.querySelectorAll('a[href*="/photo/"]')].map(a => a.href));
  for (const href of finalLinks) {
    const id = mediaId(href);
    if (id) ids.add(id);
  }
  return {date, unique_media_items: ids.size, samples};
}

(async () => {
  await fs.mkdir(OUT, {recursive: true});
  const browser = await chromium.connectOverCDP(CDP);
  const context = browser.contexts()[0];
  const page = context.pages().find(p => p.url().includes('photos.google.com'));
  if (!page) throw new Error('No authorized Google Photos page found');
  const results = [];
  for (const date of DATES) {
    const result = await sampleDate(page, date);
    results.push(result);
    console.log(`${date}: ${result.unique_media_items}`);
  }
  await fs.writeFile(path.join(OUT, 'inventory.json'), JSON.stringify({generated_at: new Date().toISOString(), mode: 'read-only browser sampling', results}, null, 2));
  await browser.close();
})().catch(error => { console.error(error); process.exitCode = 1; });
