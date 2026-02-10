/* ==============================================
   Inpainting Web Worker
   Built-in client-side inpainting algorithm:
   Stage 1: Fast boundary diffusion (structure)
   Stage 2: Patch-based texture transfer (detail)
   ============================================== */

self.onmessage = function (e) {
  const { type } = e.data;
  if (type === 'inpaint') {
    try {
      const { imageData, maskData, width, height } = e.data;
      const result = inpaint(imageData, maskData, width, height);
      self.postMessage({ type: 'result', imageData: result }, [result.buffer]);
    } catch (err) {
      self.postMessage({ type: 'error', message: err.message });
    }
  }
};

function inpaint(imgPixels, maskPixels, w, h) {
  // imgPixels: Uint8ClampedArray RGBA
  // maskPixels: Uint8ClampedArray RGBA (alpha > 128 = masked)
  const out = new Uint8ClampedArray(imgPixels);
  const mask = new Uint8Array(w * h); // 0=known, 1=to-fill

  for (let i = 0; i < w * h; i++) {
    mask[i] = maskPixels[i * 4 + 3] > 128 ? 1 : 0;
  }

  // Count masked pixels
  let totalMasked = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) totalMasked++;
  if (totalMasked === 0) {
    self.postMessage({ type: 'progress', percent: 100 });
    return out;
  }

  // Stage 1: Iterative boundary diffusion
  // Fills from edges inward using weighted neighbor averaging
  const filled = new Uint8Array(mask);
  let filledCount = 0;
  let pass = 0;
  const maxPasses = Math.max(w, h);

  while (filledCount < totalMasked && pass < maxPasses) {
    let changed = false;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        if (!filled[idx]) continue;

        let r = 0, g = 0, b = 0, weight = 0;
        // Check 8-connected neighbors
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            const ni = ny * w + nx;
            if (filled[ni] === 0) {
              const d = dx === 0 || dy === 0 ? 1.0 : 0.707;
              r += out[ni * 4] * d;
              g += out[ni * 4 + 1] * d;
              b += out[ni * 4 + 2] * d;
              weight += d;
            }
          }
        }

        if (weight > 0) {
          out[idx * 4] = r / weight;
          out[idx * 4 + 1] = g / weight;
          out[idx * 4 + 2] = b / weight;
          out[idx * 4 + 3] = 255;
          filled[idx] = 0;
          filledCount++;
          changed = true;
        }
      }
    }
    pass++;
    if (!changed) break;
    if (pass % 10 === 0) {
      const pct = Math.round((filledCount / totalMasked) * 50);
      self.postMessage({ type: 'progress', percent: pct });
    }
  }

  self.postMessage({ type: 'progress', percent: 50 });

  // Stage 2: Patch-based texture refinement
  // For each masked pixel, find best matching patch from known region
  const patchSize = 5;
  const half = Math.floor(patchSize / 2);
  const iterations = 3;

  // Build list of known (non-masked) positions for sampling
  const knownPositions = [];
  for (let y = half; y < h - half; y += 2) {
    for (let x = half; x < w - half; x += 2) {
      if (!mask[y * w + x]) {
        knownPositions.push(y * w + x);
      }
    }
  }

  if (knownPositions.length === 0) {
    self.postMessage({ type: 'progress', percent: 100 });
    return out;
  }

  // Build list of masked positions
  const maskedPositions = [];
  for (let i = 0; i < w * h; i++) {
    if (mask[i]) maskedPositions.push(i);
  }

  for (let iter = 0; iter < iterations; iter++) {
    // Shuffle masked positions for varied fill order
    for (let i = maskedPositions.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = maskedPositions[i];
      maskedPositions[i] = maskedPositions[j];
      maskedPositions[j] = tmp;
    }

    for (let mi = 0; mi < maskedPositions.length; mi++) {
      const idx = maskedPositions[mi];
      const x = idx % w;
      const y = (idx - x) / w;
      if (x < half || x >= w - half || y < half || y >= h - half) continue;

      // Find best matching patch from known region (sample a subset for speed)
      let bestDist = Infinity;
      let bestIdx = knownPositions[0];
      const sampleCount = Math.min(knownPositions.length, 50);
      const step = Math.max(1, Math.floor(knownPositions.length / sampleCount));

      for (let si = 0; si < knownPositions.length; si += step) {
        const ki = knownPositions[si];
        const kx = ki % w;
        const ky = (ki - kx) / w;
        if (kx < half || kx >= w - half || ky < half || ky >= h - half) continue;

        let dist = 0;
        for (let py = -half; py <= half; py++) {
          for (let px = -half; px <= half; px++) {
            const srcI = ((y + py) * w + (x + px)) * 4;
            const knownI = ((ky + py) * w + (kx + px)) * 4;
            const dr = out[srcI] - imgPixels[knownI];
            const dg = out[srcI + 1] - imgPixels[knownI + 1];
            const db = out[srcI + 2] - imgPixels[knownI + 2];
            dist += dr * dr + dg * dg + db * db;
          }
        }

        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = ki;
        }
      }

      // Copy center pixel from best match
      const bx = bestIdx % w;
      const by = (bestIdx - bx) / w;
      const srcP = (by * w + bx) * 4;
      out[idx * 4] = imgPixels[srcP];
      out[idx * 4 + 1] = imgPixels[srcP + 1];
      out[idx * 4 + 2] = imgPixels[srcP + 2];

      if (mi % 500 === 0) {
        const pct = 50 + Math.round(((iter * maskedPositions.length + mi) / (iterations * maskedPositions.length)) * 50);
        self.postMessage({ type: 'progress', percent: Math.min(pct, 99) });
      }
    }
  }

  self.postMessage({ type: 'progress', percent: 100 });
  return out;
}
