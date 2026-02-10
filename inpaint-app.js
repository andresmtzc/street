/* ==============================================
   Street View Inpainter — Main Application
   ============================================== */

// ---- Config ----
const CFG = {
  MAX_WORK_SIZE: 2048,     // max canvas dimension
  INFER_SIZE: 512,         // model inference resolution
  BRUSH_DEFAULT: 40,
  MASK_FILL: 'rgba(255, 40, 40, 0.45)',
  MASK_ERASE: 'destination-out',
};

// ---- State ----
const S = {
  images: [],          // [{file, name, thumbUrl, w, h, maskBlob, resultBlob, processed}]
  idx: -1,             // current image index
  drawing: false,
  lastX: 0, lastY: 0,
  brushSize: CFG.BRUSH_DEFAULT,
  brushMode: 'paint',  // 'paint' | 'erase'
  showResult: false,
  templateBlob: null,
  model: null,         // ONNX InferenceSession
  modelSize: 512,      // detected model input resolution
  processing: false,
  cancelRequested: false,
  worker: null,
};

// ---- DOM refs ----
const $ = (id) => document.getElementById(id);
const D = {};

function cacheDom() {
  const ids = [
    'loadModelBtn', 'modelStatus', 'modelInput',
    'loadImagesBtn', 'imageInput', 'imageCount',
    'brushSize', 'brushSizeVal', 'paintMode', 'eraseMode', 'clearMask',
    'saveMaskTemplate', 'applyTemplate', 'nudgeX', 'nudgeY',
    'processCurrent', 'processAll', 'cancelProcess',
    'toggleView', 'exportCurrent', 'exportAll',
    'gallery', 'editor', 'canvasWrap',
    'imageCanvas', 'maskCanvas', 'brushCursor', 'editorPlaceholder',
    'prevImage', 'nextImage', 'currentInfo',
    'progressBar', 'progressFill', 'progressText',
  ];
  ids.forEach(id => D[id] = $(id));
}

// ---- Init ----
function init() {
  cacheDom();
  initWorker();
  bindEvents();
  updateUI();
}

function initWorker() {
  // Create worker from the separate file
  S.worker = new Worker('inpaint-worker.js');
}

// ---- Events ----
function bindEvents() {
  D.loadModelBtn.addEventListener('click', () => D.modelInput.click());
  D.modelInput.addEventListener('change', onModelLoad);

  D.loadImagesBtn.addEventListener('click', () => D.imageInput.click());
  D.imageInput.addEventListener('change', onImagesLoad);

  D.brushSize.addEventListener('input', (e) => {
    S.brushSize = +e.target.value;
    D.brushSizeVal.textContent = S.brushSize;
  });

  D.paintMode.addEventListener('click', () => setBrushMode('paint'));
  D.eraseMode.addEventListener('click', () => setBrushMode('erase'));
  D.clearMask.addEventListener('click', clearCurrentMask);

  D.saveMaskTemplate.addEventListener('click', saveTemplate);
  D.applyTemplate.addEventListener('click', applyTemplateToAll);

  D.processCurrent.addEventListener('click', processCurrentImage);
  D.processAll.addEventListener('click', processAllImages);
  D.cancelProcess.addEventListener('click', () => { S.cancelRequested = true; });

  D.toggleView.addEventListener('click', toggleBeforeAfter);
  D.exportCurrent.addEventListener('click', exportCurrentImage);
  D.exportAll.addEventListener('click', exportAllAsZip);

  D.prevImage.addEventListener('click', () => navigateImage(-1));
  D.nextImage.addEventListener('click', () => navigateImage(1));

  // Canvas drawing events
  D.maskCanvas.addEventListener('pointerdown', onDrawStart);
  D.maskCanvas.addEventListener('pointermove', onDrawMove);
  D.maskCanvas.addEventListener('pointerup', onDrawEnd);
  D.maskCanvas.addEventListener('pointerleave', onDrawEnd);

  // Brush cursor
  D.canvasWrap.addEventListener('pointermove', moveBrushCursor);
  D.canvasWrap.addEventListener('pointerenter', () => D.brushCursor.style.display = 'block');
  D.canvasWrap.addEventListener('pointerleave', () => D.brushCursor.style.display = 'none');

  // Keyboard shortcuts
  document.addEventListener('keydown', onKeyDown);

  // Drag and drop on the whole page
  document.addEventListener('dragover', (e) => e.preventDefault());
  document.addEventListener('drop', onDrop);
}

// ---- Model loading (ONNX) ----
async function onModelLoad(e) {
  const file = e.target.files[0];
  if (!file) return;
  D.modelStatus.textContent = 'Loading model...';
  D.modelStatus.className = 'status-badge';
  try {
    const buf = await file.arrayBuffer();
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
    S.model = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });

    // Detect model input resolution by running a dummy inference to read expected shape
    // LaMa models have fixed input: [1, 3, H, W] for image, [1, 1, H, W] for mask
    // Try to read from the model metadata or fall back to probing
    let detectedSize = 512;
    try {
      // ONNX Runtime Web doesn't expose input shapes directly,
      // so we probe with a small test — but first try common sizes
      // The model file name often hints at the resolution
      const nameHint = file.name.toLowerCase();
      const sizeMatch = nameHint.match(/(\d{3,4})/);
      if (sizeMatch) {
        const parsed = parseInt(sizeMatch[1]);
        if (parsed >= 256 && parsed <= 4096 && parsed % 8 === 0) {
          detectedSize = parsed;
        }
      }
    } catch (_) {}
    S.modelSize = detectedSize;

    D.modelStatus.textContent = `AI model loaded (${S.modelSize}x${S.modelSize})`;
    D.modelStatus.className = 'status-badge ai-loaded';
    console.log('ONNX inputs:', S.model.inputNames, 'outputs:', S.model.outputNames, 'inferSize:', S.modelSize);
  } catch (err) {
    D.modelStatus.textContent = 'Model load failed';
    console.error('Model load error:', err);
    S.model = null;
  }
}

// ---- Image loading ----
async function onImagesLoad(e) {
  const files = Array.from(e.target.files).filter(f => f.type.startsWith('image/'));
  if (!files.length) return;
  await addImages(files);
}

async function onDrop(e) {
  e.preventDefault();
  const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
  if (files.length) await addImages(files);
}

async function addImages(files) {
  // Sort files naturally by name
  files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));

  for (const file of files) {
    const thumbUrl = URL.createObjectURL(file);
    // Get dimensions from a quick decode
    const dims = await getImageDims(file);
    S.images.push({
      file,
      name: file.name,
      thumbUrl,
      w: dims.w,
      h: dims.h,
      maskBlob: null,
      resultBlob: null,
      processed: false,
    });
  }

  renderGallery();
  D.imageCount.textContent = S.images.length + ' images';

  if (S.idx < 0) selectImage(0);
  updateUI();
}

function getImageDims(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      resolve({ w: img.naturalWidth, h: img.naturalHeight });
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  });
}

// ---- Gallery ----
function renderGallery() {
  D.gallery.innerHTML = '';
  S.images.forEach((img, i) => {
    const div = document.createElement('div');
    div.className = 'thumb' + (i === S.idx ? ' active' : '');
    div.innerHTML = `
      <img src="${img.thumbUrl}" alt="${img.name}">
      <div class="thumb-label">${i + 1}. ${img.name}</div>
      <div class="thumb-status${img.processed ? ' processed' : img.maskBlob ? ' has-mask' : ''}"></div>
    `;
    div.addEventListener('click', () => selectImage(i));
    D.gallery.appendChild(div);
  });
}

function updateGalleryItem(i) {
  const thumbs = D.gallery.querySelectorAll('.thumb');
  if (!thumbs[i]) return;
  const img = S.images[i];
  thumbs[i].className = 'thumb' + (i === S.idx ? ' active' : '');
  const dot = thumbs[i].querySelector('.thumb-status');
  dot.className = 'thumb-status' + (img.processed ? ' processed' : img.maskBlob ? ' has-mask' : '');
}

// ---- Image display ----
async function selectImage(i) {
  if (i < 0 || i >= S.images.length) return;

  // Save current mask before switching
  if (S.idx >= 0) await saveCurrentMask();

  const prevIdx = S.idx;
  S.idx = i;
  S.showResult = false;

  // Update gallery highlights
  if (prevIdx >= 0) updateGalleryItem(prevIdx);
  updateGalleryItem(i);

  // Scroll thumb into view
  const thumbs = D.gallery.querySelectorAll('.thumb');
  if (thumbs[i]) thumbs[i].scrollIntoView({ block: 'nearest' });

  await displayCurrentImage();
  await loadCurrentMask();
  updateUI();
}

async function displayCurrentImage() {
  const img = S.images[S.idx];
  if (!img) return;

  const bitmap = await createImageBitmap(img.file);

  // Cap to working size
  let w = bitmap.width, h = bitmap.height;
  if (Math.max(w, h) > CFG.MAX_WORK_SIZE) {
    const scale = CFG.MAX_WORK_SIZE / Math.max(w, h);
    w = Math.round(w * scale);
    h = Math.round(h * scale);
  }

  D.imageCanvas.width = w;
  D.imageCanvas.height = h;
  D.maskCanvas.width = w;
  D.maskCanvas.height = h;

  const ctx = D.imageCanvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();

  D.editorPlaceholder.style.display = 'none';
  D.currentInfo.textContent = `${S.idx + 1} / ${S.images.length} — ${img.name} (${w}x${h})`;
}

async function displayResult() {
  const img = S.images[S.idx];
  if (!img || !img.resultBlob) return;
  const bitmap = await createImageBitmap(img.resultBlob);
  const ctx = D.imageCanvas.getContext('2d');
  ctx.clearRect(0, 0, D.imageCanvas.width, D.imageCanvas.height);
  ctx.drawImage(bitmap, 0, 0, D.imageCanvas.width, D.imageCanvas.height);
  bitmap.close();
}

async function displayOriginal() {
  const img = S.images[S.idx];
  if (!img) return;
  const bitmap = await createImageBitmap(img.file);
  const ctx = D.imageCanvas.getContext('2d');
  ctx.clearRect(0, 0, D.imageCanvas.width, D.imageCanvas.height);
  ctx.drawImage(bitmap, 0, 0, D.imageCanvas.width, D.imageCanvas.height);
  bitmap.close();
}

// ---- Mask drawing ----
function getCanvasCoords(e) {
  const rect = D.maskCanvas.getBoundingClientRect();
  const scaleX = D.maskCanvas.width / rect.width;
  const scaleY = D.maskCanvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
}

function onDrawStart(e) {
  if (S.idx < 0) return;
  S.drawing = true;
  const { x, y } = getCanvasCoords(e);
  S.lastX = x;
  S.lastY = y;
  drawBrushStroke(x, y, x, y);
  D.maskCanvas.setPointerCapture(e.pointerId);
}

function onDrawMove(e) {
  if (!S.drawing) return;
  const { x, y } = getCanvasCoords(e);
  drawBrushStroke(S.lastX, S.lastY, x, y);
  S.lastX = x;
  S.lastY = y;
}

function onDrawEnd() {
  if (S.drawing) {
    S.drawing = false;
    // Mark image as having a mask
    saveCurrentMask();
  }
}

function drawBrushStroke(x0, y0, x1, y1) {
  const ctx = D.maskCanvas.getContext('2d');
  const r = S.brushSize / 2;
  const dx = x1 - x0, dy = y1 - y0;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const steps = Math.max(1, Math.ceil(dist / (r / 2)));

  ctx.save();
  if (S.brushMode === 'erase') {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = 'rgba(0,0,0,1)';
  } else {
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = CFG.MASK_FILL;
  }

  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const cx = x0 + dx * t;
    const cy = y0 + dy * t;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function moveBrushCursor(e) {
  const size = S.brushSize;
  const rect = D.maskCanvas.getBoundingClientRect();
  const cssScale = rect.width / (D.maskCanvas.width || 1);
  const displaySize = size * cssScale;
  D.brushCursor.style.width = displaySize + 'px';
  D.brushCursor.style.height = displaySize + 'px';
  D.brushCursor.style.left = (e.clientX - displaySize / 2) + 'px';
  D.brushCursor.style.top = (e.clientY - displaySize / 2) + 'px';
}

function clearCurrentMask() {
  const ctx = D.maskCanvas.getContext('2d');
  ctx.clearRect(0, 0, D.maskCanvas.width, D.maskCanvas.height);
  if (S.idx >= 0) {
    S.images[S.idx].maskBlob = null;
    updateGalleryItem(S.idx);
  }
}

// ---- Mask save/load ----
async function saveCurrentMask() {
  if (S.idx < 0) return;
  const blob = await canvasToBlob(D.maskCanvas);
  S.images[S.idx].maskBlob = blob;
  updateGalleryItem(S.idx);
}

async function loadCurrentMask() {
  const ctx = D.maskCanvas.getContext('2d');
  ctx.clearRect(0, 0, D.maskCanvas.width, D.maskCanvas.height);
  const img = S.images[S.idx];
  if (!img || !img.maskBlob) return;
  const bitmap = await createImageBitmap(img.maskBlob);
  ctx.drawImage(bitmap, 0, 0, D.maskCanvas.width, D.maskCanvas.height);
  bitmap.close();
}

// ---- Template system ----
async function saveTemplate() {
  if (S.idx < 0) return;
  await saveCurrentMask();
  S.templateBlob = S.images[S.idx].maskBlob;
  D.nudgeX.value = 0;
  D.nudgeY.value = 0;
  alert('Mask template saved. Click "Apply to All" to apply it to all images.');
}

async function applyTemplateToAll() {
  if (!S.templateBlob) {
    alert('Save a template first by drawing a mask and clicking "Save Template".');
    return;
  }

  const nudgeX = parseInt(D.nudgeX.value) || 0;
  const nudgeY = parseInt(D.nudgeY.value) || 0;

  // Save current mask first
  if (S.idx >= 0) await saveCurrentMask();

  const templateBitmap = await createImageBitmap(S.templateBlob);

  for (let i = 0; i < S.images.length; i++) {
    const img = S.images[i];
    // Create a temporary canvas to draw the shifted template
    const tmpCanvas = new OffscreenCanvas(
      D.imageCanvas.width || templateBitmap.width,
      D.imageCanvas.height || templateBitmap.height
    );
    const ctx = tmpCanvas.getContext('2d');
    ctx.drawImage(templateBitmap, nudgeX, nudgeY, tmpCanvas.width, tmpCanvas.height);
    const blob = await canvasToBlob(tmpCanvas);
    img.maskBlob = blob;
    img.processed = false;
    img.resultBlob = null;
    updateGalleryItem(i);
  }

  templateBitmap.close();

  // Reload current mask
  await loadCurrentMask();
  updateUI();
}

// ---- Inpainting ----
async function processCurrentImage() {
  if (S.idx < 0 || S.processing) return;
  S.processing = true;
  updateUI();
  showProgress(true);

  try {
    await inpaintImage(S.idx, true);
    // Show result
    S.showResult = true;
    await displayResult();
  } catch (err) {
    console.error('Inpaint error:', err);
    alert('Inpainting failed: ' + err.message);
  }

  S.processing = false;
  showProgress(false);
  updateUI();
}

async function processAllImages() {
  if (S.processing) return;
  S.processing = true;
  S.cancelRequested = false;
  updateUI();
  showProgress(true);

  let done = 0;
  const total = S.images.length;

  for (let i = 0; i < total; i++) {
    if (S.cancelRequested) break;
    if (!S.images[i].maskBlob) {
      done++;
      setProgress(done / total, `Skipping ${done}/${total} (no mask)`);
      continue;
    }

    setProgress(done / total, `Processing ${done + 1}/${total}: ${S.images[i].name}`);

    // Show current image being processed
    if (i !== S.idx) await selectImage(i);

    try {
      await inpaintImage(i, true);
      S.showResult = true;
      await displayResult();
    } catch (err) {
      console.error(`Error on image ${i}:`, err);
    }
    done++;
  }

  S.processing = false;
  S.cancelRequested = false;
  showProgress(false);
  setProgress(1, `Done: ${done}/${total}`);
  updateUI();
}

async function inpaintImage(index, reportProgress) {
  const img = S.images[index];
  if (!img.maskBlob) throw new Error('No mask drawn for this image');

  // Get image pixels at working resolution
  const bitmap = await createImageBitmap(img.file);
  let w = bitmap.width, h = bitmap.height;
  if (Math.max(w, h) > CFG.MAX_WORK_SIZE) {
    const scale = CFG.MAX_WORK_SIZE / Math.max(w, h);
    w = Math.round(w * scale);
    h = Math.round(h * scale);
  }

  const imgCanvas = new OffscreenCanvas(w, h);
  const imgCtx = imgCanvas.getContext('2d');
  imgCtx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();

  // Get mask pixels
  const maskBitmap = await createImageBitmap(img.maskBlob);
  const maskCanvas = new OffscreenCanvas(w, h);
  const maskCtx = maskCanvas.getContext('2d');
  maskCtx.drawImage(maskBitmap, 0, 0, w, h);
  maskBitmap.close();

  let resultCanvas;

  if (S.model) {
    resultCanvas = await inpaintWithONNX(imgCanvas, maskCanvas, w, h);
  } else {
    resultCanvas = await inpaintWithWorker(imgCanvas, maskCanvas, w, h, reportProgress);
  }

  // Composite: original where mask=0, result where mask=1
  const finalCanvas = new OffscreenCanvas(w, h);
  const fCtx = finalCanvas.getContext('2d');
  fCtx.drawImage(imgCanvas, 0, 0);

  // Get mask data to use for compositing
  const mData = maskCtx.getImageData(0, 0, w, h);
  const rBitmap = await createImageBitmap(resultCanvas instanceof OffscreenCanvas ?
    await canvasToBlob(resultCanvas) : resultCanvas);

  const rCanvas = new OffscreenCanvas(w, h);
  const rCtx = rCanvas.getContext('2d');
  rCtx.drawImage(rBitmap, 0, 0, w, h);
  rBitmap.close();

  const origData = fCtx.getImageData(0, 0, w, h);
  const resData = rCtx.getImageData(0, 0, w, h);

  // Feather the mask edges for blending (3px radius)
  const blendMask = buildBlendMask(mData.data, w, h, 3);

  for (let i = 0; i < w * h; i++) {
    const alpha = blendMask[i];
    if (alpha > 0) {
      const pi = i * 4;
      origData.data[pi] = origData.data[pi] * (1 - alpha) + resData.data[pi] * alpha;
      origData.data[pi + 1] = origData.data[pi + 1] * (1 - alpha) + resData.data[pi + 1] * alpha;
      origData.data[pi + 2] = origData.data[pi + 2] * (1 - alpha) + resData.data[pi + 2] * alpha;
    }
  }

  fCtx.putImageData(origData, 0, 0);
  img.resultBlob = await canvasToBlob(finalCanvas, 'image/jpeg', 0.92);
  img.processed = true;
  updateGalleryItem(index);
}

function buildBlendMask(maskRGBA, w, h, featherRadius) {
  // Build float mask from alpha channel, with feathered edges
  const raw = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    raw[i] = maskRGBA[i * 4 + 3] > 30 ? 1.0 : 0.0;
  }

  if (featherRadius <= 0) return raw;

  // Simple box blur for feathering
  const blurred = new Float32Array(raw);
  const r = featherRadius;
  // Horizontal pass
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let dx = -r; dx <= r; dx++) {
        const nx = x + dx;
        if (nx >= 0 && nx < w) { sum += raw[y * w + nx]; count++; }
      }
      tmp[y * w + x] = sum / count;
    }
  }
  // Vertical pass
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let dy = -r; dy <= r; dy++) {
        const ny = y + dy;
        if (ny >= 0 && ny < h) { sum += tmp[ny * w + x]; count++; }
      }
      blurred[y * w + x] = sum / count;
    }
  }

  return blurred;
}

// ---- ONNX inpainting ----
async function inpaintWithONNX(imgCanvas, maskCanvas, w, h) {
  // Find bounding box of the mask so we only process the relevant region
  const maskCtx = maskCanvas.getContext('2d');
  const fullMask = maskCtx.getImageData(0, 0, w, h);
  const bbox = getMaskBBox(fullMask.data, w, h);
  if (!bbox) throw new Error('Empty mask');

  // Expand bbox with context padding (50% on each side, min 64px)
  const padX = Math.max(64, Math.round((bbox.x2 - bbox.x1) * 0.5));
  const padY = Math.max(64, Math.round((bbox.y2 - bbox.y1) * 0.5));
  const cx1 = Math.max(0, bbox.x1 - padX);
  const cy1 = Math.max(0, bbox.y1 - padY);
  const cx2 = Math.min(w, bbox.x2 + padX);
  const cy2 = Math.min(h, bbox.y2 + padY);
  const cw = cx2 - cx1, ch = cy2 - cy1;

  // Crop image and mask to the region around the mask
  const cropImg = new OffscreenCanvas(cw, ch);
  cropImg.getContext('2d').drawImage(imgCanvas, cx1, cy1, cw, ch, 0, 0, cw, ch);

  const cropMask = new OffscreenCanvas(cw, ch);
  cropMask.getContext('2d').drawImage(maskCanvas, cx1, cy1, cw, ch, 0, 0, cw, ch);

  // Resize crop to model's input resolution
  const iw = S.modelSize, ih = S.modelSize;

  const iCanvas = new OffscreenCanvas(iw, ih);
  const iCtx = iCanvas.getContext('2d');
  iCtx.drawImage(cropImg, 0, 0, iw, ih);
  const iData = iCtx.getImageData(0, 0, iw, ih);

  // Image tensor: normalize to [0, 1], NCHW layout
  const imgFloat = new Float32Array(3 * iw * ih);
  for (let i = 0; i < iw * ih; i++) {
    imgFloat[i] = iData.data[i * 4] / 255.0;
    imgFloat[iw * ih + i] = iData.data[i * 4 + 1] / 255.0;
    imgFloat[2 * iw * ih + i] = iData.data[i * 4 + 2] / 255.0;
  }

  // Mask tensor: binarize from alpha channel
  const mCanvas = new OffscreenCanvas(iw, ih);
  const mCtx = mCanvas.getContext('2d');
  mCtx.drawImage(cropMask, 0, 0, iw, ih);
  const mData = mCtx.getImageData(0, 0, iw, ih);

  const maskFloat = new Float32Array(iw * ih);
  for (let i = 0; i < iw * ih; i++) {
    maskFloat[i] = mData.data[i * 4 + 3] > 128 ? 1.0 : 0.0;
  }

  const imgTensor = new ort.Tensor('float32', imgFloat, [1, 3, ih, iw]);
  const maskTensor = new ort.Tensor('float32', maskFloat, [1, 1, ih, iw]);

  // Build feeds using model's input names
  const inputs = S.model.inputNames;
  const feeds = {};
  if (inputs.length >= 2) {
    feeds[inputs[0]] = imgTensor;
    feeds[inputs[1]] = maskTensor;
  } else {
    throw new Error('Model must have at least 2 inputs (image, mask). Found: ' + inputs.join(', '));
  }

  setProgress(0.3, `Running AI model on ${cw}x${ch} crop...`);
  const results = await S.model.run(feeds);
  const output = results[S.model.outputNames[0]];
  setProgress(0.8, 'Processing result...');

  // Convert output to canvas (LaMa ONNX outputs [0, 255])
  const outCanvas = new OffscreenCanvas(iw, ih);
  const outCtx = outCanvas.getContext('2d');
  const outImg = outCtx.createImageData(iw, ih);

  for (let i = 0; i < iw * ih; i++) {
    outImg.data[i * 4] = Math.max(0, Math.min(255, Math.round(output.data[i])));
    outImg.data[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(output.data[iw * ih + i])));
    outImg.data[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(output.data[2 * iw * ih + i])));
    outImg.data[i * 4 + 3] = 255;
  }
  outCtx.putImageData(outImg, 0, 0);

  // Scale result back to crop size and paste into full-resolution canvas
  const resultCanvas = new OffscreenCanvas(w, h);
  const rCtx = resultCanvas.getContext('2d');
  rCtx.drawImage(imgCanvas, 0, 0); // start with original
  rCtx.drawImage(outCanvas, 0, 0, iw, ih, cx1, cy1, cw, ch); // paste inpainted crop

  return resultCanvas;
}

function getMaskBBox(maskRGBA, w, h) {
  let x1 = w, y1 = h, x2 = 0, y2 = 0;
  let found = false;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (maskRGBA[(y * w + x) * 4 + 3] > 30) {
        if (x < x1) x1 = x;
        if (x > x2) x2 = x;
        if (y < y1) y1 = y;
        if (y > y2) y2 = y;
        found = true;
      }
    }
  }
  return found ? { x1, y1, x2: x2 + 1, y2: y2 + 1 } : null;
}

// ---- Worker-based inpainting ----
function inpaintWithWorker(imgCanvas, maskCanvas, w, h, reportProgress) {
  return new Promise((resolve, reject) => {
    const imgCtx = imgCanvas.getContext('2d');
    const imgData = imgCtx.getImageData(0, 0, w, h);

    const maskCtx = maskCanvas.getContext('2d');
    const maskData = maskCtx.getImageData(0, 0, w, h);

    // For large images, process at reduced resolution for speed
    const maxWorkerSize = 800;
    let pw = w, ph = h;
    if (Math.max(w, h) > maxWorkerSize) {
      const s = maxWorkerSize / Math.max(w, h);
      pw = Math.round(w * s);
      ph = Math.round(h * s);
    }

    // Resize for worker
    const sImgCanvas = new OffscreenCanvas(pw, ph);
    const sImgCtx = sImgCanvas.getContext('2d');
    sImgCtx.drawImage(imgCanvas, 0, 0, pw, ph);
    const sImgData = sImgCtx.getImageData(0, 0, pw, ph);

    const sMaskCanvas = new OffscreenCanvas(pw, ph);
    const sMaskCtx = sMaskCanvas.getContext('2d');
    sMaskCtx.drawImage(maskCanvas, 0, 0, pw, ph);
    const sMaskData = sMaskCtx.getImageData(0, 0, pw, ph);

    const handler = (e) => {
      const { type } = e.data;
      if (type === 'progress' && reportProgress) {
        setProgress(e.data.percent / 100, `Inpainting... ${e.data.percent}%`);
      } else if (type === 'result') {
        S.worker.removeEventListener('message', handler);
        const resultData = new ImageData(new Uint8ClampedArray(e.data.imageData), pw, ph);
        const rCanvas = new OffscreenCanvas(pw, ph);
        rCanvas.getContext('2d').putImageData(resultData, 0, 0);

        // Scale back to working size
        const outCanvas = new OffscreenCanvas(w, h);
        outCanvas.getContext('2d').drawImage(rCanvas, 0, 0, w, h);
        resolve(outCanvas);
      } else if (type === 'error') {
        S.worker.removeEventListener('message', handler);
        reject(new Error(e.data.message));
      }
    };

    S.worker.addEventListener('message', handler);
    S.worker.postMessage({
      type: 'inpaint',
      imageData: sImgData.data.buffer,
      maskData: sMaskData.data.buffer,
      width: pw,
      height: ph,
    }, [sImgData.data.buffer, sMaskData.data.buffer]);
  });
}

// ---- Before / After ----
async function toggleBeforeAfter() {
  if (S.idx < 0) return;
  const img = S.images[S.idx];
  if (!img.processed || !img.resultBlob) return;

  S.showResult = !S.showResult;
  if (S.showResult) {
    await displayResult();
    // Hide mask overlay when showing result
    D.maskCanvas.style.opacity = '0';
  } else {
    await displayOriginal();
    D.maskCanvas.style.opacity = '1';
  }
}

// ---- Export ----
async function exportCurrentImage() {
  if (S.idx < 0) return;
  const img = S.images[S.idx];
  const blob = img.resultBlob || await canvasToBlob(
    getCanvasAsOffscreen(D.imageCanvas), 'image/jpeg', 0.92
  );
  downloadBlob(blob, img.name.replace(/\.\w+$/, '_inpainted.jpg'));
}

async function exportAllAsZip() {
  if (!S.images.length) return;
  S.processing = true;
  updateUI();
  showProgress(true);

  const zip = new JSZip();
  let done = 0;

  for (const img of S.images) {
    if (img.resultBlob) {
      const name = img.name.replace(/\.\w+$/, '_inpainted.jpg');
      zip.file(name, img.resultBlob);
    }
    done++;
    setProgress(done / S.images.length, `Zipping ${done}/${S.images.length}`);
  }

  setProgress(0.95, 'Generating ZIP...');
  const zipBlob = await zip.generateAsync({ type: 'blob' });
  downloadBlob(zipBlob, 'inpainted_images.zip');

  S.processing = false;
  showProgress(false);
  updateUI();
}

// ---- Navigation ----
function navigateImage(delta) {
  const newIdx = S.idx + delta;
  if (newIdx >= 0 && newIdx < S.images.length) selectImage(newIdx);
}

function setBrushMode(mode) {
  S.brushMode = mode;
  D.paintMode.classList.toggle('active', mode === 'paint');
  D.eraseMode.classList.toggle('active', mode === 'erase');
}

// ---- Keyboard shortcuts ----
function onKeyDown(e) {
  // Don't capture when typing in inputs
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  switch (e.key) {
    case 'p': case 'P': setBrushMode('paint'); break;
    case 'e': case 'E': setBrushMode('erase'); break;
    case 'c': case 'C': clearCurrentMask(); break;
    case '[':
      S.brushSize = Math.max(5, S.brushSize - 5);
      D.brushSize.value = S.brushSize;
      D.brushSizeVal.textContent = S.brushSize;
      break;
    case ']':
      S.brushSize = Math.min(200, S.brushSize + 5);
      D.brushSize.value = S.brushSize;
      D.brushSizeVal.textContent = S.brushSize;
      break;
    case 'ArrowLeft':
      if (e.shiftKey) nudgeMask(-5, 0);
      else navigateImage(-1);
      e.preventDefault();
      break;
    case 'ArrowRight':
      if (e.shiftKey) nudgeMask(5, 0);
      else navigateImage(1);
      e.preventDefault();
      break;
    case 'ArrowUp':
      if (e.shiftKey) { nudgeMask(0, -5); e.preventDefault(); }
      break;
    case 'ArrowDown':
      if (e.shiftKey) { nudgeMask(0, 5); e.preventDefault(); }
      break;
    case ' ':
      e.preventDefault();
      toggleBeforeAfter();
      break;
  }
}

async function nudgeMask(dx, dy) {
  if (S.idx < 0) return;
  const ctx = D.maskCanvas.getContext('2d');
  const data = ctx.getImageData(0, 0, D.maskCanvas.width, D.maskCanvas.height);
  ctx.clearRect(0, 0, D.maskCanvas.width, D.maskCanvas.height);
  ctx.putImageData(data, dx, dy);
  await saveCurrentMask();
}

// ---- Progress ----
function showProgress(visible) {
  D.progressBar.classList.toggle('visible', visible);
  D.cancelProcess.style.display = visible && S.processing ? 'inline-block' : 'none';
}

function setProgress(fraction, text) {
  D.progressFill.style.width = Math.round(fraction * 100) + '%';
  D.progressText.textContent = text || '';
}

// ---- UI state ----
function updateUI() {
  const hasImages = S.images.length > 0;
  const hasSelection = S.idx >= 0;

  D.processCurrent.disabled = !hasSelection || S.processing;
  D.processAll.disabled = !hasImages || S.processing;
  D.exportCurrent.disabled = !hasSelection;
  D.exportAll.disabled = !hasImages;
  D.saveMaskTemplate.disabled = !hasSelection;
  D.applyTemplate.disabled = !S.templateBlob || S.processing;
  D.clearMask.disabled = !hasSelection;
  D.toggleView.disabled = !hasSelection;
  D.prevImage.disabled = S.idx <= 0;
  D.nextImage.disabled = S.idx >= S.images.length - 1;
}

// ---- Utility ----
function canvasToBlob(canvas, type = 'image/png', quality = 1) {
  if (canvas instanceof OffscreenCanvas) {
    return canvas.convertToBlob({ type, quality });
  }
  return new Promise(r => canvas.toBlob(r, type, quality));
}

function getCanvasAsOffscreen(canvas) {
  const off = new OffscreenCanvas(canvas.width, canvas.height);
  off.getContext('2d').drawImage(canvas, 0, 0);
  return off;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// ---- Boot ----
document.addEventListener('DOMContentLoaded', init);
