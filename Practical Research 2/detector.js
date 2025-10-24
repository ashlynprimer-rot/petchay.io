// detector.js
// Frontend heuristic + optional TF.js model loader example

// ----- Utilities -----
const el = id => document.getElementById(id);

function bytesToKB(n) { return (n/1024).toFixed(1) + ' KB'; }
function clamp01(x){ return Math.max(0, Math.min(1, x)); }

// Read file -> dataURL and ArrayBuffer
async function readFile(file){
  const dataUrl = await new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
  const arrBuffer = await file.arrayBuffer();
  return { dataUrl, arrBuffer };
}

// Draw image into canvas and return ImageData
async function loadImageData(dataUrl, maxDim=1024){
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => {
      // downscale if big for performance
      let w = img.width, h = img.height;
      let scale = 1;
      if (Math.max(w,h) > maxDim) scale = maxDim / Math.max(w,h);
      const cw = Math.round(w*scale), ch = Math.round(h*scale);
      const canvas = document.createElement('canvas');
      canvas.width = cw;
      canvas.height = ch;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, cw, ch);
      const imageData = ctx.getImageData(0,0,cw,ch);
      res({ imageData, width: cw, height: ch, dataUrl });
    };
    img.onerror = rej;
    img.src = dataUrl;
  });
}

// Compute simple color stats and ExG index on masked green-ish pixels
function computeColorIndices(imageData){
  const { data, width, height } = imageData;
  const pixels = width * height;
  let sumR=0,sumG=0,sumB=0, count=0;
  let sumExG = 0;
  for (let i=0;i<data.length;i+=4){
    const r = data[i], g = data[i+1], b = data[i+2];
    // quick green-ish mask: g > r*0.9 and g > b*0.9 and g > 30
    if (g > r*0.9 && g > b*0.9 && g > 30){
      sumR += r; sumG += g; sumB += b;
      sumExG += (2*g - r - b);
      count++;
    }
  }
  if (count === 0) return { meanR:null, meanG:null, meanB:null, meanExG:null, greenPixelCount:0 };
  return {
    meanR: sumR/count,
    meanG: sumG/count,
    meanB: sumB/count,
    meanExG: sumExG/count,
    greenPixelCount: count
  };
}

// Noise proxy: compute median of local stddevs (convolution-free approx by comparing to blurred)
function noiseEstimate(imageData){
  const { data, width, height } = imageData;
  // convert to grayscale float array
  const gray = new Float32Array(width*height);
  for (let i=0, p=0;i<data.length;i+=4, p++){
    gray[p] = 0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2];
  }
  // simple blur: box blur 7x7 implemented separably
  const blur = new Float32Array(width*height);
  const radius = 3; // kernel = 7
  // horizontal pass
  for (let y=0;y<height;y++){
    let sum=0;
    const row = y*width;
    for (let x=-radius;x<=radius;x++){
      const xi = Math.min(width-1, Math.max(0, x));
      sum += gray[row + xi];
    }
    for (let x=0;x<width;x++){
      blur[row+x] = sum / (2*radius+1);
      const iRemove = x-radius;
      const iAdd = x+radius+1;
      const idxRemove = Math.min(width-1, Math.max(0, iRemove));
      const idxAdd = Math.min(width-1, Math.max(0, iAdd));
      sum += gray[row + idxAdd] - gray[row + idxRemove];
    }
  }
  // vertical pass into temp
  const blur2 = new Float32Array(width*height);
  for (let x=0;x<width;x++){
    let sum=0;
    for (let y=-radius;y<=radius;y++){
      const yi = Math.min(height-1, Math.max(0, y));
      sum += blur[yi*width + x];
    }
    for (let y=0;y<height;y++){
      blur2[y*width + x] = sum / (2*radius+1);
      const yRemove = Math.min(height-1, Math.max(0, y-radius));
      const yAdd = Math.min(height-1, Math.max(0, y+radius+1));
      sum += blur[yAdd*width + x] - blur[yRemove*width + x];
    }
  }
  // local std ~ |gray - blur2|
  const diffs = new Float32Array(width*height);
  for (let i=0;i<width*height;i++){
    diffs[i] = Math.abs(gray[i] - blur2[i]);
  }
  // median approx by sampling
  const sample = [];
  const sampleCount = Math.min(2000, diffs.length);
  for (let i=0;i<sampleCount;i++){
    sample.push(diffs[Math.floor(Math.random()*diffs.length)]);
  }
  sample.sort((a,b)=>a-b);
  const med = sample[Math.floor(sample.length/2)];
  return med;
}

// Laplacian variance proxy: simple 3x3 laplacian kernel
function laplacianVar(imageData){
  const { data, width, height } = imageData;
  const gray = new Float32Array(width*height);
  for (let i=0,p=0;i<data.length;i+=4,p++){
    gray[p] = 0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2];
  }
  const kernel = [0,1,0,1,-4,1,0,1,0];
  const out = new Float32Array(width*height);
  for (let y=1;y<height-1;y++){
    for (let x=1;x<width-1;x++){
      let s = 0;
      let k = 0;
      for (let ky=-1;ky<=1;ky++){
        for (let kx=-1;kx<=1;kx++){
          s += kernel[k] * gray[(y+ky)*width + (x+kx)];
          k++;
        }
      }
      out[y*width + x] = s;
    }
  }
  // variance
  let mean = 0;
  for (let i=0;i<out.length;i++) mean += out[i];
  mean /= out.length;
  let v = 0;
  for (let i=0;i<out.length;i++){ const d = out[i]-mean; v += d*d; }
  v /= out.length;
  return v;
}

// EXIF check using exif-js (if loaded)
function checkExifMarkers(file, callback){
  if (!window.EXIF) return callback(null);
  try {
    EXIF.getData(file, function(){
      const all = EXIF.getAllTags(this);
      const s = JSON.stringify(all).toLowerCase();
      const markers = ["midjourney","stable","sd","dalle","gpt-image","imagen","generated by","ai-generated"];
      for (let m of markers) if (s.includes(m)) return callback(m);
      return callback(null);
    });
  } catch (e){
    return callback(null);
  }
}

// ----- Heuristic scoring -----
function computeHeuristicScore(features){
  // features: {meanExG, greenPixelCount, noise, lapVar, fileSizePerMP}
  // Map each into a suspicion 0..1 where 1 is suspicious (likely AI)
  // These thresholds are example values you should tune with real data.
  const f = {};
  f.exif = features.exifMarker ? 1.0 : 0.0;
  // If ExG (green-ness) is low -> suspicious for plant images
  f.exg = (features.meanExG==null) ? 0.3 : clamp01((50 - features.meanExG) / 100); // example
  // Very low green pixel count -> suspicious (maybe synthetic background / small plant)
  f.greenCount = clamp01(1 - (features.greenPixelCount / (features.width*features.height)) * 5);
  // noise low => suspicious (many GAN outputs are too smooth)
  f.noise = clamp01((0.5 - features.noiseNormalized) * 2.0); // expects normalized noise ~ 0..1
  // laplacian low => suspicious
  f.lap = clamp01((200 - features.lapVar) / 400);

  // weighting
  const weights = { exif:3, exg:1.5, greenCount:1.2, noise:2, lap:1.8 };
  let sum = 0, wsum=0;
  for (let k in weights){ sum += (f[k]||0) * weights[k]; wsum += weights[k]; }
  const score = Math.round((sum / wsum) * 100);
  // reasons
  const reasons = [];
  if (f.exif > 0.5) reasons.push(`Found possible generator tag "${features.exifMarker}" in EXIF`);
  if (f.noise > 0.6) reasons.push("Image noise is unusually low (very smooth)");
  if (f.lap > 0.6) reasons.push("Low edge/texture variance (smoothed details)");
  if (f.exg > 0.6) reasons.push("Color indices deviate from normal plant green");
  if (f.greenCount > 0.6) reasons.push("Very few green pixels detected relative to size");
  if (reasons.length === 0) reasons.push("No strong heuristic indicators found");
  return { score, reasons, components: f };
}

// ----- Main flow -----
let loadedModel = null;

async function analyzeFile(file){
  const previewCard = el('previewCard');
  if (previewCard) previewCard.style.display = 'block';
  if (el('fileName')) el('fileName').textContent = file.name;
  if (el('fileSize')) el('fileSize').textContent = bytesToKB(file.size);

  const { dataUrl, arrBuffer } = await readFile(file);
  if (el('previewImg')) el('previewImg').src = dataUrl;

  const { imageData, width, height } = await loadImageData(dataUrl, 900);

  // compute color indices
  const color = computeColorIndices(imageData);
  const noiseRaw = noiseEstimate(imageData); // absolute value ~0..50 depending on scale
  const lapVar = laplacianVar(imageData);

  // normalize noise by simple heuristic (max expected ~30)
  const noiseNormalized = Math.min(1, noiseRaw / 30);

  const mp = (width*height)/1_000_000;
  const bytesPerMP = mp>0 ? (file.size / mp) : file.size;

  const exifPromise = new Promise(resolve => checkExifMarkers(file, resolve));
  const exifMarker = await exifPromise;

  const feats = {
    width, height,
    meanExG: color.meanExG,
    meanR: color.meanR, meanG: color.meanG, meanB: color.meanB,
    greenPixelCount: color.greenPixelCount,
    noiseRaw, noiseNormalized,
    lapVar,
    bytesPerMP,
    exifMarker
  };

  const heur = computeHeuristicScore({
    meanExG: feats.meanExG,
    greenPixelCount: feats.greenPixelCount,
    noiseNormalized: feats.noiseNormalized,
    lapVar: feats.lapVar,
    width: feats.width, height: feats.height,
    exifMarker: feats.exifMarker
  });

  // show results (guarded)
  const rc = el('resultCard');
  if (rc) rc.style.display = 'block';
  if (el('scoreOut')) el('scoreOut').textContent = heur.score;
  if (el('reasonsOut')) el('reasonsOut').innerHTML = heur.reasons.map(r=>`<li>${r}</li>`).join('');
  if (el('featuresOut')) el('featuresOut').textContent = JSON.stringify(feats, null, 2);

  // Notify page (if helper exists)
  try {
    if (window.notify) window.notify(`Analysis finished — score ${heur.score}`, 'info');
    if (window.confettiBurst && heur.score < 40) window.confettiBurst(60);
  } catch(e){ /* ignore */ }

  // Model inference block (guard loadedModel and UI location)
  if (loadedModel && typeof tf !== 'undefined'){
    try {
      const inputVec = tf.tensor2d([[ heur.score / 100 ]]); // placeholder
      const out = loadedModel.predict(inputVec);
      const outArr = await out.data();
      if (el('modelOut')) el('modelOut').textContent = `Model output (raw): ${Array.from(outArr).map(x=>x.toFixed(4)).join(', ')}`;
    } catch (e){
      if (el('modelOut')) el('modelOut').textContent = `Model run error: ${e}`;
    }
  } else {
    if (el('modelOut')) el('modelOut').textContent = 'No model loaded in-browser.';
  }

  return { feats, heur };
}

// Expose core helpers to global scope explicitly
window.detectorHelpers = {
  readFile, loadImageData, computeColorIndices,
  noiseEstimate, laplacianVar, computeHeuristicScore
};

// Safe UI hooks: attach if elements exist
if (el('analyzeBtn') && el('fileInput')) {
  el('analyzeBtn').addEventListener('click', async ()=>{
    const f = el('fileInput').files[0];
    if (!f) return alert('Select a file first');
    await analyzeFile(f);
  });
}

if (el('serverBtn') && el('fileInput')) {
  el('serverBtn').addEventListener('click', async ()=>{
    const f = el('fileInput').files[0];
    if (!f) return alert('Select a file first');
    const url = '/api/predict';
    const fd = new FormData();
    fd.append('file', f);
    try {
      const r = await fetch(url, { method:'POST', body:fd });
      if (!r.ok) throw new Error(`Server ${r.status}`);
      const json = await r.json();
      if (el('modelOut')) el('modelOut').textContent = JSON.stringify(json, null, 2);
    } catch (e){
      if (el('modelOut')) el('modelOut').textContent = `Server request failed: ${e}`;
    }
  });
}

// Optional: load TF.js model from URL into loadedModel
async function loadTFModel(url){
  try {
    loadedModel = await tf.loadLayersModel(url);
    console.log('Loaded model', loadedModel);
    return true;
  } catch (e){
    console.warn('Failed loading model', e);
    return false;
  }
}

// Example: if you host a model at /model/model.json, call:
// loadTFModel('/model/model.json');
