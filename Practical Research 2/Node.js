// server.js
const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node'); // npm i @tensorflow/tfjs-node
const fs = require('fs');
const path = require('path');

const upload = multer({ dest: 'uploads/' });
const app = express();
// allow larger JSON bodies (feedback may include base64 image)
app.use(express.json({ limit: '12mb' }));

// load your TFJS model (saved with tfjs_converter as model.json + weights)
let model = null;
(async ()=> {
  try {
    model = await tf.loadLayersModel('file://models/model.json'); // path to model.json
    console.log('Model loaded');
  } catch (e) {
    console.warn('Model not loaded:', e.message);
  }
})();

app.post('/api/predict', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'no file' });
    // For example purposes: we'll just return some file info + fake prediction
    const info = {
      filename: req.file.originalname,
      size: req.file.size,
      path: req.file.path
    };

    // If model is loaded you can pre-process file into tensor and run inference.
    // Example placeholder returns a random score:
    let modelOut = { score: Math.round(Math.random()*100) };

    // Clean up uploaded file (optional)
    fs.unlink(req.file.path, ()=>{});

    return res.json({ success:true, info, modelOut });
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
});

app.post('/api/feedback', async (req, res) => {
  try {
    const { email, msg, bgData } = req.body;
    if (!msg || typeof msg !== 'string') return res.status(400).json({ error: 'Missing feedback message' });

    let bgPath = '';
    if (bgData && typeof bgData === 'string' && bgData.startsWith('data:')) {
      // parse data URL
      const matches = bgData.match(/^data:(image\/[a-zA-Z+]+);base64,(.+)$/);
      if (matches) {
        const mime = matches[1];
        const b64 = matches[2];
        const ext = mime.split('/')[1].split('+')[0];
        const buf = Buffer.from(b64, 'base64');
        const dir = path.join(__dirname, 'backgrounds');
        fs.mkdirSync(dir, { recursive: true });
        const fname = `bg_${Date.now()}.${ext}`;
        bgPath = path.join('backgrounds', fname);
        fs.writeFileSync(path.join(__dirname, bgPath), buf);
      }
    }

    const entry = {
      email: email || '',
      msg,
      bgPath,
      time: new Date().toISOString()
    };
    fs.appendFile('feedback.log', JSON.stringify(entry) + '\n', () => {});
    return res.json({ success: true, savedBg: !!bgPath, bgPath });
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
});

const PORT = process.env.PORT || 3000;

// serve static assets (models, backgrounds) if present
app.use('/models', express.static(path.join(__dirname, 'models')));
app.use('/backgrounds', express.static(path.join(__dirname, 'backgrounds')));
app.use(express.static(path.join(__dirname, 'public')));

// simple CORS for local dev
app.use((req,res,next)=>{ res.setHeader('Access-Control-Allow-Origin','*'); res.setHeader('Access-Control-Allow-Methods','GET,POST,OPTIONS'); res.setHeader('Access-Control-Allow-Headers','Content-Type'); next(); });

app.listen(PORT, ()=>console.log('Server on', PORT));
