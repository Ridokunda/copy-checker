const express = require('express');
const multer = require('multer');
const path = require('path');
const router = express.Router();

// Set storage for uploaded files
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ storage: storage });

router.post('/upload', upload.array('files'), (req, res) => {
  if (!req.files) {
    return res.status(400).send('No files uploaded.');
  }
  res.status(200).json({
    message: 'Files uploaded successfully.',
    files: req.files.map(f => f.filename)
  });
});

module.exports = router;
