import React, { useState } from 'react';
import axios from 'axios';

function FileUpload() {
  const [originalFile, setOriginalFile] = useState(null);
  const [suspectFile, setSuspectFile] = useState(null);
  const [message, setMessage] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleOriginalChange = (e) => {
    setOriginalFile(e.target.files[0]);
    setMessage('');
    setResult(null);
  };

  const handleSuspectChange = (e) => {
    setSuspectFile(e.target.files[0]);
    setMessage('');
    setResult(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!originalFile || !suspectFile) {
      return setMessage("Please upload both the original and suspect Java files.");
    }

    const formData = new FormData();
    formData.append("original", originalFile);
    formData.append("suspect", suspectFile);

    setIsLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/api/predict", formData);
      setResult(res.data);
      setMessage("Prediction complete.");
    } catch (err) {
      console.error(err);
      setMessage("Prediction failed. Check the server.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={{ color: 'black' }}>Upload Java Files for Comparison</h2>
      <form onSubmit={handleSubmit} style={styles.form}>
        <label>Original File:</label>
        <input type="file" accept=".java" onChange={handleOriginalChange} />
        {originalFile && <p>{originalFile.name}</p>}

        <label>Suspect File:</label>
        <input type="file" accept=".java" onChange={handleSuspectChange} />
        {suspectFile && <p>{suspectFile.name}</p>}

        <button type="submit" disabled={!originalFile || !suspectFile || isLoading}>
          {isLoading ? 'Comparing...' : 'Compare Files'}
        </button>
      </form>

      {message && <p style={styles.message}>{message}</p>}

      {result && (
        <div style={styles.resultBox}>
          <h3>Prediction: {result.prediction === 1 ? 'Plagiarized' : 'Not Plagiarized'}</h3>
          {result.confidence ? (
            <p>Confidence:
              <br /> Not Plagiarized: {(result.confidence[0] * 100).toFixed(2)}%
              <br /> Plagiarized: {(result.confidence[1] * 100).toFixed(2)}%
            </p>
          ) : (
            <p>No confidence score available.</p>
          )}
        </div>
      )}

    </div>
  );
}

const styles = {
  container: {
    width: '80%',
    maxWidth: '800px',
    marginTop: '40px',
    padding: '20px',
    border: '1px solid #ddd',
    borderRadius: '8px',
    fontFamily: 'Arial, sans-serif',
    backgroundColor: '#f9f9f9',
    color: 'black'
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    marginTop: '10px'
  },
  message: {
    marginTop: '10px',
    fontWeight: 'bold'
  },
  resultBox: {
    marginTop: '20px',
    padding: '10px',
    border: '1px solid #ccc',
    borderRadius: '6px',
    backgroundColor: '#eef'
  }
};

export default FileUpload;
