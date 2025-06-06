import React, { useState } from 'react';
import axios from 'axios';

function FileUpload() {
  const [originalFile, setOriginalFile] = useState(null);
  const [suspectFile, setSuspectFile] = useState(null);
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleOriginalChange = (e) => {
    setOriginalFile(e.target.files[0]);
    setMessage('');
  };

  const handleSuspectChange = (e) => {
    setSuspectFile(e.target.files[0]);
    setMessage('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!originalFile || !suspectFile) {
      return setMessage("Please upload both the original and suspect Java files.");
    }

    const formData = new FormData();
    formData.append("files", originalFile);
    formData.append("files", suspectFile);

    setIsLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/api/files/upload", formData);
      setMessage(`Files uploaded: ${res.data.files.join(", ")}`);
    } catch (err) {
      console.error(err);
      setMessage("Upload failed. Check the console or backend.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={{ color: 'black' }}>Upload Java Files for Comparison</h2>
      <form onSubmit={handleSubmit} style={styles.form}>
        <label>File 1:</label>
        <input type="file" accept=".java" onChange={handleOriginalChange} />
        {originalFile && <p>{originalFile.name}</p>}

        <label>File 2:</label>
        <input type="file" accept=".java" onChange={handleSuspectChange} />
        {suspectFile && <p>{suspectFile.name}</p>}

        <button type="submit" disabled={!originalFile || !suspectFile || isLoading}>
          {isLoading ? 'Uploading...' : 'Upload Files'}
        </button>
        {originalFile && suspectFile && !isLoading && (
        <button
            type="button"
            //onClick={() => alert("Compare functionality coming soon!")}
            style={{ marginTop: '10px' }}
        >
            Compare Files
        </button>
        )}

      </form>
      {message && <p style={styles.message}>{message}</p>}
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
  }
};

export default FileUpload;
