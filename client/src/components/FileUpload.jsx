import React, { useState } from 'react';
import axios from 'axios';

function FileUpload() {
  const [uploadMode, setUploadMode] = useState('individual'); // 'individual' or 'batch'
  const [originalFile, setOriginalFile] = useState(null);
  const [suspectFile, setSuspectFile] = useState(null);
  const [zipFile, setZipFile] = useState(null);
  const [message, setMessage] = useState('');
  const [result, setResult] = useState(null);
  const [batchReport, setBatchReport] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleModeChange = (mode) => {
    setUploadMode(mode);
    setOriginalFile(null);
    setSuspectFile(null);
    setZipFile(null);
    setMessage('');
    setResult(null);
    setBatchReport(null);
  };

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

  const handleZipChange = (e) => {
    setZipFile(e.target.files[0]);
    setMessage('');
    setBatchReport(null);
  };

  const handleIndividualSubmit = async (e) => {
    e.preventDefault();
    if (!originalFile || !suspectFile) {
      return setMessage("Please upload both the original and suspect Java files.");
    }

    const formData = new FormData();
    formData.append("original", originalFile);
    formData.append("suspect", suspectFile);

    setIsLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/api/nn-predict", formData);
      console.log("Raw response:", res.data); 
      setResult(res.data);
      setMessage("Prediction complete.");
    } catch (err) {
      console.error(err);
      setMessage("Prediction failed. Check the server.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleBatchSubmit = async (e) => {
    e.preventDefault();
    if (!zipFile) {
      return setMessage("Please upload a ZIP file containing Java files.");
    }

    const formData = new FormData();
    formData.append("zipfile", zipFile);

    setIsLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/api/batch-upload", formData);
      console.log("Batch response:", res.data);
      setBatchReport(res.data.report);
      setMessage("Batch analysis complete.");
    } catch (err) {
      console.error(err);
      setMessage("Batch analysis failed. Check the server.");
    } finally {
      setIsLoading(false);
    }
  };

  const renderBatchResults = () => {
    if (!batchReport) return null;

    const plagiarizedPairs = batchReport.comparisons.filter(comp => comp.prediction === 1);

    return (
      <div style={styles.resultBox}>
        <h3>Batch Analysis Report</h3>
        <p><strong>Total Files Analyzed:</strong> {batchReport.totalFiles}</p>
        <p><strong>Total Comparisons:</strong> {batchReport.comparisons.length}</p>
        <p><strong>Plagiarized Pairs Found:</strong> {plagiarizedPairs.length}</p>
        <p><strong>Submitted At:</strong> {new Date(batchReport.submittedAt).toLocaleString()}</p>

        {plagiarizedPairs.length > 0 && (
          <div style={styles.plagiarizedSection}>
            <h4 style={{ color: '#d32f2f' }}> Plagiarized Pairs:</h4>
            {plagiarizedPairs.map((comp, index) => (
              <div key={index} style={styles.plagiarizedPair}>
                <strong>{comp.file1} ↔ {comp.file2}</strong>
                {comp.confidence && (
                  <div style={{ fontSize: '0.9em', marginTop: '4px' }}>
                    <span>Plagiarized: {(comp.confidence[1] * 100).toFixed(1)}%</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        <details style={{ marginTop: '15px' }}>
          <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>
            View All Comparisons ({batchReport.comparisons.length})
          </summary>
          <div style={styles.allComparisons}>
            {batchReport.comparisons.map((comp, index) => (
              <div key={index} style={{
                ...styles.comparisonItem,
                backgroundColor: comp.prediction === 1 ? '#ffebee' : '#f1f8e9'
              }}>
                <div><strong>{comp.file1} ↔ {comp.file2}</strong></div>
                <div>Status: {comp.prediction === 1 ? 'Plagiarized' : 'Not Plagiarized'}</div>
                {comp.confidence && (
                  <div style={{ fontSize: '0.85em' }}>
                    Not Plagiarized: {(comp.confidence[0] * 100).toFixed(1)}% | 
                    Plagiarized: {(comp.confidence[1] * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            ))}
          </div>
        </details>
      </div>
    );
  };

  return (
    <div style={styles.container}>
      <h2 style={{ color: 'black' }}>Java Plagiarism Detection</h2>
      
      {/* Mode Selection */}
      <div style={styles.modeSelector}>
        <h3>Choose Upload Mode:</h3>
        <div style={styles.modeButtons}>
          <button 
            type="button"
            style={{
              ...styles.modeButton,
              ...(uploadMode === 'individual' ? styles.activeMode : {})
            }}
            onClick={() => handleModeChange('individual')}
          >
            Compare Two Files
          </button>
          <button 
            type="button"
            style={{
              ...styles.modeButton,
              ...(uploadMode === 'batch' ? styles.activeMode : {})
            }}
            onClick={() => handleModeChange('batch')}
          >
            Batch Analysis (ZIP)
          </button>
        </div>
        <p style={styles.modeDescription}>
          {uploadMode === 'individual' 
            ? 'Upload two Java files to compare them for plagiarism'
            : 'Upload a ZIP file containing multiple Java files to analyze all possible pairs'
          }
        </p>
      </div>

      {/* Individual File Upload Form */}
      {uploadMode === 'individual' && (
        <form onSubmit={handleIndividualSubmit} style={styles.form}>
          <label>Original File:</label>
          <input type="file" accept=".java" onChange={handleOriginalChange} />
          {originalFile && <p style={styles.fileName}>{originalFile.name}</p>}

          <label>Suspect File:</label>
          <input type="file" accept=".java" onChange={handleSuspectChange} />
          {suspectFile && <p style={styles.fileName}>{suspectFile.name}</p>}

          <button type="submit" disabled={!originalFile || !suspectFile || isLoading} style={styles.submitButton}>
            {isLoading ? 'Comparing...' : 'Compare Files'}
          </button>
        </form>
      )}

      {/* Batch Upload Form */}
      {uploadMode === 'batch' && (
        <form onSubmit={handleBatchSubmit} style={styles.form}>
          <label>ZIP File (containing .java files):</label>
          <input type="file" accept=".zip" onChange={handleZipChange} />
          {zipFile && <p style={styles.fileName}>{zipFile.name}</p>}
          <p style={styles.note}>
            <strong>Note:</strong> ZIP file must contain at least 2 Java files. Each file will be compared with every other file.
          </p>

          <button type="submit" disabled={!zipFile || isLoading} style={styles.submitButton}>
            {isLoading ? 'Analyzing...' : 'Analyze Batch'}
          </button>
        </form>
      )}

      {message && <p style={styles.message}>{message}</p>}

      {/* Individual Comparison Result */}
      {uploadMode === 'individual' && result && (
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

      {/* Batch Analysis Results */}
      {uploadMode === 'batch' && renderBatchResults()}
    </div>
  );
}

const styles = {
  container: {
    width: '80%',
    maxWidth: '900px',
    marginTop: '40px',
    padding: '20px',
    border: '1px solid #ddd',
    borderRadius: '8px',
    fontFamily: 'Arial, sans-serif',
    backgroundColor: '#f9f9f9',
    color: 'black'
  },
  modeSelector: {
    marginBottom: '30px',
    padding: '15px',
    backgroundColor: '#fff',
    borderRadius: '6px',
    border: '1px solid #e0e0e0'
  },
  modeButtons: {
    display: 'flex',
    gap: '10px',
    marginBottom: '10px'
  },
  modeButton: {
    color: '#007bff',
    padding: '10px 20px',
    border: '2px solid #ddd',
    borderRadius: '5px',
    backgroundColor: '#fff',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold'
  },
  activeMode: {
    backgroundColor: '#007bff',
    color: 'white',
    borderColor: '#007bff'
  },
  modeDescription: {
    fontSize: '14px',
    color: '#666',
    fontStyle: 'italic'
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    marginTop: '10px'
  },
  fileName: {
    fontSize: '14px',
    color: '#007bff',
    fontStyle: 'italic'
  },
  note: {
    fontSize: '13px',
    color: '#666',
    backgroundColor: '#fff3cd',
    padding: '8px',
    borderRadius: '4px',
    border: '1px solid #ffeaa7'
  },
  submitButton: {
    padding: '12px',
    fontSize: '16px',
    fontWeight: 'bold',
    backgroundColor: '#007bff',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer'
  },
  message: {
    marginTop: '10px',
    fontWeight: 'bold'
  },
  resultBox: {
    marginTop: '20px',
    padding: '15px',
    border: '1px solid #ccc',
    borderRadius: '6px',
    backgroundColor: '#eef'
  },
  plagiarizedSection: {
    marginTop: '15px',
    padding: '10px',
    backgroundColor: '#ffebee',
    borderRadius: '5px',
    border: '1px solid #ffcdd2'
  },
  plagiarizedPair: {
    padding: '8px',
    marginBottom: '8px',
    backgroundColor: 'white',
    borderRadius: '4px',
    border: '1px solid #ffcdd2'
  },
  allComparisons: {
    marginTop: '10px',
    maxHeight: '300px',
    overflowY: 'auto'
  },
  comparisonItem: {
    padding: '8px',
    marginBottom: '5px',
    borderRadius: '4px',
    border: '1px solid #ddd',
    fontSize: '14px'
  }
};

export default FileUpload;