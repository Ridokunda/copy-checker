import React, { useState } from 'react';
import axios from 'axios';

function FileUpload() {
  const [files, setFiles] = useState([]);
  const [message, setMessage] = useState("");

  const handleChange = (e) => {
    setFiles([...e.target.files]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (files.length !== 2) {
      return setMessage("Please upload exactly 2 Java files.");
    }

    const formData = new FormData();
    files.forEach(file => formData.append("files", file));

    try {
      const res = await axios.post("http://localhost:5000/api/files/upload", formData);
      setMessage(`Uploaded: ${res.data.files.join(", ")}`);
    } catch (err) {
      console.error(err);
      setMessage("Upload failed. Check console or backend.");
    }
  };

  return (
    <div>
      <h2>Upload Java Files</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept=".java" multiple onChange={handleChange} />
        <button type="submit">Upload</button>
      </form>
      <p>{message}</p>
    </div>
  );
}

export default FileUpload;
