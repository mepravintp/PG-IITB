import React, { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleImageChange = (e) => {
    setResult(null);
    setError('');
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!image) {
      setError('Please select an image.');
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);
    const formData = new FormData();
    formData.append('image', image);
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Prediction failed.');
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>🏏 Cricket Object Detection</h1>
      <p>Upload an image to detect bats, balls, and stumps using AI</p>
      <div className="main-box">
        <div className="upload-box">
          <h3>Image Upload</h3>
          <input type="file" accept="image/*" onChange={handleImageChange} />
          <button onClick={handleUpload} disabled={loading} style={{marginTop: '10px'}}>
            {loading ? 'Detecting...' : 'Detect'}
          </button>
          {image && (
            <div style={{marginTop: '10px'}}>
              <img src={URL.createObjectURL(image)} alt="preview" width={200} />
            </div>
          )}
        </div>
        <div className="result-box">
          <h3>Detection Results</h3>
          {error && <div className="error">{error}</div>}
          {result ? (
            <div>
              <p><b>Bat:</b> <span className="bat">{result.bat_count}</span></p>
              <p><b>Ball:</b> <span className="ball">{result.ball_count}</span></p>
              <p><b>Stumps:</b> <span className="stump">{result.stump_count}</span></p>
            </div>
          ) : (
            <p>No detections yet<br/>Upload an image and click detect</p>
          )}
          <div className="legend">
            <span className="bat">● Bat</span>
            <span className="ball">● Ball</span>
            <span className="stump">● Stumps</span>
          </div>
        </div>
      </div>
      <div className="footer">
        This application uses a custom-trained object detection model to identify cricket equipment. For best results, use clear images with visible cricket gear.
      </div>
    </div>
  );
}

export default App;
