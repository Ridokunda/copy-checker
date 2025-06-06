import React from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';

function App() {
  return (
    <div style={styles.pageWrapper}>
      <Header />
      <FileUpload />
    </div>
  );
}

const styles = {
  pageWrapper: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    minHeight: '100vh',
    width: '100%',
    backgroundColor: '#f0f2f5'
  }
};

export default App;


