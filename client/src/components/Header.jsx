import React from 'react';

function Header() {
  return (
    <header style={styles.header}>
      <h1 style={styles.logo}>Copy Checker</h1>
      <div style={styles.dropdown}>
        <button style={styles.dropbtn}>Menu ▾</button>
        <div style={styles.dropdownContent}>
          <a href="#">About</a>
          <a href="#">Help</a>
          <a href="#">Logout</a>
        </div>
      </div>
    </header>
  );
}

const styles = {
  
  header: {
    width: '100%',                 
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 32px',
    backgroundColor: '#343a40',
    color: 'white',
    boxSizing: 'border-box'       
    },

  logo: {
    margin: 0,
  },
  dropbtn: {
    backgroundColor: '#343a40',
    color: 'white',
    padding: '8px 12px',
    border: 'none',
    cursor: 'pointer',
    fontSize: '16px',
  },
  dropdown: {
    position: 'relative',
    display: 'inline-block',
  },
  dropdownContent: {
    display: 'none',
    position: 'absolute',
    backgroundColor: '#f9f9f9',
    minWidth: '120px',
    boxShadow: '0px 8px 16px 0px rgba(0,0,0,0.2)',
    zIndex: 1,
    textAlign: 'left',
  },
  dropdownHover: {
    display: 'block',
  }
};

styles.dropdown[':hover'] = {
  ...styles.dropdown,
  dropdownContent: {
    display: 'block'
  }
};

export default Header;
