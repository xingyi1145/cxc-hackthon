
import React, { useState } from 'react';
import { Resizable } from 're-resizable';

const Sidebar = ({ sidebarWidth, setSidebarWidth, setCoordinates }) => {
  const [address, setAddress] = useState('');

  const handleSearch = () => {
    if (address) {
      fetch(`http://127.0.0.1:5000/geocode?address=${encodeURIComponent(address)}`)
        .then(response => response.json())
        .then(data => {
          if (data.latitude && data.longitude) {
            setCoordinates([data.longitude, data.latitude]);
          } else {
            console.error('Geocoding error:', data.error);
          }
        })
        .catch(error => console.error('Error fetching geocoding data:', error));
    }
  };

  const resizableStyle = {
    display: 'flex',
    flexDirection: 'column',
    padding: '20px',
    backgroundColor: '#f0f0f0',
    fontFamily: 'Arial, sans-serif',
    borderLeft: '2px solid #ccc',
    overflow: 'auto',
    position: 'absolute',
    top: 0,
    right: 0,
    zIndex: 1000,
  };

  const inputContainerStyle = {
    marginBottom: '15px',
  };

  const labelStyle = {
    display: 'block',
    marginBottom: '5px',
    fontWeight: 'bold',
  };

  const inputStyle = {
    width: '100%',
    padding: '8px',
    boxSizing: 'border-box',
  };

  const buttonStyle = {
    width: '100%',
    padding: '10px',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    cursor: 'pointer',
    fontSize: '16px',
  };

  return (
    <Resizable
      style={resizableStyle}
      size={{ width: sidebarWidth, height: '100vh' }}
      enable={{
        top: false,
        right: false,
        bottom: false,
        left: true,
        topRight: false,
        bottomRight: false,
        bottomLeft: false,
        topLeft: false,
      }}
      onResizeStop={(e, direction, ref, d) => {
        setSidebarWidth(ref.offsetWidth);
      }}
    >
      <h2>Address Search</h2>
      <div style={inputContainerStyle}>
        <label style={labelStyle}>
          Address:
          <input style={inputStyle} type="text" value={address} onChange={(e) => setAddress(e.target.value)} />
        </label>
      </div>
      <button style={buttonStyle} onClick={handleSearch}>Search</button>
    </Resizable>
  );
};

export default Sidebar;

