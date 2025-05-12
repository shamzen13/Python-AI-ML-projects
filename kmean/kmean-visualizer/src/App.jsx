import React, { useState } from 'react';
import './App.css';

function App() {
  const [points, setPoints] = useState([]);
  const [clusterCenters, setClusterCenters] = useState([]);
  const [clusterAssignments, setClusterAssignments] = useState([]);
  const [inInsertClusterMode, setInInsertClusterMode] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleGridClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    console.log(`Click coordinates: x=${x}, y=${y}`); 

    if (inInsertClusterMode) {
      setClusterCenters([...clusterCenters, { x, y }]);
    } else {
      setPoints([...points, { x, y }]);
    }
  };

  const calculateClusters = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/cluster', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          points,
          clusterCenters
        }),
      });
      const data = await response.json();
      setClusterAssignments(data.assignments);
    } catch (error) {
      console.error('Error fetching cluster data:', error);
    }
    setLoading(false);
  };

  // Dynamically generate unique colors for each cluster center
  const clusterColors = clusterCenters.map((_, i) =>
    `hsl(${(i * 360) / clusterCenters.length}, 100%, 50%)`
  );

  return (
    <div className="App">
      <h1>K-Means Clustering</h1>

      <button onClick={() => setInInsertClusterMode(!inInsertClusterMode)}>
        {inInsertClusterMode ? 'Switch to Add Points' : 'Insert Cluster'}
      </button>

      <button onClick={calculateClusters} disabled={loading}>
        {loading ? 'Clustering...' : 'Calculate Clusters'}
      </button>

      <button onClick={() => {
        setPoints([]);
        setClusterCenters([]);
        setClusterAssignments([]);
      }}>
        Clear All
      </button>

      <div className="grid" onClick={handleGridClick} style={{ position: 'relative', width: '600px', height: '600px', border: '1px solid #ccc' }}>

        <div style={{ position: 'absolute', top: '50%', left: 0, width: '100%', height: '1px', backgroundColor: '#888' }} />
        <div style={{ position: 'absolute', left: '50%', top: 0, width: '1px', height: '100%', backgroundColor: '#888' }} />


        {/* Points */}
{points.map((point, index) => (
  <div
    key={`point-${index}`}
    className="point"
    style={{
      left: point.x - 5,
      top: point.y - 5, 
      width: '10px', 
      height: '10px', 
      borderRadius: '50%', 
      backgroundColor: clusterAssignments[index] !== undefined
        ? clusterColors[clusterAssignments[index] % clusterColors.length]
        : 'black',
      position: 'absolute',
    }}
  ></div>
))}


        {/* Cluster Centers */}
        {clusterCenters.map((center, index) => (
          <div
            key={`center-${index}`}
            className="cluster-center"
            style={{
              left: center.x - 10, 
              top: center.y - 10,
              width: '20px', 
              height: '20px', 
              backgroundColor: clusterColors[index % clusterColors.length],
              position: 'absolute',
              borderRadius: '50%', 
            }}
          ></div>
        ))}
      </div>
    </div>
  );
}

export default App;
