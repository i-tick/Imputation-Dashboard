// client/src/App.js
import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';

function App() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [selectedColumn, setSelectedColumn] = useState("");
  const [iterations, setIterations] = useState(1);
  const [result, setResult] = useState(null);
  const d3Container = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    setFile(selectedFile);

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const parsed = d3.csvParse(text);
      if (parsed.length > 0) {
        setColumns(Object.keys(parsed[0]));
      } else {
        setColumns([]);
      }
    };
    reader.readAsText(selectedFile);
  };

  const handleSubmit = async () => {
    if (!file || !selectedColumn)
      return alert("Select a file and a column first.");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("column", selectedColumn);
    formData.append("iterations", iterations);

    try {
      const resp = await fetch("http://localhost:8000/api/impute", {
        method: "POST",
        body: formData
      });
      const json = await resp.json();
      if (json.error) {
        alert(json.error);
      } else {
        setResult(json);
      }
    } catch (err) {
      console.error(err);
      alert("Something went wrong calling the backend.");
    }
  };

  useEffect(() => {
    if (!result || !d3Container.current) return;

    d3.select(d3Container.current).selectAll("*").remove();

    const orig = result.orig_values.map((d) => +d);
    const imp = result.imputed_values.map((d) => +d);
    const all = orig.concat(imp);

    const width = 600;
    const height = 400;
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };

    const svg = d3
      .select(d3Container.current)
      .append("svg")
      .attr("width", width)
      .attr("height", height);

    const x = d3
      .scaleLinear()
      .domain(d3.extent(all))
      .nice()
      .range([margin.left, width - margin.right]);

    const binsOrig = d3
      .histogram()
      .domain(x.domain())
      .thresholds(x.ticks(30))(orig);

    const binsImp = d3
      .histogram()
      .domain(x.domain())
      .thresholds(x.ticks(30))(imp);

    const y = d3
      .scaleLinear()
      .domain([
        0,
        d3.max([
          d3.max(binsOrig, (d) => d.length),
          d3.max(binsImp, (d) => d.length),
        ]),
      ])
      .nice()
      .range([height - margin.bottom, margin.top]);

    // Combine bins by x0/x1
    const bins = binsOrig.map((bin, i) => ({
      x0: bin.x0,
      x1: bin.x1,
      orig: bin.length,
      imp: binsImp[i] ? binsImp[i].length : 0,
    }));

    // Draw stacked bars
    svg
      .append("g")
      .selectAll("g")
      .data(bins)
      .join("g")
      .each(function (d) {
        // Blue (original) bar
        d3.select(this)
          .append("rect")
          .attr("x", x(d.x0) + 1)
          .attr("y", y(d.orig))
          .attr("width", Math.max(0, x(d.x1) - x(d.x0) - 1))
          .attr("height", y(0) - y(d.orig))
          .attr("fill", "steelblue")
          .attr("opacity", 0.6);

        // Orange (imputed) bar, stacked on top
        d3.select(this)
          .append("rect")
          .attr("x", x(d.x0) + 1)
          .attr("y", y(d.orig + d.imp))
          .attr("width", Math.max(0, x(d.x1) - x(d.x0) - 1))
          .attr("height", y(d.orig) - y(d.orig + d.imp))
          .attr("fill", "orange")
          .attr("opacity", 0.6);
      });

    // X axis
    svg
      .append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x));

    // Y axis
    svg
      .append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y));

    // Legend
    const legend = svg
      .append("g")
      .attr("transform", `translate(${width - margin.right - 140}, ${margin.top})`);

    legend
      .append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", 15)
      .attr("height", 15)
      .attr("fill", "steelblue")
      .attr("opacity", 0.6);
    legend
      .append("text")
      .attr("x", 20)
      .attr("y", 12)
      .text("Original Values");

    legend
      .append("rect")
      .attr("x", 0)
      .attr("y", 20)
      .attr("width", 15)
      .attr("height", 15)
      .attr("fill", "orange")
      .attr("opacity", 0.6);
    legend
      .append("text")
      .attr("x", 20)
      .attr("y", 32)
      .text("Imputed Values");
  }, [result]);

  return (
    <div style={{ padding: "20px" }}>
      <h1>CSV Imputation Dashboard</h1>

      <div style={{ marginBottom: "1rem" }}>
        <input type="file" accept=".csv" onChange={handleFileChange} />
      </div>

      {columns.length > 0 && (
        <div style={{ marginBottom: "1rem" }}>
          <label style={{ marginRight: "1rem" }}>
            Select column to impute:
            <select
              value={selectedColumn}
              onChange={(e) => setSelectedColumn(e.target.value)}
              style={{ marginLeft: "0.5rem" }}
            >
              <option value="">--Select--</option>
              {columns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </label>

          <label style={{ marginRight: "1rem" }}>
            Number of iterations:
            <input
              type="number"
              min="1"
              value={iterations}
              onChange={(e) => setIterations(+e.target.value)}
              style={{ marginLeft: "0.5rem", width: "60px" }}
            />
          </label>

          <button onClick={handleSubmit}>Submit</button>
        </div>
      )}

      {/* D3 will mount here */}
      <div ref={d3Container}></div>
    </div>
  );
}

export default App;
