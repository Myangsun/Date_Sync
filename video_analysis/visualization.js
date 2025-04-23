/**
 * Updates the radar chart showing emotional metrics at the current time
 * @param {number} time - Current time position
 * @param {Array} data1 - Metrics from video 1
 * @param {Array} data2 - Metrics from video 2
 */
function updateRadarChart(time, data1, data2) {
  // Check if we have data for this time
  if (!data1[time] || !data2[time]) return;

  // Get current metrics
  const metrics1 = {
    valence: data1[time].valence,
    comfort: data1[time].comfort,
    engagement: data1[time].happy + data1[time].surprise,
    calm: data1[time].neutral - data1[time].fear,
    interest: 1 - data1[time].sad,
  };

  const metrics2 = {
    valence: data2[time].valence,
    comfort: data2[time].comfort,
    engagement: data2[time].happy + data2[time].surprise,
    calm: data2[time].neutral - data2[time].fear,
    interest: 1 - data2[time].sad,
  };

  // If chart doesn't exist yet, create it
  if (!window.radarChart) {
    const ctx = document.getElementById("radar-chart").getContext("2d");

    window.radarChart = new Chart(ctx, {
      type: "radar",
      data: {
        labels: ["Valence", "Comfort", "Engagement", "Calm", "Interest"],
        datasets: [
          {
            label: "Person 1",
            data: Object.values(metrics1),
            backgroundColor: "rgba(255, 99, 132, 0.2)",
            borderColor: "rgba(255, 99, 132, 1)",
            borderWidth: 2,
          },
          {
            label: "Person 2",
            data: Object.values(metrics2),
            backgroundColor: "rgba(54, 162, 235, 0.2)",
            borderColor: "rgba(54, 162, 235, 1)",
            borderWidth: 2,
          },
        ],
      },
      options: {
        scales: {
          r: {
            min: -1,
            max: 1,
            ticks: {
              stepSize: 0.5,
            },
          },
        },
        plugins: {
          title: {
            display: true,
            text: `Emotional State at ${time}s`,
          },
        },
      },
    });
  } else {
    // Update existing chart
    window.radarChart.data.datasets[0].data = Object.values(metrics1);
    window.radarChart.data.datasets[1].data = Object.values(metrics2);
    window.radarChart.options.plugins.title.text = `Emotional State at ${time}s`;
    window.radarChart.update();
  }
}

/**
 * Creates a synchrony chart showing emotional alignment over time
 * @param {Array} data1 - Metrics from video 1
 * @param {Array} data2 - Metrics from video 2
 */
function setupSynchronyChart(data1, data2) {
  // Create synchrony data
  const syncData = [];
  const maxLength = Math.min(data1.length, data2.length);

  for (let i = 0; i < maxLength; i++) {
    syncData.push(calculateSyncScore(i, data1, data2));
  }

  // Get canvas and create chart
  const canvas = document.createElement("canvas");
  canvas.id = "synchrony-chart";
  document.querySelector(".charts-container").appendChild(canvas);

  const ctx = canvas.getContext("2d");
  window.synchronyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: Array.from({ length: maxLength }, (_, i) => i),
      datasets: [
        {
          label: "Emotional Synchrony",
          data: syncData,
          borderColor: "rgba(153, 102, 255, 1)",
          backgroundColor: "rgba(153, 102, 255, 0.2)",
          borderWidth: 2,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          min: 0,
          max: 100,
          title: {
            display: true,
            text: "Sync Score",
          },
        },
        x: {
          title: {
            display: true,
            text: "Time (seconds)",
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Emotional Synchrony Over Time",
        },
      },
    },
  });

  // Add time indicator plugin
  window.synchronyChart.pluginTimeIndicator = {
    id: "syncTimeIndicator",
    beforeDraw(chart) {
      if (chart.tooltip?._active?.length) return;

      const currentTime = parseInt(
        document.getElementById("time-slider").value
      );
      if (
        currentTime >= chart.scales.x.min &&
        currentTime <= chart.scales.x.max
      ) {
        const x = chart.scales.x.getPixelForValue(currentTime);
        const yAxis = chart.scales.y;
        const ctx = chart.ctx;

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(x, yAxis.top);
        ctx.lineTo(x, yAxis.bottom);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(0, 0, 0, 0.5)";
        ctx.stroke();
        ctx.restore();
      }
    },
  };

  window.synchronyChart.options.plugins.syncTimeIndicator = true;
  window.synchronyChart.update();
}

/**
 * Updates the numeric display of metrics
 * @param {number} time - Current time position
 * @param {Array} data1 - Metrics from video 1
 * @param {Array} data2 - Metrics from video 2
 */
function updateMetricsDisplay(time, data1, data2) {
  if (data1[time]) {
    document.getElementById("val1").textContent =
      data1[time].valence.toFixed(2);
    document.getElementById("com1").textContent =
      data1[time].comfort.toFixed(2);
    document.getElementById("eng1").textContent = (
      data1[time].happy + data1[time].surprise
    ).toFixed(2);
  }

  if (data2[time]) {
    document.getElementById("val2").textContent =
      data2[time].valence.toFixed(2);
    document.getElementById("com2").textContent =
      data2[time].comfort.toFixed(2);
    document.getElementById("eng2").textContent = (
      data2[time].happy + data2[time].surprise
    ).toFixed(2);
  }
}

/**
 * Updates the synchrony score display
 * @param {number} time - Current time position
 * @param {Array} data1 - Metrics from video 1
 * @param {Array} data2 - Metrics from video 2
 */
function updateSyncScore(time, data1, data2) {
  const syncScore = calculateSyncScore(time, data1, data2);
  const syncDisplay = document.getElementById("sync-score");

  if (syncDisplay) {
    syncDisplay.textContent = `${syncScore}%`;

    // Update color based on score
    if (syncScore >= 80) {
      syncDisplay.style.color = "#4CAF50"; // Green
    } else if (syncScore >= 50) {
      syncDisplay.style.color = "#FFC107"; // Yellow
    } else {
      syncDisplay.style.color = "#F44336"; // Red
    }
  }

  return syncScore;
}

/**
 * Calculate real-time sync score based on emotional alignment
 * @param {number} time - Current time position
 * @param {Array} data1 - Metrics from video 1
 * @param {Array} data2 - Metrics from video 2
 * @returns {number} - Synchrony score between 0-100
 */
function calculateSyncScore(time, data1, data2) {
  if (!data1[time] || !data2[time]) return 0;

  // Calculate similarity between emotional metrics
  const valenceSync =
    1 - Math.abs(data1[time].valence - data2[time].valence) / 2;
  const comfortSync =
    1 - Math.abs(data1[time].comfort - data2[time].comfort) / 2;
  const engagementSync =
    1 -
    Math.abs(
      data1[time].happy +
        data1[time].surprise -
        (data2[time].happy + data2[time].surprise)
    ) /
      2;

  // Weight the metrics
  const syncScore =
    (valenceSync * 0.4 + comfortSync * 0.3 + engagementSync * 0.3) * 100;
  return Math.round(syncScore);
}

// Export functions for use in main script
window.DateSyncViz = {
  initVisualizations,
  updateRadarChart,
  calculateSyncScore,
  updateSyncScore,
};
