// Global variables
let data1 = [],
  data2 = [];
let compatibilityScore = 0;

// DOM elements - wait for DOM to be fully loaded
document.addEventListener("DOMContentLoaded", function () {
  const timeSlider = document.getElementById("time-slider");
  const currentTimeDisplay = document.getElementById("current-time");
  const video1Upload = document.getElementById("video1-upload");
  const video2Upload = document.getElementById("video2-upload");
  const analyzeBtn = document.getElementById("analyze-btn");
  const statusIndicator = document.getElementById("status-indicator");
  const playBtn = document.getElementById("play-btn");
  const v1 = document.getElementById("video1");
  const v2 = document.getElementById("video2");
  const compatibilityScoreEl = document.getElementById("compatibility-score");
  const detailedAnalysisEl = document.getElementById("detailed-analysis");

  // Enable analyze button when both videos are uploaded
  function checkUploadStatus() {
    console.log("Checking upload status...");
    if (video1Upload.files.length > 0 && video2Upload.files.length > 0) {
      analyzeBtn.disabled = false;
    } else {
      analyzeBtn.disabled = true;
    }
  }

  // Add event listeners for file uploads
  video1Upload.addEventListener("change", function (e) {
    if (e.target.files.length > 0) {
      const file = e.target.files[0];
      v1.src = URL.createObjectURL(file);
      checkUploadStatus();
    }
  });

  video2Upload.addEventListener("change", function (e) {
    if (e.target.files.length > 0) {
      const file = e.target.files[0];
      v2.src = URL.createObjectURL(file);
      checkUploadStatus();
    }
  });

  // Handle form submission
  analyzeBtn.addEventListener("click", function () {
    if (video1Upload.files.length == 0 || video2Upload.files.length == 0) {
      return;
    }

    // Show status indicator
    statusIndicator.style.display = "block";
    statusIndicator.innerHTML =
      "Analysis may take several minutes to complete. Please wait...";
    statusIndicator.style.backgroundColor = "#fff3cd";

    // Add progress bar
    statusIndicator.innerHTML +=
      '<div class="progress-bar"><div class="progress" id="progress-bar"></div></div>';
    const progressBar = document.getElementById("progress-bar");

    // Create FormData and upload videos
    const formData = new FormData();
    formData.append("video1", video1Upload.files[0]);
    formData.append("video2", video2Upload.files[0]);

    // Send data to server
    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Upload successful:", data);

        // Start polling for status
        const statusInterval = setInterval(() => {
          fetch("/status")
            .then((res) => res.json())
            .then((statusData) => {
              // Update progress bar
              progressBar.style.width = statusData.progress + "%";

              // Update status message based on progress
              if (statusData.progress < 50) {
                statusIndicator.innerHTML = `Analyzing videos (${statusData.progress}%)... This may take several minutes.`;
                statusIndicator.innerHTML +=
                  '<div class="progress-bar"><div class="progress" id="progress-bar" style="width: ' +
                  statusData.progress +
                  '%;"></div></div>';
              } else if (statusData.progress < 90) {
                statusIndicator.innerHTML = `Calculating compatibility (${statusData.progress}%)...`;
                statusIndicator.innerHTML +=
                  '<div class="progress-bar"><div class="progress" id="progress-bar" style="width: ' +
                  statusData.progress +
                  '%;"></div></div>';
              } else {
                statusIndicator.innerHTML = `Finishing up (${statusData.progress}%)...`;
                statusIndicator.innerHTML +=
                  '<div class="progress-bar"><div class="progress" id="progress-bar" style="width: ' +
                  statusData.progress +
                  '%;"></div></div>';
              }

              // Check if processing is complete
              if (statusData.status === "completed") {
                clearInterval(statusInterval);
                loadResults();
                statusIndicator.style.display = "none";
              } else if (statusData.status === "error") {
                clearInterval(statusInterval);
                statusIndicator.innerHTML =
                  "Error during analysis: " + statusData.message;
                statusIndicator.style.backgroundColor = "#f8d7da";
              }
            })
            .catch((err) => {
              console.error("Error checking status:", err);
            });
        }, 2000); // Check every 2 seconds
      })
      .catch((error) => {
        console.error("Error uploading videos:", error);
        statusIndicator.innerHTML = "Error uploading videos. Please try again.";
        statusIndicator.style.backgroundColor = "#f8d7da";
      });
  });

  // Format the feedback text to improve readability
  function formatFeedback(text) {
    // Split text into paragraphs
    const paragraphs = text.split(/\n\n|\r\n\r\n/);

    let formattedHtml = "";

    paragraphs.forEach((paragraph) => {
      // Check if paragraph is a numbered point
      if (/^\d+\.\s/.test(paragraph)) {
        // It's a numbered point, make it a list item
        const points = paragraph.split(/\d+\.\s/);
        points.shift(); // Remove the empty first element

        if (points.length > 0) {
          formattedHtml += '<ol class="feedback-list">';
          points.forEach((point) => {
            formattedHtml += `<li>${point.trim()}</li>`;
          });
          formattedHtml += "</ol>";
        }
      } else if (paragraph.trim()) {
        // Regular paragraph
        formattedHtml += `<p class="feedback-paragraph">${paragraph.trim()}</p>`;
      }
    });

    return formattedHtml;
  }

  // Format the compatibility analysis in a structured way
  function formatCompatibilityAnalysis(analysisText) {
    // Look for section headers (### Title)
    const sections = analysisText.split(/###\s+([^\n]+)/);

    if (sections.length <= 1) {
      // If no sections are found, return the text as a single section
      return `<div class="analysis-section">
                <p>${analysisText}</p>
              </div>`;
    }

    let formattedHtml = "";

    // Skip the first element which is empty text before the first header
    for (let i = 1; i < sections.length; i += 2) {
      const title = sections[i];
      const content = sections[i + 1] || "";

      // Format content - split by bullet points
      let contentHtml = "";
      const points = content.split(/\s*-\s+/);

      points.forEach((point, index) => {
        if (index === 0 && !point.trim()) return; // Skip empty first element

        if (point.trim()) {
          contentHtml += `<p class="analysis-point">${point.trim()}</p>`;
        }
      });

      formattedHtml += `
        <div class="analysis-section">
          <h3 class="analysis-title">${title}</h3>
          <div class="analysis-content">
            ${contentHtml}
          </div>
        </div>`;
    }

    return formattedHtml;
  }

  // Load results after processing is complete
  function loadResults() {
    console.log("Loading results...");

    statusIndicator.style.display = "block";
    statusIndicator.innerHTML = "Loading analysis results...";
    statusIndicator.style.backgroundColor = "#d4edda";

    // Load all data
    Promise.all([
      fetch("output/video_1/metrics_data.json").then((res) =>
        res.json().catch(() => ({}))
      ),
      fetch("output/video_2/metrics_data.json").then((res) =>
        res.json().catch(() => ({}))
      ),
      fetch("output/video_1/crossmodal_analysis.txt").then((res) =>
        res.text().catch(() => "")
      ),
      fetch("output/video_2/crossmodal_analysis.txt").then((res) =>
        res.text().catch(() => "")
      ),
      fetch("output/compatibility_score.json").then((res) =>
        res.json().catch(() => ({ score: 0 }))
      ),
      fetch("output/compatibility_analysis.txt").then((res) =>
        res.text().catch(() => "")
      ),
    ])
      .then(
        ([
          metrics1,
          metrics2,
          analysis1,
          analysis2,
          scoreData,
          detailedAnalysis,
        ]) => {
          console.log("Data loaded successfully");

          // Prepare time series data
          data1 = prepareTimeSeriesData(metrics1);
          data2 = prepareTimeSeriesData(metrics2);

          // Display analysis text with better formatting
          document.getElementById("text1").innerHTML =
            formatFeedback(analysis1);
          document.getElementById("text2").innerHTML =
            formatFeedback(analysis2);

          // Display compatibility score with original styling
          compatibilityScore = scoreData.score;
          compatibilityScoreEl.innerHTML = `Date Compatibility Score: ${compatibilityScore}/100`;

          // Display detailed compatibility analysis with better formatting
          detailedAnalysisEl.innerHTML = `
          <h2>Compatibility Analysis</h2>
          <div class="compatibility-metrics">
            <div class="metric-item">
              <span class="metric-label">Emotional Synchrony</span>
              <span class="metric-value">${
                scoreData.metrics?.emotional_synchrony?.toFixed(2) || "N/A"
              }</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">Comfort Synchrony</span>
              <span class="metric-value">${
                scoreData.metrics?.comfort_synchrony?.toFixed(2) || "N/A"
              }</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">Engagement Balance</span>
              <span class="metric-value">${
                scoreData.metrics?.engagement_balance?.toFixed(2) || "N/A"
              }</span>
            </div>
          </div>
          <div class="analysis-content">
            ${formatCompatibilityAnalysis(detailedAnalysis)}
          </div>`;

          // Create charts for each video
          createVideoChart("chart1", data1, "Video 1 Emotional Metrics");
          createVideoChart("chart2", data2, "Video 2 Emotional Metrics");

          // Update time slider max value based on video duration
          const maxTime = Math.max(
            data1.length > 0 ? Math.max(...data1.map((d) => d.time || 0)) : 0,
            data2.length > 0 ? Math.max(...data2.map((d) => d.time || 0)) : 0
          );

          timeSlider.max =
            Math.ceil(maxTime) ||
            Math.max(v1.duration || 0, v2.duration || 0) ||
            100;

          // Update video sources
          v1.src = "output/video_1.mp4";
          v2.src = "output/video_2.mp4";

          // Display metrics in a more readable format
          updateMetricsDisplay(0);

          // Create radar chart for comparison
          createRadarChart(data1, data2);

          // Hide status indicator after a brief display of success
          setTimeout(() => {
            statusIndicator.style.display = "none";
          }, 2000);
        }
      )
      .catch((err) => {
        console.error("Error loading data:", err);
        statusIndicator.innerHTML =
          "Error loading analysis data. Please try again.";
        statusIndicator.style.backgroundColor = "#f8d7da";
      });
  }

  // Get color class based on score
  function getScoreColorClass(score) {
    if (score >= 80) return "high-score";
    if (score >= 60) return "medium-score";
    return "low-score";
  }

  // Create radar chart for comparing both videos
  function createRadarChart(data1, data2) {
    // Create radar chart container if it doesn't exist
    if (!document.getElementById("radar-chart-container")) {
      const container = document.createElement("div");
      container.id = "radar-chart-container";
      container.className = "radar-chart-container";

      const canvas = document.createElement("canvas");
      canvas.id = "radar-chart";

      container.appendChild(canvas);
      document
        .querySelector(".compatibility-section")
        .insertBefore(container, document.getElementById("detailed-analysis"));
    }

    // Get data points for current time
    const dataPoint1 = data1[0] || { valence: 0, comfort: 0, engagement: 0 };
    const dataPoint2 = data2[0] || { valence: 0, comfort: 0, engagement: 0 };

    // Check if Chart.js is available
    if (typeof Chart === "undefined") {
      console.error("Chart.js is not loaded. Please include it in your HTML.");
      return;
    }

    // Destroy existing chart if it exists
    if (window.radarChart) {
      window.radarChart.destroy();
    }

    // Create radar chart
    const ctx = document.getElementById("radar-chart").getContext("2d");
    window.radarChart = new Chart(ctx, {
      type: "radar",
      data: {
        labels: ["Valence", "Comfort", "Engagement"],
        datasets: [
          {
            label: "Person 1",
            data: [
              dataPoint1.valence_score !== undefined
                ? dataPoint1.valence_score
                : dataPoint1.valence,
              dataPoint1.comfort_score !== undefined
                ? dataPoint1.comfort_score
                : dataPoint1.comfort,
              dataPoint1.engagement_score !== undefined
                ? dataPoint1.engagement_score
                : dataPoint1.happy + dataPoint1.surprise ||
                  dataPoint1.engagement ||
                  0,
            ],
            backgroundColor: "rgba(255, 99, 132, 0.2)",
            borderColor: "rgba(255, 99, 132, 1)",
            borderWidth: 2,
          },
          {
            label: "Person 2",
            data: [
              dataPoint2.valence_score !== undefined
                ? dataPoint2.valence_score
                : dataPoint2.valence,
              dataPoint2.comfort_score !== undefined
                ? dataPoint2.comfort_score
                : dataPoint2.comfort,
              dataPoint2.engagement_score !== undefined
                ? dataPoint2.engagement_score
                : dataPoint2.happy + dataPoint2.surprise ||
                  dataPoint2.engagement ||
                  0,
            ],
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
            text: "Emotional Metrics Comparison",
          },
        },
      },
    });
  }

  // Prepare time series data from metrics JSON
  function prepareTimeSeriesData(metrics) {
    // If no metrics data or empty object, return empty array
    if (!metrics) {
      console.warn("Invalid metrics data format");
      return [];
    }

    // Check if we have the new core scores time series
    if (
      metrics.valence_ts &&
      metrics.comfort_ts &&
      metrics.engagement_ts &&
      metrics.time
    ) {
      console.log("Using enhanced scores format");
      const timePoints = metrics.valence_ts.length;
      const timeSeriesData = [];

      for (let i = 0; i < timePoints; i++) {
        timeSeriesData.push({
          time: metrics.time[i],
          valence_score: metrics.valence_ts[i],
          comfort_score: metrics.comfort_ts[i],
          engagement_score: metrics.engagement_ts[i],

          // Include original data for backward compatibility
          valence: metrics.valence_ts[i],
          comfort: metrics.comfort_ts[i],

          // These are used in the original code, provide defaults
          happy: 0,
          surprise: 0,
          angry: 0,
          disgust: 0,
          fear: 0,
          sad: 0,
          neutral: 0,
        });
      }

      return timeSeriesData;
    }

    // Fallback to original format if new scores aren't available
    console.warn("Using legacy data format");
    const timePoints = metrics.valence ? metrics.valence.length : 0;
    const timeSeriesData = [];

    for (let i = 0; i < timePoints; i++) {
      timeSeriesData.push({
        time: i / 5, // Assuming 5fps in the original data
        valence: metrics.valence[i],
        comfort: metrics.comfort[i],
        saccade: metrics.saccade[i],
        happy: metrics.emotions[i]?.happy || 0,
        surprise: metrics.emotions[i]?.surprise || 0,
        angry: metrics.emotions[i]?.angry || 0,
        disgust: metrics.emotions[i]?.disgust || 0,
        fear: metrics.emotions[i]?.fear || 0,
        sad: metrics.emotions[i]?.sad || 0,
        neutral: metrics.emotions[i]?.neutral || 0,
      });
    }

    return timeSeriesData;
  }

  // Create individual charts for each video
  function createVideoChart(canvasId, data, title) {
    const ctx = document.getElementById(canvasId).getContext("2d");

    // Check if Chart.js is available
    if (typeof Chart === "undefined") {
      console.error("Chart.js is not loaded. Please include it in your HTML.");
      return;
    }

    // Prepare x-axis labels as time in seconds
    const labels = data.map((d) =>
      d.time !== undefined ? d.time.toFixed(1) : "N/A"
    );

    // Create chart
    new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Valence",
            data: data.map((d) =>
              d.valence_score !== undefined ? d.valence_score : d.valence
            ),
            borderColor: "rgba(255, 99, 132, 1)",
            backgroundColor: "rgba(255, 99, 132, 0.2)",
            tension: 0.4,
          },
          {
            label: "Comfort",
            data: data.map((d) =>
              d.comfort_score !== undefined ? d.comfort_score : d.comfort
            ),
            borderColor: "rgba(54, 162, 235, 1)",
            backgroundColor: "rgba(54, 162, 235, 0.2)",
            tension: 0.4,
          },
          {
            label: "Engagement",
            data: data.map((d) =>
              d.engagement_score !== undefined
                ? d.engagement_score
                : d.happy + d.surprise
            ),
            borderColor: "rgba(255, 206, 86, 1)",
            backgroundColor: "rgba(255, 206, 86, 0.2)",
            tension: 0.4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: title,
          },
        },
        scales: {
          y: {
            min: -1,
            max: 1,
          },
        },
      },
    });
  }

  // Update metrics display based on time
  // Update metrics display based on time with debugging
  function updateMetricsDisplay(time) {
    // Find the closest time point in the data
    function findClosestTimePoint(data, targetTime) {
      if (!data || data.length === 0) return null;

      // Find the closest match by time
      let closest = data[0];
      let closestDiff = Math.abs(data[0].time - targetTime);

      for (let i = 1; i < data.length; i++) {
        const diff = Math.abs(data[i].time - targetTime);
        if (diff < closestDiff) {
          closest = data[i];
          closestDiff = diff;
        }
      }
      return closest;
    }

    // Update metrics for video 1
    const dataPoint1 = findClosestTimePoint(data1, time);
    console.log("Video 1 data point at time " + time + ":", dataPoint1);

    if (dataPoint1) {
      const valence1 =
        dataPoint1.valence_score !== undefined
          ? dataPoint1.valence_score
          : dataPoint1.valence;
      const comfort1 =
        dataPoint1.comfort_score !== undefined
          ? dataPoint1.comfort_score
          : dataPoint1.comfort;
      const engagement1 =
        dataPoint1.engagement_score !== undefined
          ? dataPoint1.engagement_score
          : dataPoint1.happy + dataPoint1.surprise;

      console.log("Video 1 metrics:", {
        valence: valence1,
        comfort: comfort1,
        engagement: engagement1,
      });

      // Update metrics with better formatting
      document.getElementById("val1").innerHTML = `<span class="metric-value ${
        valence1 >= 0 ? "positive" : "negative"
      }">${valence1.toFixed(2)}</span>`;
      document.getElementById(
        "com1"
      ).innerHTML = `<span class="metric-value">${comfort1.toFixed(2)}</span>`;
      document.getElementById(
        "eng1"
      ).innerHTML = `<span class="metric-value">${engagement1.toFixed(
        2
      )}</span>`;
    }

    // Update metrics for video 2
    const dataPoint2 = findClosestTimePoint(data2, time);
    console.log("Video 2 data point at time " + time + ":", dataPoint2);

    if (dataPoint2) {
      const valence2 =
        dataPoint2.valence_score !== undefined
          ? dataPoint2.valence_score
          : dataPoint2.valence;
      const comfort2 =
        dataPoint2.comfort_score !== undefined
          ? dataPoint2.comfort_score
          : dataPoint2.comfort;
      const engagement2 =
        dataPoint2.engagement_score !== undefined
          ? dataPoint2.engagement_score
          : dataPoint2.happy + dataPoint2.surprise;

      console.log("Video 2 metrics:", {
        valence: valence2,
        comfort: comfort2,
        engagement: engagement2,
      });

      // Update metrics with better formatting
      document.getElementById("val2").innerHTML = `<span class="metric-value ${
        valence2 >= 0 ? "positive" : "negative"
      }">${valence2.toFixed(2)}</span>`;
      document.getElementById(
        "com2"
      ).innerHTML = `<span class="metric-value">${comfort2.toFixed(2)}</span>`;
      document.getElementById(
        "eng2"
      ).innerHTML = `<span class="metric-value">${engagement2.toFixed(
        2
      )}</span>`;
    }

    // Update radar chart if it exists
    if (window.radarChart && dataPoint1 && dataPoint2) {
      window.radarChart.data.datasets[0].data = [
        dataPoint1.valence_score !== undefined
          ? dataPoint1.valence_score
          : dataPoint1.valence,
        dataPoint1.comfort_score !== undefined
          ? dataPoint1.comfort_score
          : dataPoint1.comfort,
        dataPoint1.engagement_score !== undefined
          ? dataPoint1.engagement_score
          : dataPoint1.happy + dataPoint1.surprise ||
            dataPoint1.engagement ||
            0,
      ];

      window.radarChart.data.datasets[1].data = [
        dataPoint2.valence_score !== undefined
          ? dataPoint2.valence_score
          : dataPoint2.valence,
        dataPoint2.comfort_score !== undefined
          ? dataPoint2.comfort_score
          : dataPoint2.comfort,
        dataPoint2.engagement_score !== undefined
          ? dataPoint2.engagement_score
          : dataPoint2.happy + dataPoint2.surprise ||
            dataPoint2.engagement ||
            0,
      ];

      window.radarChart.options.plugins.title.text = `Emotional Metrics Comparison at ${time.toFixed(
        1
      )}s`;
      window.radarChart.update();
    }
  }
  // Adjust time slider width to match video width
  function adjustSliderWidth() {
    const videoWidth = v1.offsetWidth;
    timeSlider.style.width = videoWidth * 2 + 40 + "px"; // Account for gap between videos
  }

  // Call once on load and again on window resize
  window.addEventListener("resize", adjustSliderWidth);

  // Adjust slider width initially after DOM is loaded
  adjustSliderWidth();

  // Update time slider max value based on actual video duration
  v1.addEventListener("loadedmetadata", function () {
    const maxDuration = Math.max(v1.duration, v2.duration || 0);
    if (maxDuration && maxDuration > 0) {
      timeSlider.max = Math.floor(maxDuration);
    }
  });

  v2.addEventListener("loadedmetadata", function () {
    const maxDuration = Math.max(v1.duration || 0, v2.duration);
    if (maxDuration && maxDuration > 0) {
      timeSlider.max = Math.floor(maxDuration);
    }
  });

  // Enhanced slider functionality to sync with video
  timeSlider.addEventListener("input", function () {
    const time = parseInt(this.value);
    currentTimeDisplay.textContent = time;

    // Set video times and make sure playback is paused during scrubbing
    if (v1.duration) {
      v1.currentTime = time;
    }
    if (v2.duration) {
      v2.currentTime = time;
    }

    // Update metrics displays
    updateMetricsDisplay(time);
  });

  // Play both videos
  playBtn.addEventListener("click", function () {
    v1.play();
    v2.play();

    // Start periodic updates of metrics while video plays
    const updateInterval = setInterval(function () {
      if (v1.paused && v2.paused) {
        clearInterval(updateInterval);
      } else {
        const currentTime = Math.floor(v1.currentTime);
        timeSlider.value = currentTime;
        currentTimeDisplay.textContent = currentTime;
        updateMetricsDisplay(currentTime);
      }
    }, 200); // Update 5 times per second
  });

  // Sync time slider with video
  v1.addEventListener("timeupdate", function () {
    const currentTime = Math.floor(v1.currentTime);
    timeSlider.value = currentTime;
    currentTimeDisplay.textContent = currentTime;
    updateMetricsDisplay(currentTime);
  });
});
