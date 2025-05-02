// ======================================
// GLOBAL VARIABLES
// ======================================

// Analysis data
let data1 = []; // Emotional metrics for person 1
let data2 = []; // Emotional metrics for person 2
let compatibilityScore = 0;

// Recording variables
let recordingMode = false;
let stream1 = null;
let stream2 = null;
let mediaRecorder1 = null;
let mediaRecorder2 = null;
let recordedChunks1 = [];
let recordedChunks2 = [];
let recordingTimers = [null, null];
let recordingTimes = [0, 0];

// Global recording control variables
let recordBtn = null;
let stopRecordingBtn = null;
let isRecording = false;

// WebRTC variables
let localStream = null;
let peerConnection = null;
let roomId = null;
let socket = null;

// WebRTC configuration
const peerConfig = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "stun:stun1.l.google.com:19302" },
  ],
};

// ======================================
// INITIALIZATION
// ======================================

// Initialize everything when DOM is loaded
document.addEventListener("DOMContentLoaded", function () {
  // Initialize UI elements
  initUI();

  // Show initial visualizations
  showInitialVisualizations();

  // Initialize method toggle functionality
  initializeMethodToggle();

  // Create empty charts for initial display
  initializeEmptyCharts();

  // Check for previous analysis results
  // checkForPreviousResults();

  // Set up event listeners for all UI interactions
  setupEventListeners();

  // Adjust time slider width to match video width
  adjustSliderWidth();
});

// Initialize the UI elements
function initUI() {
  // Hide results sections initially
  const sectionsToHide = [
    "radar-section",
    "videos-grid",
    "compatibility-section",
  ];

  sectionsToHide.forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      element.style.display = "none";
    }
  });

  // Reset any text elements to default values
  const text1 = document.getElementById("text1");
  const text2 = document.getElementById("text2");
  if (text1) text1.innerHTML = "Upload and analyze videos to see feedback";
  if (text2) text2.innerHTML = "Upload and analyze videos to see feedback";

  const compatibilityScoreEL = document.getElementById("compatibility-score");
  if (compatibilityScoreEL)
    compatibilityScoreEL.innerHTML = "Date Compatibility Score: --";

  const detailedAnalysis = document.getElementById("detailed-analysis");
  if (detailedAnalysis) {
    detailedAnalysis.innerHTML = `
      <h2>Compatibility Analysis</h2>
      <p>Upload and analyze videos to see detailed compatibility insights</p>
    `;
  }
}

// Add this function to show empty visualizations immediately
function showInitialVisualizations() {
  // Show all visualization sections by default
  document.getElementById("radar-section").style.display = "block";
  document.getElementById("videos-grid").style.display = "grid";
  document.getElementById("compatibility-section").style.display = "block";
}

// Initialize empty charts for initial display
function initializeEmptyCharts() {
  // Create empty radar chart
  createEmptyRadarChart();

  // Create empty line charts
  createEmptyLineChart("chart1", "Video 1 Emotional Metrics");
  createEmptyLineChart("chart2", "Video 2 Emotional Metrics");
}

// Initialize method toggle buttons
function initializeMethodToggle() {
  const methodOptions = document.querySelectorAll(".method-option");
  const uploadSection = document.getElementById("upload-section");
  const recordingSection = document.getElementById("recording-section");

  if (methodOptions.length > 0) {
    methodOptions.forEach((option) => {
      option.addEventListener("click", function () {
        // Remove selected class from all options
        methodOptions.forEach((opt) => opt.classList.remove("selected"));

        // Add selected class to this option
        this.classList.add("selected");

        // Show/hide appropriate section
        const method = this.getAttribute("data-method");
        if (method === "upload") {
          uploadSection.style.display = "block";
          recordingSection.style.display = "none";
          recordingMode = false;
        } else if (method === "record") {
          uploadSection.style.display = "none";
          recordingSection.style.display = "block";
          recordingMode = true;

          // Initialize recording if needed
          if (!localStream) {
            initializeRecording();
          }
        }
      });
    });
  }
}

// Set up all event listeners
function setupEventListeners() {
  // Upload elements
  const video1Upload = document.getElementById("video1-upload");
  const video2Upload = document.getElementById("video2-upload");
  const analyzeUploadBtn = document.getElementById("analyze-upload-btn");

  // Recording elements
  const startRecording1 = document.getElementById("start-recording1");
  const stopRecording1 = document.getElementById("stop-recording1");
  const startRecording2 = document.getElementById("start-recording2");
  const stopRecording2 = document.getElementById("stop-recording2");
  const recordBothBtn = document.getElementById("record-both-btn");
  const analyzeRecordingBtn = document.getElementById("analyze-recording-btn");

  // New global recording buttons
  recordBtn = document.getElementById("record-btn");
  stopRecordingBtn = document.getElementById("stop-recording-btn");

  if (recordBtn && stopRecordingBtn) {
    recordBtn.addEventListener("click", function () {
      if (stream1 && stream2) {
        startRecordingSession(1);
        startRecordingSession(2);

        recordBtn.disabled = true;
        stopRecordingBtn.disabled = false;
        isRecording = true;

        const status1 = document.getElementById("status1");
        const status2 = document.getElementById("status2");
        if (status1) status1.textContent = "Recording";
        if (status2) status2.textContent = "Recording";

        const recordingSection = document.querySelector(".videos-grid");
        if (recordingSection) recordingSection.classList.add("recording");

        startRecordingTimer(0);
      } else {
        alert("Please establish video chat connection first.");
      }
    });

    stopRecordingBtn.addEventListener("click", function () {
      stopRecordingSession(1);
      stopRecordingSession(2);

      recordBtn.disabled = false;
      stopRecordingBtn.disabled = true;
      isRecording = false;

      const status1 = document.getElementById("status1");
      const status2 = document.getElementById("status2");
      if (status1) status1.textContent = "Ready";
      if (status2) status2.textContent = "Ready";

      const recordingSection = document.querySelector(".videos-grid");
      if (recordingSection) recordingSection.classList.remove("recording");

      stopRecordingTimer(0);

      if (recordedChunks1.length > 0 && recordedChunks2.length > 0) {
        if (analyzeRecordingBtn) analyzeRecordingBtn.disabled = false;
      }
    });
  }

  // Video playback elements
  const timeSlider = document.getElementById("time-slider");
  const currentTimeDisplay = document.getElementById("current-time");
  const playBtn = document.getElementById("play-btn");
  const v1 = document.getElementById("video1");
  const v2 = document.getElementById("video2");

  // Option selection buttons
  const selectButtons = document.querySelectorAll(".select-btn");
  if (selectButtons) {
    selectButtons.forEach((button) => {
      button.addEventListener("click", function () {
        const option = this.getAttribute("data-option");

        // Remove selected class from all options
        document.querySelectorAll(".input-option").forEach((opt) => {
          opt.classList.remove("selected");
        });

        // Add selected class to parent element
        this.closest(".input-option").classList.add("selected");

        // Show appropriate section based on selection
        if (option === "upload") {
          recordingMode = false;
          if (uploadSection) uploadSection.style.display = "block";
          if (recordingSection) recordingSection.style.display = "none";
        } else if (option === "record") {
          recordingMode = true;
          if (uploadSection) uploadSection.style.display = "none";
          if (recordingSection) recordingSection.style.display = "block";

          // Initialize recording preview
          initializeRecording();
        }
      });
    });
  }

  // File upload event listeners
  if (video1Upload && video2Upload) {
    video1Upload.addEventListener("change", function (e) {
      if (e.target.files.length > 0) {
        const file = e.target.files[0];
        if (v1) {
          v1.src = URL.createObjectURL(file);
          checkUploadStatus();
        }
      }
    });

    video2Upload.addEventListener("change", function (e) {
      if (e.target.files.length > 0) {
        const file = e.target.files[0];
        if (v2) {
          v2.src = URL.createObjectURL(file);
          checkUploadStatus();
        }
      }
    });
  }

  // Enable analyze button when both videos are uploaded
  function checkUploadStatus() {
    if (video1Upload && video2Upload && analyzeUploadBtn) {
      if (video1Upload.files.length > 0 && video2Upload.files.length > 0) {
        analyzeUploadBtn.disabled = false;
      } else {
        analyzeUploadBtn.disabled = true;
      }
    }
  }

  // Recording event listeners
  if (
    startRecording1 &&
    stopRecording1 &&
    startRecording2 &&
    stopRecording2 &&
    recordBothBtn
  ) {
    // Start recording for person 1
    startRecording1.addEventListener("click", function () {
      if (!stream1) return;
      startRecordingSession(1);
      startRecording1.disabled = true;
      stopRecording1.disabled = false;
    });

    // Stop recording for person 1
    stopRecording1.addEventListener("click", function () {
      stopRecordingSession(1);
      startRecording1.disabled = false;
      stopRecording1.disabled = true;
    });

    // Start recording for person 2
    startRecording2.addEventListener("click", function () {
      if (!stream2) return;
      startRecordingSession(2);
      startRecording2.disabled = true;
      stopRecording2.disabled = false;
    });

    // Stop recording for person 2
    stopRecording2.addEventListener("click", function () {
      stopRecordingSession(2);
      startRecording2.disabled = false;
      stopRecording2.disabled = true;
    });

    // Record both at the same time
    // Add this debugging code to the record button event listener
    if (recordBtn) {
      recordBtn.addEventListener("click", function () {
        console.log("Record button clicked");
        console.log("Stream1 exists:", !!stream1);
        console.log("Stream2 exists:", !!stream2);

        // Check if streams exist before trying to record
        if (!stream1) {
          console.error("Stream1 is not initialized");
          alert(
            "Local camera stream is not initialized. Please check camera permissions."
          );
          return;
        }

        if (!stream2) {
          console.error("Stream2 is not initialized");
          alert(
            "Remote stream is not available. Please establish a video chat connection first."
          );
          return;
        }

        try {
          // Start recording both streams
          console.log("Attempting to start recording...");
          startRecordingSession(1);
          startRecordingSession(2);

          // Update UI
          recordBtn.disabled = true;
          stopRecordingBtn.disabled = false;
          isRecording = true;

          // Update status texts
          const status1 = document.getElementById("status1");
          const status2 = document.getElementById("status2");
          if (status1) status1.textContent = "Recording";
          if (status2) status2.textContent = "Recording";

          // Add recording indicator class
          const recordingSection = document.querySelector(".videos-grid");
          if (recordingSection) recordingSection.classList.add("recording");

          // Start recording timer
          startRecordingTimer(0);
          console.log("Recording started successfully");
        } catch (error) {
          console.error("Error starting recording:", error);
          alert("Error starting recording: " + error.message);
        }
      });
    }
  }

  // Handle analyze button for recordings
  if (analyzeRecordingBtn) {
    analyzeRecordingBtn.addEventListener("click", function () {
      if (recordedChunks1.length === 0 || recordedChunks2.length === 0) {
        alert("Please record both videos before analyzing.");
        return;
      }

      uploadAndAnalyzeRecordings();
    });
  }

  // Handle form submission for uploaded videos
  if (analyzeUploadBtn) {
    console.log("Found analyze upload button:", analyzeUploadBtn); // Add debug
    analyzeUploadBtn.addEventListener("click", function () {
      console.log("Analyze button clicked"); // Add debug

      // Get the file inputs directly when clicked
      const video1Upload = document.getElementById("video1-upload");
      const video2Upload = document.getElementById("video2-upload");

      console.log(
        "Files found:",
        video1Upload?.files?.length,
        video2Upload?.files?.length
      );

      if (
        !video1Upload ||
        !video2Upload ||
        video1Upload.files.length === 0 ||
        video2Upload.files.length === 0
      ) {
        alert("Please upload both videos before analyzing.");
        return;
      }

      try {
        // Call the upload function directly
        uploadAndAnalyzeUploads(video1Upload.files[0], video2Upload.files[0]);
      } catch (error) {
        console.error("Error in uploadAndAnalyzeUploads:", error);
        alert("Error starting analysis: " + error.message);
      }
    });
  }

  // Enhanced slider functionality to sync with video
  if (timeSlider) {
    timeSlider.addEventListener("input", function () {
      const time = parseInt(this.value);
      if (currentTimeDisplay) currentTimeDisplay.textContent = time;

      // Set video times and make sure playback is paused during scrubbing
      if (v1 && v1.duration) {
        v1.currentTime = time;
      }
      if (v2 && v2.duration) {
        v2.currentTime = time;
      }

      // Update metrics displays
      updateMetricsDisplay(time);
    });
  }

  // Play both videos
  if (playBtn && v1 && v2) {
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
  }

  // Sync time slider with video
  if (v1 && timeSlider && currentTimeDisplay) {
    v1.addEventListener("timeupdate", function () {
      const currentTime = Math.floor(v1.currentTime);
      timeSlider.value = currentTime;
      currentTimeDisplay.textContent = currentTime;
      updateMetricsDisplay(currentTime);
    });
  }

  // Update time slider max value based on actual video duration
  if (v1 && timeSlider) {
    v1.addEventListener("loadedmetadata", function () {
      const maxDuration = Math.max(v1.duration, v2 ? v2.duration || 0 : 0);
      if (maxDuration && maxDuration > 0) {
        timeSlider.max = Math.floor(maxDuration);
      }
    });
  }

  if (v2 && timeSlider) {
    v2.addEventListener("loadedmetadata", function () {
      const maxDuration = Math.max(v1 ? v1.duration || 0 : 0, v2.duration);
      if (maxDuration && maxDuration > 0) {
        timeSlider.max = Math.floor(maxDuration);
      }
    });
  }

  // Window resize event for slider adjustment
  window.addEventListener("resize", adjustSliderWidth);
}

// ======================================
// UPLOADING AND ANALYSIS
// ======================================
// Add a helper function to clear charts
function clearCharts() {
  // Destroy existing charts to prevent duplicates
  if (window.radarChart) {
    window.radarChart.destroy();
    window.radarChart = null;
  }

  if (window.chart1Chart) {
    window.chart1Chart.destroy();
    window.chart1Chart = null;
  }

  if (window.chart2Chart) {
    window.chart2Chart.destroy();
    window.chart2Chart = null;
  }

  // Reset all data
  data1 = [];
  data2 = [];
  compatibilityScore = 0;

  // Reset video elements
  const v1 = document.getElementById("video1");
  const v2 = document.getElementById("video2");
  if (v1) v1.src = "";
  if (v2) v2.src = "";

  // Reset text elements
  const text1 = document.getElementById("text1");
  const text2 = document.getElementById("text2");
  if (text1) text1.innerHTML = "Analysis in progress...";
  if (text2) text2.innerHTML = "Analysis in progress...";

  const compatibilityScoreEL = document.getElementById("compatibility-score");
  if (compatibilityScoreEL)
    compatibilityScoreEL.innerHTML = "Date Compatibility Score: --";

  const detailedAnalysis = document.getElementById("detailed-analysis");
  if (detailedAnalysis) {
    detailedAnalysis.innerHTML = `
      <h2>Compatibility Analysis</h2>
      <p>Analysis in progress...</p>
    `;
  }
}

// Function to hide result sections
function hideResultSections() {
  const sectionsToHide = [
    "radar-section",
    "videos-grid",
    "compatibility-section",
  ];

  sectionsToHide.forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      element.style.display = "none";
    }
  });
}

// Upload and analyze recorded videos
function uploadAndAnalyzeRecordings() {
  // Clear any previous analysis results
  clearCharts();

  // Hide result sections - THIS WAS MISSING
  hideResultSections();

  const statusIndicator = document.getElementById("status-indicator");

  // Show status indicator
  statusIndicator.style.display = "block";
  statusIndicator.innerHTML = "Preparing videos for analysis...";
  statusIndicator.style.backgroundColor = "#fff3cd";

  // Add progress bar
  statusIndicator.innerHTML +=
    '<div class="progress-bar"><div class="progress" id="progress-bar"></div></div>';
  const progressBar = document.getElementById("progress-bar");
  progressBar.style.width = "10%";

  // Create form data
  const formData = new FormData();

  // Convert blobs to files
  const file1 = new File(
    [new Blob(recordedChunks1, { type: "video/webm" })],
    "recording1.webm"
  );
  const file2 = new File(
    [new Blob(recordedChunks2, { type: "video/webm" })],
    "recording2.webm"
  );

  formData.append("video1", file1);
  formData.append("video2", file2);

  // Upload videos to server
  // Do NOT call clearCharts and hideResultSections again inside this function
  try {
    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Upload successful:", data);
        if (progressBar) progressBar.style.width = "30%";

        // Start polling for status
        pollUntilComplete();
      })
      .catch((error) => {
        console.error("Error uploading videos:", error);
        statusIndicator.innerHTML = "Error uploading videos. Please try again.";
        statusIndicator.style.backgroundColor = "#f8d7da";
      });
  } catch (err) {
    console.error("Error in fetch operation:", err);
    statusIndicator.innerHTML = "Error: " + err.message;
    statusIndicator.style.backgroundColor = "#f8d7da";
  }

  // Function to continuously poll until analysis is complete
  function pollUntilComplete() {
    const statusInterval = setInterval(() => {
      fetch("/status")
        .then((res) => res.json())
        .then((statusData) => {
          console.log("Status update:", statusData);

          // Update progress bar
          if (progressBar) progressBar.style.width = statusData.progress + "%";

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
            console.log(
              "Analysis completed, loading new results from:",
              statusData.output_folder
            );

            // Only now load the results - AFTER the analysis is confirmed complete
            loadLatestResults(statusData.output_folder);

            // Update UI
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
  }
}

// Upload and analyze uploaded videos
function uploadAndAnalyzeUploads(file1, file2) {
  // Clear any previous analysis results
  clearCharts();

  // Hide result sections
  hideResultSections();

  const statusIndicator = document.getElementById("status-indicator");

  // Show status indicator
  statusIndicator.style.display = "block";
  statusIndicator.innerHTML =
    "Analysis may take several minutes to complete. Please wait...";
  statusIndicator.style.backgroundColor = "#fff3cd";

  // Add progress bar
  statusIndicator.innerHTML +=
    '<div class="progress-bar"><div class="progress" id="progress-bar"></div></div>';

  // Create FormData and upload videos
  const formData = new FormData();
  formData.append("video1", file1);
  formData.append("video2", file2);

  // Upload videos to server
  uploadVideosAndPollStatus(formData);
}

// Common function to upload videos and poll status
// Modify the uploadVideosAndPollStatus function to wait for completion
function uploadVideosAndPollStatus(formData) {
  const statusIndicator = document.getElementById("status-indicator");
  const progressBar = document.getElementById("progress-bar");

  // Send data to server
  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Upload successful:", data);
      if (progressBar) progressBar.style.width = "30%";

      // Start polling for status - use a continuous polling approach
      pollUntilComplete();
    })
    .catch((error) => {
      console.error("Error uploading videos:", error);
      statusIndicator.innerHTML = "Error uploading videos. Please try again.";
      statusIndicator.style.backgroundColor = "#f8d7da";
    });

  // Function to continuously poll until analysis is complete
  function pollUntilComplete() {
    const statusInterval = setInterval(() => {
      fetch("/status")
        .then((res) => res.json())
        .then((statusData) => {
          // Update progress bar
          if (progressBar) progressBar.style.width = statusData.progress + "%";

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

            // Only now load the results - AFTER the analysis is confirmed complete
            loadLatestResults(statusData.output_folder);

            // Update UI
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
  }
}

function loadLatestResults(outputFolder) {
  console.log("Loading latest results from:", outputFolder);

  // Show visualization sections now
  showVisualizationSections();

  if (!outputFolder) {
    console.error("No output folder specified");
    return;
  }

  const folderName = outputFolder.split("/").pop(); // Get the last part of the path

  // Fetch all the required data with cache-busting timestamps
  const timestamp = new Date().getTime();

  Promise.all([
    fetch(`output/${folderName}/video_1/metrics_data.json?t=${timestamp}`).then(
      (res) => res.json().catch(() => ({}))
    ),
    fetch(`output/${folderName}/video_2/metrics_data.json?t=${timestamp}`).then(
      (res) => res.json().catch(() => ({}))
    ),
    fetch(
      `output/${folderName}/video_1/crossmodal_analysis.txt?t=${timestamp}`
    ).then((res) => res.text().catch(() => "")),
    fetch(
      `output/${folderName}/video_2/crossmodal_analysis.txt?t=${timestamp}`
    ).then((res) => res.text().catch(() => "")),
    fetch(`output/${folderName}/compatibility_score.json?t=${timestamp}`).then(
      (res) => res.json().catch(() => ({ score: 0, metrics: {} }))
    ),
    fetch(
      `output/${folderName}/compatibility_analysis.txt?t=${timestamp}`
    ).then((res) => res.text().catch(() => "")),
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
        processLoadedData(
          metrics1,
          metrics2,
          analysis1,
          analysis2,
          scoreData,
          detailedAnalysis,
          folderName
        );
      }
    )
    .catch((err) => {
      console.error("Error loading analysis results:", err);
    });
}

// Show visualization sections after analysis is complete
function showVisualizationSections() {
  const sectionsToShow = [
    "radar-section",
    "videos-grid",
    "compatibility-section",
  ];
  sectionsToShow.forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      if (id === "videos-grid") {
        element.style.display = "grid";
      } else {
        element.style.display = "block";
      }
    }
  });
}

// Start polling for status updates
function startStatusPolling() {
  const statusInterval = setInterval(() => {
    fetch("/status")
      .then((res) => res.json())
      .then((statusData) => {
        // Update progress bar
        const progressBar = document.getElementById("progress-bar");
        if (progressBar) {
          progressBar.style.width = statusData.progress + "%";
        }

        // Update status message
        const statusIndicator = document.getElementById("status-indicator");
        if (statusIndicator) {
          statusIndicator.innerHTML =
            statusData.message || "Analysis in progress...";
          statusIndicator.innerHTML += `<div class="progress-bar"><div class="progress" id="progress-bar" style="width: ${statusData.progress}%;"></div></div>`;
        }

        // Check if processing is complete
        if (statusData.status === "completed") {
          clearInterval(statusInterval);
          loadResults();
        } else if (statusData.status === "error") {
          clearInterval(statusInterval);
          if (statusIndicator) {
            statusIndicator.innerHTML =
              "Error during analysis: " +
              (statusData.message || "Unknown error");
            statusIndicator.style.backgroundColor = "#f8d7da";
          }
        }
      })
      .catch((err) => {
        console.error("Error checking status:", err);
      });
  }, 2000); // Check every 2 seconds
}

// ======================================
// RECORDING FUNCTIONALITY
// ======================================

// Initialize recording preview
function initializeRecording() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: true,
      })
      .then(function (stream) {
        localStream = stream;
        stream1 = stream;
        const video1 = document.getElementById("recording1");
        if (video1) {
          video1.srcObject = stream;
          video1.play();
        }
        initializeVideoChat();
        console.log("Camera preview initialized");
      })
      .catch(function (error) {
        console.error("Error accessing media devices:", error);
        alert(
          "Error accessing camera and microphone. Please check permissions."
        );
      });
  } else {
    alert(
      "Your browser doesn't support media recording. Please try a different browser."
    );
  }
}

// Start recording session for a person
function startRecordingTimer(personIndex) {
  // Use a single timer display
  const timerElement = document.getElementById("record-timer");

  // Reset time
  recordingTimes[personIndex] = 0;

  // Clear existing timer
  if (recordingTimers[personIndex]) {
    clearInterval(recordingTimers[personIndex]);
  }

  // Update timer display
  if (timerElement) timerElement.textContent = formatTime(0);

  // Start new timer
  recordingTimers[personIndex] = setInterval(function () {
    recordingTimes[personIndex]++;
    if (timerElement)
      timerElement.textContent = formatTime(recordingTimes[personIndex]);
  }, 1000);
}

// Stop recording session for a person
function stopRecordingTimer(personIndex) {
  // Clear timer interval
  if (recordingTimers[personIndex]) {
    clearInterval(recordingTimers[personIndex]);
    recordingTimers[personIndex] = null;
  }
}

// Start recording timer
function startRecordingTimer(personIndex) {
  // Reset time
  recordingTimes[personIndex - 1] = 0;

  // Clear existing timer
  if (recordingTimers[personIndex - 1]) {
    clearInterval(recordingTimers[personIndex - 1]);
  }

  // Update timer display
  const timerElement = document.getElementById(`timer${personIndex}`);
  if (timerElement) timerElement.textContent = formatTime(0);

  // Start new timer
  recordingTimers[personIndex - 1] = setInterval(function () {
    recordingTimes[personIndex - 1]++;
    if (timerElement)
      timerElement.textContent = formatTime(recordingTimes[personIndex - 1]);
  }, 1000);
}

// Stop recording timer
function stopRecordingTimer(personIndex) {
  // Clear timer interval
  if (recordingTimers[personIndex - 1]) {
    clearInterval(recordingTimers[personIndex - 1]);
    recordingTimers[personIndex - 1] = null;
  }
}

// Format time in MM:SS
function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes.toString().padStart(2, "0")}:${remainingSeconds
    .toString()
    .padStart(2, "0")}`;
}

// ======================================
// WEBRTC VIDEO CHAT
// ======================================

// Initialize WebRTC socket for video chat
function initializeVideoChat() {
  // Connect to the signaling server if socket.io is loaded
  if (typeof io !== "undefined") {
    socket = io.connect(window.location.origin);

    // Socket event handlers
    socket.on("connect", () => {
      console.log("Connected to signaling server");
    });

    socket.on("room-created", (data) => {
      roomId = data.roomId;
      const roomInfo = document.getElementById("room-info");
      if (roomInfo) {
        roomInfo.style.display = "block";
        roomInfo.textContent = `Room created! Share this ID: ${roomId}`;
      }
    });

    socket.on("room-joined", (data) => {
      roomId = data.roomId;
      const roomInfo = document.getElementById("room-info");
      if (roomInfo) {
        roomInfo.style.display = "block";
        roomInfo.textContent = `Joined room: ${roomId}`;
      }

      // Create an offer when joining a room
      createOffer();
    });

    socket.on("new-ice-candidate", (data) => {
      if (peerConnection) {
        const candidate = new RTCIceCandidate(data.candidate);
        peerConnection
          .addIceCandidate(candidate)
          .catch((error) =>
            console.error("Error adding ice candidate:", error)
          );
      }
    });

    socket.on("offer", async (data) => {
      if (!peerConnection) {
        createPeerConnection();
      }

      await peerConnection.setRemoteDescription(
        new RTCSessionDescription(data.offer)
      );

      // Create answer
      const answer = await peerConnection.createAnswer();
      await peerConnection.setLocalDescription(answer);

      socket.emit("answer", {
        answer: answer,
        roomId: roomId,
      });
    });

    socket.on("answer", async (data) => {
      if (peerConnection) {
        await peerConnection.setRemoteDescription(
          new RTCSessionDescription(data.answer)
        );
      }
    });

    // Set up video chat UI buttons
    setupVideoChatButtons();
  } else {
    console.warn(
      "Socket.io not loaded. Video chat functionality will not work."
    );
  }
}

// Set up video chat UI buttons
function setupVideoChatButtons() {
  const videoChatBtn = document.getElementById("video-chat-btn");
  const createRoomBtn = document.getElementById("create-room-btn");
  const joinRoomBtn = document.getElementById("join-room-btn");

  if (videoChatBtn) {
    videoChatBtn.addEventListener("click", function () {
      const roomContainer = document.querySelector(".room-id-container");
      if (roomContainer) {
        if (roomContainer.style.display === "none") {
          roomContainer.style.display = "flex";
          this.textContent = "Hide Video Chat";
        } else {
          roomContainer.style.display = "none";
          this.textContent = "Start Video Chat";
        }
      }
    });
  }

  if (createRoomBtn) {
    createRoomBtn.addEventListener("click", function () {
      if (!localStream) {
        alert("Please allow camera access first by clicking Start on Person 1");
        return;
      }

      // Create a new room
      const randomRoomId = Math.floor(
        100000 + Math.random() * 900000
      ).toString();
      socket.emit("create-room", { roomId: randomRoomId });

      // Create peer connection
      createPeerConnection();
    });
  }

  if (joinRoomBtn) {
    joinRoomBtn.addEventListener("click", function () {
      if (!localStream) {
        alert("Please allow camera access first by clicking Start on Person 1");
        return;
      }

      const roomIdInput = document.getElementById("room-id");
      if (roomIdInput) {
        const roomIdValue = roomIdInput.value.trim();
        if (!roomIdValue) {
          alert("Please enter a room ID");
          return;
        }

        // Join the room
        socket.emit("join-room", { roomId: roomIdValue });

        // Create peer connection
        createPeerConnection();
      }
    });
  }
}

// Create and set up the RTCPeerConnection
function createPeerConnection() {
  peerConnection = new RTCPeerConnection(peerConfig);

  if (localStream) {
    localStream.getTracks().forEach((track) => {
      peerConnection.addTrack(track, localStream);
    });
  }

  peerConnection.onicecandidate = (event) => {
    if (event.candidate && socket) {
      socket.emit("ice-candidate", {
        candidate: event.candidate,
        roomId: roomId,
      });
    }
  };

  peerConnection.onconnectionstatechange = (event) => {
    if (peerConnection.connectionState === "connected") {
      console.log("Peers connected!");
      const col1 = document.querySelector(".recording-column:nth-child(1)");
      const col2 = document.querySelector(".recording-column:nth-child(2)");
      if (col1) col1.classList.add("connected");
      if (col2) col2.classList.add("connected");
    }
  };

  peerConnection.ontrack = (event) => {
    const remoteVideo = document.getElementById("recording2");
    if (remoteVideo && remoteVideo.srcObject !== event.streams[0]) {
      remoteVideo.srcObject = event.streams[0];
      stream2 = event.streams[0];
      console.log("Received remote stream");

      const status2 = document.getElementById("status2");
      if (status2) status2.textContent = "Connected";

      const recordingSection = document.querySelector(".videos-grid");
      if (recordingSection) recordingSection.classList.add("connected");
    }
  };
}

// Create an offer to initiate connection
async function createOffer() {
  if (!peerConnection) {
    createPeerConnection();
  }

  try {
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);

    if (socket) {
      socket.emit("offer", {
        offer: offer,
        roomId: roomId,
      });
    }
  } catch (error) {
    console.error("Error creating offer:", error);
  }
}

// ======================================
// RESULTS LOADING AND VISUALIZATION
// ======================================
// Add timestamp to all fetch requests to prevent caching
function fetchWithTimestamp(url) {
  const timestamp = new Date().getTime();
  const separator = url.includes("?") ? "&" : "?";
  return fetch(`${url}${separator}t=${timestamp}`);
}

// Load results after processing is complete
function loadResults() {
  console.log("Loading results...");

  const statusIndicator = document.getElementById("status-indicator");
  if (statusIndicator) {
    statusIndicator.style.display = "block";
    statusIndicator.innerHTML = "Loading analysis results...";
    statusIndicator.style.backgroundColor = "#d4edda";
  }

  // First get the status to find the output folder
  fetchWithTimestamp("/status")
    .then((res) => res.json())
    .then((statusData) => {
      // Extract the output folder path from status data
      const outputFolder = statusData.output_folder || "";

      // If no output folder is specified in the status, use the default paths
      if (!outputFolder) {
        console.log("No output folder specified, using default paths");
        loadFromDefaultPaths();
        return;
      }

      const folderName = outputFolder.split("/").pop(); // Get the last part of the path

      console.log("Loading results from folder:", outputFolder);

      // Adjust all the fetch URLs to include the folder name and timestamp
      Promise.all([
        fetchWithTimestamp(
          `output/${folderName}/video_1/metrics_data.json`
        ).then((res) => res.json().catch(() => ({}))),
        fetchWithTimestamp(
          `output/${folderName}/video_2/metrics_data.json`
        ).then((res) => res.json().catch(() => ({}))),
        fetchWithTimestamp(
          `output/${folderName}/video_1/crossmodal_analysis.txt`
        ).then((res) => res.text().catch(() => "")),
        fetchWithTimestamp(
          `output/${folderName}/video_2/crossmodal_analysis.txt`
        ).then((res) => res.text().catch(() => "")),
        fetchWithTimestamp(
          `output/${folderName}/compatibility_score.json`
        ).then((res) => res.json().catch(() => ({ score: 0, metrics: {} }))),
        fetchWithTimestamp(
          `output/${folderName}/compatibility_analysis.txt`
        ).then((res) => res.text().catch(() => "")),
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
            // Clear any previous data
            data1 = [];
            data2 = [];
            compatibilityScore = 0;

            // Process the loaded data
            processLoadedData(
              metrics1,
              metrics2,
              analysis1,
              analysis2,
              scoreData,
              detailedAnalysis,
              folderName
            );
          }
        )
        .catch((err) => {
          console.error("Error loading data from output folder:", err);
          // Try the default paths as a fallback
          loadFromDefaultPaths();
        });
    })
    .catch((err) => {
      console.error("Error checking status:", err);
      // Try the default paths as a fallback
      loadFromDefaultPaths();
    });
}

// Load results from default paths (for backward compatibility)
function loadFromDefaultPaths() {
  console.log("Trying to load results from default paths");

  Promise.all([
    fetchWithTimestamp("output/video_1/metrics_data.json").then((res) =>
      res.json().catch(() => ({}))
    ),
    fetchWithTimestamp("output/video_2/metrics_data.json").then((res) =>
      res.json().catch(() => ({}))
    ),
    fetchWithTimestamp("output/video_1/crossmodal_analysis.txt").then((res) =>
      res.text().catch(() => "")
    ),
    fetchWithTimestamp("output/video_2/crossmodal_analysis.txt").then((res) =>
      res.text().catch(() => "")
    ),
    fetchWithTimestamp("output/compatibility_score.json").then((res) =>
      res.json().catch(() => ({ score: 0, metrics: {} }))
    ),
    fetchWithTimestamp("output/compatibility_analysis.txt").then((res) =>
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
        // Clear any previous data
        data1 = [];
        data2 = [];
        compatibilityScore = 0;

        processLoadedData(
          metrics1,
          metrics2,
          analysis1,
          analysis2,
          scoreData,
          detailedAnalysis,
          ""
        );
      }
    )
    .catch((err) => {
      console.error("Error loading data from default paths:", err);
      const statusIndicator = document.getElementById("status-indicator");
      if (statusIndicator) {
        statusIndicator.innerHTML =
          "Error loading analysis data. No results found.";
        statusIndicator.style.backgroundColor = "#f8d7da";
      }
    });
}

// Process the loaded data and update the UI
function processLoadedData(
  metrics1,
  metrics2,
  analysis1,
  analysis2,
  scoreData,
  detailedAnalysis,
  folderName
) {
  console.log("Data loaded successfully");

  // Prepare time series data
  data1 = prepareTimeSeriesData(metrics1);
  data2 = prepareTimeSeriesData(metrics2);

  // Display analysis text with better formatting
  const text1El = document.getElementById("text1");
  const text2El = document.getElementById("text2");
  if (text1El) text1El.innerHTML = formatFeedback(analysis1);
  if (text2El) text2El.innerHTML = formatFeedback(analysis2);

  // Update video sources with the folder path
  const v1 = document.getElementById("video1");
  const v2 = document.getElementById("video2");

  if (v1 && v2) {
    if (folderName) {
      v1.src = `output/${folderName}/video_1.mp4`;
      v2.src = `output/${folderName}/video_2.mp4`;
    } else {
      v1.src = "output/video_1.mp4";
      v2.src = "output/video_2.mp4";
    }
  }

  // Create radar chart first
  createRadarChart(data1, data2);

  // Then create individual charts
  createVideoChart("chart1", data1, "Video 1 Emotional Metrics");
  createVideoChart("chart2", data2, "Video 2 Emotional Metrics");

  // Display compatibility score with original styling
  compatibilityScore = scoreData.score;
  const compatibilityScoreEl = document.getElementById("compatibility-score");
  if (compatibilityScoreEl) {
    compatibilityScoreEl.innerHTML = `Date Compatibility Score: ${compatibilityScore}/100`;
  }

  // Make sure we have default values for metrics if they're not in the data
  if (!scoreData.metrics) {
    scoreData.metrics = {
      emotional_synchrony: 0.0,
      comfort_synchrony: 0.0,
      engagement_balance: 0.0,
      emotional_stability_1: 0.0,
      emotional_stability_2: 0.0,
      mutual_responsiveness: 0.0,
    };
  }

  // Display detailed compatibility analysis with better formatting
  const detailedAnalysisEl = document.getElementById("detailed-analysis");
  if (detailedAnalysisEl) {
    detailedAnalysisEl.innerHTML = `
      <h2>Compatibility Analysis</h2>
      <div class="compatibility-metrics">
        <div class="metric-item">
          <span class="metric-label">Emotional Synchrony</span>
          <span class="metric-value">${(
            scoreData.metrics.emotional_synchrony || 0
          ).toFixed(2)}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Comfort Synchrony</span>
          <span class="metric-value">${(
            scoreData.metrics.comfort_synchrony || 0
          ).toFixed(2)}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Engagement Balance</span>
          <span class="metric-value">${(
            scoreData.metrics.engagement_balance || 0
          ).toFixed(2)}</span>
        </div>
      </div>
      <div class="analysis-content">
        ${formatCompatibilityAnalysis(detailedAnalysis)}
      </div>`;
  }

  // Update time slider max value based on video duration
  const maxTime = Math.max(
    data1.length > 0 ? Math.max(...data1.map((d) => d.time || 0)) : 0,
    data2.length > 0 ? Math.max(...data2.map((d) => d.time || 0)) : 0
  );

  const timeSlider = document.getElementById("time-slider");
  if (timeSlider) {
    timeSlider.max =
      Math.ceil(maxTime) || Math.max(v1.duration || 0, v2.duration || 0) || 100;
  }

  // Display metrics in a more readable format
  updateMetricsDisplay(0);

  // Show visualization sections
  showVisualizationSections();

  // Store the current folder name for future reference
  if (folderName) {
    localStorage.setItem("currentAnalysisFolder", folderName);
  }

  // Hide status indicator after a brief display of success
  setTimeout(() => {
    const statusIndicator = document.getElementById("status-indicator");
    if (statusIndicator) {
      statusIndicator.style.display = "none";
    }
  }, 2000);
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
    // Check if emotions array exists and has the current index
    const emotions =
      metrics.emotions && metrics.emotions[i] ? metrics.emotions[i] : {};

    timeSeriesData.push({
      time: i / 5, // Assuming 5fps in the original data
      valence: metrics.valence[i],
      comfort: metrics.comfort[i],
      saccade: metrics.saccade ? metrics.saccade[i] : 0,
      happy: emotions.happy || 0,
      surprise: emotions.surprise || 0,
      angry: emotions.angry || 0,
      disgust: emotions.disgust || 0,
      fear: emotions.fear || 0,
      sad: emotions.sad || 0,
      neutral: emotions.neutral || 0,
    });
  }

  return timeSeriesData;
}

// Update metrics display based on time
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

    // Update metrics with better formatting and consistent decimal places
    const val1El = document.getElementById("val1");
    const com1El = document.getElementById("com1");
    const eng1El = document.getElementById("eng1");

    if (val1El) {
      val1El.innerHTML = `<span class="metric-value ${
        valence1 >= 0 ? "positive" : "negative"
      }">${valence1.toFixed(2)}</span>`;
    }
    if (com1El) {
      com1El.innerHTML = `<span class="metric-value">${comfort1.toFixed(
        2
      )}</span>`;
    }
    if (eng1El) {
      eng1El.innerHTML = `<span class="metric-value">${engagement1.toFixed(
        2
      )}</span>`;
    }
  }

  // Update metrics for video 2
  const dataPoint2 = findClosestTimePoint(data2, time);
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

    // Update metrics with better formatting and consistent decimal places
    const val2El = document.getElementById("val2");
    const com2El = document.getElementById("com2");
    const eng2El = document.getElementById("eng2");

    if (val2El) {
      val2El.innerHTML = `<span class="metric-value ${
        valence2 >= 0 ? "positive" : "negative"
      }">${valence2.toFixed(2)}</span>`;
    }
    if (com2El) {
      com2El.innerHTML = `<span class="metric-value">${comfort2.toFixed(
        2
      )}</span>`;
    }
    if (eng2El) {
      eng2El.innerHTML = `<span class="metric-value">${engagement2.toFixed(
        2
      )}</span>`;
    }
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
        : dataPoint1.happy + dataPoint1.surprise || dataPoint1.engagement || 0,
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
        : dataPoint2.happy + dataPoint2.surprise || dataPoint2.engagement || 0,
    ];

    window.radarChart.options.plugins.title.text = `Emotional Metrics Comparison at ${time.toFixed(
      1
    )}s`;
    window.radarChart.update();
  }
}

// ======================================
// CHART CREATION AND FORMATTING
// ======================================

// Create an empty radar chart
function createEmptyRadarChart() {
  const ctx = document.getElementById("radar-chart");
  if (!ctx) return;

  // Check if Chart.js is available
  if (typeof Chart === "undefined") {
    console.error("Chart.js is not loaded. Please include it in your HTML.");
    return;
  }

  // Destroy existing chart if it exists
  if (window.radarChart) {
    window.radarChart.destroy();
  }

  // Sample placeholder data
  const placeholderData = [
    {
      label: "Person 1",
      data: [0, 0, 0],
      color: "rgba(255, 99, 132, 0.2)",
      border: "rgba(255, 99, 132, 1)",
    },
    {
      label: "Person 2",
      data: [0, 0, 0],
      color: "rgba(54, 162, 235, 0.2)",
      border: "rgba(54, 162, 235, 1)",
    },
  ];

  // Create radar chart
  window.radarChart = new Chart(ctx.getContext("2d"), {
    type: "radar",
    data: {
      labels: ["Valence", "Comfort", "Engagement"],
      datasets: placeholderData.map((item) => ({
        label: item.label,
        data: item.data,
        backgroundColor: item.color,
        borderColor: item.border,
        borderWidth: 2,
        pointBackgroundColor: item.border,
        pointRadius: 4,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: -1,
          max: 1,
          ticks: {
            stepSize: 0.2,
            backdropColor: "rgba(255, 255, 255, 0.6)",
            callback: function (value) {
              return value.toFixed(1);
            },
          },
          angleLines: {
            color: "rgba(0, 0, 0, 0.1)",
          },
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
          pointLabels: {
            font: {
              size: 14,
              weight: "bold",
            },
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Emotional Metrics Comparison",
          font: {
            size: 16,
            weight: "bold",
          },
        },
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              const label = context.dataset.label || "";
              const value = parseFloat(context.raw).toFixed(2);
              return `${label}: ${value}`;
            },
          },
        },
      },
    },
  });
}

// Create an empty line chart
function createEmptyLineChart(canvasId, title) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  // Check if Chart.js is available
  if (typeof Chart === "undefined") {
    console.error("Chart.js is not loaded. Please include it in your HTML.");
    return;
  }

  // Destroy existing chart if it exists
  if (window[canvasId + "Chart"]) {
    window[canvasId + "Chart"].destroy();
  }

  // Sample time points
  const timePoints = 10;
  const labels = Array.from({ length: timePoints }, (_, i) => i.toFixed(1));

  // Create empty datasets
  const datasets = [
    {
      label: "Valence",
      data: Array(timePoints).fill(0),
      borderColor: "rgba(255, 99, 132, 1)",
      backgroundColor: "rgba(255, 99, 132, 0.2)",
      tension: 0.4,
      borderWidth: 2,
    },
    {
      label: "Comfort",
      data: Array(timePoints).fill(0),
      borderColor: "rgba(54, 162, 235, 1)",
      backgroundColor: "rgba(54, 162, 235, 0.2)",
      tension: 0.4,
      borderWidth: 2,
    },
    {
      label: "Engagement",
      data: Array(timePoints).fill(0),
      borderColor: "rgba(255, 206, 86, 1)",
      backgroundColor: "rgba(255, 206, 86, 0.2)",
      tension: 0.4,
      borderWidth: 2,
    },
  ];

  // Create chart with FIXED scale
  window[canvasId + "Chart"] = new Chart(ctx.getContext("2d"), {
    type: "line",
    data: {
      labels: labels,
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: title,
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              const label = context.dataset.label || "";
              const value = parseFloat(context.raw).toFixed(2);
              return `${label}: ${value}`;
            },
          },
        },
      },
      scales: {
        y: {
          type: "linear",
          min: -1.0, // FIXED min value
          max: 1.0, // FIXED max value
          ticks: {
            stepSize: 0.2, // FIXED step size
            callback: function (value) {
              return value.toFixed(1); // Show one decimal place
            },
          },
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
        },
        x: {
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
        },
      },
    },
  });
}

// Create radar chart for comparing both videos
function createRadarChart(data1, data2) {
  console.log("Creating radar chart");

  const ctx = document.getElementById("radar-chart");
  if (!ctx) return;

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
  window.radarChart = new Chart(ctx.getContext("2d"), {
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
          pointBackgroundColor: "rgba(255, 99, 132, 1)",
          pointRadius: 4,
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
          pointBackgroundColor: "rgba(54, 162, 235, 1)",
          pointRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: -1,
          max: 1,
          ticks: {
            stepSize: 0.2, // Consistent step size with line charts
            backdropColor: "rgba(255, 255, 255, 0.6)",
            callback: function (value) {
              return value.toFixed(1); // Display with 1 decimal place consistently
            },
          },
          angleLines: {
            color: "rgba(0, 0, 0, 0.1)",
          },
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
          pointLabels: {
            font: {
              size: 14,
              weight: "bold",
            },
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Real-time Emotional Metrics Comparison",
          font: {
            size: 16,
            weight: "bold",
          },
        },
        legend: {
          display: false, // We're using our custom legend below the chart
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              const label = context.dataset.label || "";
              const value = parseFloat(context.raw).toFixed(2); // Always show 2 decimal places
              return `${label}: ${value}`;
            },
          },
        },
      },
    },
  });

  console.log("Radar chart created successfully");
}

// Create individual charts for each video with FIXED y-scale
function createVideoChart(canvasId, data, title) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  // Check if Chart.js is available
  if (typeof Chart === "undefined") {
    console.error("Chart.js is not loaded. Please include it in your HTML.");
    return;
  }

  // Prepare x-axis labels as time in seconds
  const labels = data.map((d) =>
    d.time !== undefined ? d.time.toFixed(1) : "N/A"
  );

  // Destroy existing chart if it exists
  if (window[canvasId + "Chart"]) {
    window[canvasId + "Chart"].destroy();
  }

  // Create chart with FIXED scale
  window[canvasId + "Chart"] = new Chart(ctx.getContext("2d"), {
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
          borderWidth: 2,
        },
        {
          label: "Comfort",
          data: data.map((d) =>
            d.comfort_score !== undefined ? d.comfort_score : d.comfort
          ),
          borderColor: "rgba(54, 162, 235, 1)",
          backgroundColor: "rgba(54, 162, 235, 0.2)",
          tension: 0.4,
          borderWidth: 2,
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
          borderWidth: 2,
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
        tooltip: {
          callbacks: {
            label: function (context) {
              const label = context.dataset.label || "";
              const value = parseFloat(context.raw).toFixed(2);
              return `${label}: ${value}`;
            },
          },
        },
      },
      scales: {
        y: {
          type: "linear",
          min: -1.0, // FIXED min value
          max: 1.0, // FIXED max value
          ticks: {
            stepSize: 0.2, // FIXED step size
            callback: function (value) {
              return value.toFixed(1); // Show one decimal place
            },
          },
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
        },
        x: {
          grid: {
            color: "rgba(0, 0, 0, 0.1)",
          },
        },
      },
    },
  });
}

// ======================================
// UTILITY FUNCTIONS
// ======================================

// Format the compatibility analysis in a structured way
function formatCompatibilityAnalysis(analysisText) {
  if (!analysisText) return "";

  // Remove markdown ** formatting
  analysisText = analysisText.replace(/\*\*/g, "");

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

// Format the feedback text to improve readability
function formatFeedback(text) {
  if (!text) return "";

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

// Adjust time slider width to match video width
function adjustSliderWidth() {
  const v1 = document.getElementById("video1");
  const timeSlider = document.getElementById("time-slider");

  if (v1 && timeSlider) {
    const videoWidth = v1.offsetWidth;
    timeSlider.style.width = videoWidth * 2 + 40 + "px"; // Account for gap between videos
  }
}

// Function to start recording for a specific person
function startRecordingSession(personIndex) {
  console.log(`Starting recording session for person ${personIndex}`);

  let stream;
  if (personIndex === 1) {
    stream = stream1;
    console.log("Using stream1 for person 1");
  } else {
    stream = stream2;
    console.log("Using stream2 for person 2");
  }

  if (!stream) {
    console.error(`No stream available for person ${personIndex}`);
    return;
  }

  // Check if stream has tracks
  if (stream.getTracks().length === 0) {
    console.error(`Stream for person ${personIndex} has no tracks`);
    return;
  }

  console.log(
    `Stream for person ${personIndex} has tracks:`,
    stream
      .getTracks()
      .map((t) => t.kind)
      .join(", ")
  );

  // Clear previous recorded chunks
  if (personIndex === 1) {
    recordedChunks1 = [];
  } else {
    recordedChunks2 = [];
  }

  // Try different MIME types for better browser compatibility
  const mimeTypes = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
    "video/mp4",
  ];

  let options = null;
  for (const mimeType of mimeTypes) {
    if (MediaRecorder.isTypeSupported(mimeType)) {
      options = { mimeType };
      console.log(`Using MIME type: ${mimeType}`);
      break;
    }
  }

  try {
    console.log(`Creating MediaRecorder for person ${personIndex}`);
    const mediaRecorder = new MediaRecorder(stream, options);

    // Store the media recorder
    if (personIndex === 1) {
      mediaRecorder1 = mediaRecorder;
    } else {
      mediaRecorder2 = mediaRecorder;
    }

    // Handle data available event
    mediaRecorder.ondataavailable = function (event) {
      console.log(
        `Data available for person ${personIndex}, size: ${event.data.size}`
      );
      if (event.data.size > 0) {
        if (personIndex === 1) {
          recordedChunks1.push(event.data);
        } else {
          recordedChunks2.push(event.data);
        }
      }
    };

    // Handle recording stop event
    mediaRecorder.onstop = function () {
      console.log(`Recording ${personIndex} stopped`);

      // Create the recorded video blob
      const blob = new Blob(
        personIndex === 1 ? recordedChunks1 : recordedChunks2,
        { type: "video/webm" }
      );

      // Create URL for the blob
      const url = URL.createObjectURL(blob);

      // Display the recorded video
      const videoElement = document.getElementById(`recording${personIndex}`);
      if (videoElement) {
        videoElement.srcObject = null;
        videoElement.src = url;
        videoElement.controls = true;
        videoElement.play();
      }

      // Enable analyze button if both videos are recorded
      if (recordedChunks1.length > 0 && recordedChunks2.length > 0) {
        const analyzeBtn = document.getElementById("analyze-recording-btn");
        if (analyzeBtn) analyzeBtn.disabled = false;
      }
    };

    // Start recording
    mediaRecorder.start(1000); // Capture in 1-second chunks
    console.log(`Recording ${personIndex} started`);

    return true;
  } catch (error) {
    console.error(
      `Error creating MediaRecorder for person ${personIndex}:`,
      error
    );
    alert(
      `Error creating recording for person ${personIndex}: ${error.message}`
    );
    return false;
  }
}

// Function to stop recording for a specific person
function stopRecordingSession(personIndex) {
  console.log(`Stopping recording session for person ${personIndex}`);

  // Stop media recorder
  if (
    personIndex === 1 &&
    mediaRecorder1 &&
    mediaRecorder1.state !== "inactive"
  ) {
    mediaRecorder1.stop();
    console.log("Stopped mediaRecorder1");
  } else if (
    personIndex === 2 &&
    mediaRecorder2 &&
    mediaRecorder2.state !== "inactive"
  ) {
    mediaRecorder2.stop();
    console.log("Stopped mediaRecorder2");
  } else {
    console.log(`MediaRecorder ${personIndex} not active or not found`);
  }

  // Update status display
  const status = document.getElementById(`status${personIndex}`);
  if (status) status.textContent = "Ready";
}
