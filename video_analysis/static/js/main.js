let metrics = [], analysisText = "";
let chart, lastSecond = -1;

window.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("fileInput");
  const startBtn  = document.getElementById("startBtn");
  const pauseBtn  = document.getElementById("pauseBtn");
  const video     = document.getElementById("videoPlayer");
  const toggleDash= document.getElementById("toggleDash");
  const dashContent = document.getElementById("dashContent");
  const comfortV  = document.getElementById("comfortValue");
  const valenceV  = document.getElementById("valenceValue");
  const engageV   = document.getElementById("engagementValue");
  const feedback  = document.getElementById("feedbackText");
  const ctx       = document.getElementById("metricsChart").getContext("2d");

  // 初始化 Chart.js
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Valence',   data: [], borderColor: 'blue', fill: false },
        { label: 'Comfort',   data: [], borderColor: 'green', fill: false },
        { label: 'Engagement',data: [], borderColor: 'red', fill: false },
      ]
    },
    options: {
      animation: false,
      scales: { x: { title: { display: true, text: 'Time (s)' } } }
    }
  });

  // 折叠面板
  toggleDash.addEventListener("click", () => {
    dashContent.style.display = dashContent.style.display === "none" ? "block" : "none";
    toggleDash.textContent = dashContent.style.display === "none" ? "Dashboard ▼" : "Dashboard ▲";
  });

  // 上传视频并触发分析
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("video", file);

    fetch("/upload", { method: "POST", body: form })
      .then(r => r.json())
      .then(resp => {
        if (resp.status === "ok") {
          video.src = URL.createObjectURL(file);
          // 加载 metrics & analysis
          return Promise.all([
            fetch("/metrics").then(r=>r.json()),
            fetch("/analysis").then(r=>r.text())
          ]);
        }
      })
      .then(([mData, aText]) => {
        metrics = mData;
        analysisText = aText;
        startBtn.disabled = false;
      });
  });

  // 开始播放 & 同步更新
  startBtn.addEventListener("click", () => {
    video.play();
  });

  // 暂停
  pauseBtn.addEventListener("click", () => {
    video.pause();
  });

  // 每秒更新一次 UI
  video.addEventListener("timeupdate", () => {
    const sec = Math.floor(video.currentTime);
    if (sec !== lastSecond && metrics.length) {
      lastSecond = sec;
      const m = metrics[sec] || {valence:0, comfort:0, engagement:0};
      // 更新圆圈
      valenceV.textContent = m.valence.toFixed(2);
      comfortV.textContent = m.comfort.toFixed(2);
      engageV.textContent  = m.engagement.toFixed(2);
      // 更新图表
      chart.data.labels.push(sec);
      chart.data.datasets[0].data.push(m.valence);
      chart.data.datasets[1].data.push(m.comfort);
      chart.data.datasets[2].data.push(m.engagement);
      chart.update();
    }
  });

  // 播放完毕后显示建议
  video.addEventListener("ended", () => {
    feedback.textContent = analysisText;
    pauseBtn.disabled = true;
  });
});
