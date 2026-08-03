const learningRate = document.getElementById('learning-rate');
const optimizer = document.getElementById('optimizer');
const epoch = document.getElementById('epoch');
const loss = document.getElementById('loss');
const accuracy = document.getElementById('accuracy-label');
const batchSize = document.getElementById('batch-size');
const duration = document.getElementById('duration');
const version = document.getElementById('version');

(async () => {
    const socket = new WebSocket("ws://localhost:7001");

    let hasAccuracyDataset = false;
    let isUserInteracting = false; 

    const accuracyDatasetConfig = {
        label: 'Accuracy (%)',
        data: [], 
        borderColor: 'rgba(255, 159, 64, 1)',
        backgroundColor: '#ffffff00',
        borderWidth: 3, 
        tension: 0.2, 
        fill: false,
        yAxisID: 'yAccuracy'
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        // Update basic UI labels
        if (epoch) epoch.textContent = `${data.epoch} / ${data.totalEpoch}`;
        if (loss) loss.textContent = data.loss.toFixed(6);
        if (optimizer && data.optimizer) optimizer.textContent = data.optimizer;
        if (data.learningRate) learningRate.textContent = data.learningRate;
        if (data.totalBatchSize) batchSize.textContent = data.totalBatchSize;
        if (data.duration) duration.textContent = data.duration;
        if (data.version) version.textContent = data.version;

        // Push data to chart
        myLineChart.data.labels.push(data.epoch);
        myLineChart.data.datasets[0].data.push(data.loss);
        
        if (data.accuracy !== undefined && data.accuracy !== null) {
            if (accuracy) {
                accuracy.style.display = 'block';
                accuracy.innerHTML = `Accuracy: <span>${data.accuracy.toFixed(2)}%</span>`;
            }

            if (!hasAccuracyDataset) {
                myLineChart.data.datasets.push(accuracyDatasetConfig);
                myLineChart.options.scales.yAccuracy.display = true;
                hasAccuracyDataset = true;
            }

            const accDatasetIndex = myLineChart.data.datasets.findIndex(ds => ds.yAxisID === 'yAccuracy');
            if (accDatasetIndex !== -1) {
                myLineChart.data.datasets[accDatasetIndex].data.push(data.accuracy);
            }
        } else if (accuracy) {
            accuracy.style.display = 'none';
        }

        // Only move view window if user is NOT scrolling back in history
        if (!isUserInteracting) {
            const windowSize = 20;
            if (data.epoch > windowSize) {
                myLineChart.options.scales.x.min = data.epoch - windowSize;
                myLineChart.options.scales.x.max = data.epoch;
            } else {
                myLineChart.options.scales.x.min = 1;
                myLineChart.options.scales.x.max = windowSize;
            }
        }

        // Use 'none' mode for fast updates (prevents jittering at ~0.14ms/epoch)
        myLineChart.update('none');
    };

    const ctx = document.getElementById('chart');

    const myLineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [], 
            datasets: [
                {
                    label: 'Loss',
                    data: [], 
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: '#ffffff00',
                    borderWidth: 3, 
                    tension: 0.2, 
                    fill: true,
                    yAxisID: 'yLoss'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear', // Use linear scale for numerical epoch spacing
                    min: 1,
                    max: 20
                },
                yLoss: {
                    type: 'linear',
                    position: 'left',
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'Loss' }
                },
                yAccuracy: {
                    type: 'linear',
                    position: 'right',
                    min: 0,
                    max: 100,
                    display: false,
                    grid: { drawOnChartArea: false },
                    title: { display: true, text: 'Accuracy (%)' }
                }
            },
            plugins: {
                zoom: {
                    pan: { 
                        enabled: true, 
                        mode: 'x',
                        onPanStart: () => { isUserInteracting = true; },
                        onPanComplete: ({ chart }) => {
                            // If user scrolled back to the very end, resume autoscroll
                            const currentMax = chart.scales.x.max;
                            const totalDataPoints = chart.data.labels.length;
                            if (currentMax >= totalDataPoints - 1) {
                                isUserInteracting = false;
                            }
                        }
                    },
                    zoom: {
                        wheel: { enabled: true },
                        pinch: { enabled: true },
                        mode: 'x',
                        onZoomStart: () => { isUserInteracting = true; }
                    }
                }
            }
        }
    });
})();