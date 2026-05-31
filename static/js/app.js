// ============================================================
        //  GLOBAL STATE
        // ============================================================
        const STATE = {
            history: [],
            totalGraded: 0,
            grades: { 'Grade A': 0, 'Grade B': 0, 'Grade C': 0, 'Reject': 0 },
            latencies: [],
            confidences: [],
            volumeData: [],
            cameraActive: false,
            cameraStream: null,
            cameraCaptures: 0,
            cameraAutoGraded: 0,
            cameraFlagged: 0,
            cameraLatencies: [],
            currentFilter: 'all',
            batchFiles: [],
            flaggedItem: null,
            animatedAccuracy: 0,
            animatedLatency: 0,
        };

        // Grade definitions
        const GRADES = {
            'Grade A': { color: 'var(--grade-premium)', class: 'grade-premium', icon: 'fa-crown', desc: 'Premium quality, uniform color, no defects' },
            'Grade B': { color: 'var(--grade-standard)', class: 'grade-standard', icon: 'fa-star', desc: 'Standard quality, minor color variation' },
            'Grade C': { color: 'var(--grade-substandard)', class: 'grade-substandard', icon: 'fa-exclamation', desc: 'Sub-standard, visible defects or discoloration' },
            'Reject': { color: 'var(--grade-reject)', class: 'grade-reject', icon: 'fa-times-circle', desc: 'Rejected — severe damage, rot, or contamination' },
        };

        // Sample chilli data for simulation
        const SAMPLE_CHILLIES = [
            { name: 'Premium Red Chilli', seed: 'premium1', expectedGrade: 'Grade A' },
            { name: 'Guntur Sannam', seed: 'guntur1', expectedGrade: 'Grade A' },
            { name: 'Byadgi Chilli', seed: 'byadgi1', expectedGrade: 'Grade B' },
            { name: 'Kashmiri Chilli', seed: 'kashmiri1', expectedGrade: 'Grade A' },
            { name: 'Jwala Chilli', seed: 'jwala1', expectedGrade: 'Grade B' },
            { name: 'Bird Eye Chilli', seed: 'birdseye1', expectedGrade: 'Grade A' },
            { name: 'Discolored Lot', seed: 'discolor1', expectedGrade: 'Grade C' },
            { name: 'Sun-damaged Batch', seed: 'sun_damage1', expectedGrade: 'Grade C' },
            { name: 'Mold Spotted', seed: 'mold1', expectedGrade: 'Reject' },
            { name: 'Rotting Sample', seed: 'rot1', expectedGrade: 'Reject' },
            { name: 'Green Chilli Mix', seed: 'green1', expectedGrade: 'Grade B' },
            { name: 'Wrinkled Batch', seed: 'wrinkle1', expectedGrade: 'Grade C' },
        ];

        // ============================================================
        //  UTILITY FUNCTIONS
        // ============================================================
        function showToast(message, type = 'info') {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = 'toast';
            const icons = { success: 'fa-check-circle', error: 'fa-exclamation-circle', info: 'fa-info-circle', warning: 'fa-exclamation-triangle' };
            const colors = { success: 'var(--grade-premium)', error: 'var(--grade-reject)', info: 'var(--accent)', warning: 'var(--accent-gold)' };
            toast.innerHTML = `<i class="fas ${icons[type]}" style="color:${colors[type]};font-size:16px;"></i><span>${message}</span>`;
            container.appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
        }

        function scrollToSection(id) {
            document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
        }

        function formatTime(date) {
            return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }

        // ============================================================
        //  SIMULATED ML GRADING ENGINE
        //  Analyzes pixel data from canvas to produce a realistic grading result
        // ============================================================
        function analyzeImage(canvas) {
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            const imageData = ctx.getImageData(0, 0, w, h);
            const data = imageData.data;

            let totalR = 0, totalG = 0, totalB = 0;
            let redPixels = 0, greenPixels = 0, darkPixels = 0, brightPixels = 0;
            let totalSaturation = 0;
            let pixelCount = 0;
            // Sample every 4th pixel for speed
            for (let i = 0; i < data.length; i += 16) {
                const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
                if (a < 128) continue;
                pixelCount++;
                totalR += r; totalG += g; totalB += b;

                const max = Math.max(r, g, b), min = Math.min(r, g, b);
                const sat = max === 0 ? 0 : (max - min) / max;
                totalSaturation += sat;

                if (r > 150 && g < 100 && b < 80) redPixels++;
                if (g > r && g > 100) greenPixels++;
                if (r < 60 && g < 60 && b < 60) darkPixels++;
                if (r > 200 && g > 180 && b > 150) brightPixels++;
            }

            if (pixelCount === 0) pixelCount = 1;
            const avgR = totalR / pixelCount;
            const avgG = totalG / pixelCount;
            const avgB = totalB / pixelCount;
            const avgSat = totalSaturation / pixelCount;
            const redRatio = redPixels / pixelCount;
            const greenRatio = greenPixels / pixelCount;
            const darkRatio = darkPixels / pixelCount;
            const brightRatio = brightPixels / pixelCount;
            const colorVariance = Math.sqrt(((avgR - 180) ** 2 + (avgG - 60) ** 2 + (avgB - 40) ** 2) / 3);

            // Determine grade based on color analysis
            let scores = { 'Grade A': 0, 'Grade B': 0, 'Grade C': 0, 'Reject': 0 };

            // Grade A: Deep red, high saturation, uniform
            scores['Grade A'] = redRatio * 60 + avgSat * 30 + (1 - colorVariance / 150) * 20 - greenRatio * 40 - darkRatio * 30;

            // Grade B: Moderate red, some variation
            scores['Grade B'] = redRatio * 30 + avgSat * 15 + (1 - Math.abs(redRatio - 0.3)) * 25 - darkRatio * 15;

            // Grade C: Low red, visible green/discoloration
            scores['Grade C'] = greenRatio * 40 + (1 - redRatio) * 20 + colorVariance / 150 * 15 - avgSat * 10;

            // Reject: Dark spots, extreme discoloration, mold-like colors
            scores['Reject'] = darkRatio * 50 + greenRatio * 20 + (avgB > avgR ? 15 : 0) - redRatio * 30;

            // Add small random variation for realism
            for (let g in scores) {
                scores[g] += (Math.random() - 0.5) * 8;
                scores[g] = Math.max(0, scores[g]);
            }

            // Softmax to get probabilities
            const maxScore = Math.max(...Object.values(scores));
            const expScores = {};
            let expSum = 0;
            for (let g in scores) {
                expScores[g] = Math.exp(scores[g] - maxScore);
                expSum += expScores[g];
            }
            const probabilities = {};
            for (let g in scores) {
                probabilities[g] = expScores[g] / expSum;
            }

            // Pick the grade with highest probability
            let bestGrade = 'Grade A', bestProb = 0;
            for (let g in probabilities) {
                if (probabilities[g] > bestProb) { bestProb = probabilities[g]; bestGrade = g; }
            }

            // Detect attributes
            const attributes = [];
            if (redRatio > 0.4) attributes.push({ label: 'Deep Red', icon: 'fa-circle', color: '#e63926' });
            if (greenRatio > 0.15) attributes.push({ label: 'Green Spots', icon: 'fa-leaf', color: '#22c55e' });
            if (darkRatio > 0.2) attributes.push({ label: 'Dark Areas', icon: 'fa-moon', color: '#6b7280' });
            if (avgSat > 0.6) attributes.push({ label: 'High Saturation', icon: 'fa-palette', color: '#f4a623' });
            if (brightRatio > 0.15) attributes.push({ label: 'Overexposed', icon: 'fa-sun', color: '#fbbf24' });
            if (colorVariance > 80) attributes.push({ label: 'Color Variation', icon: 'fa-adjust', color: '#a78bfa' });
            if (avgR < 100) attributes.push({ label: 'Low Red Content', icon: 'fa-tint-slash', color: '#94a3b8' });
            if (attributes.length === 0) attributes.push({ label: 'Uniform Color', icon: 'fa-check', color: 'var(--grade-premium)' });

            // Simulate latency (real CNN would be 5-50ms)
            const latency = Math.round(8 + Math.random() * 25);

            return {
                grade: bestGrade,
                confidence: bestProb,
                probabilities,
                latency,
                attributes,
                metrics: { avgR: Math.round(avgR), avgG: Math.round(avgG), avgB: Math.round(avgB), avgSat: avgSat.toFixed(2), redRatio: (redRatio * 100).toFixed(1), darkRatio: (darkRatio * 100).toFixed(1) }
            };
        }

        // ============================================================
        //  GRADING RESULT DISPLAY
        // ============================================================
        function displayResult(result, source = 'upload') {
            const gradeInfo = GRADES[result.grade];

            // Update result panel
            document.getElementById('resultPlaceholder').style.display = 'none';
            document.getElementById('resultContent').style.display = 'block';

            // Grade badge
            const gradeEl = document.getElementById('resultGrade');
            gradeEl.className = `grade-badge ${gradeInfo.class}`;
            gradeEl.innerHTML = `<i class="fas ${gradeInfo.icon}"></i> ${result.grade}`;

            // Confidence and latency
            document.getElementById('resultConfidence').textContent = (result.confidence * 100).toFixed(1) + '%';
            document.getElementById('resultConfidence').style.color = result.confidence > 0.85 ? 'var(--grade-premium)' : result.confidence > 0.7 ? 'var(--accent-gold)' : 'var(--grade-reject)';
            document.getElementById('resultLatency').textContent = result.latency + 'ms';

            // Probability bars
            const barsContainer = document.getElementById('probabilityBars');
            barsContainer.innerHTML = '';
            const sortedGrades = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
            sortedGrades.forEach(([grade, prob]) => {
                const gInfo = GRADES[grade];
                const pct = (prob * 100).toFixed(1);
                barsContainer.innerHTML += `
      <div style="margin-bottom:8px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
          <span style="font-size:12px;color:var(--fg-muted);">${grade}</span>
          <span style="font-size:12px;font-weight:600;">${pct}%</span>
        </div>
        <div style="height:5px;border-radius:3px;background:var(--border);overflow:hidden;">
          <div style="height:100%;width:${pct}%;border-radius:3px;background:${gInfo.color};transition:width 0.5s ease;"></div>
        </div>
      </div>`;
            });

            // Attributes
            const attrContainer = document.getElementById('detectedAttributes');
            attrContainer.innerHTML = '';
            result.attributes.forEach(attr => {
                attrContainer.innerHTML += `<span style="padding:4px 10px;border-radius:6px;background:rgba(33,26,20,0.8);border:1px solid var(--border);font-size:12px;display:inline-flex;align-items:center;gap:4px;">
      <i class="fas ${attr.icon}" style="color:${attr.color};font-size:10px;"></i>${attr.label}
    </span>`;
            });

            // Record to history
            addToHistory(result, source);
        }

        function addToHistory(result, source) {
            STATE.totalGraded++;
            STATE.grades[result.grade]++;
            STATE.latencies.push(result.latency);
            STATE.confidences.push(result.confidence);

            // Volume data (simulated time buckets)
            const now = new Date();
            const timeKey = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0');
            if (STATE.volumeData.length > 0 && STATE.volumeData[STATE.volumeData.length - 1].time === timeKey) {
                STATE.volumeData[STATE.volumeData.length - 1].count++;
            } else {
                STATE.volumeData.push({ time: timeKey, count: 1 });
                if (STATE.volumeData.length > 20) STATE.volumeData.shift();
            }

            const isFlagged = result.confidence < parseFloat(document.getElementById('confidenceThreshold').value);
            const entry = {
                id: STATE.totalGraded,
                grade: result.grade,
                confidence: result.confidence,
                latency: result.latency,
                source: source,
                timestamp: now,
                flagged: isFlagged,
                attributes: result.attributes,
            };
            STATE.history.unshift(entry);

            updateDashboard();
            updateHeroStats();
        }

        // ============================================================
        //  FILE UPLOAD HANDLING
        // ============================================================
        function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            processSingleFile(file);
        }

        function processSingleFile(file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                const img = new Image();
                img.onload = function () {
                    // Show preview with scan animation
                    const preview = document.getElementById('previewImage');
                    const container = document.getElementById('previewContainer');
                    const scanLine = document.getElementById('scanLine');
                    preview.src = e.target.result;
                    container.style.display = 'block';
                    scanLine.style.display = 'block';

                    // Draw to canvas for analysis
                    const canvas = document.createElement('canvas');
                    const maxSize = 300;
                    const scale = Math.min(maxSize / img.width, maxSize / img.height, 1);
                    canvas.width = Math.max(1, Math.round(img.width * scale));
                    canvas.height = Math.max(1, Math.round(img.height * scale));
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                    // Simulate processing delay
                    setTimeout(() => {
                        scanLine.style.display = 'none';
                        const result = analyzeImage(canvas);
                        displayResult(result, 'upload');
                        showToast(`Graded as ${result.grade} (${(result.confidence * 100).toFixed(1)}% confidence)`, result.grade === 'Reject' ? 'warning' : 'success');
                    }, 1500);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        // Drag and drop
        const uploadZone = document.getElementById('uploadZone');
        uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
        uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) processSingleFile(e.dataTransfer.files[0]);
        });

        function resetGrading() {
            document.getElementById('previewContainer').style.display = 'none';
            document.getElementById('resultPlaceholder').style.display = 'block';
            document.getElementById('resultContent').style.display = 'none';
            document.getElementById('fileInput').value = '';
        }

        // ============================================================
        //  BATCH PROCESSING
        // ============================================================
        function handleBatchUpload(event) {
            STATE.batchFiles = Array.from(event.target.files);
            document.getElementById('batchCount').textContent = STATE.batchFiles.length + ' images selected';
            document.getElementById('batchStartBtn').style.display = STATE.batchFiles.length > 0 ? 'block' : 'none';
            document.getElementById('batchResults').innerHTML = '';
        }

        async function startBatchProcessing() {
            if (STATE.batchFiles.length === 0) return;
            const btn = document.getElementById('batchStartBtn');
            btn.disabled = true;
            const progress = document.getElementById('batchProgress');
            const bar = document.getElementById('batchProgressBar');
            const text = document.getElementById('batchProgressText');
            const results = document.getElementById('batchResults');
            progress.style.display = 'block';
            results.innerHTML = '';

            for (let i = 0; i < STATE.batchFiles.length; i++) {
                text.textContent = `${i + 1}/${STATE.batchFiles.length}`;
                bar.style.width = ((i + 1) / STATE.batchFiles.length * 100) + '%';

                await new Promise((resolve) => {
                    const reader = new FileReader();
                    reader.onload = function (e) {
                        const img = new Image();
                        img.onload = function () {
                            const canvas = document.createElement('canvas');
                            canvas.width = Math.max(1, Math.round(img.width * 0.3));
                            canvas.height = Math.max(1, Math.round(img.height * 0.3));
                            canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);

                            setTimeout(() => {
                                const result = analyzeImage(canvas);
                                const gInfo = GRADES[result.grade];
                                const card = document.createElement('div');
                                card.className = 'glass-card';
                                card.style.padding = '12px';
                                card.style.animation = 'fadeIn 0.3s ease';
                                card.innerHTML = `
              <img src="${e.target.result}" style="width:100%;height:100px;object-fit:cover;border-radius:8px;margin-bottom:8px;">
              <div class="grade-badge ${gInfo.class}" style="margin-bottom:4px;font-size:11px;padding:3px 8px;">
                <i class="fas ${gInfo.icon}"></i>${result.grade}
              </div>
              <div style="font-size:11px;color:var(--fg-muted);">${(result.confidence * 100).toFixed(1)}% conf · ${result.latency}ms</div>
            `;
                                results.appendChild(card);
                                addToHistory(result, 'batch');
                                resolve();
                            }, 100 + Math.random() * 200);
                        };
                        img.src = e.target.result;
                    };
                    reader.readAsDataURL(STATE.batchFiles[i]);
                });
            }

            btn.disabled = false;
            progress.style.display = 'none';
            showToast(`Batch complete: ${STATE.batchFiles.length} images graded`, 'success');
            STATE.batchFiles = [];
        }

        // ============================================================
        //  SAMPLE DATA TAB
        // ============================================================
        function initSamples() {
            const grid = document.getElementById('samplesGrid');
            SAMPLE_CHILLIES.forEach((sample, idx) => {
                const gInfo = GRADES[sample.expectedGrade];
                const card = document.createElement('div');
                card.className = 'glass-card';
                card.style.padding = '16px';
                card.style.cursor = 'pointer';
                card.onclick = () => gradeSample(idx, card);
                card.innerHTML = `
      <div style="width:100%;height:120px;border-radius:10px;margin-bottom:12px;overflow:hidden;position:relative;background:linear-gradient(135deg,
        ${sample.expectedGrade === 'Grade A' ? '#8b1a1a,#c0392b' :
                        sample.expectedGrade === 'Grade B' ? '#935116,#d4770b' :
                            sample.expectedGrade === 'Grade C' ? '#5c4a1e,#8b7d3c' :
                                '#2d2d2d,#4a3a3a'});">
        <canvas class="sample-canvas" width="220" height="120" data-index="${idx}" style="width:100%;height:100%;"></canvas>
      </div>
      <div style="font-size:14px;font-weight:600;margin-bottom:4px;">${sample.name}</div>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:12px;color:var(--fg-muted);">Expected: ${sample.expectedGrade}</span>
        <i class="fas fa-play-circle" style="color:var(--accent);font-size:18px;"></i>
      </div>
    `;
                grid.appendChild(card);
            });

            // Draw simulated chilli images on sample canvases
            document.querySelectorAll('.sample-canvas').forEach(canvas => {
                drawSimulatedChilli(canvas, parseInt(canvas.dataset.index));
            });
        }

        function drawSimulatedChilli(canvas, index) {
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            const sample = SAMPLE_CHILLIES[index];

            // Use seed for consistent randomness
            let seed = 0;
            for (let i = 0; i < sample.seed.length; i++) seed += sample.seed.charCodeAt(i) * (i + 1);
            const seededRandom = () => { seed = (seed * 16807 + 0) % 2147483647; return (seed - 1) / 2147483646; };

            // Background
            const bgColors = {
                'Grade A': ['#1a0808', '#2d0f0f'],
                'Grade B': ['#1a1208', '#2d1f0f'],
                'Grade C': ['#14140a', '#2d2a10'],
                'Reject': ['#0f0f0f', '#1f1a1a'],
            };
            const [bg1, bg2] = bgColors[sample.expectedGrade];
            const grad = ctx.createLinearGradient(0, 0, w, h);
            grad.addColorStop(0, bg1);
            grad.addColorStop(1, bg2);
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, w, h);

            // Draw chilli shapes
            const chiliCount = 2 + Math.floor(seededRandom() * 3);
            for (let c = 0; c < chiliCount; c++) {
                const cx = 40 + seededRandom() * (w - 80);
                const cy = 20 + seededRandom() * (h - 40);
                const angle = -0.5 + seededRandom() * 1;
                const length = 40 + seededRandom() * 50;
                const width = 8 + seededRandom() * 8;

                ctx.save();
                ctx.translate(cx, cy);
                ctx.rotate(angle);

                // Chilli body color
                let r, g, b;
                if (sample.expectedGrade === 'Grade A') {
                    r = 180 + Math.floor(seededRandom() * 60);
                    g = 20 + Math.floor(seededRandom() * 30);
                    b = 10 + Math.floor(seededRandom() * 20);
                } else if (sample.expectedGrade === 'Grade B') {
                    r = 160 + Math.floor(seededRandom() * 60);
                    g = 40 + Math.floor(seededRandom() * 50);
                    b = 10 + Math.floor(seededRandom() * 20);
                } else if (sample.expectedGrade === 'Grade C') {
                    r = 120 + Math.floor(seededRandom() * 60);
                    g = 60 + Math.floor(seededRandom() * 60);
                    b = 10 + Math.floor(seededRandom() * 30);
                } else {
                    r = 60 + Math.floor(seededRandom() * 40);
                    g = 40 + Math.floor(seededRandom() * 30);
                    b = 30 + Math.floor(seededRandom() * 20);
                }

                // Draw chilli body (ellipse with taper)
                ctx.beginPath();
                ctx.moveTo(0, -width / 2);
                ctx.bezierCurveTo(length * 0.3, -width / 2 - 2, length * 0.7, -width / 3, length, 0);
                ctx.bezierCurveTo(length * 0.7, width / 3, length * 0.3, width / 2 + 2, 0, width / 2);
                ctx.closePath();

                const bodyGrad = ctx.createLinearGradient(0, -width, 0, width);
                bodyGrad.addColorStop(0, `rgb(${Math.min(255, r + 30)},${g + 10},${b})`);
                bodyGrad.addColorStop(0.5, `rgb(${r},${g},${b})`);
                bodyGrad.addColorStop(1, `rgb(${Math.max(0, r - 40)},${Math.max(0, g - 10)},${Math.max(0, b - 10)})`);
                ctx.fillStyle = bodyGrad;
                ctx.fill();

                // Stem
                ctx.beginPath();
                ctx.moveTo(-2, -width / 3);
                ctx.lineTo(-10 - seededRandom() * 8, -width / 2 - 5 - seededRandom() * 5);
                ctx.lineTo(2, -width / 4);
                ctx.closePath();
                ctx.fillStyle = '#2d5016';
                ctx.fill();

                // Add defects for lower grades
                if (sample.expectedGrade === 'Grade C' || sample.expectedGrade === 'Reject') {
                    for (let d = 0; d < 3 + Math.floor(seededRandom() * 4); d++) {
                        const dx = seededRandom() * length * 0.8;
                        const dy = (seededRandom() - 0.5) * width * 0.8;
                        ctx.beginPath();
                        ctx.arc(dx, dy, 2 + seededRandom() * 4, 0, Math.PI * 2);
                        ctx.fillStyle = sample.expectedGrade === 'Reject' ? `rgba(20,20,10,${0.5 + seededRandom() * 0.4})` : `rgba(80,100,20,${0.3 + seededRandom() * 0.3})`;
                        ctx.fill();
                    }
                }

                ctx.restore();
            }

            // Add subtle noise
            const imageData = ctx.getImageData(0, 0, w, h);
            for (let i = 0; i < imageData.data.length; i += 4) {
                const noise = (seededRandom() - 0.5) * 15;
                imageData.data[i] = Math.max(0, Math.min(255, imageData.data[i] + noise));
                imageData.data[i + 1] = Math.max(0, Math.min(255, imageData.data[i + 1] + noise));
                imageData.data[i + 2] = Math.max(0, Math.min(255, imageData.data[i + 2] + noise));
            }
            ctx.putImageData(imageData, 0, 0);
        }

        function gradeSample(index, card) {
            const canvas = card.querySelector('.sample-canvas');
            card.style.opacity = '0.6';
            card.style.pointerEvents = 'none';

            // Show scan effect
            const scanDiv = document.createElement('div');
            scanDiv.className = 'scan-line';
            card.querySelector('div').style.position = 'relative';
            card.querySelector('div').appendChild(scanDiv);

            setTimeout(() => {
                scanDiv.remove();
                const result = analyzeImage(canvas);
                displayResult(result, 'sample');
                card.style.opacity = '1';
                card.style.pointerEvents = 'auto';
                showToast(`${SAMPLE_CHILLIES[index].name}: ${result.grade} (${(result.confidence * 100).toFixed(1)}%)`, 'success');

                // Switch to upload tab to show result
                switchGradingTab('upload', document.querySelector('.tab-btn'));
            }, 1800);
        }

        // ============================================================
        //  TAB SWITCHING
        // ============================================================
        function switchGradingTab(tab, btn) {
            document.querySelectorAll('[id^="tab-"]').forEach(t => t.style.display = 'none');
            document.getElementById('tab-' + tab).style.display = tab === 'batch' ? 'block' : (tab === 'samples' ? 'block' : 'grid');
            btn.parentElement.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        }

        // ============================================================
        //  CAMERA SYSTEM
        // ============================================================
        let cameraAnimFrame = null;

        async function toggleCamera() {
            const btn = document.getElementById('cameraStartBtn');
            const captureBtn = document.getElementById('captureBtn');

            if (STATE.cameraActive) {
                // Stop camera
                STATE.cameraActive = false;
                if (STATE.cameraStream) {
                    STATE.cameraStream.getTracks().forEach(t => t.stop());
                    STATE.cameraStream = null;
                }
                if (cameraAnimFrame) cancelAnimationFrame(cameraAnimFrame);
                btn.innerHTML = '<i class="fas fa-video" style="margin-right:8px;"></i>Start Camera';
                captureBtn.disabled = true;
                document.getElementById('cameraStatusText').textContent = 'Offline';
                document.getElementById('cameraStatusDot').style.background = 'var(--grade-reject)';
                document.getElementById('cameraGradeOverlay').style.display = 'none';

                // Draw placeholder
                const canvas = document.getElementById('cameraCanvas');
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#333';
                ctx.font = '16px "DM Sans", sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('Camera offline — click Start Camera to begin', canvas.width / 2, canvas.height / 2);
                return;
            }

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 800 }, height: { ideal: 500 } } });
                STATE.cameraStream = stream;
                STATE.cameraActive = true;
                btn.innerHTML = '<i class="fas fa-video-slash" style="margin-right:8px;"></i>Stop Camera';
                captureBtn.disabled = false;
                document.getElementById('cameraStatusText').textContent = 'Live';
                document.getElementById('cameraStatusDot').style.background = 'var(--grade-premium)';
                showToast('Camera activated — real-time grading ready', 'success');

                const video = document.createElement('video');
                video.srcObject = stream;
                video.play();

                const canvas = document.getElementById('cameraCanvas');
                const ctx = canvas.getContext('2d');
                let lastTime = performance.now(), frameCount = 0;

                function renderCamera() {
                    if (!STATE.cameraActive) return;
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                    // Draw overlay grid
                    ctx.strokeStyle = 'rgba(230,57,38,0.15)';
                    ctx.lineWidth = 1;
                    for (let x = 0; x < canvas.width; x += 80) {
                        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
                    }
                    for (let y = 0; y < canvas.height; y += 80) {
                        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
                    }

                    // Corner markers
                    const m = 20, ml = 40;
                    ctx.strokeStyle = 'rgba(230,57,38,0.6)';
                    ctx.lineWidth = 2;
                    // Top-left
                    ctx.beginPath(); ctx.moveTo(m, m + ml); ctx.lineTo(m, m); ctx.lineTo(m + ml, m); ctx.stroke();
                    // Top-right
                    ctx.beginPath(); ctx.moveTo(canvas.width - m - ml, m); ctx.lineTo(canvas.width - m, m); ctx.lineTo(canvas.width - m, m + ml); ctx.stroke();
                    // Bottom-left
                    ctx.beginPath(); ctx.moveTo(m, canvas.height - m - ml); ctx.lineTo(m, canvas.height - m); ctx.lineTo(m + ml, canvas.height - m); ctx.stroke();
                    // Bottom-right
                    ctx.beginPath(); ctx.moveTo(canvas.width - m, canvas.height - m - ml); ctx.lineTo(canvas.width - m, canvas.height - m); ctx.lineTo(canvas.width - m - ml, canvas.height - m); ctx.stroke();

                    // FPS counter
                    frameCount++;
                    const now = performance.now();
                    if (now - lastTime >= 1000) {
                        document.getElementById('cameraFps').textContent = frameCount + ' FPS';
                        frameCount = 0;
                        lastTime = now;
                    }

                    cameraAnimFrame = requestAnimationFrame(renderCamera);
                }
                renderCamera();
            } catch (err) {
                showToast('Camera access denied — using simulated feed instead', 'warning');
                startSimulatedCamera();
            }
        }

        function startSimulatedCamera() {
            STATE.cameraActive = true;
            document.getElementById('cameraStartBtn').innerHTML = '<i class="fas fa-video-slash" style="margin-right:8px;"></i>Stop Camera';
            document.getElementById('captureBtn').disabled = false;
            document.getElementById('cameraStatusText').textContent = 'Simulated';
            document.getElementById('cameraStatusDot').style.background = 'var(--accent-gold)';

            const canvas = document.getElementById('cameraCanvas');
            const ctx = canvas.getContext('2d');
            let frame = 0;

            function renderSim() {
                if (!STATE.cameraActive) return;
                frame++;

                // Dark background
                ctx.fillStyle = '#0a0806';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Conveyor belt
                ctx.fillStyle = '#1a1510';
                ctx.fillRect(0, canvas.height - 80, canvas.width, 80);
                const beltOffset = (frame * 2) % 40;
                ctx.strokeStyle = '#2a2018';
                ctx.lineWidth = 1;
                for (let x = -40 + beltOffset; x < canvas.width; x += 40) {
                    ctx.beginPath(); ctx.moveTo(x, canvas.height - 80); ctx.lineTo(x, canvas.height); ctx.stroke();
                }

                // Moving chillies
                const chilliX = ((frame * 3) % (canvas.width + 200)) - 100;
                for (let i = 0; i < 3; i++) {
                    const cx = chilliX - i * 200;
                    if (cx > -100 && cx < canvas.width + 100) {
                        drawCameraChilli(ctx, cx, canvas.height - 120, 50 + i * 10, frame + i * 100);
                    }
                }

                // Overlay grid
                ctx.strokeStyle = 'rgba(230,57,38,0.15)';
                ctx.lineWidth = 1;
                for (let x = 0; x < canvas.width; x += 80) {
                    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
                }
                for (let y = 0; y < canvas.height; y += 80) {
                    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
                }

                // Corner markers
                const m = 20, ml = 40;
                ctx.strokeStyle = 'rgba(230,57,38,0.6)';
                ctx.lineWidth = 2;
                ctx.beginPath(); ctx.moveTo(m, m + ml); ctx.lineTo(m, m); ctx.lineTo(m + ml, m); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(canvas.width - m - ml, m); ctx.lineTo(canvas.width - m, m); ctx.lineTo(canvas.width - m, m + ml); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(m, canvas.height - m - ml); ctx.lineTo(m, canvas.height - m); ctx.lineTo(m + ml, canvas.height - m); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(canvas.width - m, canvas.height - m - ml); ctx.lineTo(canvas.width - m, canvas.height - m); ctx.lineTo(canvas.width - m - ml, canvas.height - m); ctx.stroke();

                // Detection zone indicator
                ctx.strokeStyle = frame % 60 < 30 ? 'rgba(230,57,38,0.4)' : 'rgba(230,57,38,0.1)';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(canvas.width / 2 - 80, canvas.height / 2 - 60, 160, 120);
                ctx.setLineDash([]);

                document.getElementById('cameraFps').textContent = '30 FPS';
                cameraAnimFrame = requestAnimationFrame(renderSim);
            }
            renderSim();
        }

        function drawCameraChilli(ctx, x, y, size, seed) {
            const r = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
            const gradeRand = r();
            let red, green, blue;
            if (gradeRand < 0.4) { red = 180 + Math.floor(r() * 60); green = 20 + Math.floor(r() * 30); blue = 10; }
            else if (gradeRand < 0.7) { red = 160 + Math.floor(r() * 50); green = 40 + Math.floor(r() * 40); blue = 15; }
            else if (gradeRand < 0.9) { red = 120 + Math.floor(r() * 50); green = 60 + Math.floor(r() * 40); blue = 20; }
            else { red = 60 + Math.floor(r() * 40); green = 40 + Math.floor(r() * 20); blue = 30; }

            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(-0.3 + r() * 0.6);
            ctx.beginPath();
            ctx.ellipse(0, 0, size, size * 0.25, 0, 0, Math.PI * 2);
            const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, size);
            grad.addColorStop(0, `rgb(${Math.min(255, red + 20)},${green + 5},${blue})`);
            grad.addColorStop(1, `rgb(${Math.max(0, red - 30)},${Math.max(0, green - 10)},${Math.max(0, blue - 10)})`);
            ctx.fillStyle = grad;
            ctx.fill();
            // Stem
            ctx.beginPath();
            ctx.moveTo(-size + 2, -3);
            ctx.lineTo(-size - 10, -8 - r() * 5);
            ctx.lineTo(-size + 5, 0);
            ctx.closePath();
            ctx.fillStyle = '#2d5016';
            ctx.fill();
            ctx.restore();
        }

        function captureAndGrade() {
            if (!STATE.cameraActive) return;
            const canvas = document.getElementById('cameraCanvas');
            const result = analyzeImage(canvas);

            STATE.cameraCaptures++;
            const threshold = parseFloat(document.getElementById('confidenceThreshold').value);

            if (result.confidence < threshold) {
                STATE.cameraFlagged++;
                showFlagModal(result, canvas);
            } else {
                STATE.cameraAutoGraded++;
                addToHistory(result, 'camera');
                showToast(`Auto-graded: ${result.grade} (${(result.confidence * 100).toFixed(1)}%)`, 'success');
            }

            STATE.cameraLatencies.push(result.latency);

            // Update camera overlay
            const overlay = document.getElementById('cameraGradeOverlay');
            overlay.style.display = 'block';
            const gInfo = GRADES[result.grade];
            document.getElementById('cameraGradeText').textContent = result.grade;
            document.getElementById('cameraGradeText').style.color = gInfo.color;
            document.getElementById('cameraConfText').textContent = (result.confidence * 100).toFixed(1) + '%';
            document.getElementById('cameraConfText').style.color = result.confidence > threshold ? 'var(--grade-premium)' : 'var(--accent-gold)';

            // Update camera metrics
            document.getElementById('camTotalCaptured').textContent = STATE.cameraCaptures;
            document.getElementById('camAutoGraded').textContent = STATE.cameraAutoGraded;
            document.getElementById('camFlagged').textContent = STATE.cameraFlagged;
            if (STATE.cameraLatencies.length > 0) {
                const avg = STATE.cameraLatencies.reduce((a, b) => a + b, 0) / STATE.cameraLatencies.length;
                document.getElementById('camAvgLatency').textContent = avg.toFixed(1) + 'ms';
            }

            // Add to camera history
            const histContainer = document.getElementById('cameraHistory');
            if (STATE.cameraCaptures === 1) histContainer.innerHTML = '';
            const item = document.createElement('div');
            item.className = 'history-item';
            item.style.animation = 'fadeIn 0.3s ease';
            const isFlagged = result.confidence < threshold;
            item.innerHTML = `
    <div class="grade-badge ${gInfo.class}" style="font-size:10px;padding:3px 8px;">
      <i class="fas ${gInfo.icon}"></i>${result.grade}
    </div>
    <div style="flex:1;">
      <div style="font-size:12px;font-weight:600;">${(result.confidence * 100).toFixed(1)}% · ${result.latency}ms</div>
      <div style="font-size:11px;color:var(--fg-muted);">${formatTime(new Date())}</div>
    </div>
    ${isFlagged ? '<i class="fas fa-flag" style="color:var(--accent-gold);font-size:12px;"></i>' : ''}
  `;
            histContainer.insertBefore(item, histContainer.firstChild);
        }

        // ============================================================
        //  FLAG MODAL (Human-in-the-Loop)
        // ============================================================
        function showFlagModal(result, sourceCanvas) {
            STATE.flaggedItem = { result, sourceCanvas };
            const modal = document.getElementById('flagModal');
            modal.classList.add('show');

            const gInfo = GRADES[result.grade];
            document.getElementById('flagModalGrade').textContent = result.grade;
            document.getElementById('flagModalGrade').style.color = gInfo.color;
            document.getElementById('flagModalConf').textContent = (result.confidence * 100).toFixed(1) + '%';

            // Draw thumbnail
            const thumbCanvas = document.getElementById('flagModalCanvas');
            const tCtx = thumbCanvas.getContext('2d');
            tCtx.drawImage(sourceCanvas, 0, 0, thumbCanvas.width, thumbCanvas.height);
        }

        function closeFlagModal() {
            document.getElementById('flagModal').classList.remove('show');
            STATE.flaggedItem = null;
        }

        function overrideGrade(grade) {
            if (!STATE.flaggedItem) return;
            const result = { ...STATE.flaggedItem.result, grade, confidence: 1.0 };
            result.probabilities = {};
            for (let g of ['Grade A', 'Grade B', 'Grade C', 'Reject']) {
                result.probabilities[g] = g === grade ? 1.0 : 0.0;
            }
            result.attributes.push({ label: 'Human Override', icon: 'fa-user-check', color: 'var(--accent-gold)' });
            addToHistory(result, 'camera-override');
            showToast(`Grade overridden to ${grade} by human review`, 'info');
            closeFlagModal();
        }

        function confirmFlaggedGrade() {
            if (!STATE.flaggedItem) return;
            const result = STATE.flaggedItem.result;
            result.attributes.push({ label: 'Human Confirmed', icon: 'fa-user-check', color: 'var(--grade-premium)' });
            addToHistory(result, 'camera-confirmed');
            showToast(`Low-confidence ${result.grade} confirmed by human review`, 'info');
            closeFlagModal();
        }

        // ============================================================
        //  DASHBOARD & CHARTS
        // ============================================================
        let volumeChart, distributionChart, latencyChart;

        function initCharts() {
            const chartDefaults = {
                color: '#a89684',
                borderColor: '#3a2e22',
                font: { family: "'DM Sans', sans-serif" },
            };
            Chart.defaults.color = chartDefaults.color;
            Chart.defaults.borderColor = chartDefaults.borderColor;
            Chart.defaults.font.family = chartDefaults.font.family;

            // Volume chart
            volumeChart = new Chart(document.getElementById('volumeChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Graded',
                        data: [],
                        borderColor: '#e63926',
                        backgroundColor: 'rgba(230,57,38,0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 3,
                        pointBackgroundColor: '#e63926',
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: 'rgba(58,46,34,0.5)' } },
                        y: { grid: { color: 'rgba(58,46,34,0.5)' }, beginAtZero: true }
                    }
                }
            });

            // Distribution chart
            distributionChart = new Chart(document.getElementById('distributionChart'), {
                type: 'doughnut',
                data: {
                    labels: ['Grade A', 'Grade B', 'Grade C', 'Reject'],
                    datasets: [{
                        data: [0, 0, 0, 0],
                        backgroundColor: ['#22c55e', '#f4a623', '#f97316', '#ef4444'],
                        borderColor: '#211a14',
                        borderWidth: 3,
                    }]
                },
                options: {
                    responsive: true,
                    cutout: '65%',
                    plugins: {
                        legend: { position: 'bottom', labels: { padding: 16, usePointStyle: true, pointStyleWidth: 8 } }
                    }
                }
            });

            // Latency chart
            latencyChart = new Chart(document.getElementById('latencyChart'), {
                type: 'bar',
                data: {
                    labels: ['0-10ms', '10-15ms', '15-20ms', '20-25ms', '25-30ms', '30ms+'],
                    datasets: [{
                        label: 'Count',
                        data: [0, 0, 0, 0, 0, 0],
                        backgroundColor: [
                            'rgba(34,197,94,0.6)', 'rgba(34,197,94,0.4)',
                            'rgba(244,166,35,0.6)', 'rgba(244,166,35,0.4)',
                            'rgba(249,115,22,0.6)', 'rgba(239,68,68,0.6)'
                        ],
                        borderRadius: 6,
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false } },
                        y: { grid: { color: 'rgba(58,46,34,0.5)' }, beginAtZero: true }
                    }
                }
            });
        }

        function updateDashboard() {
            const total = STATE.totalGraded;

            // KPI cards
            document.getElementById('dashTotal').textContent = total;
            const premiumPct = total > 0 ? ((STATE.grades['Grade A'] / total) * 100).toFixed(1) : 0;
            const rejectPct = total > 0 ? ((STATE.grades['Reject'] / total) * 100).toFixed(1) : 0;
            document.getElementById('dashPremium').textContent = premiumPct + '%';
            document.getElementById('dashReject').textContent = rejectPct + '%';
            const avgConf = STATE.confidences.length > 0 ? (STATE.confidences.reduce((a, b) => a + b, 0) / STATE.confidences.length * 100).toFixed(1) + '%' : '—';
            document.getElementById('dashConfidence').textContent = avgConf;

            // Volume chart
            volumeChart.data.labels = STATE.volumeData.map(d => d.time);
            volumeChart.data.datasets[0].data = STATE.volumeData.map(d => d.count);
            volumeChart.update('none');

            // Distribution chart
            distributionChart.data.datasets[0].data = [STATE.grades['Grade A'], STATE.grades['Grade B'], STATE.grades['Grade C'], STATE.grades['Reject']];
            distributionChart.update('none');

            // Latency chart
            const buckets = [0, 0, 0, 0, 0, 0];
            STATE.latencies.forEach(l => {
                if (l < 10) buckets[0]++;
                else if (l < 15) buckets[1]++;
                else if (l < 20) buckets[2]++;
                else if (l < 25) buckets[3]++;
                else if (l < 30) buckets[4]++;
                else buckets[5]++;
            });
            latencyChart.data.datasets[0].data = buckets;
            latencyChart.update('none');

            // Business impact (simulated proportional to total graded)
            const factor = Math.min(1, total / 100); // Scale over 100 items
            const lossReduction = (factor * 15).toFixed(1);
            const satisfactionGain = (factor * 10).toFixed(1);
            const cogsReduction = (factor * 8).toFixed(1);
            const throughputGain = (factor * 10).toFixed(1);

            document.getElementById('impactLoss').textContent = lossReduction + '%';
            document.getElementById('impactLossBar').style.width = (factor * 100) + '%';
            document.getElementById('impactSatisfaction').textContent = satisfactionGain + '%';
            document.getElementById('impactSatisfactionBar').style.width = (factor * 100) + '%';
            document.getElementById('impactCOGS').textContent = cogsReduction + '%';
            document.getElementById('impactCOGSBar').style.width = (factor * 100) + '%';
            document.getElementById('impactThroughput').textContent = throughputGain + 'x';
            document.getElementById('impactThroughputBar').style.width = (factor * 100) + '%';

            // Update history
            renderHistory();
        }

        function updateHeroStats() {
            // Animate accuracy (simulated around 94-97%)
            const targetAccuracy = 94 + (STATE.totalGraded % 4);
            const targetLatency = STATE.latencies.length > 0 ? (STATE.latencies.reduce((a, b) => a + b, 0) / STATE.latencies.length).toFixed(0) : 0;

            animateValue('statAccuracy', STATE.animatedAccuracy, targetAccuracy, '%', 600);
            animateValue('statLatency', STATE.animatedLatency, parseInt(targetLatency), 'ms', 400);
            document.getElementById('statGraded').textContent = STATE.totalGraded;
        }

        function animateValue(elementId, start, end, suffix, duration) {
            const el = document.getElementById(elementId);
            const range = end - start;
            const startTime = performance.now();
            function update(currentTime) {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3); // ease out cubic
                const current = Math.round(start + range * eased);
                el.textContent = current + suffix;
                if (progress < 1) requestAnimationFrame(update);
            }
            requestAnimationFrame(update);
            if (elementId === 'statAccuracy') STATE.animatedAccuracy = end;
            if (elementId === 'statLatency') STATE.animatedLatency = end;
        }

        // ============================================================
        //  HISTORY
        // ============================================================
        function renderHistory() {
            const container = document.getElementById('historyList');
            const filtered = STATE.currentFilter === 'all'
                ? STATE.history
                : STATE.currentFilter === 'flagged'
                    ? STATE.history.filter(h => h.flagged)
                    : STATE.history.filter(h => h.grade === STATE.currentFilter);

            if (filtered.length === 0) {
                container.innerHTML = `<div style="text-align:center;padding:40px 0;color:var(--fg-muted);font-size:15px;">
      <i class="fas fa-inbox" style="font-size:36px;display:block;margin-bottom:12px;color:var(--border);"></i>
      No ${STATE.currentFilter === 'all' ? '' : STATE.currentFilter + ' '}records found
    </div>`;
                return;
            }

            container.innerHTML = filtered.map(item => {
                const gInfo = GRADES[item.grade];
                const sourceIcons = { upload: 'fa-upload', batch: 'fa-layer-group', sample: 'fa-flask', camera: 'fa-video', 'camera-override': 'fa-user-edit', 'camera-confirmed': 'fa-user-check' };
                return `<div class="history-item" style="animation:fadeIn 0.2s ease;">
      <div style="width:36px;height:36px;border-radius:8px;background:rgba(33,26,20,0.8);display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <i class="fas ${sourceIcons[item.source] || 'fa-box'}" style="color:var(--fg-muted);font-size:12px;"></i>
      </div>
      <div style="flex:1;min-width:0;">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:13px;font-weight:600;">#${item.id}</span>
          <div class="grade-badge ${gInfo.class}" style="font-size:10px;padding:2px 8px;">
            <i class="fas ${gInfo.icon}"></i>${item.grade}
          </div>
          ${item.flagged ? '<i class="fas fa-flag" style="color:var(--accent-gold);font-size:11px;" title="Flagged for review"></i>' : ''}
        </div>
        <div style="font-size:12px;color:var(--fg-muted);margin-top:2px;">
          ${(item.confidence * 100).toFixed(1)}% confidence · ${item.latency}ms · ${formatTime(item.timestamp)}
        </div>
      </div>
    </div>`;
            }).join('');
        }

        function filterHistory(filter, btn) {
            STATE.currentFilter = filter;
            document.querySelectorAll('#historyFilters .tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderHistory();
        }

        function clearHistory() {
            STATE.history = [];
            STATE.totalGraded = 0;
            STATE.grades = { 'Grade A': 0, 'Grade B': 0, 'Grade C': 0, 'Reject': 0 };
            STATE.latencies = [];
            STATE.confidences = [];
            STATE.volumeData = [];
            updateDashboard();
            updateHeroStats();
            showToast('History cleared', 'info');
        }

        // ============================================================
        //  HERO CANVAS ANIMATION
        // ============================================================
        function initHeroCanvas() {
            const canvas = document.getElementById('heroCanvas');
            const ctx = canvas.getContext('2d');
            let frame = 0;

            function draw() {
                frame++;
                ctx.fillStyle = '#0a0806';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Draw simulated sorting visualization
                // Conveyor belt
                ctx.fillStyle = '#151008';
                ctx.fillRect(0, 320, canvas.width, 80);
                const beltOff = (frame * 1.5) % 30;
                ctx.strokeStyle = '#1f1a10';
                ctx.lineWidth = 1;
                for (let x = -30 + beltOff; x < canvas.width; x += 30) {
                    ctx.beginPath(); ctx.moveTo(x, 320); ctx.lineTo(x, 400); ctx.stroke();
                }

                // Bins at bottom
                const bins = [
                    { x: 80, label: 'Grade A', color: '#22c55e' },
                    { x: 200, label: 'Grade B', color: '#f4a623' },
                    { x: 320, label: 'Grade C', color: '#f97316' },
                    { x: 440, label: 'Reject', color: '#ef4444' },
                ];
                bins.forEach(bin => {
                    ctx.fillStyle = 'rgba(33,26,20,0.8)';
                    ctx.strokeStyle = bin.color;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.roundRect(bin.x - 30, 350, 60, 45, 6);
                    ctx.fill(); ctx.stroke();
                    ctx.fillStyle = bin.color;
                    ctx.font = '9px "Space Grotesk", sans-serif';
                    ctx.textAlign = 'center';
                    ctx.fillText(bin.label, bin.x, 375);
                });

                // Moving chillies
                for (let i = 0; i < 5; i++) {
                    const phase = ((frame * 2 + i * 120) % 700);
                    const x = phase - 100;
                    const y = 290 - Math.sin(phase * 0.01 + i) * 5;

                    if (x > -50 && x < canvas.width + 50) {
                        const seed = i * 1000 + Math.floor(phase / 700) * 7;
                        let sr = seed;
                        const rng = () => { sr = (sr * 16807) % 2147483647; return (sr - 1) / 2147483646; };

                        const r = 160 + Math.floor(rng() * 80);
                        const g = 15 + Math.floor(rng() * 40);
                        const b = 10 + Math.floor(rng() * 15);

                        ctx.save();
                        ctx.translate(x, y);
                        ctx.rotate(-0.2 + rng() * 0.4);
                        ctx.beginPath();
                        ctx.ellipse(0, 0, 25, 8, 0, 0, Math.PI * 2);
                        const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, 25);
                        grad.addColorStop(0, `rgb(${Math.min(255, r + 20)},${g + 5},${b})`);
                        grad.addColorStop(1, `rgb(${Math.max(0, r - 30)},${Math.max(0, g - 10)},${Math.max(0, b - 10)})`);
                        ctx.fillStyle = grad;
                        ctx.fill();
                        // Stem
                        ctx.beginPath();
                        ctx.moveTo(-23, -2);
                        ctx.lineTo(-30, -6 - rng() * 4);
                        ctx.lineTo(-21, 1);
                        ctx.closePath();
                        ctx.fillStyle = '#2d5016';
                        ctx.fill();

                        // Detection box when near center
                        if (Math.abs(x - 250) < 60) {
                            ctx.strokeStyle = 'rgba(230,57,38,0.7)';
                            ctx.lineWidth = 1;
                            ctx.setLineDash([3, 3]);
                            ctx.strokeRect(-30, -15, 60, 30);
                            ctx.setLineDash([]);
                            // Label
                            ctx.fillStyle = 'rgba(0,0,0,0.7)';
                            ctx.fillRect(-25, -30, 50, 14);
                            ctx.fillStyle = '#e63926';
                            ctx.font = '9px "Space Grotesk", sans-serif';
                            ctx.textAlign = 'center';
                            const labels = ['Grade A', 'Grade B', 'Grade A', 'Grade C', 'Grade A'];
                            ctx.fillText(labels[i], 0, -20);
                        }

                        ctx.restore();
                    }
                }

                // Scanning effect line
                const scanY = 100 + Math.sin(frame * 0.03) * 80;
                const scanGrad = ctx.createLinearGradient(0, scanY - 1, 0, scanY + 1);
                scanGrad.addColorStop(0, 'rgba(230,57,38,0)');
                scanGrad.addColorStop(0.5, 'rgba(230,57,38,0.3)');
                scanGrad.addColorStop(1, 'rgba(230,57,38,0)');
                ctx.fillStyle = scanGrad;
                ctx.fillRect(0, scanY - 1, canvas.width, 2);

                // Title overlay
                ctx.fillStyle = 'rgba(0,0,0,0.5)';
                ctx.fillRect(0, 0, canvas.width, 36);
                ctx.fillStyle = '#a89684';
                ctx.font = '11px "DM Sans", sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText('CONVEYOR BELT VIEW — BAY 3', 12, 23);
                ctx.fillStyle = '#22c55e';
                ctx.beginPath();
                ctx.arc(canvas.width - 20, 18, 4, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#a89684';
                ctx.textAlign = 'right';
                ctx.fillText('LIVE', canvas.width - 30, 22);

                requestAnimationFrame(draw);
            }
            draw();
        }

        // ============================================================
        //  FLOATING PARTICLES
        // ============================================================
        function initParticles() {
            const container = document.getElementById('particles');
            const colors = ['#e63926', '#ff6b35', '#f4a623', '#c0392b'];
            for (let i = 0; i < 15; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                const size = 2 + Math.random() * 4;
                particle.style.width = size + 'px';
                particle.style.height = size + 'px';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.background = colors[Math.floor(Math.random() * colors.length)];
                particle.style.animationDuration = (15 + Math.random() * 25) + 's';
                particle.style.animationDelay = (Math.random() * 20) + 's';
                particle.style.opacity = 0.3 + Math.random() * 0.4;
                container.appendChild(particle);
            }
        }

        // ============================================================
        //  SECTION REVEAL ON SCROLL
        // ============================================================
        function initScrollReveal() {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                    }
                });
            }, { threshold: 0.1 });

            document.querySelectorAll('.reveal-section').forEach(el => observer.observe(el));
        }

        // ============================================================
        //  NAV ACTIVE STATE
        // ============================================================
        function initNavHighlight() {
            const sections = ['hero', 'grading', 'camera', 'dashboard', 'history'];
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        document.querySelectorAll('#navLinks .nav-link').forEach(l => l.classList.remove('active'));
                        const link = document.querySelector(`[data-section="${entry.target.id}"]`);
                        if (link) link.classList.add('active');
                    }
                });
            }, { threshold: 0.3 });

            sections.forEach(id => {
                const el = document.getElementById(id);
                if (el) observer.observe(el);
            });
        }

        // ============================================================
        //  RESPONSIVE GRID ADJUSTMENTS
        // ============================================================
        function initResponsive() {
            const style = document.createElement('style');
            style.textContent = `
    @media (max-width: 768px) {
      .hero-grid { grid-template-columns: 1fr !important; gap: 40px !important; }
      .grading-grid { grid-template-columns: 1fr !important; }
      .camera-grid { grid-template-columns: 1fr !important; }
      .kpi-grid { grid-template-columns: repeat(2, 1fr) !important; }
      .chart-grid { grid-template-columns: 1fr !important; }
      .chart-grid-2 { grid-template-columns: 1fr !important; }
      #navLinks { display: none !important; }
    }
  `;
            document.head.appendChild(style);
        }

        // ============================================================
        //  INITIALIZATION
        // ============================================================
        function init() {
            initParticles();
            initHeroCanvas();
            initSamples();
            initCharts();
            initScrollReveal();
            initNavHighlight();
            initResponsive();
            updateHeroStats();

            // Draw initial camera placeholder
            const camCanvas = document.getElementById('cameraCanvas');
            const camCtx = camCanvas.getContext('2d');
            camCtx.fillStyle = '#000';
            camCtx.fillRect(0, 0, camCanvas.width, camCanvas.height);
            camCtx.fillStyle = '#333';
            camCtx.font = '16px "DM Sans", sans-serif';
            camCtx.textAlign = 'center';
            camCtx.fillText('Camera offline — click Start Camera to begin', camCanvas.width / 2, camCanvas.height / 2);

            showToast('ML Mirchi system initialized successfully', 'success');
        }

        document.addEventListener('DOMContentLoaded', init);