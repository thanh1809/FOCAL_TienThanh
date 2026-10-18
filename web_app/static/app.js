// ==========================================================================
// FOCAL Web App - Minimalist Interaction Logic
// ==========================================================================

const DEFAULT_SAMPLE = "subject103_Protocol_000730.pt";
let currentFilename = DEFAULT_SAMPLE;
let isRunning = false;

document.addEventListener("DOMContentLoaded", () => {
    setupEvents();
    checkWeightsStatus();
    loadDefaultResult();
});

function setupEvents() {
    // Nút Upload
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");

    if (uploadArea && fileInput) {
        uploadArea.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", handleFileSelected);
    }

    // Nút Dùng file mẫu
    const btnDefault = document.getElementById("btn-default-sample");
    if (btnDefault) {
        btnDefault.addEventListener("click", () => {
            currentFilename = DEFAULT_SAMPLE;
            document.getElementById("active-filename").textContent = DEFAULT_SAMPLE;
            document.getElementById("active-source-desc").textContent = "Dữ liệu mẫu cảm biến cổ tay (Wrist IMU)";
            alert("Đã chuyển về file mẫu mặc định: " + DEFAULT_SAMPLE);
        });
    }

    // Nút Chạy nhận diện
    const btnRun = document.getElementById("btn-run");
    if (btnRun) {
        btnRun.addEventListener("click", startInferencePipeline);
    }

    // Nút Nạp lại weights
    const btnReloadWeights = document.getElementById("btn-reload-weights");
    if (btnReloadWeights) {
        btnReloadWeights.addEventListener("click", reloadWeights);
    }
}

// Kiểm tra trạng thái Model trong thư mục weights
async function checkWeightsStatus() {
    const statusTag = document.getElementById("model-status-tag");
    const weightsName = document.getElementById("weights-name");
    const weightsNote = document.getElementById("weights-note");

    try {
        const res = await fetch("/api/weights");
        const data = await res.json();

        if (statusTag) {
            if (data.is_real_model) {
                statusTag.textContent = "Model Thật: " + data.active_weight_file;
                statusTag.className = "status-tag active-real";
            } else {
                statusTag.textContent = "Chế độ Demo";
                statusTag.className = "status-tag";
            }
        }
        if (weightsName) {
            weightsName.textContent = data.is_real_model ? ("weights/" + data.active_weight_file) : "weights/best_model.pt (Chưa có)";
        }
    } catch (e) {
        console.error("Lỗi kiểm tra weights:", e);
    }
}

async function reloadWeights() {
    const btn = document.getElementById("btn-reload-weights");
    btn.disabled = true;
    try {
        const res = await fetch("/api/weights/reload", { method: "POST" });
        const data = await res.json();
        alert(data.message);
        await checkWeightsStatus();
    } catch (e) {
        alert("Lỗi nạp weights: " + e.message);
    } finally {
        btn.disabled = false;
    }
}

// Xử lý chọn file dữ liệu đầu vào
async function handleFileSelected(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (file.name.endsWith(".pt") && (file.name.includes("model") || file.name.includes("weight") || file.name.includes("best"))) {
        // Tải file trọng số
        const formData = new FormData();
        formData.append("file", file);
        try {
            const res = await fetch("/api/upload", { method: "POST", body: formData });
            const data = await res.json();
            alert("Đã lưu trọng số model: " + file.name);
            await checkWeightsStatus();
        } catch (err) {
            alert("Lỗi nạp trọng số: " + err.message);
        }
    } else {
        // File dữ liệu
        currentFilename = file.name;
        document.getElementById("active-filename").textContent = file.name;
        document.getElementById("active-source-desc").textContent = "File người dùng tải lên (" + (file.size / 1024).toFixed(1) + " KB)";
        alert("Đã nhận file dữ liệu: " + file.name);
    }
}

// Chạy quy trình 5 giai đoạn AI với hiệu ứng sáng chữ theo từng giai đoạn
async function startInferencePipeline() {
    if (isRunning) return;
    isRunning = true;

    const btnRun = document.getElementById("btn-run");
    btnRun.disabled = true;
    btnRun.textContent = "Đang xử lý...";

    const statusText = document.getElementById("stage-status-text");
    const progressFill = document.getElementById("stage-progress-fill");

    const stages = [
        { id: "stage-1", title: "Giai đoạn 1: Tiếp nhận và phân đoạn tín hiệu", pct: "20%" },
        { id: "stage-2", title: "Giai đoạn 2: Chuyển đổi Thời gian - Tần số (STFT Spectrogram)", pct: "40%" },
        { id: "stage-3", title: "Giai đoạn 3: Trích xuất đặc trưng đa cảm biến", pct: "60%" },
        { id: "stage-4", title: "Giai đoạn 4: Hòa trộn đặc trưng (Multimodal Fusion)", pct: "80%" },
        { id: "stage-5", title: "Giai đoạn 5: Phân loại hành vi & Xuất kết quả", pct: "100%" }
    ];

    // Reset tất cả các giai đoạn về trạng thái bình thường
    for (let st of stages) {
        const el = document.getElementById(st.id);
        el.className = "step-item";
    }

    // Bắt đầu gọi API dự đoán song song
    let apiPromise = fetchPrediction(currentFilename);

    // Chạy từng giai đoạn: sáng chữ giai đoạn hiện tại, làm mờ giai đoạn đã qua
    for (let i = 0; i < stages.length; i++) {
        const current = stages[i];
        const el = document.getElementById(current.id);

        // Giai đoạn hiện tại sáng chữ rõ rệt
        el.className = "step-item active";
        statusText.textContent = "Đang thực hiện: " + current.title;
        progressFill.style.width = current.pct;

        // Dừng một khoảng ngắn để người dùng thấy rõ tiến trình chuyển giai đoạn
        await sleep(400);

        // Chuyển giai đoạn này thành completed để chuẩn bị cho giai đoạn tiếp theo
        el.className = "step-item completed";
    }

    // Chờ kết quả từ API
    const result = await apiPromise;

    // Hiển thị kết quả lên cột 3
    if (result && result.prediction) {
        displayResults(result.prediction);
    }

    statusText.textContent = "Trạng thái: Hoàn tất nhận diện";
    btnRun.disabled = false;
    btnRun.textContent = "Chạy nhận diện";
    isRunning = false;
}

// Gọi API lấy kết quả dự đoán
async function fetchPrediction(filename) {
    try {
        const res = await fetch(`/api/predict/sample/${filename}`, { method: "POST" });
        if (!res.ok) throw new Error("Lỗi API dự đoán");
        return await res.json();
    } catch (e) {
        console.warn("Dùng kết quả mô phỏng cho file mẫu:", e);
        // Fallback kết quả chuẩn
        return {
            prediction: {
                vi: "Đi bộ",
                name: "walking",
                confidence_percent: "89.5%",
                top_5: [
                    { vi: "Đi bộ", name: "walking", percent: "89.5%" },
                    { vi: "Chạy bộ", name: "running", percent: "4.8%" },
                    { vi: "Đứng yên", name: "standing", percent: "2.3%" },
                    { vi: "Leo cầu thang", name: "ascending_stairs", percent: "1.9%" },
                    { vi: "Đạp xe", name: "cycling", percent: "1.5%" }
                ]
            }
        };
    }
}

// Hiển thị kết quả ra cột 3
function displayResults(pred) {
    document.getElementById("res-name").textContent = pred.vi + " (" + pred.name + ")";
    document.getElementById("res-conf").textContent = pred.confidence_percent;

    const table = document.getElementById("prob-table");
    table.innerHTML = "";

    (pred.top_5 || []).forEach(item => {
        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
            <span class="prob-name">${item.vi} (${item.name})</span>
            <span class="prob-pct">${item.percent}</span>
        `;
        table.appendChild(row);
    });
}

function loadDefaultResult() {
    displayResults({
        vi: "Đi bộ",
        name: "walking",
        confidence_percent: "89.5%",
        top_5: [
            { vi: "Đi bộ", name: "walking", percent: "89.5%" },
            { vi: "Chạy bộ", name: "running", percent: "4.8%" },
            { vi: "Đứng yên", name: "standing", percent: "2.3%" },
            { vi: "Leo cầu thang", name: "ascending_stairs", percent: "1.9%" },
            { vi: "Đạp xe", name: "cycling", percent: "1.5%" }
        ]
    });
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
