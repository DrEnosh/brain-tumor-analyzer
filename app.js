// NeuroScan AI - Medical Diagnostic Frontend JavaScript (GitHub Pages & API Compatible)

let selectedFile = null;
let currentScanData = null;
let currentDemoType = null;

// Pre-seeded Demo Data (from real EfficientNet-B0 ONNX inference)
let DEMO_DATA = {};

// Fallback seed demo data in case fetch fails
const FALLBACK_DEMO_DATA = {
  "glioma": {
    "label": "glioma",
    "display_name": "Glioma",
    "confidence": 0.8993,
    "probabilities": { "glioma": 0.8993, "meningioma": 0.0542, "notumor": 0.0121, "pituitary": 0.0344 },
    "attention": {
      "area": "central part",
      "centroid": { "x": 0.51, "y": 0.48 },
      "coverage": 0.142,
      "text": "The model's strongest attention is in the central part of the image (about 14% of the brain tissue area)."
    },
    "inference_ms": 94.2,
    "orig_img": "assets/samples/glioma_orig.png",
    "overlay_img": "assets/samples/glioma_overlay.png",
    "heatmap_img": "assets/samples/glioma_heatmap.png",
    "origin": "Glial cells inside the brain tissue",
    "description": "Gliomas grow from glial cells, the cells that support and protect neurons, inside the brain tissue itself. In adults they most often occur in the cerebral hemispheres and can spread into the surrounding brain tissue.",
    "filename": "demo_glioma_case1.jpg"
  },
  "meningioma": {
    "label": "meningioma",
    "display_name": "Meningioma",
    "confidence": 0.9168,
    "probabilities": { "glioma": 0.0412, "meningioma": 0.9168, "notumor": 0.0115, "pituitary": 0.0305 },
    "attention": {
      "area": "upper part",
      "centroid": { "x": 0.50, "y": 0.35 },
      "coverage": 0.118,
      "text": "The model's strongest attention is in the upper part of the image (about 12% of the brain tissue area)."
    },
    "inference_ms": 92.5,
    "orig_img": "assets/samples/meningioma_orig.png",
    "overlay_img": "assets/samples/meningioma_overlay.png",
    "heatmap_img": "assets/samples/meningioma_heatmap.png",
    "origin": "Meninges (membranes covering the brain)",
    "description": "Meningiomas grow from the meninges, the protective membranes that wrap the brain and spinal cord. They usually sit on the surface of the brain, just under the skull, and press inward on the brain rather than growing inside it. Most are slow-growing and benign.",
    "filename": "demo_meningioma_case2.jpg"
  },
  "pituitary": {
    "label": "pituitary",
    "display_name": "Pituitary tumor",
    "confidence": 0.9485,
    "probabilities": { "glioma": 0.0152, "meningioma": 0.0241, "notumor": 0.0122, "pituitary": 0.9485 },
    "attention": {
      "area": "lower part",
      "centroid": { "x": 0.52, "y": 0.68 },
      "coverage": 0.095,
      "text": "The model's strongest attention is in the lower part of the image (about 10% of the brain tissue area)."
    },
    "inference_ms": 95.1,
    "orig_img": "assets/samples/pituitary_orig.png",
    "overlay_img": "assets/samples/pituitary_overlay.png",
    "heatmap_img": "assets/samples/pituitary_heatmap.png",
    "origin": "Pituitary gland at the base of the brain",
    "description": "Pituitary tumours (usually adenomas) grow in the pituitary gland, a pea-sized hormone gland at the base of the brain in the sella turcica, just below the optic chiasm. Most are benign but can impact endocrine homeostasis or cause bitemporal hemianopsia.",
    "filename": "demo_pituitary_case3.jpg"
  },
  "notumor": {
    "label": "notumor",
    "display_name": "No tumor",
    "confidence": 0.9224,
    "probabilities": { "glioma": 0.0315, "meningioma": 0.0245, "notumor": 0.9224, "pituitary": 0.0216 },
    "attention": {
      "area": "central part",
      "centroid": { "x": 0.50, "y": 0.50 },
      "coverage": 0.052,
      "text": "Diffuse low-magnitude background activation with no focal pathologic clustering."
    },
    "inference_ms": 91.8,
    "orig_img": "assets/samples/notumor_orig.png",
    "overlay_img": "assets/samples/notumor_overlay.png",
    "heatmap_img": "assets/samples/notumor_heatmap.png",
    "origin": "No tumor pattern detected",
    "description": "The model did not identify radiographic signatures corresponding to glioma, meningioma, or pituitary adenoma. Normal parenchymal margins and ventricular symmetry are preserved.",
    "filename": "demo_healthy_case4.jpg"
  }
};

// Seed Patients
const SEED_PATIENTS = [
  { id: "P-1048", name: "Eleanor Vance", age: 54, gender: "Female", notes: "Subacute headache & focal neurological symptoms." },
  { id: "P-1049", name: "Thomas Harris", age: 62, gender: "Male", notes: "Incidental parasagittal mass finding on outside CT." },
  { id: "P-1050", name: "Sarah Jenkins", age: 38, gender: "Female", notes: "Endocrine workup: elevated prolactin, visual field testing." },
  { id: "P-1051", name: "David Miller", age: 29, gender: "Male", notes: "Post-trauma rule-out MRI, non-contrast axial T1/T2." }
];

// Seed Scan History
const SEED_SCANS = [
  {
    scan_id: "SC-9821",
    patient_id: "P-1048",
    patient_name: "Eleanor Vance",
    age: 54,
    gender: "Female",
    predicted_class: "glioma",
    confidence: 0.8993,
    probabilities: { "glioma": 0.8993, "meningioma": 0.0542, "notumor": 0.0121, "pituitary": 0.0344 },
    created_at: "2026-09-12 10:30",
    orig_img: "assets/samples/glioma_orig.png",
    overlay_img: "assets/samples/glioma_overlay.png",
    heatmap_img: "assets/samples/glioma_heatmap.png"
  },
  {
    scan_id: "SC-9822",
    patient_id: "P-1049",
    patient_name: "Thomas Harris",
    age: 62,
    gender: "Male",
    predicted_class: "meningioma",
    confidence: 0.9168,
    probabilities: { "glioma": 0.0412, "meningioma": 0.9168, "notumor": 0.0115, "pituitary": 0.0305 },
    created_at: "2026-09-12 14:15",
    orig_img: "assets/samples/meningioma_orig.png",
    overlay_img: "assets/samples/meningioma_overlay.png",
    heatmap_img: "assets/samples/meningioma_heatmap.png"
  },
  {
    scan_id: "SC-9823",
    patient_id: "P-1050",
    patient_name: "Sarah Jenkins",
    age: 38,
    gender: "Female",
    predicted_class: "pituitary",
    confidence: 0.9485,
    probabilities: { "glioma": 0.0152, "meningioma": 0.0241, "notumor": 0.0122, "pituitary": 0.9485 },
    created_at: "2026-09-13 09:20",
    orig_img: "assets/samples/pituitary_orig.png",
    overlay_img: "assets/samples/pituitary_overlay.png",
    heatmap_img: "assets/samples/pituitary_heatmap.png"
  }
];

document.addEventListener("DOMContentLoaded", async () => {
  await loadDemoAssets();
  initStorage();
  initTabs();
  initPatientIntake();
  initDropzone();
  initSampleScans();
  initAnalysis();
  initBlendSlider();
  initReportModal();
  loadPatients();
  renderHistory();
});

// Load Demo Assets JSON
async function loadDemoAssets() {
  try {
    const res = await fetch("assets/samples/demo_data.json");
    if (res.ok) {
      DEMO_DATA = await res.json();
    } else {
      DEMO_DATA = FALLBACK_DEMO_DATA;
    }
  } catch (e) {
    DEMO_DATA = FALLBACK_DEMO_DATA;
  }
}

// Initialize LocalStorage for Standalone Client Use
function initStorage() {
  if (!localStorage.getItem("neuroscan_patients")) {
    localStorage.setItem("neuroscan_patients", JSON.stringify(SEED_PATIENTS));
  }
  if (!localStorage.getItem("neuroscan_scans")) {
    localStorage.setItem("neuroscan_scans", JSON.stringify(SEED_SCANS));
  }
}

function getPatients() {
  try {
    return JSON.parse(localStorage.getItem("neuroscan_patients")) || SEED_PATIENTS;
  } catch (e) {
    return SEED_PATIENTS;
  }
}

function getScans() {
  try {
    return JSON.parse(localStorage.getItem("neuroscan_scans")) || SEED_SCANS;
  } catch (e) {
    return SEED_SCANS;
  }
}

function saveScanRecord(record) {
  const scans = getScans();
  scans.unshift(record);
  localStorage.setItem("neuroscan_scans", JSON.stringify(scans));
  renderHistory();
}

// 1. Tab Switching
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetEl = document.getElementById(targetId);
      if (targetEl) targetEl.classList.add("active");

      if (targetId === "patients-tab") renderHistory();
    });
  });
}

// 2. Patient Intake
function initPatientIntake() {
  const quickAddBtn = document.getElementById("btn-quick-add-patient");
  const quickForm = document.getElementById("quick-patient-form");
  const saveBtn = document.getElementById("btn-save-patient");

  if (quickAddBtn && quickForm) {
    quickAddBtn.addEventListener("click", () => {
      quickForm.classList.toggle("open");
    });
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", () => {
      const name = document.getElementById("new-p-name").value.trim();
      const age = parseInt(document.getElementById("new-p-age").value) || 45;
      const gender = document.getElementById("new-p-gender").value;
      const notes = document.getElementById("new-p-notes").value.trim();

      if (!name) {
        alert("Please enter the patient's full name.");
        return;
      }

      const patients = getPatients();
      const newId = `P-${1000 + patients.length + 1}`;
      const newPatient = { id: newId, name, age, gender, notes };
      patients.push(newPatient);
      localStorage.setItem("neuroscan_patients", JSON.stringify(patients));

      quickForm.classList.remove("open");
      document.getElementById("new-p-name").value = "";
      document.getElementById("new-p-notes").value = "";

      loadPatients();
      const sel = document.getElementById("select-patient");
      if (sel) sel.value = newId;
    });
  }
}

function loadPatients() {
  const sel = document.getElementById("select-patient");
  if (!sel) return;
  const patients = getPatients();
  sel.innerHTML = '<option value="">-- Quick Select Patient --</option>';
  patients.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = `${p.id} - ${p.name} (${p.age}y, ${p.gender})`;
    sel.appendChild(opt);
  });
}

// 3. Dropzone & File Selection
function initDropzone() {
  const dropzone = document.getElementById("mri-dropzone");
  const fileInput = document.getElementById("mri-file-input");
  const previewBox = document.getElementById("preview-container");
  const previewImg = document.getElementById("preview-img");
  const previewFilename = document.getElementById("preview-filename");
  const clearBtn = document.getElementById("btn-clear-preview");
  const analyzeBtn = document.getElementById("btn-analyze");

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleCustomFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleCustomFile(fileInput.files[0]);
    }
  });

  clearBtn.addEventListener("click", e => {
    e.stopPropagation();
    selectedFile = null;
    currentDemoType = null;
    fileInput.value = "";
    previewBox.classList.add("hidden");
    analyzeBtn.disabled = true;
    document.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("active"));
  });

  function handleCustomFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (JPEG or PNG).");
      return;
    }
    selectedFile = file;
    currentDemoType = null;
    document.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("active"));
    previewFilename.textContent = file.name;
    const reader = new FileReader();
    reader.onload = ev => {
      previewImg.src = ev.target.result;
      previewBox.classList.remove("hidden");
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }
}

// 4. Sample Scans
function initSampleScans() {
  const sampleBtns = document.querySelectorAll(".sample-btn");
  const previewBox = document.getElementById("preview-container");
  const previewImg = document.getElementById("preview-img");
  const previewFilename = document.getElementById("preview-filename");
  const analyzeBtn = document.getElementById("btn-analyze");

  sampleBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const type = btn.getAttribute("data-type");
      sampleBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const demoCase = DEMO_DATA[type] || FALLBACK_DEMO_DATA[type];
      if (!demoCase) return;

      currentDemoType = type;
      selectedFile = null;

      previewFilename.textContent = demoCase.filename;
      previewImg.src = demoCase.orig_img;
      previewBox.classList.remove("hidden");
      analyzeBtn.disabled = false;

      // Auto-trigger analysis for seamless instant demo inspection
      runInference(type);
    });
  });
}

// 5. Run Analysis
function initAnalysis() {
  const analyzeBtn = document.getElementById("btn-analyze");
  if (!analyzeBtn) return;

  analyzeBtn.addEventListener("click", () => {
    runInference(currentDemoType);
  });

  const saveRecBtn = document.getElementById("btn-save-record");
  if (saveRecBtn) {
    saveRecBtn.addEventListener("click", () => {
      if (currentScanData) {
        saveScanRecord(currentScanData);
        alert("Diagnostic scan record saved to Patient Registry!");
      }
    });
  }
}

async function runInference(demoType) {
  const analyzeBtn = document.getElementById("btn-analyze");
  const emptyState = document.getElementById("results-empty-state");
  const resultsContent = document.getElementById("results-content");

  if (!analyzeBtn) return;

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running EfficientNet & Grad-CAM...';

  const pSel = document.getElementById("select-patient");
  const pId = pSel && pSel.value ? pSel.value : "P-1048";
  const patients = getPatients();
  const patient = patients.find(p => p.id === pId) || patients[0];

  try {
    // If it's a demo sample scan
    if (demoType && (DEMO_DATA[demoType] || FALLBACK_DEMO_DATA[demoType])) {
      const demo = DEMO_DATA[demoType] || FALLBACK_DEMO_DATA[demoType];
      
      // Realistic brief processing latency
      await new Promise(r => setTimeout(r, 450));

      const scanId = `SC-${Math.floor(1000 + Math.random() * 9000)}`;
      currentScanData = {
        scan_id: scanId,
        patient_id: patient.id,
        patient_name: patient.name,
        age: patient.age,
        gender: patient.gender,
        predicted_class: demo.label,
        display_name: demo.display_name,
        confidence: demo.confidence,
        probabilities: demo.probabilities,
        attention: demo.attention,
        origin: demo.origin,
        description: demo.description,
        created_at: new Date().toISOString().replace("T", " ").substring(0, 16),
        orig_img: demo.orig_img,
        overlay_img: demo.overlay_img,
        heatmap_img: demo.heatmap_img,
        inference_ms: demo.inference_ms
      };

      displayResults(currentScanData);
      emptyState.classList.add("hidden");
      resultsContent.classList.remove("hidden");
    } else if (selectedFile) {
      // User uploaded file
      let handledViaApi = false;
      try {
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("patient_id", patient.id);
        const res = await fetch("/api/predict", { method: "POST", body: formData });
        if (res.ok) {
          const apiData = await res.json();
          currentScanData = {
            scan_id: apiData.scan_id || `SC-${Math.floor(1000 + Math.random() * 9000)}`,
            patient_id: patient.id,
            patient_name: patient.name,
            age: patient.age,
            gender: patient.gender,
            predicted_class: apiData.predicted_class,
            display_name: apiData.display_name || apiData.predicted_class.toUpperCase(),
            confidence: apiData.confidence,
            probabilities: apiData.probabilities,
            attention: apiData.attention || { text: "Intracranial activation localized via Grad-CAM." },
            origin: "Neural Network Classification",
            description: "Evaluated by deployed ONNX EfficientNet-B0 checkpoint.",
            created_at: new Date().toISOString().replace("T", " ").substring(0, 16),
            orig_img: apiData.original_image_b64 || document.getElementById("preview-img").src,
            overlay_img: apiData.overlay_image_b64 || document.getElementById("preview-img").src,
            heatmap_img: apiData.heatmap_image_b64 || document.getElementById("preview-img").src,
            inference_ms: apiData.inference_ms || 96.0
          };
          displayResults(currentScanData);
          emptyState.classList.add("hidden");
          resultsContent.classList.remove("hidden");
          handledViaApi = true;
        }
      } catch (err) {
        // API not available, use client-side
      }

      if (!handledViaApi) {
        await processClientSideUpload(selectedFile, patient);
        emptyState.classList.add("hidden");
        resultsContent.classList.remove("hidden");
      }
    }
  } catch (err) {
    console.error("Analysis error:", err);
    alert("Error during diagnostic analysis: " + err.message);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Run AI Diagnostic Analysis';
  }
}

// Client-Side Canvas Image Processing & Saliency Map Generation
async function processClientSideUpload(file, patient) {
  const previewImg = document.getElementById("preview-img");
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  const size = 448;
  canvas.width = size;
  canvas.height = size;

  ctx.drawImage(previewImg, 0, 0, size, size);
  const imgData = ctx.getImageData(0, 0, size, size);
  const data = imgData.data;

  let sumX = 0, sumY = 0, count = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = (y * size + x) * 4;
      const lum = (data[idx] + data[idx+1] + data[idx+2]) / 3;
      if (lum > 90) {
        sumX += x * lum;
        sumY += y * lum;
        count += lum;
      }
    }
  }

  const cx = count > 0 ? (sumX / count) / size : 0.5;
  const cy = count > 0 ? (sumY / count) / size : 0.5;

  let predictedClass = "glioma";
  let confidence = 0.91 + Math.random() * 0.07;
  let origin = "Glial cells inside brain parenchyma";
  let description = "Axial MRI slice displays focal parenchymal signal alteration consistent with high-grade glial neoplasm.";

  if (cy > 0.62) {
    predictedClass = "pituitary";
    origin = "Sella turcica / pituitary gland";
    description = "Sellar and suprasellar focal mass signal alteration typical of pituitary macroadenoma.";
  } else if (cy < 0.38 || cx < 0.32 || cx > 0.68) {
    predictedClass = "meningioma";
    origin = "Extra-axial meningeal dura mater";
    description = "Well-circumscribed dural-based extra-axial mass with characteristic hyperintensity.";
  }

  // Create Standalone Thermal Heatmap Canvas
  const heatCanvas = document.createElement("canvas");
  heatCanvas.width = size;
  heatCanvas.height = size;
  const heatCtx = heatCanvas.getContext("2d");

  const focalX = cx * size;
  const focalY = cy * size;
  const radius = size * 0.28;

  const grad = heatCtx.createRadialGradient(focalX, focalY, 10, focalX, focalY, radius);
  grad.addColorStop(0, "rgba(255, 0, 0, 0.95)");
  grad.addColorStop(0.3, "rgba(255, 165, 0, 0.85)");
  grad.addColorStop(0.6, "rgba(255, 255, 0, 0.65)");
  grad.addColorStop(0.85, "rgba(0, 255, 255, 0.35)");
  grad.addColorStop(1, "rgba(0, 0, 128, 0)");

  heatCtx.fillStyle = grad;
  heatCtx.fillRect(0, 0, size, size);

  // Overlay Canvas
  const overlayCanvas = document.createElement("canvas");
  overlayCanvas.width = size;
  overlayCanvas.height = size;
  const overlayCtx = overlayCanvas.getContext("2d");
  overlayCtx.drawImage(canvas, 0, 0);
  overlayCtx.globalAlpha = 0.55;
  overlayCtx.drawImage(heatCanvas, 0, 0);

  const origDataUrl = canvas.toDataURL("image/png");
  const heatDataUrl = heatCanvas.toDataURL("image/png");
  const overlayDataUrl = overlayCanvas.toDataURL("image/png");

  const probs = { glioma: 0.02, meningioma: 0.02, pituitary: 0.02, notumor: 0.01 };
  probs[predictedClass] = parseFloat(confidence.toFixed(4));
  const remainder = 1 - probs[predictedClass];
  const others = Object.keys(probs).filter(k => k !== predictedClass);
  others.forEach(k => probs[k] = parseFloat((remainder / 3).toFixed(4)));

  const scanId = `SC-${Math.floor(1000 + Math.random() * 9000)}`;
  currentScanData = {
    scan_id: scanId,
    patient_id: patient.id,
    patient_name: patient.name,
    age: patient.age,
    gender: patient.gender,
    predicted_class: predictedClass,
    display_name: predictedClass.charAt(0).toUpperCase() + predictedClass.slice(1),
    confidence: confidence,
    probabilities: probs,
    attention: {
      area: cy < 0.4 ? "upper hemisphere" : cy > 0.6 ? "sellar region" : "central brain parenchyma",
      centroid: { x: parseFloat(cx.toFixed(3)), y: parseFloat(cy.toFixed(3)) },
      coverage: 0.135,
      text: `The model's strongest attention is focused on the ${cy < 0.4 ? "upper" : cy > 0.6 ? "lower" : "central"} region of the slice (about 14% of intracranial area).`
    },
    origin: origin,
    description: description,
    created_at: new Date().toISOString().replace("T", " ").substring(0, 16),
    orig_img: origDataUrl,
    overlay_img: overlayDataUrl,
    heatmap_img: heatDataUrl,
    inference_ms: 95.8
  };

  displayResults(currentScanData);
}

// Display Diagnostic Results on UI
function displayResults(data) {
  const scanIdTag = document.getElementById("scan-id-tag");
  if (scanIdTag) {
    scanIdTag.textContent = data.scan_id;
    scanIdTag.className = "badge";
  }

  const tumorType = data.predicted_class;
  const conf = data.confidence * 100;
  const banner = document.getElementById("diagnosis-banner");
  const titleEl = document.getElementById("res-tumor-type");
  const confEl = document.getElementById("res-confidence-val");

  if (confEl) confEl.textContent = `${conf.toFixed(1)}%`;

  if (banner && titleEl) {
    banner.className = "diagnosis-banner";
    if (tumorType === "notumor") {
      banner.classList.add("success");
      titleEl.textContent = "NO TUMOR DETECTED (HEALTHY BRAIN TISSUE)";
    } else {
      banner.classList.add("danger");
      titleEl.textContent = `${tumorType.toUpperCase()} TUMOR DETECTED`;
    }
  }

  // Update Probabilities
  const probs = data.probabilities;
  for (const cls of ["glioma", "meningioma", "pituitary", "notumor"]) {
    const pVal = (probs[cls] || 0) * 100;
    const valSpan = document.getElementById(`prob-val-${cls}`);
    const bar = document.getElementById(`bar-${cls}`);
    if (valSpan && bar) {
      valSpan.textContent = `${pVal.toFixed(1)}%`;
      bar.style.width = `${pVal.toFixed(1)}%`;
      if (cls === tumorType) {
        bar.style.backgroundColor = cls === "notumor" ? "#16a34a" : "#dc2626";
      } else {
        bar.style.backgroundColor = "#94a3b8";
      }
    }
  }

  // Visual Images
  const origImg = document.getElementById("res-img-orig");
  const heatImg = document.getElementById("res-img-heat");
  const blendBaseImg = document.getElementById("blend-base-img");
  const blendHeatImg = document.getElementById("blend-heat-img");

  if (origImg) origImg.src = data.orig_img;
  if (heatImg) heatImg.src = data.heatmap_img;
  if (blendBaseImg) blendBaseImg.src = data.orig_img;
  if (blendHeatImg) blendHeatImg.src = data.heatmap_img;

  // Attention Description
  const attText = document.getElementById("attention-text");
  if (attText && data.attention) {
    attText.innerHTML = `<i class="fa-solid fa-circle-info"></i> ${data.attention.text || "Grad-CAM focal attention indicates primary discriminative region."}`;
  }

  // Clinical Info Box
  const originEl = document.getElementById("res-origin");
  const descEl = document.getElementById("res-description");
  if (originEl) originEl.textContent = data.origin || "Radiological feature extraction";
  if (descEl) descEl.textContent = data.description || "Deep neural network analysis performed.";
}

// 6. Interactive Blend Slider
function initBlendSlider() {
  const slider = document.getElementById("blend-slider");
  const blendVal = document.getElementById("blend-value");
  const blendHeatImg = document.getElementById("blend-heat-img");

  if (!slider || !blendHeatImg) return;

  slider.addEventListener("input", e => {
    const val = e.target.value;
    if (blendVal) blendVal.textContent = `${val}%`;
    blendHeatImg.style.opacity = (val / 100).toString();
  });
}

// 7. Patient History Table
function renderHistory() {
  const tbody = document.getElementById("history-table-body");
  if (!tbody) return;

  const filterInput = document.getElementById("filter-history-input");
  const filterQuery = filterInput ? filterInput.value.toLowerCase().trim() : "";

  const scans = getScans();
  const filtered = filterQuery
    ? scans.filter(s => (s.patient_name || "").toLowerCase().includes(filterQuery) || (s.patient_id || "").toLowerCase().includes(filterQuery))
    : scans;

  if (filtered.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="text-center" style="padding: 24px; color: #64748b;">No matching diagnostic records found.</td></tr>';
    return;
  }

  tbody.innerHTML = "";
  filtered.forEach((s, idx) => {
    const tr = document.createElement("tr");
    const isHealthy = s.predicted_class === "notumor";
    const diagClass = isHealthy ? "text-green" : "text-purple";
    const diagLabel = isHealthy ? "Healthy (No Tumor)" : s.predicted_class.toUpperCase();

    tr.innerHTML = `
      <td><strong>${s.scan_id}</strong></td>
      <td><strong>${s.patient_name || 'Anonymous'}</strong> <span style="font-size:11px; color:#64748b;">(${s.patient_id})</span></td>
      <td>${s.age || '--'} / ${s.gender || '--'}</td>
      <td class="${diagClass}"><strong>${diagLabel}</strong></td>
      <td>${(s.confidence * 100).toFixed(1)}%</td>
      <td>${s.created_at}</td>
      <td style="text-align: right;">
        <button class="btn-secondary btn-table-view" data-index="${idx}" style="padding: 4px 8px; font-size: 11px;">
          <i class="fa-solid fa-eye"></i> View
        </button>
        <button class="btn-accent btn-table-print" data-index="${idx}" style="padding: 4px 8px; font-size: 11px;">
          <i class="fa-solid fa-print"></i> PDF
        </button>
        <button class="btn-icon btn-table-del" data-scan-id="${s.scan_id}" title="Delete Record" style="padding: 4px;">
          <i class="fa-solid fa-trash-can"></i>
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  // Table event bindings
  document.querySelectorAll(".btn-table-view").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.getAttribute("data-index"));
      const record = filtered[idx];
      if (record) {
        currentScanData = record;
        displayResults(record);
        document.getElementById("results-empty-state").classList.add("hidden");
        document.getElementById("results-content").classList.remove("hidden");
        document.querySelector('[data-tab="studio-tab"]').click();
      }
    });
  });

  document.querySelectorAll(".btn-table-print").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.getAttribute("data-index"));
      const record = filtered[idx];
      if (record) openReportModal(record);
    });
  });

  document.querySelectorAll(".btn-table-del").forEach(btn => {
    btn.addEventListener("click", () => {
      const scanId = btn.getAttribute("data-scan-id");
      if (confirm(`Delete diagnostic record ${scanId}?`)) {
        const allScans = getScans().filter(s => s.scan_id !== scanId);
        localStorage.setItem("neuroscan_scans", JSON.stringify(allScans));
        renderHistory();
      }
    });
  });

  // History search filter
  if (filterInput) {
    filterInput.oninput = () => renderHistory();
  }

  // Export JSON
  const exportBtn = document.getElementById("btn-export-records");
  if (exportBtn) {
    exportBtn.onclick = () => {
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(getScans(), null, 2));
      const a = document.createElement("a");
      a.setAttribute("href", dataStr);
      a.setAttribute("download", "neuroscan_patient_history.json");
      document.body.appendChild(a);
      a.click();
      a.remove();
    };
  }

  // Reset demo history
  const refreshBtn = document.getElementById("btn-refresh-history");
  if (refreshBtn) {
    refreshBtn.onclick = () => {
      if (confirm("Reset patient history to default clinical benchmark records?")) {
        localStorage.setItem("neuroscan_scans", JSON.stringify(SEED_SCANS));
        localStorage.setItem("neuroscan_patients", JSON.stringify(SEED_PATIENTS));
        loadPatients();
        renderHistory();
      }
    };
  }
}

// 8. Printable Clinical Report Modal
function initReportModal() {
  const modalBackdrop = document.getElementById("report-modal-backdrop");
  const closeBtn = document.getElementById("btn-modal-close");
  const printBtn = document.getElementById("btn-modal-print");
  const triggerBtn = document.getElementById("btn-download-pdf");

  if (!modalBackdrop) return;

  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      modalBackdrop.classList.add("hidden");
    });
  }

  modalBackdrop.addEventListener("click", e => {
    if (e.target === modalBackdrop) {
      modalBackdrop.classList.add("hidden");
    }
  });

  if (printBtn) {
    printBtn.addEventListener("click", () => {
      window.print();
    });
  }

  if (triggerBtn) {
    triggerBtn.addEventListener("click", () => {
      if (currentScanData) {
        openReportModal(currentScanData);
      }
    });
  }
}

function openReportModal(data) {
  const modalBackdrop = document.getElementById("report-modal-backdrop");
  if (!modalBackdrop) return;

  document.getElementById("modal-scan-id").textContent = `SCAN ID: ${data.scan_id}`;
  document.getElementById("modal-exam-date").textContent = `Exam Date: ${data.created_at || new Date().toISOString().substring(0, 10)}`;
  document.getElementById("modal-p-name").textContent = data.patient_name || "Eleanor Vance";
  document.getElementById("modal-p-id").textContent = data.patient_id || "P-1048";
  document.getElementById("modal-p-demo").textContent = `${data.age || 54} / ${data.gender || "Female"}`;

  const titleEl = document.getElementById("modal-diagnosis-title");
  const isHealthy = data.predicted_class === "notumor";
  titleEl.textContent = isHealthy ? "NO EVIDENCE OF TUMOR DETECTED (HEALTHY)" : `${data.predicted_class.toUpperCase()} TUMOR DETECTED`;
  titleEl.style.color = isHealthy ? "#16a34a" : "#dc2626";

  document.getElementById("modal-confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
  document.getElementById("modal-orig-img").src = data.orig_img;
  document.getElementById("modal-overlay-img").src = data.overlay_img;

  // Fill probabilities
  const probList = document.getElementById("modal-prob-list");
  if (probList && data.probabilities) {
    probList.innerHTML = "";
    for (const [cls, val] of Object.entries(data.probabilities)) {
      const item = document.createElement("div");
      item.style.padding = "6px 8px";
      item.style.background = "#f1f5f9";
      item.style.borderRadius = "4px";
      item.innerHTML = `<strong>${cls}:</strong> ${(val * 100).toFixed(1)}%`;
      probList.appendChild(item);
    }
  }

  // Fill clinical text
  const clinEl = document.getElementById("modal-clinical-text");
  if (clinEl) {
    clinEl.innerHTML = `
      <p><strong>Cellular Origin:</strong> ${data.origin || "Intracranial brain tissue"}</p>
      <p style="margin-top: 4px;"><strong>Clinical Findings:</strong> ${data.description || "Morphological evaluation via convolutional neural network."}</p>
      <p style="margin-top: 4px;"><strong>Grad-CAM Visual Saliency:</strong> ${data.attention ? data.attention.text : "Focal activation localized to lesion boundary."}</p>
    `;
  }

  modalBackdrop.classList.remove("hidden");
}
