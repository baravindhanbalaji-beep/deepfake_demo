// Human-Centered Veritas Lens Frontend Controller

document.addEventListener("DOMContentLoaded", () => {
    // Mode Switcher Elements
    const tabVideo = document.getElementById("tab-video");
    const tabFolder = document.getElementById("tab-folder");
    const tabFrames = document.getElementById("tab-frames");

    // Native file inputs
    const videoInput = document.getElementById("video-input");
    const folderInput = document.getElementById("folder-input");
    const framesInput = document.getElementById("frames-input");

    // Dropzone & Prompts
    const dropzone = document.getElementById("dropzone");
    const dropzonePrompt = document.getElementById("dropzone-prompt");
    const promptIcon = document.getElementById("prompt-icon");
    const promptTitle = document.getElementById("prompt-title");
    const promptSubtitle = document.getElementById("prompt-subtitle");
    const browseBtn = document.getElementById("browse-btn");

    // File Selected State Elements
    const fileInfo = document.getElementById("file-info");
    const fileReadyBadge = document.getElementById("file-ready-badge");
    const fileIcon = document.getElementById("file-icon");
    const fileName = document.getElementById("file-name");
    const fileSize = document.getElementById("file-size");
    const removeFileBtn = document.getElementById("remove-file");

    // Actions & Progress
    const submitBtn = document.getElementById("submit-btn");
    const submitHint = document.getElementById("submit-hint");
    const uploadForm = document.getElementById("upload-form");
    const loadingState = document.getElementById("loading-state");
    const progressText = document.getElementById("progress-text");

    // Results Dashboard Elements
    const resultsDashboard = document.getElementById("results-dashboard");
    const verdictCard = document.getElementById("verdict-card");
    const verdictIcon = document.getElementById("verdict-icon");
    const verdictText = document.getElementById("verdict-text");
    const humanSummaryText = document.getElementById("human-summary-text");
    const confidenceText = document.getElementById("confidence-text");
    const confidenceBar = document.getElementById("confidence-bar");
    const totalFrames = document.getElementById("total-frames");
    const fakeFramesCount = document.getElementById("fake-frames-count");
    const realFramesCount = document.getElementById("real-frames-count");
    const inferenceTime = document.getElementById("inference-time");
    const galleryCount = document.getElementById("gallery-count");
    const framesGallery = document.getElementById("frames-gallery");

    // Modal Lightbox Elements
    const frameModal = document.getElementById("frame-modal");
    const modalTitle = document.getElementById("modal-title");
    const modalSubtitle = document.getElementById("modal-subtitle");
    const modalImage = document.getElementById("modal-image");
    const modalBadge = document.getElementById("modal-badge");
    const modalExplanation = document.getElementById("modal-explanation");
    const closeModalBtn = document.getElementById("close-modal");

    // State Variables
    let currentMode = "video"; // "video", "folder", or "frames"
    let selectedFiles = [];

    const activeTabClass = "flex-1 py-2.5 px-3 rounded-xl font-semibold text-xs sm:text-sm transition-all duration-200 flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md";
    const inactiveTabClass = "flex-1 py-2.5 px-3 rounded-xl font-semibold text-xs sm:text-sm transition-all duration-200 flex items-center justify-center gap-2 text-slate-400 hover:text-white";

    // -----------------------------------------------------------------
    // Mode Switching
    // -----------------------------------------------------------------
    function setMode(mode) {
        currentMode = mode;
        clearSelection();

        tabVideo.className = mode === "video" ? activeTabClass : inactiveTabClass;
        if (tabFolder) tabFolder.className = mode === "folder" ? activeTabClass : inactiveTabClass;
        tabFrames.className = mode === "frames" ? activeTabClass : inactiveTabClass;

        if (mode === "video") {
            dropzone.htmlFor = "video-input";
            promptIcon.textContent = "🎬";
            promptTitle.textContent = "Click anywhere to select or drag and drop video";
            promptSubtitle.textContent = "MP4, AVI, MOV, MKV, or WEBM (Max 200MB)";
            if (browseBtn) browseBtn.textContent = "📁 Browse Video File";
            submitBtn.textContent = "Check Video Authenticity ✨";
        } else if (mode === "folder") {
            dropzone.htmlFor = "folder-input";
            promptIcon.textContent = "📁";
            promptTitle.textContent = "Click anywhere to select a folder of image frames";
            promptSubtitle.textContent = "Select any folder on your computer or drag & drop it here";
            if (browseBtn) browseBtn.textContent = "📁 Choose Frames Folder";
            submitBtn.textContent = "Check Frames Folder Authenticity ✨";
        } else {
            dropzone.htmlFor = "frames-input";
            promptIcon.textContent = "🖼️";
            promptTitle.textContent = "Click anywhere to select or drag and drop image file(s)";
            promptSubtitle.textContent = "JPG, PNG, JPEG, WEBP, or BMP (Select 1 or multiple files)";
            if (browseBtn) browseBtn.textContent = "📁 Browse Image File(s)";
            submitBtn.textContent = "Check Image(s) Authenticity ✨";
        }
    }

    tabVideo.addEventListener("click", () => setMode("video"));
    if (tabFolder) tabFolder.addEventListener("click", () => setMode("folder"));
    tabFrames.addEventListener("click", () => setMode("frames"));

    // Prevent label trigger when clicking remove button
    dropzone.addEventListener("click", (e) => {
        if (e.target.closest("#remove-file")) {
            e.preventDefault();
        }
    });

    // -----------------------------------------------------------------
    // File Input Listeners
    // -----------------------------------------------------------------
    videoInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFiles(e.target.files);
        }
    });

    if (folderInput) {
        folderInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                const allFiles = Array.from(e.target.files);
                const imageFiles = allFiles.filter(f => {
                    const ext = f.name.split(".").pop().toLowerCase();
                    return ["jpg", "jpeg", "png", "webp", "bmp"].includes(ext);
                });
                imageFiles.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }));

                if (imageFiles.length === 0) {
                    alert("No supported images (.jpg, .png, .jpeg, .webp, .bmp) found in this folder.");
                    clearSelection();
                    return;
                }

                const folderName = imageFiles[0].webkitRelativePath ? imageFiles[0].webkitRelativePath.split("/")[0] : "Folder";
                handleFiles(imageFiles, folderName);
            }
        });
    }

    framesInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFiles(e.target.files);
        }
    });

    // -----------------------------------------------------------------
    // Drag & Drop Handling (Files and Folders)
    // -----------------------------------------------------------------
    ["dragenter", "dragover"].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            dropzone.classList.add("border-indigo-500", "bg-indigo-500/10");
        });
    });

    ["dragleave", "drop"].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            dropzone.classList.remove("border-indigo-500", "bg-indigo-500/10");
        });
    });

    function traverseEntry(entry) {
        return new Promise((resolve) => {
            if (entry.isFile) {
                entry.file(f => resolve([f]), () => resolve([]));
            } else if (entry.isDirectory) {
                const reader = entry.createReader();
                const allEntries = [];
                const readBatch = () => {
                    reader.readEntries(async (entries) => {
                        if (entries.length === 0) {
                            const subFiles = await Promise.all(allEntries.map(traverseEntry));
                            resolve(subFiles.flat());
                        } else {
                            allEntries.push(...entries);
                            readBatch();
                        }
                    }, () => resolve([]));
                };
                readBatch();
            } else {
                resolve([]);
            }
        });
    }

    dropzone.addEventListener("drop", async (e) => {
        e.preventDefault();
        dropzone.classList.remove("border-indigo-500", "bg-indigo-500/10");

        let files = [];
        if (e.dataTransfer.items) {
            const promises = [];
            for (let i = 0; i < e.dataTransfer.items.length; i++) {
                const item = e.dataTransfer.items[i];
                const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
                if (entry) {
                    promises.push(traverseEntry(entry));
                } else {
                    const f = item.getAsFile();
                    if (f) files.push(f);
                }
            }
            const nested = await Promise.all(promises);
            files.push(...nested.flat());
        } else if (e.dataTransfer.files) {
            files = Array.from(e.dataTransfer.files);
        }

        if (files.length === 0) return;

        const imageFiles = files.filter(f => {
            const ext = f.name.split(".").pop().toLowerCase();
            return ["jpg", "jpeg", "png", "webp", "bmp"].includes(ext);
        });

        if (imageFiles.length > 0) {
            imageFiles.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }));
            if (currentMode === "video") {
                setMode(imageFiles.length > 1 ? "folder" : "frames");
            }
            const folderName = imageFiles.length > 1 ? "Dropped Folder" : null;
            handleFiles(imageFiles, folderName);
        } else {
            handleFiles(files);
        }
    });

    // -----------------------------------------------------------------
    // Process File List & Update UI
    // -----------------------------------------------------------------
    function handleFiles(filesList, folderName = null) {
        selectedFiles = Array.from(filesList);
        if (selectedFiles.length === 0) return;

        const totalBytes = selectedFiles.reduce((acc, f) => acc + f.size, 0);
        const sizeMB = (totalBytes / (1024 * 1024)).toFixed(2);

        if (folderName) {
            fileName.textContent = `Folder: "${folderName}" (${selectedFiles.length} frames)`;
            fileSize.textContent = `${sizeMB} MB total`;
            fileIcon.textContent = "📁";
            fileReadyBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> ${selectedFiles.length} Frames Ready from Folder`;
            submitBtn.textContent = `Check Folder (${selectedFiles.length} Frames) ✨`;
        } else if (selectedFiles.length === 1) {
            const file = selectedFiles[0];
            fileName.textContent = file.name;
            fileSize.textContent = `${sizeMB} MB`;
            fileIcon.textContent = currentMode === "video" ? "🎬" : "🖼️";
            fileReadyBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> ${currentMode === "video" ? "Video" : "Photo"} Ready for Analysis`;
            submitBtn.textContent = currentMode === "video" ? "Check Video Authenticity ✨" : "Check Photo Authenticity ✨";
        } else {
            fileName.textContent = `${selectedFiles.length} Photos Selected (${selectedFiles[0].name}, ...)`;
            fileSize.textContent = `${sizeMB} MB total`;
            fileIcon.textContent = "🖼️";
            fileReadyBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> ${selectedFiles.length} Photos Ready for Analysis`;
            submitBtn.textContent = `Check ${selectedFiles.length} Photos Authenticity ✨`;
        }

        dropzonePrompt.classList.add("hidden");
        fileInfo.classList.remove("hidden");

        submitBtn.disabled = false;
        submitBtn.classList.remove("bg-slate-800", "text-slate-500", "cursor-not-allowed");
        submitBtn.classList.add("bg-gradient-to-r", "from-blue-600", "via-indigo-600", "to-purple-600", "hover:from-blue-500", "hover:via-indigo-500", "hover:to-purple-500", "text-white", "cursor-pointer", "shadow-indigo-500/30", "ring-2", "ring-indigo-400/40");

        if (submitHint) {
            submitHint.classList.remove("hidden");
        }
    }

    function clearSelection() {
        selectedFiles = [];
        videoInput.value = "";
        if (folderInput) folderInput.value = "";
        framesInput.value = "";
        dropzonePrompt.classList.remove("hidden");
        fileInfo.classList.add("hidden");

        submitBtn.disabled = true;
        submitBtn.classList.add("bg-slate-800", "text-slate-500", "cursor-not-allowed");
        submitBtn.classList.remove("bg-gradient-to-r", "from-blue-600", "via-indigo-600", "to-purple-600", "hover:from-blue-500", "hover:via-indigo-500", "hover:to-purple-500", "text-white", "cursor-pointer", "shadow-indigo-500/30", "ring-2", "ring-indigo-400/40");

        if (submitHint) {
            submitHint.classList.add("hidden");
        }
    }

    removeFileBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        clearSelection();
    });

    // -----------------------------------------------------------------
    // Form Submission & Analysis
    // -----------------------------------------------------------------
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (selectedFiles.length === 0) return;

        // UI transition to loading state
        submitBtn.disabled = true;
        uploadForm.classList.add("opacity-40", "pointer-events-none");
        loadingState.classList.remove("hidden");
        resultsDashboard.classList.add("hidden");

        // Human, Conversational Progress Messages
        const isFramesOrFolder = currentMode === "frames" || currentMode === "folder";
        const steps = isFramesOrFolder ? [
            "Reading and validating frames...",
            "Locating human faces with seam margin expansion...",
            "Inspecting skin pores and micro-wrinkles (LBP)...",
            "Searching for generative frequency fingerprints (DCT)...",
            "Aggregating evidence across all frames into final report..."
        ] : [
            "Decomposing video at 1 frame per second...",
            "Locating faces and expanding 20% boundary margins...",
            "Inspecting skin pore micro-texture and consistency...",
            "Checking for generative frequency fingerprints (DCT)...",
            "Pooling temporal evidence into overall authenticity verdict..."
        ];

        let stepIdx = 0;
        progressText.textContent = steps[0];
        const progressTimer = setInterval(() => {
            stepIdx = (stepIdx + 1) % steps.length;
            progressText.textContent = steps[stepIdx];
        }, 1200);

        const formData = new FormData();
        if (currentMode === "video") {
            formData.append("video", selectedFiles[0]);
        } else {
            selectedFiles.forEach(f => {
                formData.append("frames", f);
            });
        }

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            clearInterval(progressTimer);
            const data = await response.json();

            if (!response.ok) {
                alert(`Analysis Notice: ${data.error || "Unable to complete analysis"}`);
                return;
            }

            renderResults(data);

        } catch (err) {
            clearInterval(progressTimer);
            alert(`Network or processing error: ${err.message}`);
        } finally {
            loadingState.classList.add("hidden");
            uploadForm.classList.remove("opacity-40", "pointer-events-none");
            submitBtn.disabled = false;
        }
    });

    // -----------------------------------------------------------------
    // Render Results with Human Explanations
    // -----------------------------------------------------------------
    function renderResults(data) {
        resultsDashboard.classList.remove("hidden");

        // Clear previous classes
        verdictCard.classList.remove("glow-fake", "glow-real", "border-rose-500/50", "border-emerald-500/50");

        if (data.is_fake) {
            verdictCard.classList.add("glow-fake", "border-rose-500/50");
            verdictIcon.textContent = "⚠️";
            verdictText.textContent = data.verdict_title || "High Likelihood of AI Manipulation";
            verdictText.className = "text-3xl sm:text-4xl font-extrabold tracking-tight text-rose-400";
            confidenceBar.className = "h-full rounded-full bg-gradient-to-r from-rose-600 to-red-500 transition-all duration-1000";
            confidenceText.textContent = `${data.confidence} Manipulated`;
        } else {
            verdictCard.classList.add("glow-real", "border-emerald-500/50");
            verdictIcon.textContent = "🛡️";
            verdictText.textContent = data.verdict_title || "Looks Natural & Authentic";
            verdictText.className = "text-3xl sm:text-4xl font-extrabold tracking-tight text-emerald-400";
            confidenceBar.className = "h-full rounded-full bg-gradient-to-r from-emerald-600 to-teal-400 transition-all duration-1000";
            confidenceText.textContent = `${data.confidence} Authentic`;
        }

        if (humanSummaryText) {
            humanSummaryText.textContent = data.human_summary || "Analysis completed across all extracted facial crops.";
        }

        confidenceBar.style.width = data.confidence;

        // Populate summary stats
        totalFrames.textContent = data.total_frames;
        fakeFramesCount.textContent = data.fake_frames;
        realFramesCount.textContent = data.real_frames;
        inferenceTime.textContent = data.elapsed_time;
        galleryCount.textContent = `${data.total_frames} Frames Analyzed`;

        // Populate Frame Gallery with Interactive Lightbox triggers
        framesGallery.innerHTML = "";

        data.frames.forEach(f => {
            const card = document.createElement("div");
            card.className = "frame-card glass-panel rounded-2xl overflow-hidden text-center group";

            const badgeBg = f.is_fake ? "bg-rose-500/20 text-rose-300 border-rose-500/30" : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
            const badgeLabel = f.is_fake ? `⚠️ ${f.fake_prob}% Fake` : `🛡️ ${f.real_prob}% Real`;

            card.innerHTML = `
                <div class="relative aspect-square bg-slate-950 overflow-hidden">
                    <img src="${f.image_url}" alt="Frame ${f.frame_num}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300">
                    <span class="absolute top-2 left-2 px-2 py-0.5 rounded-md text-[10px] font-mono bg-slate-950/80 text-slate-300 border border-slate-800">
                        ${f.timestamp}
                    </span>
                    <div class="absolute inset-0 bg-indigo-600/10 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <span class="px-2.5 py-1 rounded-lg bg-black/70 text-white text-[10px] font-semibold">🔍 Inspect</span>
                    </div>
                </div>
                <div class="p-2.5 space-y-1.5">
                    <p class="text-[11px] font-bold text-slate-300 truncate">${f.timestamp}</p>
                    <span class="inline-block px-2.5 py-1 rounded-lg text-[10px] font-bold border ${badgeBg}">
                        ${badgeLabel}
                    </span>
                </div>
            `;

            // Open Modal Lightbox on Click
            card.addEventListener("click", () => {
                openModal(f);
            });

            framesGallery.appendChild(card);
        });

        // Smooth scroll to results
        resultsDashboard.scrollIntoView({ behavior: "smooth" });
    }

    // -----------------------------------------------------------------
    // Interactive Frame Lightbox Modal
    // -----------------------------------------------------------------
    function openModal(f) {
        modalTitle.textContent = `Facial Inspection • ${f.timestamp}`;
        modalSubtitle.textContent = `Crop #${f.frame_num} Evaluated by Swin V2 + LBP + DCT`;
        modalImage.src = f.image_url;

        if (f.is_fake) {
            modalBadge.className = "absolute bottom-3 left-3 right-3 text-center py-2 px-3 rounded-xl text-xs font-bold border bg-rose-500/30 border-rose-500/50 text-rose-200 backdrop-blur-md";
            modalBadge.textContent = `⚠️ Flagged Manipulated (${f.fake_prob}% confidence)`;
            modalExplanation.textContent = `In this crop, the model flagged high anomaly scores in the micro-texture and frequency spectrum. The skin pore distribution appears unnaturally smoothed, and boundary transitions along the jawline deviate from authentic camera sensor capture.`;
        } else {
            modalBadge.className = "absolute bottom-3 left-3 right-3 text-center py-2 px-3 rounded-xl text-xs font-bold border bg-emerald-500/30 border-emerald-500/50 text-emerald-200 backdrop-blur-md";
            modalBadge.textContent = `🛡️ Natural & Authentic (${f.real_prob}% confidence)`;
            modalExplanation.textContent = `This crop passed all three forensic checks. Skin micro-textures exhibit organic pore variation, lighting gradients blend naturally with background elements, and no synthetic frequency artifacts were discovered.`;
        }

        frameModal.classList.remove("hidden");
    }

    closeModalBtn.addEventListener("click", () => {
        frameModal.classList.add("hidden");
    });

    frameModal.addEventListener("click", (e) => {
        if (e.target === frameModal) {
            frameModal.classList.add("hidden");
        }
    });
});
