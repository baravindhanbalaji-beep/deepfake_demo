
document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("dropzone");
    const videoInput = document.getElementById("video-input");
    const dropzonePrompt = document.getElementById("dropzone-prompt");
    const fileInfo = document.getElementById("file-info");
    const fileName = document.getElementById("file-name");
    const fileSize = document.getElementById("file-size");
    const removeFileBtn = document.getElementById("remove-file");

    const videoModeBtn = document.getElementById("video-mode-btn");
    const folderModeBtn = document.getElementById("folder-mode-btn");
    const folderPanel = document.getElementById("folder-upload-panel");
    const folderInput = document.getElementById("folder-input");
    const folderInfo = document.getElementById("folder-info");

    const submitBtn = document.getElementById("submit-btn");
    const uploadForm = document.getElementById("upload-form");
    const loadingState = document.getElementById("loading-state");
    const loadingHeading = document.getElementById("loading-heading");
    const progressText = document.getElementById("progress-text");
    const resultsDashboard = document.getElementById("results-dashboard");
    const errorMessage = document.getElementById("error-message");
    const analyzeAnotherBtn = document.getElementById("analyze-another");
    const samplingHint = document.getElementById("sampling-hint");
    const verdictSectionLabel = document.getElementById("verdict-section-label");

    const MAX_FILE_SIZE = 200 * 1024 * 1024;
    const MAX_FOLDER_SIZE = 200 * 1024 * 1024;

    const VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"];
    const IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp"];

    let selectedFile = null;
    let selectedImages = [];
    let mode = "video";
    let progressTimer = null;

    function getExtension(name) {
        return name && name.includes(".")
            ? name.split(".").pop().toLowerCase()
            : "";
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove("hidden");
    }

    function hideError() {
        errorMessage.classList.add("hidden");
    }

    function updateSubmitState() {
        submitBtn.disabled = mode === "video"
            ? !selectedFile
            : selectedImages.length === 0;
    }

    function setMode(nextMode) {
        mode = nextMode;

        const videoMode = mode === "video";

        videoModeBtn.classList.toggle("active", videoMode);
        folderModeBtn.classList.toggle("active", !videoMode);

        videoModeBtn.setAttribute("aria-pressed", String(videoMode));
        folderModeBtn.setAttribute("aria-pressed", String(!videoMode));

        dropzone.classList.toggle("hidden", !videoMode);
        folderPanel.classList.toggle("hidden", videoMode);

        if (submitBtn) {
            submitBtn.textContent = videoMode ? "Analyze video" : "Analyze image folder";
        }
        if (samplingHint) {
            samplingHint.textContent = videoMode
                ? "Up to 30 frames are sampled at approximately 1 FPS."
                : "Preprocessed face crops (up to 60) are analyzed sequentially.";
        }
        if (loadingHeading) {
            loadingHeading.textContent = videoMode ? "Analyzing video" : "Analyzing image folder";
        }

        hideError();
        updateSubmitState();
    }

    videoModeBtn.addEventListener("click", () => setMode("video"));
    folderModeBtn.addEventListener("click", () => setMode("folder"));

    // ---------------- Video selection ----------------

    dropzone.addEventListener("click", (event) => {
        if (event.target.closest("button")) return;
        videoInput.click();
    });

    dropzone.addEventListener("keydown", (event) => {
        if (event.target !== dropzone) return;

        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            videoInput.click();
        }
    });

    videoInput.addEventListener("change", () => {
        const file = videoInput.files?.[0];
        if (file) handleVideoFile(file);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.add("is-dragging");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.remove("is-dragging");
        });
    });

    dropzone.addEventListener("drop", (event) => {
        const file = event.dataTransfer.files?.[0];
        if (file) handleVideoFile(file);
    });

    function handleVideoFile(file) {
        if (!VIDEO_EXTENSIONS.includes(getExtension(file.name))) {
            showError("Choose an MP4, AVI, MOV, MKV or WEBM video.");
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            showError("The selected video exceeds the 200 MB limit.");
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent =
            `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

        dropzonePrompt.classList.add("hidden");
        fileInfo.classList.remove("hidden");

        hideError();
        updateSubmitState();
    }

    removeFileBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        clearVideoFile();
    });

    function clearVideoFile() {
        selectedFile = null;
        videoInput.value = "";

        dropzonePrompt.classList.remove("hidden");
        fileInfo.classList.add("hidden");

        updateSubmitState();
    }

    // ---------------- Folder selection ----------------

    folderInput.addEventListener("change", () => {
        const files = Array.from(folderInput.files || []);

        const images = files.filter((file) =>
            IMAGE_EXTENSIONS.includes(getExtension(file.name))
        );

        const totalSize = images.reduce(
            (total, file) => total + file.size, 0
        );

        if (images.length === 0) {
            selectedImages = [];
            folderInfo.textContent =
                "No supported images found. Select a folder containing JPG, PNG, WEBP or BMP images.";
            hideError();
            updateSubmitState();
            return;
        }

        if (totalSize > MAX_FOLDER_SIZE) {
            selectedImages = [];
            folderInfo.textContent = "Folder exceeds the 200 MB limit.";
            showError("The total size of the selected images exceeds 200 MB.");
            updateSubmitState();
            return;
        }

        // Sort by relative path for a predictable processing order.
        selectedImages = images.sort((a, b) => {
            const pathA = a.webkitRelativePath || a.name;
            const pathB = b.webkitRelativePath || b.name;
            return pathA.localeCompare(pathB, undefined, {
                numeric: true,
                sensitivity: "base"
            });
        });

        const firstPath = selectedImages[0].webkitRelativePath || "";
        const folderName = firstPath.includes("/")
            ? firstPath.split("/")[0]
            : "Selected folder";

        folderInfo.textContent =
            `${folderName}: ${selectedImages.length} images selected · ` +
            `${(totalSize / (1024 * 1024)).toFixed(2)} MB`;

        hideError();
        updateSubmitState();
    });

    // ---------------- Progress indicator ----------------

    function stopProgressTimer() {
        if (progressTimer !== null) {
            window.clearInterval(progressTimer);
            progressTimer = null;
        }
    }

    // ---------------- Submit video or folder ----------------

    uploadForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        if (mode === "video" && !selectedFile) return;
        if (mode === "folder" && selectedImages.length === 0) return;

        hideError();
        resultsDashboard.classList.add("hidden");

        submitBtn.disabled = true;
        loadingState.classList.remove("hidden");

        const messages = mode === "video"
            ? [
                "Extracting sampled frames...",
                "Processing video frames...",
                "Computing model predictions...",
                "Aggregating frame-level results..."
            ]
            : [
                "Preparing uploaded face crops...",
                "Running the detector on each image...",
                "Collecting frame-level predictions...",
                "Aggregating folder-level results..."
            ];

        let step = 0;
        progressText.textContent = messages[0];

        stopProgressTimer();
        progressTimer = window.setInterval(() => {
            step = (step + 1) % messages.length;
            progressText.textContent = messages[step];
        }, 1800);

        const formData = new FormData();
        let endpoint;

        if (mode === "video") {
            formData.append("video", selectedFile);
            endpoint = "/predict";
        } else {
            selectedImages.forEach((file) => {
                formData.append("images", file, file.name);
            });
            endpoint = "/predict-folder";
        }

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error || "The analysis could not be completed."
                );
            }

            if (!Array.isArray(data.frames)) {
                throw new Error("The server returned an invalid result.");
            }

            renderResults(data);

        } catch (error) {
            showError(error.message || "An unexpected error occurred.");

        } finally {
            stopProgressTimer();
            loadingState.classList.add("hidden");
            updateSubmitState();
        }
    });

    // ---------------- Render aggregate results ----------------

    function renderResults(data) {
        resultsDashboard.classList.remove("hidden");

        const verdictCard = document.getElementById("verdict-card");
        const verdictText = document.getElementById("verdict-text");
        const verdictDescription =
            document.getElementById("verdict-description");

        const confidenceText =
            document.getElementById("confidence-text");
        const confidenceBar =
            document.getElementById("confidence-bar");
        const confidenceTrack =
            document.getElementById("confidence-track");

        verdictCard.classList.remove("verdict-fake", "verdict-real");

        if (data.is_fake) {
            verdictCard.classList.add("verdict-fake");
            verdictText.textContent = "Possible manipulation detected";
            verdictDescription.textContent =
                "The model classified the input as more consistent with manipulated content.";
        } else {
            verdictCard.classList.add("verdict-real");
            verdictText.textContent = "Likely authentic";
            verdictDescription.textContent =
                "The model classified the input as more consistent with authentic content.";
        }

        if (verdictSectionLabel) {
            verdictSectionLabel.textContent = mode === "video"
                ? "VIDEO-LEVEL PREDICTION"
                : "FOLDER-LEVEL PREDICTION";
        }

        const confidence = Math.max(
            0,
            Math.min(100, Number.parseFloat(data.confidence) || 0)
        );

        confidenceText.textContent =
            `${confidence.toFixed(2)}%`;

        confidenceBar.style.width = `${confidence}%`;
        confidenceTrack.setAttribute("aria-valuenow", String(confidence));

        document.getElementById("total-frames").textContent =
            data.total_frames;

        document.getElementById("fake-frames-count").textContent =
            data.fake_frames;

        document.getElementById("real-frames-count").textContent =
            data.real_frames;

        document.getElementById("inference-time").textContent =
            data.elapsed_time;

        document.getElementById("gallery-count").textContent =
            `${data.total_frames} frame${data.total_frames === 1 ? "" : "s"}`;

        renderFrameGallery(data.frames);

        resultsDashboard.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }

    // ---------------- Render individual frame predictions ----------------

    function renderFrameGallery(frames) {
        const gallery = document.getElementById("frames-gallery");
        gallery.replaceChildren();

        frames.forEach((frame) => {
            const card = document.createElement("article");
            card.className = "frame-card";

            const imageWrap = document.createElement("div");
            imageWrap.className = "frame-image-wrap";

            const image = document.createElement("img");
            image.className = "frame-image";
            image.src = frame.image_url;
            image.alt = `Face crop from frame ${frame.frame_num}`;
            image.loading = "lazy";

            const timestamp = document.createElement("span");
            timestamp.className = "frame-time";
            timestamp.textContent = frame.timestamp || "";

            imageWrap.append(image, timestamp);

            const details = document.createElement("div");
            details.className = "frame-details";

            const number = document.createElement("p");
            number.className = "frame-number";
            number.textContent = `Frame ${frame.frame_num}`;

            const result = document.createElement("span");
            result.className = frame.is_fake
                ? "frame-result fake"
                : "frame-result real";

            const probability = frame.is_fake
                ? frame.fake_prob
                : frame.real_prob;

            result.textContent =
                `${frame.is_fake ? "Manipulated" : "Authentic"} · ${probability}%`;

            details.append(number, result);
            card.append(imageWrap, details);
            gallery.appendChild(card);
        });
    }

    // ---------------- Reset for another analysis ----------------

    analyzeAnotherBtn.addEventListener("click", () => {
        resultsDashboard.classList.add("hidden");
        hideError();

        clearVideoFile();

        selectedImages = [];
        folderInput.value = "";
        folderInfo.textContent =
            "Select a folder containing JPG, PNG, WEBP or BMP images.";

        updateSubmitState();

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    });

    setMode("video");
});
