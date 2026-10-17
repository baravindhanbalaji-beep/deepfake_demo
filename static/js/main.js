document.addEventListener("DOMContentLoaded", () => {
    const dropzone = document.getElementById("dropzone");
    const videoInput = document.getElementById("video-input");
    const dropzonePrompt = document.getElementById("dropzone-prompt");
    const fileInfo = document.getElementById("file-info");
    const fileName = document.getElementById("file-name");
    const fileSize = document.getElementById("file-size");
    const removeFileBtn = document.getElementById("remove-file");
    const submitBtn = document.getElementById("submit-btn");
    const uploadForm = document.getElementById("upload-form");

    const loadingState = document.getElementById("loading-state");
    const progressText = document.getElementById("progress-text");
    const resultsDashboard = document.getElementById("results-dashboard");

    let selectedFile = null;

    // Handle Dropzone clicks
    dropzone.addEventListener("click", () => videoInput.click());

    // File selection
    videoInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & Drop events
    ["dragenter", "dragover"].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            dropzone.classList.add("border-blue-500", "bg-blue-500/5");
        });
    });

    ["dragleave", "drop"].forEach(event => {
        dropzone.addEventListener(event, (e) => {
            e.preventDefault();
            dropzone.classList.remove("border-blue-500", "bg-blue-500/5");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFile(file) {
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

        dropzonePrompt.classList.add("hidden");
        fileInfo.classList.remove("hidden");

        submitBtn.disabled = false;
        submitBtn.classList.remove("bg-slate-800", "text-slate-500", "cursor-not-allowed");
        submitBtn.classList.add("bg-blue-600", "hover:bg-blue-500", "text-white", "cursor-pointer");
    }

    removeFileBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        selectedFile = null;
        videoInput.value = "";
        dropzonePrompt.classList.remove("hidden");
        fileInfo.classList.add("hidden");

        submitBtn.disabled = true;
        submitBtn.classList.add("bg-slate-800", "text-slate-500", "cursor-not-allowed");
        submitBtn.classList.remove("bg-blue-600", "hover:bg-blue-500", "text-white", "cursor-pointer");
    });

    // Form submission
    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        // UI transition to loading state
        submitBtn.disabled = true;
        uploadForm.classList.add("opacity-50", "pointer-events-none");
        loadingState.classList.remove("hidden");
        resultsDashboard.classList.add("hidden");

        // Progress text animation
        const steps = [
            "Decomposing Video at 1 FPS...",
            "Detecting Facial Crops (MTCNN/Cascade)...",
            "Extracting Swin V2 Shifted-Window Attention...",
            "Computing LBP Texture & DCT Frequency Maps...",
            "Aggregating Temporal Mean Pooling..."
        ];
        let stepIdx = 0;
        const progressTimer = setInterval(() => {
            stepIdx = (stepIdx + 1) % steps.length;
            progressText.textContent = steps[stepIdx];
        }, 1200);

        const formData = new FormData();
        formData.append("video", selectedFile);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            clearInterval(progressTimer);
            const data = await response.json();

            if (!response.ok) {
                alert(`Error: ${data.error || "Analysis failed"}`);
                return;
            }

            renderResults(data);

        } catch (err) {
            clearInterval(progressTimer);
            alert(`Network or processing error: ${err.message}`);
        } finally {
            loadingState.classList.add("hidden");
            uploadForm.classList.remove("opacity-50", "pointer-events-none");
            submitBtn.disabled = false;
        }
    });

    function renderResults(data) {
        resultsDashboard.classList.remove("hidden");

        const verdictCard = document.getElementById("verdict-card");
        const verdictIcon = document.getElementById("verdict-icon");
        const verdictText = document.getElementById("verdict-text");
        const confidenceText = document.getElementById("confidence-text");
        const confidenceBar = document.getElementById("confidence-bar");

        // Clear previous classes
        verdictCard.classList.remove("glow-fake", "glow-real", "border-rose-500", "border-emerald-500");

        if (data.is_fake) {
            verdictCard.classList.add("glow-fake", "border-rose-500/50");
            verdictIcon.textContent = "⚠️";
            verdictText.textContent = "DEEPFAKE DETECTED";
            verdictText.className = "text-3xl sm:text-4xl font-extrabold tracking-tight text-rose-400";
            confidenceBar.className = "h-3 rounded-full bg-rose-500 transition-all duration-700";
            confidenceText.textContent = `${data.confidence} Manipulated`;
        } else {
            verdictCard.classList.add("glow-real", "border-emerald-500/50");
            verdictIcon.textContent = "🛡️";
            verdictText.textContent = "AUTHENTIC VIDEO";
            verdictText.className = "text-3xl sm:text-4xl font-extrabold tracking-tight text-emerald-400";
            confidenceBar.className = "h-3 rounded-full bg-emerald-500 transition-all duration-700";
            confidenceText.textContent = `${data.confidence} Authentic`;
        }

        confidenceBar.style.width = data.confidence;

        // Populate summary stats
        document.getElementById("total-frames").textContent = data.total_frames;
        document.getElementById("fake-frames-count").textContent = data.fake_frames;
        document.getElementById("real-frames-count").textContent = data.real_frames;
        document.getElementById("inference-time").textContent = data.elapsed_time;
        document.getElementById("gallery-count").textContent = `${data.total_frames} Frames`;

        // Populate Frame Gallery
        const gallery = document.getElementById("frames-gallery");
        gallery.innerHTML = "";

        data.frames.forEach(f => {
            const card = document.createElement("div");
            card.className = "frame-card bg-slate-950 border border-slate-800 rounded-xl overflow-hidden text-center";

            const badgeBg = f.is_fake ? "bg-rose-500/20 text-rose-400 border-rose-500/30" : "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
            const badgeText = f.is_fake ? `Fake ${f.fake_prob}%` : `Real ${f.real_prob}%`;

            card.innerHTML = `
                <div class="relative aspect-square bg-slate-900">
                    <img src="${f.image_url}" alt="Frame ${f.frame_num}" class="w-full h-full object-cover">
                    <span class="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded text-[10px] font-mono bg-black/70 text-slate-300">
                        ${f.timestamp}
                    </span>
                </div>
                <div class="p-2 space-y-1">
                    <p class="text-[11px] font-semibold text-slate-300">Frame #${f.frame_num}</p>
                    <span class="inline-block px-2 py-0.5 rounded text-[10px] font-bold border ${badgeBg}">
                        ${badgeText}
                    </span>
                </div>
            `;
            gallery.appendChild(card);
        });

        // Smooth scroll to results
        resultsDashboard.scrollIntoView({ behavior: "smooth" });
    }
});
