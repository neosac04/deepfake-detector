/* ============================================================================
   DEEPGUARD - AI Deepfake Detector
   Vanilla JavaScript - No frameworks or libraries
   ============================================================================ */

class DeepGuard {
    constructor() {
        // DOM Elements
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('fileInput');
        this.uploadButton = document.getElementById('uploadButton');
        this.previewContainer = document.getElementById('previewContainer');
        this.previewImage = document.getElementById('previewImage');
        this.previewFilename = document.getElementById('previewFilename');
        this.previewSize = document.getElementById('previewSize');
        this.changeImageBtn = document.getElementById('changeImageBtn');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        
        this.uploadPanel = document.getElementById('uploadPanel');
        this.analysisPanel = document.getElementById('analysisPanel');
        this.resultsContainer = document.getElementById('resultsContainer');
        this.predictionPanel = document.getElementById('predictionPanel');
        this.explanationPanel = document.getElementById('explanationPanel');
        this.actionsPanel = document.getElementById('actionsPanel');
        
        this.resultBadge = document.getElementById('resultBadge');
        this.confidencePercentage = document.getElementById('confidencePercentage');
        this.confidenceFill = document.getElementById('confidenceFill');
        
        this.analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
        this.downloadReportBtn = document.getElementById('downloadReportBtn');
        
        // State
        this.selectedFile = null;
        this.currentStep = 1;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Upload zone interactions
        this.uploadZone.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));
        
        // Drag and drop
        this.uploadZone.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadZone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadZone.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Upload button
        this.uploadButton.addEventListener('click', () => this.fileInput.click());
        this.changeImageBtn.addEventListener('click', () => this.resetUpload());
        
        // Analyze button
        this.analyzeBtn.addEventListener('click', () => this.startAnalysis());
        
        // Action buttons
        this.analyzeAnotherBtn.addEventListener('click', () => this.reset());
        this.downloadReportBtn.addEventListener('click', () => this.downloadReport());
    }
    
    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadZone.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadZone.classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (this.isValidImageFile(file)) {
                this.handleFileSelect(file);
            } else {
                alert('Please drop a valid image file (JPG, PNG, or WebP)');
            }
        }
    }
    
    handleFileSelect(file) {
        if (!file) return;
        
        if (!this.isValidImageFile(file)) {
            alert('Invalid file type. Please select a JPG, PNG, or WebP image.');
            return;
        }
        
        this.selectedFile = file;
        this.displayPreview(file);
        this.analyzeBtn.disabled = false;
    }
    
    isValidImageFile(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
        return file && validTypes.includes(file.type);
    }
    
    displayPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.previewFilename.textContent = file.name;
            this.previewSize.textContent = this.formatFileSize(file.size);
            this.previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }
    
    resetUpload() {
        this.selectedFile = null;
        this.fileInput.value = '';
        this.previewContainer.style.display = 'none';
        this.analyzeBtn.disabled = true;
    }
    
    async startAnalysis() {
        // Update stepper
        this.updateStepper(2);
        
        // Show analysis panel
        this.hidePanel(this.uploadPanel);
        this.showPanel(this.analysisPanel);
        
        // Simulate analysis delay (2-3 seconds)
        const delay = 2000 + Math.random() * 1000;
        await this.sleep(delay);
        
        // Generate results
        this.generatePrediction();
        
        // Show results
        this.hidePanel(this.analysisPanel);
        this.showPanel(this.resultsContainer);
        this.showPanel(this.actionsPanel);
        
        // Update stepper
        this.updateStepper(3);
        setTimeout(() => this.updateStepper(4), 500);
    }
    
    generatePrediction() {
        // Random result: REAL or DEEPFAKE
        const isReal = Math.random() > 0.5;
        const confidence = 85 + Math.random() * 14; // 85-99%
        
        // Update result badge
        this.resultBadge.textContent = isReal ? 'REAL' : 'DEEPFAKE DETECTED';
        this.resultBadge.className = `result-badge ${isReal ? 'real' : 'deepfake'}`;
        
        // Update confidence
        this.confidencePercentage.textContent = confidence.toFixed(1) + '%';
        this.confidenceFill.style.width = confidence + '%';
        
        // Generate breakdown metrics
        this.generateMetrics();
        
        // Store result for report
        this.lastResult = {
            isReal,
            confidence,
            timestamp: new Date().toLocaleString()
        };
    }
    
    generateMetrics() {
        // Generate random but realistic metrics
        const authenticity = 80 + Math.random() * 18;
        const texture = 82 + Math.random() * 16;
        const artifact = 78 + Math.random() * 20;
        
        // Update metrics
        document.getElementById('metricAuthenticity').textContent = Math.round(authenticity) + '%';
        document.getElementById('metricAuthenticityFill').style.width = authenticity + '%';
        
        document.getElementById('metricTexture').textContent = Math.round(texture) + '%';
        document.getElementById('metricTextureFill').style.width = texture + '%';
        
        document.getElementById('metricArtifact').textContent = Math.round(artifact) + '%';
        document.getElementById('metricArtifactFill').style.width = artifact + '%';
    }
    
    updateStepper(step) {
        // Update all steps
        const steps = document.querySelectorAll('.step');
        steps.forEach((stepEl, index) => {
            stepEl.classList.remove('active');
            if (index < step) {
                stepEl.classList.add('active');
            }
        });
        
        this.currentStep = step;
    }
    
    showPanel(panel) {
        panel.style.display = panel.classList.contains('results-container') ? 'grid' : 'block';
    }
    
    hidePanel(panel) {
        panel.style.display = 'none';
    }
    
    reset() {
        // Reset everything
        this.resetUpload();
        
        // Hide all panels
        this.hidePanel(this.analysisPanel);
        this.hidePanel(this.resultsContainer);
        this.hidePanel(this.actionsPanel);
        
        // Show upload panel
        this.showPanel(this.uploadPanel);
        
        // Reset stepper
        this.updateStepper(1);
        
        // Reset file input
        this.fileInput.value = '';
        this.selectedFile = null;
    }
    
    downloadReport() {
        if (!this.lastResult) {
            alert('No analysis results to download');
            return;
        }
        
        // Generate mock report
        const report = this.generateReport();
        
        // Create and download file
        const element = document.createElement('a');
        element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(report));
        element.setAttribute('download', 'deepguard_report.txt');
        element.style.display = 'none';
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
        
        alert('Report downloaded successfully!');
    }
    
    generateReport() {
        const result = this.lastResult;
        const filename = this.selectedFile ? this.selectedFile.name : 'unknown';
        
        const report = `
═══════════════════════════════════════════════════════════
                    DEEPGUARD ANALYSIS REPORT
═══════════════════════════════════════════════════════════

ANALYSIS DETAILS
─────────────────────────────────────────────────────────
Timestamp:      ${result.timestamp}
Image File:     ${filename}
Model Used:     MobileNetV2
Model Version:  2.1.0

PREDICTION RESULT
─────────────────────────────────────────────────────────
Classification: ${result.isReal ? 'REAL' : 'DEEPFAKE DETECTED'}
Confidence:     ${result.confidence.toFixed(1)}%

CONFIDENCE BREAKDOWN
─────────────────────────────────────────────────────────
Face Authenticity:  ${Math.round(80 + Math.random() * 18)}%
Texture Analysis:   ${Math.round(82 + Math.random() * 16)}%
Artifact Score:     ${Math.round(78 + Math.random() * 20)}%

KEY FINDINGS
─────────────────────────────────────────────────────────
• Inconsistent facial blending around the jawline
• Unnatural eye reflection patterns detected
• GAN compression artifacts found in high-frequency regions
• Skin texture inconsistency in cheek area
• Asymmetric facial feature distribution

DISCLAIMER
─────────────────────────────────────────────────────────
This analysis is for research purposes only. DeepGuard 
provides AI-assisted detection but should not be relied 
upon as the sole source of truth for critical decisions. 
Always verify results independently.

═══════════════════════════════════════════════════════════
Generated by DeepGuard © 2025
═══════════════════════════════════════════════════════════
        `;
        
        return report.trim();
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new DeepGuard();
});
