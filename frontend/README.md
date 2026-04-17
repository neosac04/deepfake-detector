# DeepGuard — AI Deepfake Detector

A professional, responsive single-page web application for detecting AI-generated or manipulated faces. Built with pure HTML, CSS, and vanilla JavaScript — no frameworks or libraries.

## Features

### 🎯 User Experience
- **Clean, minimal design** with a dark navy header and light gray background
- **Professional stepper** showing 4 steps: Upload → Analysis → Prediction → Explanation
- **Fully responsive** design that works on desktop, tablet, and mobile devices
- **Smooth animations** with fade-in transitions between panels

### 📤 Upload Panel
- Large drag-and-drop zone with cloud icon
- Click to browse file selector
- Accepts JPG, PNG, and WebP images only
- Shows image preview with filename and file size
- Option to change/replace selected image
- "Analyze Image" button (disabled until image is uploaded)

### ⚙️ Analysis Panel
- Animated loading spinner
- "Analyzing image for deepfake indicators..." message
- Simulated 2–3 second analysis delay using `setTimeout`

### 🎲 Prediction Panel
- Large colored badge showing **REAL** (green) or **DEEPFAKE DETECTED** (red)
- Confidence score displayed as percentage + horizontal progress bar
- Model info showing "Model: MobileNetV2"
- Smooth animations when result appears

### 📊 Explanation Panel
- "Why this prediction?" section with 5 detailed bullet points
- Each reason has a colored importance indicator bar on the left
- **Confidence Breakdown** section with 3 metric cards:
  - Face Authenticity
  - Texture Analysis
  - Artifact Score
- Each metric shows a percentage and visual progress bar

### 🔄 Action Buttons
- "Analyze Another Image" button — resets the entire UI
- "Download Report" button — generates and downloads a text report

## Project Structure

```
frontend/
├── index.html          # Main HTML file with all sections
├── css/
│   └── style.css       # All styling and responsive design
├── js/
│   └── app.js          # Vanilla JavaScript functionality
└── assets/             # (Future) Icons, images, etc.
```

## How to Use

1. **Open the app**: Simply open `index.html` in any modern web browser
   - No server required
   - No build process
   - Works offline

2. **Upload an image**:
   - Drag & drop an image onto the zone, or
   - Click the upload button to browse files
   - Only JPG, PNG, and WebP files are accepted

3. **Analyze**:
   - Click "Analyze Image" button
   - The stepper updates to show progress
   - Loading spinner appears for 2–3 seconds
   - Watch the stepper advance through steps

4. **Review Results**:
   - See the prediction badge (REAL or DEEPFAKE)
   - Check confidence score with progress bar
   - Read the explanation with key findings
   - View confidence breakdown metrics

5. **Next Steps**:
   - Download the report (saves as .txt file)
   - Analyze another image (full reset)

## Technical Details

### HTML
- Semantic HTML5 structure
- All required sections pre-built in the markup
- Hidden panels shown via JavaScript on demand

### CSS
- **CSS Variables** for consistent theming
- **CSS Grid & Flexbox** for responsive layouts
- **Animations** using `@keyframes` (fade-in, slide-down, spin)
- **Mobile-first** responsive breakpoints
- Dark navy (`#1a2332`) + light gray (`#f5f5f5`) color scheme

### JavaScript (Vanilla)
- **Class-based architecture** with `DeepGuard` class
- **No dependencies** — pure ES6+ JavaScript
- **Event listeners** for upload, drag-drop, clicks
- **File validation** for image types
- **FileReader API** for image preview
- **Randomization** for realistic results:
  - 50/50 chance of REAL or DEEPFAKE
  - Confidence scores between 85–99%
  - Metrics vary each analysis
- **Local storage** of results for report generation
- **Report generation** as downloadable text file

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Modern mobile browsers

## Color Scheme

| Color | HEX | Usage |
|-------|-----|-------|
| Navy | `#1a2332` | Header background |
| Light Gray | `#f5f5f5` | Main background |
| Blue | `#0066ff` | Primary buttons, active states |
| Green | `#10b981` | REAL result badge |
| Red | `#ef4444` | DEEPFAKE result badge |
| White | `#ffffff` | Text on dark backgrounds |

## Font

- **System UI font stack**: `system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif`
- Clean, modern, platform-native appearance

## Future Enhancements

- Backend integration for real ML model predictions
- Camera capture from webcam
- Batch image processing
- Result history storage in localStorage
- Export reports as PDF
- Dark mode toggle
- Multi-language support
- API integration with actual deepfake detection models

## Notes

- All predictions are **simulated** and for demonstration purposes
- Results are randomly generated to show UI/UX flow
- No real ML inference happens on the frontend
- The app is designed to be easily integrated with a backend service
- For production use, connect to an actual deepfake detection API

## License

Open source. Feel free to modify and use as needed.

---

**DeepGuard © 2025** — Detect AI-generated or manipulated faces with confidence.
