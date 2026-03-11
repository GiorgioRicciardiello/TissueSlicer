# TissueSlicer Visualization Improvement Plan
## Auto Contrast/Brightness Calibration for Web GUI

**Status:** Planning Phase
**Date:** 2026-03-11
**Priority:** High (improves usability for tissue selection)

---

## Executive Summary

Implement **automatic contrast/brightness calibration** for the web GUI preview image, similar to QuPath's Brightness/Contrast pane. This enables users to see tissue details clearly during cropping, improving region-of-interest selection accuracy.

**Recommended Approach:** Hybrid server-side + browser-side implementation
- **Server:** Initial percentile-based stretch on image load (fast, cached)
- **Browser:** Interactive sliders for real-time fine-tuning (no latency)

---

## Problem Statement

**Current Issue:**
- Users see low-contrast, dark preview images in the web interface
- Difficult to identify tissue boundaries and anatomical features
- Manual polygon drawing over poorly-visible features is error-prone

**Root Cause:**
- Preview image served as 8-bit PNG (0-255 range)
- Original 16-bit image data (0-65535 range) compressed to 8-bit without calibration
- No contrast stretching or histogram equalization applied
- No interactive controls for users to adjust display

**Impact:**
- Poor user experience during region selection
- May lead to incorrect ROI boundaries
- Requires users to zoom in frequently to see details

---

## Solution Architecture

### Phase 1: Server-Side Initial Calibration (Backend)
**Goal:** Pre-compute optimal brightness/contrast for each channel on image load

#### 1.1 Modify Image Loading Pipeline
**File:** `gui/backend/utils/image_loader.py`

**Changes:**
- Add calibration metadata to session when image is loaded
- Compute percentile values (2nd and 98th) for each channel
- Store calibration data in session

**Algorithm:** Percentile-based stretching (2-98)
```python
def compute_calibration_for_channel(channel_data: np.ndarray) -> dict:
    """Compute optimal display range for a 16-bit channel."""
    p2, p98 = np.percentile(channel_data, (2, 98))
    return {
        'p2': float(p2),
        'p98': float(p98),
        'min': float(np.min(channel_data)),
        'max': float(np.max(channel_data))
    }
```

#### 1.2 Add Calibration Endpoint
**File:** `gui/backend/app.py`

**New Endpoint:** `POST /api/get-calibration`
```
Request: { "channel_idx": 0, "timestamp": "session-id" }
Response: {
    "channel": 0,
    "name": "01_Nucleus_Hoechst",
    "p2": 150,
    "p98": 3200,
    "min": 0,
    "max": 4095,
    "suggested_stretch": [150, 3200]
}
```

### Phase 2: Browser-Side Interactive Calibration (Frontend)
**Goal:** Real-time contrast/brightness adjustment with instant preview

#### 2.1 Update Frontend Image Handling
**File:** `gui/frontend/app.js`

**Add Calibration UI:**
1. Two sliders for contrast control (min/max intensity mapping)
2. Toggle for auto-calibration
3. Reset button
4. Display current min/max values

**UI Layout:**
```
[Channel Selector Dropdown]

[Auto Calibration Button] [Reset Button]

Min Intensity: [=====|======]  Value: 150
Max Intensity: [=====|======]  Value: 3200

[Preview Image Canvas]
```

#### 2.2 Implement Canvas Rendering with Calibration
**Approach:** Modify image canvas rendering to apply contrast stretching

**Algorithm (JavaScript):**
```javascript
function stretchImageContrast(imageData, minVal, maxVal) {
    const data = imageData.data;
    const range = maxVal - minVal;

    for (let i = 0; i < data.length; i += 4) {
        // Assume single-channel grayscale in R, G, B
        const pixel = data[i];
        const stretched = Math.max(0, Math.min(255,
            ((pixel - minVal) / range) * 255
        ));
        data[i] = data[i+1] = data[i+2] = stretched;  // RGB
        // data[i+3] = alpha (unchanged)
    }
    return imageData;
}
```

#### 2.3 Handle 16-bit Data
**Challenge:** Canvas natively supports 8-bit (0-255) only. Need workaround.

**Solution:** Use Web Worker to process 16-bit data on load
```javascript
// Convert 16-bit preview to 8-bit with calibration applied
const worker = new Worker('calibration-worker.js');
worker.postMessage({
    imageData: uint16Array,
    width: 4096,
    height: 4096,
    minVal: 150,
    maxVal: 3200
});
worker.onmessage = (event) => {
    canvas.putImageData(event.data.imageData);
};
```

---

## Implementation Phases

### Phase 1: Backend Calibration Computation (2-3 hours)
**Deliverables:**
1. ✓ Calibration utility function in `image_loader.py`
2. ✓ Store calibration in session
3. ✓ New `/api/get-calibration` endpoint
4. ✓ Integration with `/api/load-image` response

**Files to Modify:**
- `gui/backend/utils/image_loader.py` — Add `compute_calibration_for_channel()`
- `gui/backend/utils/session_manager.py` — Store calibration metadata
- `gui/backend/app.py` — Add endpoint

**Testing:**
- Verify calibration values reasonable (p2 < p98 < max)
- Check performance impact (should be <100ms per channel)

---

### Phase 2: Frontend Calibration UI (3-4 hours)
**Deliverables:**
1. ✓ HTML controls (sliders, buttons)
2. ✓ CSS styling matching dark theme
3. ✓ JavaScript slider event handlers
4. ✓ Real-time canvas updates

**Files to Modify:**
- `gui/frontend/index.html` — Add calibration UI controls
- `gui/frontend/style.css` — Style controls
- `gui/frontend/app.js` — Implement calibration logic

**Features:**
- Dual-slider min/max control
- Auto-calibration button (pre-compute on server)
- Reset to defaults
- Display current values
- Channel-specific calibration (each channel independent)

**Testing:**
- Slider adjustments trigger instant preview update
- Values persist while scrolling/zooming
- Works across all channels

---

### Phase 3: Web Worker Integration (Optional, 1-2 hours)
**For performance optimization (if needed)**

**Deliverables:**
1. ✓ `gui/frontend/calibration-worker.js` — Offload histogram computation
2. ✓ Non-blocking UI during large image processing

**Rationale:**
- Only if preview rendering is slow (< 100ms acceptable for 4K images)
- Monitor performance first, add worker if needed

---

## Technical Details

### Calibration Algorithm: Percentile Stretching

**Why Percentile-based?**
- Industry standard in microscopy (ignores outliers)
- Works with 16-bit data
- Fast (O(n) single pass)
- Preserves dynamic range

**Formula:**
```
stretched_pixel = ((raw_pixel - p2) / (p98 - p2)) × 255
```

**Example:**
- Raw 16-bit value: 2500 (range 0-65535)
- p2 = 150, p98 = 3200
- Stretched = ((2500 - 150) / (3200 - 150)) × 255 = 205/8-bit

### Data Flow

```
┌─────────────────────┐
│  User loads image   │
│  (16-bit OME-TIFF)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  Backend: Compute calibration           │
│  - Load downsampled channel             │
│  - Compute p2, p98, min, max            │
│  - Cache in session                     │
└──────────┬──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  Frontend: Display preview with sliders │
│  - Convert 16-bit → 8-bit with stretch  │
│  - Show initial calibration             │
│  - User adjusts sliders                 │
│  - Real-time canvas update              │
└─────────────────────────────────────────┘
```

---

## Dependencies & Resources

### Python (Backend)
- **numpy** (already installed) — percentile computation
- **scikit-image** (already installed) — optional for future enhancements

### JavaScript (Frontend)
- **No new dependencies** — use vanilla JS
- **Optional:** Bootstrap/jQuery for slider UI (if not already present)

### Browser APIs
- **Canvas API** — image rendering
- **Web Workers** — optional, for performance

---

## Testing Strategy

### Unit Tests
1. **Backend calibration function**
   ```python
   def test_percentile_calibration():
       # Test with synthetic 16-bit data
       data = np.arange(0, 65536)
       cal = compute_calibration_for_channel(data)
       assert cal['p2'] < cal['p98']
       assert cal['p2'] >= 0
   ```

2. **Endpoint validation**
   ```
   POST /api/get-calibration
   → Verify response contains p2, p98, min, max
   ```

### Integration Tests
1. Load multi-channel image → verify all channels calibrated
2. Compare preview before/after calibration → should be more visible
3. Adjust sliders → canvas updates in <100ms

### Visual Verification
1. Open tissue image in GUI
2. Verify preview is clearly visible (not dark/washed out)
3. Adjust sliders → contrast increases/decreases smoothly
4. Switch channels → calibration applies per-channel

---

## Success Criteria

✓ **Usability:** Users can clearly see tissue details in preview
✓ **Performance:** Slider adjustment → canvas update < 100ms
✓ **Compatibility:** Works across all 20 channels
✓ **Robustness:** Handles edge cases (saturated images, empty regions)
✓ **Documentation:** Clear UI labels and tooltips for controls

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Canvas performance degrades | Medium | High | Implement Web Worker |
| 16-bit → 8-bit conversion artifacts | Low | Medium | Use proper stretching formula |
| Slider responsiveness lag | Low | Medium | Debounce slider events |
| Memory issues with large preview | Low | Medium | Cap preview size at 4K |

---

## Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Backend calibration | 2-3 hrs | Day 1 | Day 1 |
| Frontend UI + logic | 3-4 hrs | Day 2 | Day 2 |
| Testing & refinement | 1-2 hrs | Day 2-3 | Day 3 |
| **Total** | **6-9 hrs** | — | — |

---

## Future Enhancements (Not in Scope)

1. **Histogram equalization** — More advanced contrast enhancement
2. **CLAHE (Adaptive)** — Local contrast enhancement
3. **Presets** — Save/load calibration profiles
4. **Batch calibration** — Apply same settings across multiple channels
5. **Auto-levels** — AI-based optimal calibration suggestion

---

## Sign-Off

**Approval Status:** ⏳ Pending Review

- [ ] Backend team approval
- [ ] Frontend team approval
- [ ] UX review
- [ ] Performance review

---

## References

- Percentile stretching: Industry standard in microscopy (napari, Icy)
- Canvas API: MDN Web Docs
- Web Workers: Web Workers API specification
- scikit-image contrast functions: https://scikit-image.org/docs/0.25.x/auto_examples/color_exposure/plot_equalize.html
