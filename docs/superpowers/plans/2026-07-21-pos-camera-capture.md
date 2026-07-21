# POS Camera Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one-frame browser camera capture to the POS while preserving image upload, the existing YOLO checkout endpoint, manual correction, and checkout behavior.

**Architecture:** A focused `pos_camera.js` module owns browser media APIs, modal state, still-image capture, camera switching, and stream cleanup. The existing POS script extracts one shared checkout-image submission function so uploaded files and camera JPEG blobs follow the same authenticated `/s1/checkout` path. No camera image is persisted and no backend camera service is introduced.

**Tech Stack:** Vanilla JavaScript, browser MediaDevices API, HTML5 video/canvas, FastAPI, existing YOLO checkout endpoint, pytest static frontend contracts, Playwright smoke verification, Windows 10/11 camera hardware.

---

## File Structure

- Create `api/module4_frontend/static/pos_camera.js`: camera permissions, stream lifecycle, device switching, capture, retake, recognize, and cleanup.
- Modify `api/module4_frontend/static/index.html`: camera modal and styles, separate camera/upload commands, shared checkout-image submission, lifecycle hooks, translations, and script loading.
- Create `tests/test_s1_pos_camera_frontend.py`: repository-level contracts for the module boundary, shared submission path, privacy, cleanup, and fallback behavior.

No backend route, database table, model file, or Python dependency changes are required.

### Task 1: Lock The Frontend Contract With Failing Tests

**Files:**
- Create: `tests/test_s1_pos_camera_frontend.py`
- Inspect: `api/module4_frontend/static/index.html`
- Inspect: `api/module4_frontend/static/pos_camera.js`

- [ ] **Step 1: Write the failing camera contract tests**

Create `tests/test_s1_pos_camera_frontend.py` with:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "api" / "module4_frontend" / "static" / "index.html"
CAMERA = ROOT / "api" / "module4_frontend" / "static" / "pos_camera.js"


def test_pos_loads_focused_camera_module_and_exposes_two_image_sources():
    html = INDEX.read_text(encoding="utf-8")

    assert '<script src="/pos_camera.js"></script>' in html
    assert "openPOSCamera()" in html
    assert "triggerScan()" in html
    assert "Open Camera" in html
    assert "Upload Image" in html


def test_camera_module_uses_one_frame_media_capture_and_device_switching():
    source = CAMERA.read_text(encoding="utf-8")

    assert "navigator.mediaDevices.getUserMedia" in source
    assert "navigator.mediaDevices.enumerateDevices" in source
    assert 'facingMode:{ideal:"environment"}' in source
    assert "deviceId:{exact:deviceId}" in source
    assert 'canvas.toBlob' in source
    assert '"image/jpeg"' in source
    assert "switchCamera" in source


def test_camera_module_releases_streams_and_does_not_persist_images():
    source = CAMERA.read_text(encoding="utf-8")

    assert "stream.getTracks().forEach" in source
    assert "track.stop()" in source
    assert "beforeunload" in source
    assert "localStorage" not in source
    assert "sessionStorage" not in source
    assert "indexedDB" not in source
    assert "fetch(" not in source


def test_upload_and_camera_share_checkout_image_submission():
    html = INDEX.read_text(encoding="utf-8")

    assert "async function submitCheckoutImage(imageFile)" in html
    assert "return submitCheckoutImage(file);" in html
    assert "submitImage:submitCheckoutImage" in html
    assert "API+'/s1/checkout'" in html
    assert "if(scanning)return;" in html


def test_pos_navigation_and_logout_close_the_camera():
    html = INDEX.read_text(encoding="utf-8")

    assert "if(panel!=='pos'&&window.POSCamera)window.POSCamera.close();" in html
    assert "if(window.POSCamera)window.POSCamera.close();" in html
```

- [ ] **Step 2: Run the new tests and verify the expected failure**

Run:

```powershell
& 'C:\Users\Curtis\AppData\Local\BakeryAI\venv313\Scripts\python.exe' -m pytest tests\test_s1_pos_camera_frontend.py -q
```

Expected: FAIL because `pos_camera.js`, the camera commands, and the shared submission function do not exist.

- [ ] **Step 3: Commit the failing contract tests**

```powershell
git add tests/test_s1_pos_camera_frontend.py
git commit -m "test: define POS camera capture contract"
```

### Task 2: Implement The Camera Module And Modal

**Files:**
- Create: `api/module4_frontend/static/pos_camera.js`
- Modify: `api/module4_frontend/static/index.html:56-74`
- Modify: `api/module4_frontend/static/index.html:289`
- Modify: `api/module4_frontend/static/index.html:4816`
- Test: `tests/test_s1_pos_camera_frontend.py`

- [ ] **Step 1: Add stable camera modal styles**

Add these styles beside the existing scan styles in `index.html`:

```css
.scan-actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.scan-action{min-height:68px;border:1px solid #d4c4a8;background:#fff;color:#8b6914;display:flex;align-items:center;justify-content:center;gap:8px;font-weight:700;cursor:pointer}
.scan-action:hover,.scan-action:focus-visible{border-color:#8b6914;background:#fdf9f3;outline:2px solid rgba(139,105,20,.18);outline-offset:1px}
.scan-action-icon{font-size:24px;line-height:1}
.camera-modal-box{width:min(760px,calc(100vw - 32px));max-height:calc(100vh - 32px);overflow:auto}
.camera-preview{position:relative;width:100%;aspect-ratio:16/9;background:#111;overflow:hidden;border:1px solid #d4c4a8}
.camera-preview video,.camera-preview img{width:100%;height:100%;object-fit:contain;display:block}
.camera-preview canvas{display:none}
.camera-error{min-height:22px;margin-top:8px;color:#c0392b;font-size:13px}
.camera-controls{display:flex;gap:8px;justify-content:flex-end;flex-wrap:wrap;margin-top:12px}
.camera-controls .btn{min-width:108px}
@media(max-width:640px){.scan-actions{grid-template-columns:1fr}.camera-controls .btn{flex:1;min-width:120px}}
```

- [ ] **Step 2: Add the static modal markup**

Insert the modal after the hidden `scan-file` input:

```html
<div id="camera-modal" class="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="camera-modal-title" onclick="if(event.target===this&&window.POSCamera)window.POSCamera.close()">
<div class="modal-box camera-modal-box">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
<h3 id="camera-modal-title" style="margin:0" data-i18n="Camera Capture">Camera Capture</h3>
<button type="button" onclick="window.POSCamera.close()" title="Close" aria-label="Close" style="background:none;border:none;font-size:24px;cursor:pointer;color:#8b6914;padding:0;line-height:1">&times;</button>
</div>
<div class="camera-preview">
<video id="camera-video" autoplay playsinline muted></video>
<img id="camera-photo" alt="Captured tray" hidden>
<canvas id="camera-canvas"></canvas>
</div>
<div id="camera-error" class="camera-error" role="status" aria-live="polite"></div>
<div class="camera-controls">
<button id="camera-switch" type="button" class="btn btn-outline" onclick="window.POSCamera.switchCamera()" hidden data-i18n="Switch Camera">Switch Camera</button>
<button id="camera-retake" type="button" class="btn btn-outline" onclick="window.POSCamera.retake()" hidden data-i18n="Retake">Retake</button>
<button id="camera-capture" type="button" class="btn btn-primary" onclick="window.POSCamera.capture()" data-i18n="Capture">Capture</button>
<button id="camera-recognize" type="button" class="btn btn-primary" onclick="window.POSCamera.recognize()" hidden data-i18n="Recognize">Recognize</button>
</div>
</div>
</div>
```

- [ ] **Step 3: Implement `pos_camera.js`**

Create `api/module4_frontend/static/pos_camera.js`:

```javascript
(function(global){
"use strict";

var stream=null;
var capturedBlob=null;
var capturedUrl="";
var deviceIds=[];
var deviceIndex=0;
var submitImage=null;
var translate=function(value){return value;};

function byId(id){return document.getElementById(id);}

function setError(message){
var element=byId("camera-error");
if(element)element.textContent=message||"";
}

function stopStream(){
if(!stream)return;
stream.getTracks().forEach(function(track){track.stop();});
stream=null;
var video=byId("camera-video");
if(video)video.srcObject=null;
}

function clearCapture(){
capturedBlob=null;
if(capturedUrl){URL.revokeObjectURL(capturedUrl);capturedUrl="";}
var photo=byId("camera-photo");
if(photo){photo.removeAttribute("src");photo.hidden=true;}
var video=byId("camera-video");
if(video)video.hidden=false;
}

function showCaptureControls(captured){
byId("camera-capture").hidden=captured;
byId("camera-retake").hidden=!captured;
byId("camera-recognize").hidden=!captured;
}

function cameraMessage(error){
if(!error)return translate("Camera unavailable");
if(error.name==="NotAllowedError")return translate("Camera permission denied");
if(error.name==="NotFoundError")return translate("No camera found");
if(error.name==="NotReadableError")return translate("Camera is already in use");
return translate("Camera unavailable");
}

async function refreshDevices(){
var devices=await navigator.mediaDevices.enumerateDevices();
deviceIds=devices.filter(function(device){return device.kind==="videoinput";}).map(function(device){return device.deviceId;}).filter(Boolean);
var switchButton=byId("camera-switch");
switchButton.hidden=deviceIds.length<2;
}

async function startStream(deviceId){
stopStream();
var videoConstraints=deviceId?{
deviceId:{exact:deviceId},width:{ideal:1280},height:{ideal:720}
}:{
facingMode:{ideal:"environment"},width:{ideal:1280},height:{ideal:720}
};
stream=await navigator.mediaDevices.getUserMedia({video:videoConstraints,audio:false});
var video=byId("camera-video");
video.srcObject=stream;
await video.play();
await refreshDevices();
}

async function open(options){
submitImage=options&&options.submitImage;
translate=options&&options.translate?options.translate:translate;
if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){
alert(translate("Camera is not supported by this browser"));
return;
}
clearCapture();
showCaptureControls(false);
setError("");
byId("camera-modal").classList.add("show");
try{
await startStream("");
}catch(error){
stopStream();
setError(cameraMessage(error));
}
}

function capture(){
var video=byId("camera-video");
if(!stream||!video.videoWidth||!video.videoHeight){setError(translate("Camera frame is not ready"));return;}
var canvas=byId("camera-canvas");
canvas.width=video.videoWidth;
canvas.height=video.videoHeight;
canvas.getContext("2d").drawImage(video,0,0,canvas.width,canvas.height);
canvas.toBlob(function(blob){
if(!blob){setError(translate("Camera capture failed"));return;}
capturedBlob=blob;
capturedUrl=URL.createObjectURL(blob);
var photo=byId("camera-photo");
photo.src=capturedUrl;
photo.hidden=false;
video.hidden=true;
showCaptureControls(true);
setError("");
},"image/jpeg",0.9);
}

function retake(){
clearCapture();
showCaptureControls(false);
setError("");
}

async function recognize(){
if(!capturedBlob||typeof submitImage!=="function")return;
var imageFile=new File([capturedBlob],"camera-capture.jpg",{type:"image/jpeg"});
close();
await submitImage(imageFile);
}

async function switchCamera(){
if(deviceIds.length<2)return;
deviceIndex=(deviceIndex+1)%deviceIds.length;
clearCapture();
showCaptureControls(false);
setError("");
try{
await startStream(deviceIds[deviceIndex]);
}catch(error){
stopStream();
setError(cameraMessage(error));
}
}

function close(){
stopStream();
clearCapture();
showCaptureControls(false);
setError("");
var modal=byId("camera-modal");
if(modal)modal.classList.remove("show");
}

document.addEventListener("keydown",function(event){
if(event.key==="Escape")close();
});
global.addEventListener("beforeunload",close);

global.POSCamera={open:open,close:close,capture:capture,retake:retake,recognize:recognize,switchCamera:switchCamera};
})(window);
```

- [ ] **Step 4: Load the module before the existing S5 frontend script**

At the bottom of `index.html`, use:

```html
<script src="/pos_camera.js"></script>
<script src="/s5_analysis.js"></script>
```

- [ ] **Step 5: Run the focused tests and inspect the expected remaining failures**

```powershell
& 'C:\Users\Curtis\AppData\Local\BakeryAI\venv313\Scripts\python.exe' -m pytest tests\test_s1_pos_camera_frontend.py -q
```

Expected: module lifecycle tests PASS; tests for POS commands, shared submission, and navigation hooks remain FAIL.

- [ ] **Step 6: Commit the camera module and modal**

```powershell
git add api/module4_frontend/static/pos_camera.js api/module4_frontend/static/index.html
git commit -m "feat: add POS camera capture modal"
```

### Task 3: Share The Checkout Recognition Path

**Files:**
- Modify: `api/module4_frontend/static/index.html:2053-2057`
- Modify: `api/module4_frontend/static/index.html:2438-2450`
- Modify: `api/module4_frontend/static/index.html:4440`
- Test: `tests/test_s1_pos_camera_frontend.py`

- [ ] **Step 1: Replace the single scan area with camera and upload commands**

In `renderPOS()`, replace the current clickable scan area with:

```javascript
h+='<div class="panel"><h4>'+t('Camera - Visual Recognition')+'</h4>';
h+='<div class="scan-actions">';
h+='<button type="button" class="scan-action" onclick="openPOSCamera()"><span class="scan-action-icon">&#128247;</span><span>'+t('Open Camera')+'</span></button>';
h+='<button type="button" class="scan-action" onclick="triggerScan()"><span class="scan-action-icon">&#8682;</span><span>'+t('Upload Image')+'</span></button>';
h+='</div>';
h+='<div id="detect-list">';
```

Keep the existing detection-list rendering immediately after this block.

- [ ] **Step 2: Extract result mapping and shared submission**

Replace `handleScan()` with:

```javascript
function applyCheckoutDetections(rows){
lastScanResult=rows||[];
detections=[];
for(var i=0;i<lastScanResult.length;i++){
var det=lastScanResult[i];
detections.push({product_name:det.product_name,quantity:det.quantity||1,confidence:det.confidence||0.5,bbox:det.bbox||[],tray_color:det.tray_color||'green',status:det.confidence>=0.85?'confirmed':(det.confidence>=0.7?'review':'pending'),stock:det.stock||'--'});
}
hitlLog=[];
}

async function submitCheckoutImage(imageFile){
if(scanning)return;
scanning=true;
renderPOS(document.getElementById('panel-pos')||document.getElementById('content-area'));
try{
var fd=new FormData();fd.append('file',imageFile);
var response=await fetchWithTimeout(API+'/s1/checkout',{method:'POST',headers:{'Authorization':'Bearer '+token},body:fd},120000);
var payload=await response.json();
if(!response.ok)throw new Error(payload.detail||t('Scan failed'));
applyCheckoutDetections(payload.detections||[]);
if(!detections.length)alert(t('No products detected'));
}catch(error){
alert(t('Scan error')+': '+error.message);
}finally{
scanning=false;
var input=document.getElementById('scan-file');if(input)input.value='';
renderPOS(document.getElementById('panel-pos')||document.getElementById('content-area'));
}
}

function handleScan(event){
var file=event.target.files[0];
if(!file)return;
return submitCheckoutImage(file);
}

function openPOSCamera(){
if(!window.POSCamera){alert(t('Camera is not supported by this browser'));return;}
window.POSCamera.open({submitImage:submitCheckoutImage,translate:t});
}
```

- [ ] **Step 3: Keep upload behavior explicit**

Keep `triggerScan()` as the upload-only command:

```javascript
function triggerScan(){
var input=document.getElementById("scan-file");
input.onchange=handleScan;
input.click();
}
```

Do not route fresh-batch inflow through the new POS camera. Its existing upload/manual flow remains unchanged because this feature is checkout-scoped.

- [ ] **Step 4: Run focused tests**

```powershell
& 'C:\Users\Curtis\AppData\Local\BakeryAI\venv313\Scripts\python.exe' -m pytest tests\test_s1_pos_camera_frontend.py -q
```

Expected: image-source and shared-submission tests PASS; navigation and logout cleanup tests remain FAIL.

- [ ] **Step 5: Commit the shared recognition path**

```powershell
git add api/module4_frontend/static/index.html tests/test_s1_pos_camera_frontend.py
git commit -m "refactor: share POS checkout image recognition"
```

### Task 4: Add Lifecycle Hooks And Translations

**Files:**
- Modify: `api/module4_frontend/static/index.html:478-579`
- Modify: `api/module4_frontend/static/index.html:960-1062`
- Modify: `api/module4_frontend/static/index.html:1699`
- Modify: `api/module4_frontend/static/index.html:1751`
- Test: `tests/test_s1_pos_camera_frontend.py`

- [ ] **Step 1: Close the camera on sign-out**

Start `doLogout()` with:

```javascript
function doLogout(){
if(window.POSCamera)window.POSCamera.close();
token='';role='';username='';cartItems=[];detections=[];hitlLog=[];bundleRecs=[];resetRecommendationSelectionState();
```

Keep the remaining logout statements unchanged.

- [ ] **Step 2: Close the camera when leaving POS**

Start `showPanel(panel)` with:

```javascript
async function showPanel(panel){
if(!canAccessPanel(panel))panel='pos';
if(panel!=='pos'&&window.POSCamera)window.POSCamera.close();
```

Keep the remaining panel-switch behavior unchanged.

- [ ] **Step 3: Add English and Chinese translation entries**

Add these English keys to the existing English translation object:

```javascript
'Open Camera':'Open Camera','Upload Image':'Upload Image','Camera Capture':'Camera Capture',
'Switch Camera':'Switch Camera','Capture':'Capture','Retake':'Retake','Recognize':'Recognize',
'Camera unavailable':'Camera unavailable','Camera permission denied':'Camera permission denied',
'No camera found':'No camera found','Camera is already in use':'Camera is already in use',
'Camera is not supported by this browser':'Camera is not supported by this browser',
'Camera frame is not ready':'Camera frame is not ready','Camera capture failed':'Camera capture failed',
'No products detected':'No products detected'
```

Add these values to the existing Chinese translation object using escaped Unicode so the source remains ASCII:

```javascript
'Open Camera':'\u6253\u5f00\u6444\u50cf\u5934','Upload Image':'\u4e0a\u4f20\u56fe\u7247','Camera Capture':'\u6444\u50cf\u5934\u62cd\u6444',
'Switch Camera':'\u5207\u6362\u6444\u50cf\u5934','Capture':'\u62cd\u6444','Retake':'\u91cd\u62cd','Recognize':'\u8bc6\u522b',
'Camera unavailable':'\u6444\u50cf\u5934\u4e0d\u53ef\u7528','Camera permission denied':'\u6444\u50cf\u5934\u6743\u9650\u88ab\u62d2\u7edd',
'No camera found':'\u672a\u627e\u5230\u6444\u50cf\u5934','Camera is already in use':'\u6444\u50cf\u5934\u6b63\u88ab\u5176\u4ed6\u7a0b\u5e8f\u4f7f\u7528',
'Camera is not supported by this browser':'\u5f53\u524d\u6d4f\u89c8\u5668\u4e0d\u652f\u6301\u6444\u50cf\u5934',
'Camera frame is not ready':'\u6444\u50cf\u5934\u753b\u9762\u5c1a\u672a\u51c6\u5907\u5b8c\u6210','Camera capture failed':'\u62cd\u6444\u5931\u8d25',
'No products detected':'\u672a\u8bc6\u522b\u5230\u4ea7\u54c1'
```

- [ ] **Step 4: Run all focused frontend contracts**

```powershell
& 'C:\Users\Curtis\AppData\Local\BakeryAI\venv313\Scripts\python.exe' -m pytest tests\test_s1_pos_camera_frontend.py tests\test_auth_boundaries.py tests\test_s4_beverage_customization_frontend.py -q
```

Expected: PASS.

- [ ] **Step 5: Scan edited frontend files for accidental non-ASCII source additions**

```powershell
rg -n "[\p{Han}]" api/module4_frontend/static/index.html api/module4_frontend/static/pos_camera.js tests/test_s1_pos_camera_frontend.py
```

Expected: no new literal Chinese characters in the changed code.

- [ ] **Step 6: Commit lifecycle integration**

```powershell
git add api/module4_frontend/static/index.html api/module4_frontend/static/pos_camera.js tests/test_s1_pos_camera_frontend.py
git commit -m "feat: finish POS camera lifecycle"
```

### Task 5: Browser And Real-Camera Acceptance

**Files:**
- Verify: `api/module4_frontend/static/index.html`
- Verify: `api/module4_frontend/static/pos_camera.js`
- Verify: `api/module1_yolo.py`
- Verify: `tests/test_s1_pos_camera_frontend.py`

- [ ] **Step 1: Run the full automated test suite**

```powershell
& 'C:\Users\Curtis\AppData\Local\BakeryAI\venv313\Scripts\python.exe' -m pytest -q
```

Expected: all tests PASS with no new warnings caused by the camera feature.

- [ ] **Step 2: Confirm both services are healthy**

```powershell
Invoke-WebRequest http://127.0.0.1:8002/ping -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:8001/health -UseBasicParsing
```

Expected: both requests return HTTP 200.

- [ ] **Step 3: Verify the mocked browser flow**

Use Playwright against `http://127.0.0.1:8002/` and inject a mocked `navigator.mediaDevices` before page load. The mock must return a stream with one stoppable track, a playable video source, and two `videoinput` devices. Verify:

1. `Open Camera` requests permission only after a click.
2. The preview modal opens.
3. The switch control appears for two cameras.
4. Capture changes the controls to `Retake` and `Recognize`.
5. Retake restores the video preview.
6. Recognize invokes one `/s1/checkout` request.
7. Repeated Recognize clicks do not create duplicate requests.
8. Close, Escape, navigation, and sign-out call `track.stop()`.
9. Permission denial leaves `Upload Image` usable.

Expected: every assertion passes and no browser console error remains.

- [ ] **Step 4: Verify with a real Windows camera**

Open the POS through Microsoft Edge App Mode and perform this acceptance sequence:

1. Allow camera permission.
2. Confirm the live preview is correctly framed.
3. Switch devices if more than one camera is connected.
4. Photograph a tray containing supported bakery products.
5. Retake once, then submit the second frame.
6. Confirm that YOLO detection cards appear.
7. Edit one detection and verify the HITL correction entry.
8. Close and reopen the camera; confirm the camera indicator turns off between sessions.
9. Leave POS and sign out; confirm the camera indicator turns off.
10. Upload an image and confirm the fallback still reaches the same detection cards.

Expected: the complete workflow succeeds without a console window, retained image, duplicate request, blocked checkout, or camera left active.

- [ ] **Step 5: Inspect final changes for duplication and residue**

```powershell
rg -n "function handleScan|function submitCheckoutImage|function openPOSCamera|function triggerScan" api/module4_frontend/static/index.html
rg -n "getUserMedia|enumerateDevices|toBlob|track.stop" api/module4_frontend/static/pos_camera.js
git diff --check
git status --short
```

Expected:

- one shared checkout-image submission function;
- one upload trigger;
- one camera module;
- no duplicate scan request implementation;
- no whitespace errors;
- only intended files changed.

- [ ] **Step 6: Commit any acceptance-only corrections**

If real-camera verification required a correction, stage only the camera files and focused test:

```powershell
git add api/module4_frontend/static/index.html api/module4_frontend/static/pos_camera.js tests/test_s1_pos_camera_frontend.py
git commit -m "fix: stabilize POS camera acceptance flow"
```

If no correction was required, do not create an empty commit.
