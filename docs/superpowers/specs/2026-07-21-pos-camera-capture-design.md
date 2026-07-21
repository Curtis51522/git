# POS Camera Capture Design

## Purpose

Add a real camera capture workflow to the existing POS visual-recognition area before the final Windows installer is built. The feature must make S1 useful at checkout without introducing continuous video inference, a camera backend service, or persistent customer images.

## Scope

The POS will support two image sources:

- a live browser camera for one-frame tray capture;
- the existing local image upload fallback.

Both sources will submit one image to the existing authenticated `POST /s1/checkout` endpoint and will continue through the current detection review, correction, cart, inventory, and checkout flows.

The following are explicitly out of scope:

- continuous or streaming YOLO inference;
- direct OpenCV camera control from Python;
- customer-image storage;
- changes to YOLO training, model weights, inventory deduction, or checkout pricing;
- a new camera-specific backend endpoint.

## Architecture

Camera behavior will live in a focused frontend module:

`api/module4_frontend/static/pos_camera.js`

The module will own:

- camera capability detection;
- permission requests;
- media-stream lifecycle;
- camera enumeration and switching;
- still-frame capture;
- in-memory JPEG creation;
- modal state and cleanup.

The existing scan request inside `handleScan()` will be extracted into one shared image-submission function. The file input and camera module will call that function with a `File` or `Blob`. Detection rendering and HITL correction will remain owned by the current POS code.

No runtime path may depend on a developer workstation. The module will be loaded from the installed static application directory and will call the same-origin `/s1/checkout` route. It must work when the UI is opened in Microsoft Edge App Mode and in the default-browser fallback.

## User Interface

The visual-recognition area will expose two equally clear commands:

- `Open Camera`, with a camera icon;
- `Upload Image`, with an upload icon.

Opening the camera displays a modal with a stable 16:9 preview area. The initial primary action is `Capture`. After capture, the preview freezes and the available commands become `Retake` and `Recognize`. A close control remains available throughout the flow.

The interface will use familiar icons with tooltips where the existing frontend icon facilities allow them. Controls will have fixed dimensions, responsive wrapping, visible keyboard focus, and labels that fit on desktop and narrow viewports.

If multiple cameras are available after permission is granted, the modal will show a compact `Switch Camera` control. It will not show a permanent device-management panel. The first request will prefer an environment-facing camera while allowing the browser to select the best available device.

## Capture And Recognition Flow

1. The cashier opens the camera.
2. The browser requests a video stream with an ideal resolution of 1280 by 720 pixels.
3. The module displays the stream and enumerates available video-input devices after permission is granted.
4. The cashier captures one frame.
5. The frame is drawn to an in-memory canvas and encoded as JPEG.
6. The cashier may retake the frame or submit it for recognition.
7. Submission stops the active stream, closes the modal, and invokes the shared POS image-submission function.
8. The existing scanning state prevents duplicate submissions while `/s1/checkout` is running.
9. The current detection cards allow the cashier to confirm, edit, or delete results before checkout.

The captured image will not be written to local storage, IndexedDB, the application directory, or MySQL. Existing detection metadata remains available through `detection_log`, including the image identifier, predicted class, bounding box, confidence, inference time, review requirement, and correction evidence.

## Camera Lifecycle

Every active `MediaStreamTrack` must be stopped when any of the following occurs:

- the user closes the modal;
- the user presses Escape;
- the user submits a captured frame;
- camera initialization fails;
- the user leaves the POS view;
- the user signs out;
- the page is unloaded.

Opening the camera a second time must create a fresh stream. Switching devices must stop the previous stream before requesting the selected device. Cleanup must be idempotent so repeated calls cannot fail or leave a camera indicator active.

## Error Handling

Camera failures affect only the camera flow. Manual entry, image upload, cart operations, and checkout must remain available.

The UI will distinguish these cases with concise messages:

- camera APIs are unsupported;
- permission was denied;
- no camera was found;
- the selected camera is already in use or unavailable;
- frame capture or JPEG conversion failed;
- the recognition request failed or timed out;
- the image produced no detections.

The UI must not repeatedly request permission after a denial. Every failure path must release any acquired stream and restore actionable POS controls. Existing authentication, file-type validation, the 10 MB request limit, and backend error responses remain authoritative.

## Security And Privacy

The browser will request camera permission only after an explicit cashier action. No camera opens during login, page load, or POS navigation. Captured frames remain in memory for the current recognition request and are released afterward.

The implementation will not add analytics, remote image upload, background recording, or image retention. The local `127.0.0.1` application origin and Microsoft Edge App Mode must use the same authenticated API boundary as the current upload workflow.

## Compatibility

The supported deployment target is Windows 10 or Windows 11, 64-bit. The primary shell will be Microsoft Edge App Mode, with the default browser as fallback. A standard USB camera and a laptop-integrated camera must both be supported through browser media APIs.

If a browser does not expose `navigator.mediaDevices.getUserMedia`, the camera command will be unavailable or will show an unsupported message while the upload command remains usable.

## Testing

Automated coverage will include:

- a frontend contract test confirming that `pos_camera.js` is loaded;
- a contract test confirming separate camera and upload commands;
- a test confirming that both image sources use the shared checkout-scan submission function;
- lifecycle tests or source contracts for closing, switching, submitting, signing out, and leaving the POS view;
- a contract confirming that the camera module does not persist captured images;
- existing `/s1/checkout` tests for authenticated JPEG submission and error handling;
- the complete repository test suite for regression coverage.

Browser verification will use a mocked media-device implementation to cover permission success, denial, multiple cameras, capture, retake, submission, duplicate-submit prevention, and cleanup. Final acceptance will also use a real Windows camera to verify:

1. camera permission;
2. live preview;
3. camera switching when more than one device exists;
4. capture and retake;
5. YOLO recognition;
6. manual correction;
7. camera release after close, recognition, navigation, and sign-out;
8. image-upload fallback;
9. operation inside Microsoft Edge App Mode.

## Acceptance Criteria

The feature is complete when:

- one desktop command opens a real camera preview from the POS;
- a still frame reaches the existing YOLO checkout endpoint;
- upload and camera capture share one recognition path;
- no image is persisted;
- every exit and error path releases the camera;
- camera failure never blocks manual POS operation;
- the existing detection-review and checkout behavior remains unchanged;
- all automated tests pass;
- the real-camera acceptance flow passes on the target Windows computer.
