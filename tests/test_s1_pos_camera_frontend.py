from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "api" / "module4_frontend" / "static" / "index.html"
CAMERA = ROOT / "api" / "module4_frontend" / "static" / "pos_camera.js"


def _run_camera_scenario(scenario: str):
    source = CAMERA.as_posix()
    script = textwrap.dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        const assert = require("assert");

        function element(id) {{
          return {{
            id,
            hidden: false,
            src: "",
            srcObject: null,
            width: 0,
            height: 0,
            textContent: "",
            classList: {{
              values: new Set(),
              add(value) {{ this.values.add(value); }},
              remove(value) {{ this.values.delete(value); }},
              contains(value) {{ return this.values.has(value); }},
            }},
            focus() {{ document.activeElement = this; }},
            play() {{ return Promise.resolve(); }},
            removeAttribute(name) {{ if (name === "src") this.src = ""; }},
            getContext() {{
              return {{
                drawImage() {{}},
                clearRect() {{}},
              }};
            }},
            toBlob(callback) {{ callback(new Blob(["frame"], {{type: "image/jpeg"}})); }},
          }};
        }}

        const ids = ["camera-error", "camera-video", "camera-photo", "camera-capture",
          "camera-retake", "camera-recognize", "camera-switch", "camera-canvas",
          "camera-modal", "camera-close"];
        const elements = Object.fromEntries(ids.map(id => [id, element(id)]));
        const opener = element("camera-opener");
        global.document = {{
          activeElement: opener,
          getElementById(id) {{ return elements[id] || null; }},
          addEventListener() {{}},
        }};
        global.window = global;
        Object.defineProperty(global, "navigator", {{
          value: {{mediaDevices: {{}}}},
          configurable: true,
        }});
        global.URL = {{
          createObjectURL() {{ return "blob:test"; }},
          revokeObjectURL() {{}},
        }};
        global.alert = function() {{}};
        global.addEventListener = function() {{}};
        global.File = class File extends Blob {{
          constructor(parts, name, options) {{ super(parts, options); this.name = name; }}
        }};
        vm.runInThisContext(fs.readFileSync("{source}", "utf8"));

        (async () => {{
        {textwrap.indent(textwrap.dedent(scenario), '  ')}
        }})().catch(error => {{ console.error(error); process.exit(1); }});
        """
    )
    result = subprocess.run(
        ["node", "-"],
        input=script,
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_pos_loads_focused_camera_module_and_exposes_two_image_sources():
    html = INDEX.read_text(encoding="utf-8")

    assert '<script src="/pos_camera.js"></script>' in html
    assert "openPOSCamera()" in html
    assert "triggerScan()" in html
    assert "Open Camera" in html
    assert "Upload Image" in html


def test_camera_module_uses_one_frame_capture_and_device_switching():
    source = CAMERA.read_text(encoding="utf-8")

    assert "navigator.mediaDevices.getUserMedia" in source
    assert "navigator.mediaDevices.enumerateDevices" in source
    assert 'facingMode:{ideal:"environment"}' in source
    assert "deviceId:{exact:deviceId}" in source
    assert "canvas.toBlob" in source
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


def test_camera_lifecycle_closes_on_logout_and_when_leaving_pos():
    html = INDEX.read_text(encoding="utf-8")

    assert "if(window.POSCamera)window.POSCamera.close();" in html
    assert "if(panel!=='pos'&&window.POSCamera)window.POSCamera.close();" in html


def test_camera_modal_has_capture_review_and_fallback_commands():
    html = INDEX.read_text(encoding="utf-8")

    assert 'id="camera-modal"' in html
    assert 'id="camera-video"' in html
    assert 'id="camera-photo"' in html
    assert "window.POSCamera.capture()" in html
    assert "window.POSCamera.retake()" in html
    assert "window.POSCamera.recognize()" in html
    assert "window.POSCamera.switchCamera()" in html
    assert "window.POSCamera.close()" in html
    assert ".camera-controls [hidden],.camera-preview [hidden]{display:none!important}" in html


def test_concurrent_camera_switches_stop_every_superseded_stream():
    _run_camera_scenario(
        """
        function makeStream(name) {
          const track = {name, stops: 0, stop() { this.stops += 1; }};
          return {track, getTracks() { return [track]; }};
        }
        const initial = makeStream("initial");
        const first = makeStream("first");
        const second = makeStream("second");
        const pending = [];
        navigator.mediaDevices.enumerateDevices = async () => [
          {kind: "videoinput", deviceId: "one"},
          {kind: "videoinput", deviceId: "two"},
        ];
        navigator.mediaDevices.getUserMedia = () => {
          if (!pending.length) {
            pending.push("initial-resolved");
            return Promise.resolve(initial);
          }
          return new Promise(resolve => pending.push(resolve));
        };
        await POSCamera.open({submitImage() {}, translate(value) { return value; }});
        const switchOne = POSCamera.switchCamera();
        const switchTwo = POSCamera.switchCamera();
        pending[2](second);
        await Promise.resolve();
        pending[1](first);
        await Promise.all([switchOne, switchTwo]);
        POSCamera.close();
        assert.strictEqual(initial.track.stops, 1);
        assert.strictEqual(first.track.stops, 1);
        assert.strictEqual(second.track.stops, 1);
        """
    )


def test_delayed_capture_cannot_restore_a_closed_frame_and_canvas_is_erased():
    _run_camera_scenario(
        """
        const track = {stops: 0, stop() { this.stops += 1; }};
        const stream = {getTracks() { return [track]; }};
        navigator.mediaDevices.enumerateDevices = async () => [];
        navigator.mediaDevices.getUserMedia = async () => stream;
        elements["camera-video"].videoWidth = 1280;
        elements["camera-video"].videoHeight = 720;
        let finishEncoding;
        elements["camera-canvas"].toBlob = callback => { finishEncoding = callback; };
        await POSCamera.open({submitImage() {}, translate(value) { return value; }});
        POSCamera.capture();
        POSCamera.close();
        finishEncoding(new Blob(["old-frame"], {type: "image/jpeg"}));
        await Promise.resolve();
        assert.strictEqual(elements["camera-photo"].hidden, true);
        assert.strictEqual(elements["camera-recognize"].hidden, true);
        assert.strictEqual(elements["camera-canvas"].width, 0);
        assert.strictEqual(elements["camera-canvas"].height, 0);
        """
    )


def test_permission_denial_is_not_reprompted_and_focus_is_restored():
    _run_camera_scenario(
        """
        let requests = 0;
        navigator.mediaDevices.enumerateDevices = async () => [];
        navigator.mediaDevices.getUserMedia = async () => {
          requests += 1;
          const error = new Error("denied");
          error.name = "NotAllowedError";
          throw error;
        };
        await POSCamera.open({submitImage() {}, translate(value) { return value; }});
        await POSCamera.open({submitImage() {}, translate(value) { return value; }});
        assert.strictEqual(requests, 1);
        assert.strictEqual(document.activeElement, elements["camera-close"]);
        POSCamera.close();
        assert.strictEqual(document.activeElement, opener);
        """
    )


def test_superseded_stream_error_does_not_stop_the_current_stream():
    _run_camera_scenario(
        """
        function makeStream(name) {
          const track = {name, stops: 0, stop() { this.stops += 1; }};
          return {track, getTracks() { return [track]; }};
        }
        const initial = makeStream("initial");
        const first = makeStream("first");
        const second = makeStream("second");
        const streams = [initial, first, second];
        let rejectFirstPlay;
        navigator.mediaDevices.enumerateDevices = async () => [
          {kind: "videoinput", deviceId: "one"},
          {kind: "videoinput", deviceId: "two"},
        ];
        navigator.mediaDevices.getUserMedia = async () => streams.shift();
        elements["camera-video"].play = function() {
          if (this.srcObject === first) {
            return new Promise((resolve, reject) => { rejectFirstPlay = reject; });
          }
          return Promise.resolve();
        };
        await POSCamera.open({submitImage() {}, translate(value) { return value; }});
        const switchOne = POSCamera.switchCamera();
        await Promise.resolve();
        const switchTwo = POSCamera.switchCamera();
        await switchTwo;
        rejectFirstPlay(new Error("old stream failed"));
        await switchOne;
        assert.strictEqual(second.track.stops, 0);
        POSCamera.close();
        assert.strictEqual(second.track.stops, 1);
        """
    )


def test_permission_denial_while_switching_is_not_reprompted():
    _run_camera_scenario(
        """
        const track = {stops: 0, stop() { this.stops += 1; }};
        const stream = {getTracks() { return [track]; }};
        let requests = 0;
        navigator.mediaDevices.enumerateDevices = async () => [
          {kind: "videoinput", deviceId: "one"},
          {kind: "videoinput", deviceId: "two"},
        ];
        navigator.mediaDevices.getUserMedia = async () => {
          requests += 1;
          if (requests === 1) return stream;
          const error = new Error("denied");
          error.name = "NotAllowedError";
          throw error;
        };
        await POSCamera.open({submitImage() {}, translate(value) { return value; }});
        await POSCamera.switchCamera();
        POSCamera.close();
        await POSCamera.open({submitImage() {}, translate(value) { return value; }});
        assert.strictEqual(requests, 2);
        POSCamera.close();
        """
    )
