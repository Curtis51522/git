from pathlib import Path


INDEX_HTML = Path("api/module4_frontend/static/index.html")


def _generate_bundle_source():
    html = INDEX_HTML.read_text(encoding="utf-8")
    start = html.index("async function generateBundle()")
    end = html.index("function addBundleToCart", start)
    return html, html[start:end]


def test_top3_requests_have_bounded_wait_time():
    html, generate_bundle = _generate_bundle_source()

    assert "function fetchWithTimeout" in html
    assert "fetchWithTimeout(S5_API+'/priorities'" in generate_bundle
    assert "fetchWithTimeout(S5_API+'/discounts'" in generate_bundle
    assert "fetchWithTimeout(API+'/s4/combo'" in generate_bundle


def test_top3_failure_clears_loading_placeholder():
    _, generate_bundle = _generate_bundle_source()
    catch_start = generate_bundle.index("catch(e)")
    catch_body = generate_bundle[catch_start:]

    assert "bundleRecs=[]" in catch_body
    assert "renderPOS(" in catch_body


def test_top3_timeout_has_readable_error_message():
    html, _ = _generate_bundle_source()

    assert "throw new Error(t('Request timed out'))" in html
    assert "'Request timed out':'Request timed out'" in html


def test_pairing_prompt_uses_dynamic_beverage_catalog(monkeypatch):
    from api.module4_frontend import pairing_llm

    monkeypatch.setattr(
        pairing_llm,
        "_get_bakery",
        lambda: [{"key": "croissant", "name": "Croissant", "desc": "Croissant"}],
    )
    monkeypatch.setattr(
        pairing_llm,
        "_get_coffee",
        lambda: [{"key": "latte", "name": "Latte", "desc": "Latte"}],
    )
    prompt = pairing_llm._build_pairing_prompt()

    assert '"latte": 0.5' in prompt


def test_deepseek_pairing_request_has_short_timeout(monkeypatch):
    from api.module4_frontend import pairing_llm

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}

    def fake_post(*args, **kwargs):
        captured["timeout"] = kwargs["timeout"]
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(pairing_llm.httpx, "post", fake_post)

    assert pairing_llm._call_deepseek("prompt") == "{}"
    assert captured["timeout"] <= 5
