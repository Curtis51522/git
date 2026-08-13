(function () {
  "use strict";

  var REFERENCE_WIDTH = 2560;
  var REFERENCE_HEIGHT = 1262;
  var MIN_DESKTOP_WIDTH = 1025;
  var dashboard = document.getElementById("dashboard");
  var contentArea = document.getElementById("content-area");
  var root = document.documentElement;
  var resizeFrame = 0;

  if (!dashboard) {
    return;
  }

  function clearDesktopFit() {
    root.removeAttribute("data-desktop-fit");
    dashboard.removeAttribute("data-desktop-fit");
    dashboard.style.removeProperty("--desktop-fit-scale");
    dashboard.style.removeProperty("--desktop-fit-width");
    dashboard.style.removeProperty("--desktop-fit-height");
  }

  function applyDesktopFit() {
    resizeFrame = 0;

    var posPanel = document.getElementById("panel-pos");
    var posWorkspaceIsActive =
      posPanel &&
      window.getComputedStyle(posPanel).display !== "none";

    if (window.innerWidth < MIN_DESKTOP_WIDTH || !posWorkspaceIsActive) {
      clearDesktopFit();
      return;
    }

    var scale = Math.min(
      1,
      window.innerWidth / REFERENCE_WIDTH,
      window.innerHeight / REFERENCE_HEIGHT
    );

    if (scale >= 0.995) {
      clearDesktopFit();
      return;
    }

    var logicalWidth = window.innerWidth / scale;
    var logicalHeight = window.innerHeight / scale;
    var scaleValue = scale.toFixed(6);

    root.setAttribute("data-desktop-fit", "true");
    dashboard.setAttribute("data-desktop-fit", "true");
    dashboard.style.setProperty("--desktop-fit-scale", scaleValue);
    dashboard.style.setProperty("--desktop-fit-width", logicalWidth.toFixed(2) + "px");
    dashboard.style.setProperty("--desktop-fit-height", logicalHeight.toFixed(2) + "px");
  }

  function scheduleDesktopFit() {
    if (resizeFrame) {
      return;
    }
    resizeFrame = window.requestAnimationFrame(applyDesktopFit);
  }

  window.addEventListener("resize", scheduleDesktopFit, { passive: true });
  window.addEventListener("orientationchange", scheduleDesktopFit, { passive: true });
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) {
      scheduleDesktopFit();
    }
  });

  if (contentArea && window.MutationObserver) {
    new MutationObserver(scheduleDesktopFit).observe(contentArea, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["class", "style"]
    });
  }

  window.BakeryDesktopFit = {
    refresh: applyDesktopFit
  };

  applyDesktopFit();
})();
