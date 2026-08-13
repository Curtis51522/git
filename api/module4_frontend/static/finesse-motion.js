(function () {
  "use strict";

  var palette = {
    "#8b6914": "#9a4e2e",
    "#d4a853": "#c46f46",
    "#6b4f10": "#7f3d28",
    "#5c3d2e": "#3f4754",
    "#3d322b": "#303743",
    "#6b5b4f": "#687386",
    "#d4c4a8": "#d7dce5",
    "#d4c5b2": "#d7dce5",
    "#e0d5c7": "#dfe3ea",
    "#f0e8d8": "#e9edf3",
    "#f5f0eb": "#f1f3f7",
    "#fdfaf5": "#f8f9fb",
    "#3498db": "#2d6cdf",
    "#2980b9": "#245ab9",
    "#27ae60": "#2a8a66",
    "#2ecc71": "#42a77d",
    "#e74c3c": "#c74646",
    "#c0392b": "#aa3636",
    "#f39c12": "#d4932f",
    "#e67e22": "#c46f46",
    "#d35400": "#a95635"
  };

  function recolor(value) {
    if (typeof value === "string") {
      return palette[value.toLowerCase()] || value;
    }
    if (Array.isArray(value)) {
      return value.map(recolor);
    }
    if (value && Object.prototype.toString.call(value) === "[object Object]") {
      var result = {};
      Object.keys(value).forEach(function (key) {
        result[key] = recolor(value[key]);
      });
      return result;
    }
    return value;
  }

  function normalizeAxes(axis) {
    if (!axis) return axis;
    var axes = Array.isArray(axis) ? axis : [axis];
    var normalized = axes.map(function (item) {
      return Object.assign({}, item, {
        axisLabel: Object.assign({ color: "#687386", fontSize: 11 }, item.axisLabel || {}),
        axisLine: Object.assign(
          { lineStyle: { color: "rgba(48,55,67,.14)" } },
          item.axisLine || {}
        ),
        splitLine: Object.assign(
          { lineStyle: { color: "rgba(48,55,67,.07)" } },
          item.splitLine || {}
        )
      });
    });
    return Array.isArray(axis) ? normalized : normalized[0];
  }

  function themeChartOption(rawOption) {
    var option = recolor(rawOption || {});
    option.textStyle = Object.assign(
      {
        color: "#303743",
        fontFamily: '-apple-system, BlinkMacSystemFont, "PingFang SC", "Segoe UI", sans-serif'
      },
      option.textStyle || {}
    );
    option.tooltip = Object.assign(
      {
        backgroundColor: "rgba(24,27,34,.96)",
        borderColor: "transparent",
        borderWidth: 0,
        padding: [9, 11],
        textStyle: { color: "#f7f9fc", fontSize: 12 },
        extraCssText: "border-radius:12px;box-shadow:0 18px 42px -20px rgba(11,15,24,.74);"
      },
      option.tooltip || {}
    );
    if (option.legend) {
      option.legend = Object.assign(
        { textStyle: { color: "#687386", fontSize: 11 } },
        option.legend
      );
    }
    option.xAxis = normalizeAxes(option.xAxis);
    option.yAxis = normalizeAxes(option.yAxis);
    return option;
  }

  function installChartTheme() {
    if (!window.echarts || window.echarts.__bakeryThemeInstalled) return;
    var originalInit = window.echarts.init;
    window.echarts.init = function () {
      var chart = originalInit.apply(window.echarts, arguments);
      if (!chart.__bakerySetOption) {
        var originalSetOption = chart.setOption;
        chart.setOption = function (option) {
          var args = Array.prototype.slice.call(arguments);
          args[0] = themeChartOption(option);
          return originalSetOption.apply(chart, args);
        };
        chart.__bakerySetOption = true;
      }
      return chart;
    };
    window.echarts.__bakeryThemeInstalled = true;
  }

  function visiblePanel(panelName) {
    if (panelName) {
      var named = document.getElementById(panelName + "-panel");
      if (named && named.offsetParent !== null) return named;
    }
    return Array.prototype.find.call(
      document.querySelectorAll(".panel-container"),
      function (panel) { return panel.offsetParent !== null; }
    ) || document.getElementById("content-area");
  }

  function panelItems(panel) {
    if (!panel) return [];
    var selector = ".kpi-card, .panel, .chart-panel, .table-panel, .scan-action, .detect-card, .bundle-card";
    return Array.prototype.slice.call(panel.querySelectorAll(selector), 0, 18)
      .filter(function (item) { return item.offsetParent !== null; });
  }

  function installMotion() {
    if (!window.gsap) return;

    var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    window.gsap.defaults({ duration: 0.38, ease: "power2.out" });

    var copyMap = {
      "Scan a tray to begin": { en: "Scan a tray to begin", zh: "扫描托盘以开始" },
      "Business Events": { en: "Business Events", zh: "经营事件" },
      "+ New Product Launch": { en: "+ New Product Launch", zh: "+ 新品上市" },
      "+ Competitor Activity": { en: "+ Competitor Activity", zh: "+ 竞品动态" },
      "New Product Launch": { en: "New Product Launch", zh: "新品上市" },
      "Competitor Activity": { en: "Competitor Activity", zh: "竞品动态" },
      "No active business events for the selected date.": { en: "No active business events for the selected date.", zh: "选定日期暂无生效的经营事件。" },
      "Loading business events...": { en: "Loading business events...", zh: "正在加载经营事件..." },
      "Stock Risk AI Analysis": { en: "Stock Risk AI Analysis", zh: "库存风险 AI 分析" },
      "Promotion & Product Mix AI": { en: "Promotion & Product Mix AI", zh: "促销与商品组合 AI" }
    };

    function localizeOperationalCopy(root) {
      root = root || document.getElementById("content-area");
      if (!root) return;
      var language = window.LANG === "zh" ? "zh" : "en";
      var leaves = root.querySelectorAll("button, p, span, h4, div");
      Array.prototype.forEach.call(leaves, function (element) {
        if (element.children.length) return;
        var raw = (element.textContent || "").trim();
        var key = element.dataset.finesseCopy || raw;
        var copy = copyMap[key];
        if (!copy) return;
        element.dataset.finesseCopy = key;
        if (raw !== copy[language]) element.textContent = copy[language];
      });
    }

    function revealPanel(panelName) {
      localizeOperationalCopy();
      if (reduceMotion.matches) return;
      var panel = visiblePanel(panelName);
      var items = panelItems(panel);
      if (!items.length) return;
      window.gsap.killTweensOf(items);
      window.gsap.fromTo(
        items,
        { autoAlpha: 0.35, y: 9 },
        {
          autoAlpha: 1,
          y: 0,
          duration: 0.34,
          stagger: 0.025,
          clearProps: "opacity,visibility,transform"
        }
      );
    }

    function revealDashboard() {
      if (reduceMotion.matches) return;
      var chrome = document.querySelectorAll(".topnav, .topbar");
      window.gsap.fromTo(
        chrome,
        { autoAlpha: 0, y: -7 },
        { autoAlpha: 1, y: 0, duration: 0.32, stagger: 0.05, clearProps: "opacity,visibility,transform" }
      );
      window.setTimeout(function () { revealPanel("pos"); }, 80);
    }

    var loginCard = document.querySelector("#login-page .card");
    if (loginCard && !reduceMotion.matches) {
      window.gsap.fromTo(
        loginCard,
        { autoAlpha: 0, y: 18, scale: 0.985 },
        { autoAlpha: 1, y: 0, scale: 1, duration: 0.58, ease: "power3.out", clearProps: "opacity,visibility,transform" }
      );
    }

    if (typeof window.showPanel === "function" && !window.showPanel.__bakeryMotionWrapped) {
      var originalShowPanel = window.showPanel;
      var wrappedShowPanel = function () {
        var result = originalShowPanel.apply(this, arguments);
        var panelName = arguments[0];
        window.requestAnimationFrame(function () { revealPanel(panelName); });
        return result;
      };
      wrappedShowPanel.__bakeryMotionWrapped = true;
      window.showPanel = wrappedShowPanel;
    }

    var dashboard = document.getElementById("dashboard");
    if (dashboard) {
      var wasHidden = dashboard.classList.contains("hidden");
      new MutationObserver(function () {
        var isHidden = dashboard.classList.contains("hidden");
        if (wasHidden && !isHidden) revealDashboard();
        wasHidden = isHidden;
      }).observe(dashboard, { attributes: true, attributeFilter: ["class"] });
    }

    var contentArea = document.getElementById("content-area");
    if (contentArea) {
      var copyFrame = 0;
      new MutationObserver(function () {
        if (copyFrame) return;
        copyFrame = window.requestAnimationFrame(function () {
          copyFrame = 0;
          localizeOperationalCopy(contentArea);
        });
      }).observe(contentArea, { childList: true, subtree: true, characterData: true });
    }

    document.addEventListener("click", function (event) {
      if (reduceMotion.matches) return;
      var control = event.target.closest("button, .scan-action, .drink-btn, .bundle-card");
      if (!control || control.disabled) return;
      window.gsap.killTweensOf(control);
      window.gsap.fromTo(
        control,
        { scale: 0.985 },
        { scale: 1, duration: 0.22, ease: "back.out(2)", clearProps: "transform" }
      );
    }, true);

    window.BakeryUIMotion = {
      revealPanel: revealPanel,
      localize: localizeOperationalCopy,
      reducedMotion: function () { return reduceMotion.matches; }
    };
  }

  installChartTheme();
  installMotion();
}());
