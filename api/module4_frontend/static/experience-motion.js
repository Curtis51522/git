(function () {
  "use strict";

  function installLoginDepth() {
    var root = document.getElementById("login-page");
    var stage = root && root.querySelector(".login-product-stage");
    if (!root || !stage || !window.gsap) return;

    var products = Array.prototype.slice.call(stage.querySelectorAll(".login-product"));
    var chips = Array.prototype.slice.call(stage.querySelectorAll(".login-stage-chip"));
    var context = window.gsap.context(function () {
      var media = window.gsap.matchMedia();

      media.add(
        {
          motion: "(prefers-reduced-motion: no-preference)",
          precision: "(hover: hover) and (pointer: fine)"
        },
        function (mediaContext) {
          var conditions = mediaContext.conditions || {};
          if (!conditions.motion || !conditions.precision) return;

          var stageX = window.gsap.quickTo(stage, "x", {
            duration: 0.75,
            ease: "power3.out"
          });
          var stageY = window.gsap.quickTo(stage, "y", {
            duration: 0.75,
            ease: "power3.out"
          });
          var stageRotationX = window.gsap.quickTo(stage, "rotationX", {
            duration: 0.9,
            ease: "power3.out"
          });
          var stageRotationY = window.gsap.quickTo(stage, "rotationY", {
            duration: 0.9,
            ease: "power3.out"
          });

          var productSetters = products.map(function (product) {
            return {
              element: product,
              depth: Number(product.getAttribute("data-depth") || 0.6),
              x: window.gsap.quickTo(product, "x", {
                duration: 0.82,
                ease: "power3.out"
              }),
              y: window.gsap.quickTo(product, "y", {
                duration: 0.82,
                ease: "power3.out"
              })
            };
          });

          function move(event) {
            if (root.classList.contains("hidden")) return;
            var bounds = root.getBoundingClientRect();
            var normalizedX = ((event.clientX - bounds.left) / bounds.width - 0.5) * 2;
            var normalizedY = ((event.clientY - bounds.top) / bounds.height - 0.5) * 2;

            stageX(normalizedX * 10);
            stageY(normalizedY * 6);
            stageRotationX(normalizedY * -4.8);
            stageRotationY(normalizedX * 7.2);

            productSetters.forEach(function (setter) {
              setter.x(normalizedX * 13 * setter.depth);
              setter.y(normalizedY * 9 * setter.depth);
            });

            root.style.setProperty("--login-x", ((normalizedX + 1) * 50).toFixed(1) + "%");
            root.style.setProperty("--login-y", ((normalizedY + 1) * 50).toFixed(1) + "%");
          }

          function reset() {
            stageX(0);
            stageY(0);
            stageRotationX(0);
            stageRotationY(0);
            productSetters.forEach(function (setter) {
              setter.x(0);
              setter.y(0);
            });
            root.style.setProperty("--login-x", "50%");
            root.style.setProperty("--login-y", "48%");
          }

          root.addEventListener("pointermove", move, { passive: true });
          root.addEventListener("pointerleave", reset, { passive: true });

          var entrance = window.gsap.timeline({
            defaults: { ease: "power3.out" }
          });
          entrance
            .fromTo(
              ".login-story > *",
              { autoAlpha: 0, y: 18 },
              { autoAlpha: 1, y: 0, duration: 0.65, stagger: 0.07 },
              0
            )
            .fromTo(
              products,
              { autoAlpha: 0, y: 34, scale: 0.88 },
              { autoAlpha: 1, y: 0, scale: 1, duration: 0.92, stagger: 0.075 },
              0.12
            )
            .fromTo(
              chips,
              { autoAlpha: 0, scale: 0.9 },
              { autoAlpha: 1, scale: 1, duration: 0.48, stagger: 0.08 },
              0.52
            )
            .fromTo(
              "#login-page .login-console",
              { autoAlpha: 0, x: 24, scale: 0.985 },
              { autoAlpha: 1, x: 0, scale: 1, duration: 0.72 },
              0.2
            );

          return function () {
            root.removeEventListener("pointermove", move);
            root.removeEventListener("pointerleave", reset);
          };
        }
      );

      window.BakeryExperienceMotion = {
        destroy: function () {
          media.revert();
          context.revert();
        }
      };
    }, root);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", installLoginDepth, { once: true });
  } else {
    installLoginDepth();
  }
})();
