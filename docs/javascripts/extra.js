(function () {
  "use strict";

  function initAnimations() {
    var els = document.querySelectorAll(".vbn-animate");
    if (!els.length) return;

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("vbn-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.08, rootMargin: "0px 0px -20px 0px" }
    );

    els.forEach(function (el) { observer.observe(el); });
  }

  function initPipeline() {
    document.querySelectorAll(".vbn-pipeline-node[data-href]").forEach(function (node) {
      node.style.cursor = "pointer";
      node.addEventListener("click", function () {
        var href = node.getAttribute("data-href");
        if (href) window.location.href = href;
      });
    });
  }

  function init() {
    initAnimations();
    initPipeline();
  }

  if (typeof document$ !== "undefined") {
    document$.subscribe(function () { init(); });
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
