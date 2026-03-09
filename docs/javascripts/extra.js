/* VBN Analysis Suite - Interactive Documentation */
(function () {
  "use strict";

  /* ── Scroll-reveal animations ───────────────────────────── */
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
      { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
    );

    els.forEach(function (el) {
      observer.observe(el);
    });
  }

  /* ── Interactive pipeline clicks ────────────────────────── */
  function initPipeline() {
    document.querySelectorAll(".vbn-pipeline-node[data-href]").forEach(function (node) {
      node.style.cursor = "pointer";
      node.setAttribute("role", "link");
      node.setAttribute("tabindex", "0");

      node.addEventListener("click", function () {
        var href = node.getAttribute("data-href");
        if (href) window.location.href = href;
      });

      node.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          node.click();
        }
      });
    });
  }

  /* ── Expandable sections ────────────────────────────────── */
  function initExpandables() {
    document.querySelectorAll(".vbn-expand-trigger").forEach(function (trigger) {
      var target = trigger.nextElementSibling;
      if (!target || !target.classList.contains("vbn-expand-content")) return;

      trigger.addEventListener("click", function () {
        var isOpen = target.classList.contains("vbn-expanded");
        target.classList.toggle("vbn-expanded");
        trigger.classList.toggle("vbn-open");
        target.style.maxHeight = isOpen ? "0" : target.scrollHeight + "px";
      });
    });
  }

  /* ── Counter animation for hero stats ───────────────────── */
  function initCounters() {
    var stats = document.querySelectorAll(".vbn-stat-number");
    if (!stats.length) return;

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          var el = entry.target;
          var target = parseInt(el.textContent, 10);
          if (isNaN(target)) return;
          observer.unobserve(el);

          var current = 0;
          var step = Math.max(1, Math.floor(target / 20));
          var interval = setInterval(function () {
            current += step;
            if (current >= target) {
              current = target;
              clearInterval(interval);
            }
            el.textContent = current;
          }, 40);
        });
      },
      { threshold: 0.5 }
    );

    stats.forEach(function (el) {
      observer.observe(el);
    });
  }

  /* ── Initialize everything ──────────────────────────────── */
  function init() {
    initAnimations();
    initPipeline();
    initExpandables();
    initCounters();
  }

  /* MkDocs Material instant navigation compatibility */
  if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
      init();
    });
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
