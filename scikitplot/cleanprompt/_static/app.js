/* CleanPrompt web interface behaviour.
 *
 * Self-contained, no dependencies, no remote requests. Three jobs:
 * autosizing the textareas, copy buttons, and turning a suggestion chip into
 * an entry in the "also hide these" box.
 *
 * Two rules this file keeps:
 *
 * 1. It only ever ADDS terms to the hide box. A script on a privacy page must
 *    not be able to alter the text the user is about to send, so nothing here
 *    writes to the main textarea or to the redacted output.
 * 2. Listeners are attached here rather than inlined as onclick attributes, so
 *    the page needs no 'unsafe-inline' in a Content-Security-Policy.
 */
(function () {
  "use strict";

  var MAX_PX = 900;

  function fit(area) {
    area.style.height = "auto";
    area.style.height = Math.min(area.scrollHeight + 2, MAX_PX) + "px";
  }

  function autosize() {
    var areas = document.querySelectorAll("textarea");
    for (var i = 0; i < areas.length; i++) {
      (function (area) {
        fit(area);
        area.addEventListener("input", function () { fit(area); });
      })(areas[i]);
    }
  }

  function flash(button, message) {
    var original = button.textContent;
    button.textContent = message;
    button.classList.add("done");
    window.setTimeout(function () {
      button.textContent = original;
      button.classList.remove("done");
    }, 1200);
  }

  function copiers() {
    var buttons = document.querySelectorAll("[data-copy]");
    for (var i = 0; i < buttons.length; i++) {
      (function (button) {
        button.addEventListener("click", function () {
          var target = document.getElementById(button.getAttribute("data-copy"));
          if (!target) { return; }
          target.select();
          target.setSelectionRange(0, target.value.length);
          // The async clipboard API needs a secure context, which plain http
          // on loopback is not in every browser. Fall back to the legacy call
          // rather than failing silently on the one transport this tool uses.
          if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(target.value).then(
              function () { flash(button, "Copied"); },
              function () { flash(button, "Press Ctrl+C"); }
            );
          } else {
            var ok = false;
            try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
            flash(button, ok ? "Copied" : "Press Ctrl+C");
          }
        });
      })(buttons[i]);
    }
  }

  function chips() {
    var box = document.getElementById("additional_words");
    var all = document.querySelectorAll(".chip");
    for (var i = 0; i < all.length; i++) {
      (function (chip) {
        chip.addEventListener("click", function () {
          if (!box) { return; }
          var term = chip.getAttribute("data-term");
          var current = box.value.split(",")
            .map(function (s) { return s.trim(); })
            .filter(function (s) { return s.length > 0; });
          var at = current.indexOf(term);
          if (at === -1) {
            current.push(term);
            chip.classList.add("picked");
          } else {
            current.splice(at, 1);
            chip.classList.remove("picked");
          }
          box.value = current.join(", ");
          fit(box);
        });
      })(all[i]);
    }
  }

  function shortcuts() {
    document.addEventListener("keydown", function (event) {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        var form = document.getElementById("redact-form");
        if (form) { form.requestSubmit ? form.requestSubmit() : form.submit(); }
      }
    });
  }

  function init() {
    autosize();
    copiers();
    chips();
    shortcuts();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
