(() => {
  "use strict";

  const ROOT_SELECTOR =
    "[data-sp-principles]";

  const ACTION_SELECTOR =
    "[data-sp-action]";

  const SPEED_SELECTOR =
    "[data-sp-speed]";

  const MENU_SELECTOR =
    "[data-sp-menu]";

  const DEFAULT_SPEED =
    "normal";

  const DEFAULT_DURATIONS = {
    slow: 42,
    normal: 28,
    fast: 16,
  };

  const VALID_SPEEDS =
    new Set(
      Object.keys(
        DEFAULT_DURATIONS
      )
    );

  const tickerState =
    new WeakMap();


  function shuffle(items) {
    const shuffled = [...items];

    for (
      let i = shuffled.length - 1;
      i > 0;
      i -= 1
    ) {
      const j =
        Math.floor(
          Math.random() * (i + 1)
        );

      [
        shuffled[i],
        shuffled[j],
      ] = [
        shuffled[j],
        shuffled[i],
      ];
    }

    return shuffled;
  }


  function removeClone(track) {
    track
      .querySelectorAll(
        ':scope > .sp-principles__group[aria-hidden="true"]'
      )
      .forEach((node) => {
        node.remove();
      });
  }


  function createClone(
    track,
    group
  ) {
    removeClone(track);

    const clone =
      group.cloneNode(true);

    clone.removeAttribute("id");

    clone.setAttribute(
      "aria-hidden",
      "true"
    );

    track.appendChild(clone);
  }


  function restartAnimation(track) {
    const animation =
      track.style.animation;

    track.style.animation =
      "none";

    void track.offsetWidth;

    track.style.animation =
      animation;
  }


  function cloneOriginalItems(group) {
    return Array
      .from(
        group.querySelectorAll(
          ":scope > .sp-principles__item"
        )
      )
      .map(
        (item) =>
          item.cloneNode(true)
      );
  }


  function restoreOriginalOrder(
    state
  ) {
    state.group.replaceChildren(
      ...state.originalItems.map(
        (item) =>
          item.cloneNode(true)
      )
    );

    createClone(
      state.track,
      state.group
    );

    restartAnimation(
      state.track
    );
  }


  function numberFromDataset(
    value,
    fallback
  ) {
    const parsed =
      Number.parseFloat(value);

    return Number.isFinite(parsed)
      ? parsed
      : fallback;
  }


  function getDurations(root) {
    return {
      slow:
        numberFromDataset(
          root.dataset.spSpeedSlow,
          DEFAULT_DURATIONS.slow
        ),

      normal:
        numberFromDataset(
          root.dataset.spSpeedNormal,
          DEFAULT_DURATIONS.normal
        ),

      fast:
        numberFromDataset(
          root.dataset.spSpeedFast,
          DEFAULT_DURATIONS.fast
        ),
    };
  }


  function getDefaultSpeed(root) {
    const requested =
      root.dataset.spSpeedDefault;

    return VALID_SPEEDS.has(requested)
      ? requested
      : DEFAULT_SPEED;
  }


  function readStoredPreferences(
    root
  ) {
    if (
      root.dataset.spPersist !==
      "true"
    ) {
      return null;
    }

    const key =
      root.dataset.spStorageKey;

    if (!key) {
      return null;
    }

    try {
      const raw =
        window.localStorage.getItem(
          key
        );

      if (!raw) {
        return null;
      }

      const value =
        JSON.parse(raw);

      if (
        !value ||
        typeof value !== "object"
      ) {
        return null;
      }

      return value;
    } catch {
      return null;
    }
  }


  function writePreferences(
    root,
    state
  ) {
    if (
      root.dataset.spPersist !==
      "true"
    ) {
      return;
    }

    const key =
      root.dataset.spStorageKey;

    if (!key) {
      return;
    }

    const payload = {
      paused: state.paused,
      reverse: state.reverse,
      speed: state.speed,
    };

    try {
      window.localStorage.setItem(
        key,
        JSON.stringify(payload)
      );
    } catch {
      /*
       * Storage is an optional enhancement.
       * The ticker remains functional when blocked.
       */
    }
  }


  function clearPreferences(root) {
    const key =
      root.dataset.spStorageKey;

    if (!key) {
      return;
    }

    try {
      window.localStorage.removeItem(
        key
      );
    } catch {
      /* Ignore restricted storage contexts. */
    }
  }


  function normalizePreferences(
    root,
    stored
  ) {
    const defaultSpeed =
      getDefaultSpeed(root);

    return {
      paused:
        stored?.paused === true,

      reverse:
        stored?.reverse === true,

      speed:
        VALID_SPEEDS.has(
          stored?.speed
        )
          ? stored.speed
          : defaultSpeed,
    };
  }


  function updateControls(
    root,
    state
  ) {
    const toggle =
      root.querySelector(
        '[data-sp-action="toggle"]'
      );

    const reverse =
      root.querySelector(
        '[data-sp-action="reverse"]'
      );

    const toggleTitle =
      root.querySelector(
        "[data-sp-toggle-title]"
      );

    const toggleDescription =
      root.querySelector(
        "[data-sp-toggle-description]"
      );

    const directionDescription =
      root.querySelector(
        "[data-sp-direction-description]"
      );

    if (
      toggle instanceof
      HTMLButtonElement
    ) {
      toggle.setAttribute(
        "aria-pressed",
        String(state.paused)
      );
    }

    if (
      reverse instanceof
      HTMLButtonElement
    ) {
      reverse.setAttribute(
        "aria-pressed",
        String(state.reverse)
      );
    }

    if (
      toggleTitle instanceof
      HTMLElement
    ) {
      toggleTitle.textContent =
        state.paused
          ? "Play"
          : "Pause";
    }

    if (
      toggleDescription instanceof
      HTMLElement
    ) {
      toggleDescription.textContent =
        state.paused
          ? "Resume ticker motion"
          : "Stop ticker motion";
    }

    if (
      directionDescription instanceof
      HTMLElement
    ) {
      directionDescription.textContent =
        state.reverse
          ? "Reverse"
          : "Forward";
    }

    root
      .querySelectorAll(
        SPEED_SELECTOR
      )
      .forEach((button) => {
        button.setAttribute(
          "aria-pressed",
          String(
            button.dataset.spSpeed ===
            state.speed
          )
        );
      });

    const summary =
      root.querySelector(
        `${MENU_SELECTOR} > summary`
      );

    if (
      summary instanceof
      HTMLElement
    ) {
      const motion =
        state.paused
          ? "Paused"
          : "Playing";

      const direction =
        state.reverse
          ? "Reverse"
          : "Forward";

      const speed =
        state.speed
          .charAt(0)
          .toUpperCase() +
        state.speed.slice(1);

      const label =
        `Ticker settings. ${motion}, ${direction}, ${speed} speed`;

      summary.setAttribute(
        "aria-label",
        label
      );

      summary.setAttribute(
        "title",
        label
      );
    }
  }


  function applyState(
    root,
    state,
    {
      persist = true,
      restart = false,
    } = {}
  ) {
    state.track.dataset.paused =
      String(state.paused);

    state.track.dataset.reverse =
      String(state.reverse);

    root.dataset.spSpeed =
      state.speed;

    const duration =
      state.durations[state.speed];

    root.style.setProperty(
      "--sp-principles-speed",
      `${duration}s`
    );

    updateControls(
      root,
      state
    );

    if (restart) {
      restartAnimation(
        state.track
      );
    }

    if (persist) {
      writePreferences(
        root,
        state
      );
    }

    root.dispatchEvent(
      new CustomEvent(
        "spprincipleschange",
        {
          bubbles: false,
          detail: {
            paused: state.paused,
            reverse: state.reverse,
            speed: state.speed,
          },
        }
      )
    );
  }


  function closeMenu(root) {
    const menu =
      root.querySelector(
        MENU_SELECTOR
      );

    if (
      menu instanceof
      HTMLDetailsElement
    ) {
      menu.open = false;
    }
  }


  function runAction(
    root,
    action
  ) {
    const state =
      tickerState.get(root);

    if (!state) {
      return;
    }

    switch (action) {
      case "toggle":
        state.paused =
          !state.paused;

        applyState(
          root,
          state
        );
        break;

      case "reverse":
        state.reverse =
          !state.reverse;

        applyState(
          root,
          state
        );
        break;

      case "shuffle": {
        const items =
          Array.from(
            state.group.querySelectorAll(
              ":scope > .sp-principles__item"
            )
          );

        state.group.replaceChildren(
          ...shuffle(items)
        );

        createClone(
          state.track,
          state.group
        );

        restartAnimation(
          state.track
        );
        break;
      }

      case "reset":
        state.paused = false;
        state.reverse = false;
        state.speed =
          getDefaultSpeed(root);

        restoreOriginalOrder(state);

        clearPreferences(root);

        applyState(
          root,
          state,
          {
            persist: false,
            restart: true,
          }
        );
        break;

      default:
        return;
    }

    closeMenu(root);
  }


  function setSpeed(
    root,
    requestedSpeed
  ) {
    const state =
      tickerState.get(root);

    if (
      !state ||
      !VALID_SPEEDS.has(
        requestedSpeed
      )
    ) {
      return;
    }

    state.speed =
      requestedSpeed;

    applyState(
      root,
      state,
      {
        persist: true,
        restart: true,
      }
    );

    closeMenu(root);
  }


  function initTicker(root) {
    if (
      root.dataset
        .spPrinciplesReady ===
      "true"
    ) {
      return;
    }

    const track =
      root.querySelector(
        ".sp-principles__track"
      );

    const group =
      root.querySelector(
        ".sp-principles__group"
      );

    if (
      !(track instanceof HTMLElement) ||
      !(group instanceof HTMLElement)
    ) {
      return;
    }

    const stored =
      readStoredPreferences(root);

    const preferences =
      normalizePreferences(
        root,
        stored
      );

    const state = {
      track,
      group,
      durations:
        getDurations(root),
      originalItems:
        cloneOriginalItems(group),
      ...preferences,
    };

    tickerState.set(
      root,
      state
    );

    createClone(
      track,
      group
    );

    applyState(
      root,
      state,
      {
        persist: false,
        restart: false,
      }
    );

    root.dataset.spPrinciplesReady =
      "true";
  }


  function initWithin(scope) {
    if (
      scope instanceof Element &&
      scope.matches(ROOT_SELECTOR)
    ) {
      initTicker(scope);
    }

    scope
      .querySelectorAll?.(
        ROOT_SELECTOR
      )
      .forEach(initTicker);
  }


  /*
   * Event delegation:
   * one listener supports every ticker and future menu action.
   */
  document.addEventListener(
    "click",
    (event) => {
      const target =
        event.target;

      if (
        !(target instanceof Element)
      ) {
        return;
      }

      const actionButton =
        target.closest(
          ACTION_SELECTOR
        );

      if (
        actionButton instanceof
        HTMLButtonElement
      ) {
        const root =
          actionButton.closest(
            ROOT_SELECTOR
          );

        if (
          root instanceof
          HTMLElement
        ) {
          runAction(
            root,
            actionButton.dataset.spAction
          );
        }

        return;
      }

      const speedButton =
        target.closest(
          SPEED_SELECTOR
        );

      if (
        speedButton instanceof
        HTMLButtonElement
      ) {
        const root =
          speedButton.closest(
            ROOT_SELECTOR
          );

        if (
          root instanceof
          HTMLElement
        ) {
          setSpeed(
            root,
            speedButton.dataset.spSpeed
          );
        }

        return;
      }

      /*
       * Native <details> does not close automatically when the user clicks
       * elsewhere. Close any open ticker menus on outside click.
       */
      document
        .querySelectorAll(
          `${MENU_SELECTOR}[open]`
        )
        .forEach((menu) => {
          if (
            !menu.contains(target) &&
            menu instanceof
            HTMLDetailsElement
          ) {
            menu.open = false;
          }
        });
    }
  );


  document.addEventListener(
    "keydown",
    (event) => {
      if (
        event.key !== "Escape"
      ) {
        return;
      }

      document
        .querySelectorAll(
          `${MENU_SELECTOR}[open]`
        )
        .forEach((menu) => {
          if (
            menu instanceof
            HTMLDetailsElement
          ) {
            menu.open = false;
          }
        });
    }
  );


  function init() {
    initWithin(document);

    /*
     * Future-proof dynamic content support for documentation systems that
     * inject or replace sections after initial page load.
     */
    if (
      "MutationObserver" in window
    ) {
      const observer =
        new MutationObserver(
          (mutations) => {
            for (
              const mutation
              of mutations
            ) {
              for (
                const node
                of mutation.addedNodes
              ) {
                if (
                  node instanceof Element
                ) {
                  initWithin(node);
                }
              }
            }
          }
        );

      observer.observe(
        document.body,
        {
          childList: true,
          subtree: true,
        }
      );
    }
  }


  if (
    document.readyState ===
    "loading"
  ) {
    document.addEventListener(
      "DOMContentLoaded",
      init,
      {
        once: true,
      }
    );
  } else {
    init();
  }
})();
