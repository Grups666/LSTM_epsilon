/**
 * Render scheduler
 * Coalesces map redraws to animation frames so high-frequency interactions
 * such as drag, wheel, and hover do not force synchronous full-canvas renders.
 */
window.Foundation = window.Foundation || {};

Foundation.RenderScheduler = class RenderScheduler {
  constructor(render) {
    this.render = render;
    this.pending = false;
    this.reason = "initial";
    this.interacting = false;
    this.interactionEndTimer = null;
  }

  request(reason = "update") {
    this.reason = reason;
    if (this.pending) return;

    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      this.render({
        reason: this.reason,
        interacting: this.interacting
      });
    });
  }

  beginInteraction() {
    if (this.interactionEndTimer) {
      clearTimeout(this.interactionEndTimer);
      this.interactionEndTimer = null;
    }
    this.interacting = true;
  }

  endInteraction(delay = 80) {
    if (this.interactionEndTimer) clearTimeout(this.interactionEndTimer);
    this.interactionEndTimer = setTimeout(() => {
      this.interacting = false;
      this.request("interaction-end");
    }, delay);
  }
};
