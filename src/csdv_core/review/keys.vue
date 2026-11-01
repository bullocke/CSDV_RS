<template>
  <span style="display: none"></span>
</template>

<script>
// Sends review shortcut keys to Python. Keys typed into text fields, and keys
// with Ctrl, Alt or Meta held, are left alone.
module.exports = {
  mounted() {
    const keys = [
      "1", "2", "3", "4", "a", "b", "h", "w", "c", "n", "p", "s",
      "Enter", "ArrowLeft", "ArrowRight",
    ];
    this._onKey = (e) => {
      const t = e.target;
      const tag = ((t && t.tagName) || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (t && t.isContentEditable) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (!keys.includes(e.key)) return;
      e.preventDefault();
      this.event_key(e.key);
    };
    document.addEventListener("keydown", this._onKey);
  },
  beforeDestroy() {
    document.removeEventListener("keydown", this._onKey);
  },
};
</script>
