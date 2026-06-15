/**
 * Foundation UI Utilities
 */
window.Foundation = window.Foundation || {};

Foundation.UI = {
  /**
   * Escape HTML special characters
   */
  escapeHtml(value) {
    if (value == null) return '';
    return String(value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#039;');
  },

  /**
   * Convert hex color to rgba
   */
  hexToRgba(hex, alpha) {
    const value = hex.replace('#', '');
    const r = parseInt(value.slice(0, 2), 16);
    const g = parseInt(value.slice(2, 4), 16);
    const b = parseInt(value.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  },

  /**
   * Show toast notification
   */
  showToast(message, duration = 2600) {
    const toast = document.querySelector('.toast');
    if (!toast) return;

    toast.textContent = message;
    toast.classList.remove('hidden');

    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => {
      toast.classList.add('hidden');
    }, duration);
  },

  /**
   * Convert Markdown to HTML (simple)
   */
  markdownToHtml(text) {
    let html = text
      .replace(/^### (.*)$/gm, '<h3>$1</h3>')
      .replace(/^## (.*)$/gm, '<h2>$1</h2>')
      .replace(/^# (.*)$/gm, '<h1>$1</h1>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/^- (.*)$/gm, '<li>$1</li>')
      .replace(/^(\d+)\. (.*)$/gm, '<li>$2</li>');

    // Wrap consecutive <li> in <ul>
    html = html.replace(/(<li>.*?<\/li>\n?)+/g, (match) => {
      const hasNumbers = /^\d+\./m.test(match);
      return `<${hasNumbers ? 'ol' : 'ul'}>${match}</${hasNumbers ? 'ol' : 'ul'}>`;
    });

    // Convert remaining double newlines to paragraph breaks
    html = html
      .split(/\n\n+/)
      .map(block => {
        block = block.trim();
        if (!block) return '';
        if (block.startsWith('<')) return block;
        return `<p>${block.replace(/\n/g, '<br>')}</p>`;
      })
      .join('\n');

    return html;
  },

  /**
   * Format number with locale
   */
  formatNumber(num, options = {}) {
    if (num == null) return '—';
    return num.toLocaleString(undefined, options);
  },

  /**
   * Format area in km²
   */
  formatArea(areaKm2) {
    if (areaKm2 == null) return '—';
    if (areaKm2 >= 1000000) {
      return `${(areaKm2 / 1000000).toFixed(2)}M km²`;
    } else if (areaKm2 >= 1000) {
      return `${(areaKm2 / 1000).toFixed(1)}K km²`;
    }
    return `${Math.round(areaKm2).toLocaleString()} km²`;
  }
};