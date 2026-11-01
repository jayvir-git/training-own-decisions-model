import { readFileSync, writeFileSync } from 'node:fs';
import { render } from 'pele';

const figures = new URL('../../docs/figures/', import.meta.url);
const source = readFileSync(new URL('pipeline.mmd', figures), 'utf8');
const themes = {
  light: { bg: '#fcfcfb', fg: '#0b0b0b', muted: '#52514e', line: '#92938f', surface: '#ffffff', alt: '#f4f4f1', border: '#deded8', accent: '#2a78d6', tint: '#edf4fd' },
  dark: { bg: '#0d1117', fg: '#e6edf3', muted: '#a6b0bd', line: '#7d8998', surface: '#161b22', alt: '#111820', border: '#303b49', accent: '#79b5ff', tint: '#192c43' },
};

for (const [name, t] of Object.entries(themes)) {
  const result = render(source, {
    fontFamily: 'Segoe UI, Arial, sans-serif', fontSize: 15,
    padding: 24, autoDirection: false, idPrefix: `pipeline-${name}`,
    config: { flowchart: { rankSpacing: 24, nodeSpacing: 28 } },
    links: false, images: false,
  });
  const css = `
    .pele {
      ${Object.entries({ bg: t.bg, fg: t.fg, muted: t.muted, line: t.line, surface: t.surface, 'surface-alt': t.alt, border: t.border, accent: t.accent }).map(([key, value]) => `--pele-${key}: ${value};`).join('\n      ')}
      --pele-radius: 8px;
    }
    .pele-cluster-label { fill: ${t.muted}; font-weight: 600; letter-spacing: 1px; }
    .pele-node[data-id="G"], .pele-node[data-id="H"], .pele-node[data-id="I"] {
      --pele-surface: ${t.tint}; --pele-border: ${t.accent};
    }
    .pele-node[data-id="G"] .pele-label,
    .pele-node[data-id="H"] .pele-label,
    .pele-node[data-id="I"] .pele-label { fill: ${t.accent}; font-weight: 600; }
  `;
  // Inline CSS makes each SVG self-contained when GitHub displays it as an image.
  const svg = result.svg.replace(/(<svg\b[^>]*>)/, `$1<style>${css}</style>`);
  writeFileSync(new URL(`pipeline-${name}.svg`, figures), svg + '\n');
  console.log(`${name}: ${Math.round(result.width)} x ${Math.round(result.height)}`);
}
