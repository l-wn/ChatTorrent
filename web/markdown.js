// A small markdown renderer for model replies. The text is HTML-escaped
// before any markup is added, so the output can only contain the tags made
// here. Links are kept only for http:, https: and mailto: URLs. Handles
// paragraphs, headings, lists, fenced and inline code, bold, italic,
// strikethrough, blockquotes, rules and simple tables. An unclosed fence (a
// reply still streaming) renders as a code block to the end.

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ESC[c]);

// Inline markup on one already-escaped line. Code spans are cut out first so
// nothing inside them is formatted.
function inline(text) {
    const codes = [];
    let s = text.replace(/`([^`\n]+)`/g, (_, c) => {
        codes.push(c);
        return `\u0000${codes.length - 1}\u0000`;
    });
    s = s.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (m, label, url) => {
        // `url` is escaped already; &amp; is the only entity a URL needs.
        const raw = url.replace(/&amp;/g, '&');
        if (!/^(https?:|mailto:)/i.test(raw)) return m;
        return `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`;
    });
    s = s.replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>')
        .replace(/__([^_\n]+?)__/g, '<strong>$1</strong>')
        .replace(/(^|[^*\w])\*([^*\s][^*\n]*?)\*(?![*\w])/g, '$1<em>$2</em>')
        .replace(/(^|[^_\w])_([^_\s][^_\n]*?)_(?![_\w])/g, '$1<em>$2</em>')
        .replace(/~~([^~\n]+?)~~/g, '<del>$1</del>');
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[i]}</code>`);
}

const isRule = (l) => /^ {0,3}([-*_])( *\1){2,} *$/.test(l);
const heading = (l) => /^ {0,3}(#{1,6}) +(.*?)(?: +#+)? *$/.exec(l);
const bullet = (l) => /^( *)([-*+]|\d{1,9}[.)]) +(.*)$/.exec(l);
const fence = (l) => /^ {0,3}(```|~~~)(.*)$/.exec(l);
const tableRow = (l) => /^ *\|.*\| *$/.test(l);
const tableSep = (l) => /^ *\|( *:?-+:? *\|)+ *$/.test(l);
const cells = (l) => l.trim().replace(/^\||\|$/g, '').split('|').map((c) => inline(c.trim()));

// Markdown to an HTML string.
export function renderMarkdown(src) {
    const lines = escapeHtml(String(src ?? '')).replace(/\r\n?/g, '\n').split('\n');
    const out = [];
    let para = [];
    const flush = () => {
        if (para.length) out.push(`<p>${para.map(inline).join('<br>')}</p>`);
        para = [];
    };

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        const f = fence(line);
        if (f) {
            flush();
            const lang = f[2].trim().split(/\s/)[0];
            const body = [];
            for (i++; i < lines.length && !new RegExp(`^ {0,3}${f[1]}\\s*$`).test(lines[i]); i++) body.push(lines[i]);
            const cls = /^[\w+#-]+$/.test(lang) ? ` class="language-${lang}"` : '';
            out.push(`<pre><code${cls}>${body.join('\n')}</code></pre>`);
            continue;
        }
        if (!line.trim()) { flush(); continue; }
        if (isRule(line)) { flush(); out.push('<hr>'); continue; }

        const h = heading(line);
        if (h) { flush(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }

        if (/^ {0,3}&gt; ?/.test(line)) {
            flush();
            const quote = [];
            for (; i < lines.length && /^ {0,3}&gt; ?/.test(lines[i]); i++) quote.push(lines[i].replace(/^ {0,3}&gt; ?/, ''));
            i--;
            out.push(`<blockquote>${renderMarkdownEscaped(quote)}</blockquote>`);
            continue;
        }

        if (tableRow(line) && i + 1 < lines.length && tableSep(lines[i + 1])) {
            flush();
            const head = cells(line);
            const rows = [];
            for (i += 2; i < lines.length && tableRow(lines[i]); i++) rows.push(cells(lines[i]));
            i--;
            out.push(`<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>` +
                rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('') + '</tbody></table>');
            continue;
        }

        const b = bullet(line);
        if (b) {
            flush();
            i = list(lines, i, out);
            continue;
        }

        para.push(line.trim());
    }
    flush();
    return out.join('');
}

// Lines that are already escaped (a blockquote's body).
function renderMarkdownEscaped(lines) {
    const ent = { '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'" };
    return renderMarkdown(lines.join('\n').replace(/&(amp|lt|gt|quot|#39);/g, (m) => ent[m]));
}

// A list starting at lines[i], with nested lists by indent. Pushes its HTML
// and returns the index of its last line.
function list(lines, i, out) {
    const first = bullet(lines[i]);
    const indent = first[1].length;
    const ordered = /\d/.test(first[2]);
    const start = ordered ? parseInt(first[2], 10) : 1;
    const items = [];
    for (; i < lines.length; i++) {
        const b = bullet(lines[i]);
        if (b && b[1].length === indent) { items.push({ text: [b[3]], sub: [] }); continue; }
        if (b && b[1].length > indent) {
            const sub = [];
            i = list(lines, i, sub);
            items.at(-1).sub.push(...sub);
            continue;
        }
        // A wrapped continuation line of the item above.
        if (lines[i].trim() && /^ +\S/.test(lines[i]) && !b) { items.at(-1).text.push(lines[i].trim()); continue; }
        break;
    }
    const tag = ordered ? 'ol' : 'ul';
    const attr = ordered && start !== 1 ? ` start="${start}"` : '';
    out.push(`<${tag}${attr}>${items.map((it) => `<li>${it.text.map(inline).join(' ')}${it.sub.join('')}</li>`).join('')}</${tag}>`);
    return i - 1;
}
