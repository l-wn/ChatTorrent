// Tests web/markdown.js: model replies rendered as HTML. Escaping comes first,
// so no input can produce a tag the renderer did not make.
//
//   node tools/markdown.test.mjs

import { renderMarkdown as md } from '../web/markdown.js';

let pass = 0, fail = 0;
const eq = (name, got, want) => {
    if (got === want) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name}\n        got  ${got}\n        want ${want}`); }
};
const no = (name, got, re) => {
    if (!re.test(got)) { pass++; console.log(`  ok    ${name}`); }
    else { fail++; console.log(`  FAIL  ${name}\n        got  ${got}`); }
};

console.log('\nescaping');
eq('HTML in the text is shown, not run', md('<script>alert(1)</script>'), '<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>');
eq('an attribute cannot be injected through a link', md('[x](https://a.b/"onmouseover="alert(1))'),
    '<p><a href="https://a.b/&quot;onmouseover=&quot;alert(1" target="_blank" rel="noopener noreferrer">x</a>)</p>');
eq('a javascript: link stays text', md('[x](javascript:alert(1))'), '<p>[x](javascript:alert(1))</p>');
eq('a data: link stays text', md('[x](data:text/html,hi)'), '<p>[x](data:text/html,hi)</p>');
no('HTML inside code stays escaped', md('`<b>` and\n```\n<img src=x onerror=alert(1)>\n```'), /<img|<b>/);
no('an image tag in a table cell stays escaped', md('| a |\n|---|\n| <img src=x> |'), /<img/);
no('a quote cannot open a tag', md('> <iframe src=x>'), /<iframe/);

console.log('\ninline');
eq('bold and italic', md('**bold** and *it* and _it_'), '<p><strong>bold</strong> and <em>it</em> and <em>it</em></p>');
eq('inline code is not formatted inside', md('`**x**`'), '<p><code>**x**</code></p>');
eq('snake_case words are left alone', md('a_b_c and 2*3*4'), '<p>a_b_c and 2*3*4</p>');
eq('an https link', md('see [docs](https://x.y/a?b=1&c=2)'),
    '<p>see <a href="https://x.y/a?b=1&amp;c=2" target="_blank" rel="noopener noreferrer">docs</a></p>');
eq('lines in a paragraph keep their breaks', md('one\ntwo'), '<p>one<br>two</p>');

console.log('\nblocks');
eq('headings', md('# A\n### B'), '<h1>A</h1><h3>B</h3>');
eq('a fenced block with a language', md('```js\nconst a = 1;\n```'), '<pre><code class="language-js">const a = 1;</code></pre>');
eq('an unclosed fence runs to the end (streaming)', md('```\nline'), '<pre><code>line</code></pre>');
eq('a bullet list', md('- a\n- b'), '<ul><li>a</li><li>b</li></ul>');
eq('an ordered list keeps its start', md('3. c\n4. d'), '<ol start="3"><li>c</li><li>d</li></ol>');
eq('a nested list', md('- a\n  - b\n- c'), '<ul><li>a<ul><li>b</li></ul></li><li>c</li></ul>');
eq('a table', md('| a | b |\n|---|:-:|\n| 1 | 2 |'),
    '<table><thead><tr><th>a</th><th>b</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table>');
eq('a blockquote', md('> hi'), '<blockquote><p>hi</p></blockquote>');
eq('a rule', md('a\n\n---\n\nb'), '<p>a</p><hr><p>b</p>');
eq('empty input', md(''), '');

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
