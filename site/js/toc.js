// A contents column for the long pages (about, privacy, terms): one link per
// section heading (h2), kept in view while the page scrolls, with the
// section being read highlighted. Built from the page's own headings, so the
// text is not edited. Headings keep any id they have (links such as
// /terms#conduct rely on them); the rest get one from their text.

(() => {
    const main = document.querySelector('main');
    const heads = main ? [...main.querySelectorAll('h2')] : [];
    if (heads.length < 3) return;

    const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    for (const h of heads) if (!h.id) h.id = slug(h.textContent);

    // main holds two columns: the contents, and the page as it was.
    const body = document.createElement('div');
    body.className = 'toc-body';
    body.append(...main.childNodes);
    const aside = document.createElement('aside');
    aside.className = 'toc';
    aside.setAttribute('aria-label', 'On this page');
    const nav = aside.appendChild(document.createElement('nav'));
    const title = nav.appendChild(document.createElement('p'));
    title.className = 'toc-title';
    title.textContent = 'On this page';
    const links = heads.map((h) => {
        const a = document.createElement('a');
        a.href = `#${h.id}`;
        a.textContent = h.textContent;
        nav.appendChild(a);
        return a;
    });
    main.append(aside, body);
    main.classList.add('with-toc');

    // The section being read: the last heading above a line near the top,
    // or the last one once the page cannot scroll further. A clicked link
    // stays highlighted until the reader scrolls themselves, since the last
    // sections are too short to reach the top.
    let pinned = -1;
    const show = (at) => links.forEach((a, i) => {
        if (i === at) a.setAttribute('aria-current', 'true');
        else a.removeAttribute('aria-current');
    });
    const mark = () => {
        if (pinned >= 0) return show(pinned);
        const bottom = innerHeight + scrollY >= document.documentElement.scrollHeight - 4;
        let at = 0;
        heads.forEach((h, i) => { if (h.getBoundingClientRect().top < 140) at = i; });
        show(bottom ? heads.length - 1 : at);
    };
    links.forEach((a, i) => a.addEventListener('click', () => { pinned = i; show(i); }));
    const release = () => { if (pinned >= 0) { pinned = -1; mark(); } };
    for (const ev of ['wheel', 'touchmove', 'keydown']) addEventListener(ev, release, { passive: true });
    addEventListener('scroll', mark, { passive: true });
    addEventListener('resize', mark);
    mark();
})();
