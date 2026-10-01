/* Authors: The scikit-plots developers
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Independent learning UI. No fetch, telemetry, training or publication calls.
 * The optional assistant.tasks.v1 adapter transfers only an explicitly selected
 * result. All user content is rendered through textContent, never innerHTML.
 */
(function () {
    "use strict";
    if (window.SPHINX_AI_LEARN && window.SPHINX_AI_LEARN.contract === "learn.ui.v1") {
        window.SPHINX_AI_LEARN.mountAll();
        return;
    }
    const mounts = new Map();
    const ID = /^[a-z][a-z0-9_-]{0,63}$/;
    const CONTRACT = "learn.contribution.v1";
    const MAX_SAVED = 100;
    const DRAFT_KINDS = ["topic", "problem", "source", "skill", "whiteboard", "video"];
    const CATALOG_KINDS = [...DRAFT_KINDS, "audio", "document"];
    const RESOURCE_KINDS = ["source", "whiteboard", "video"];
    const utcNow = () => new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
    function timestamp(value) {
        if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/.test(value) || !Number.isFinite(Date.parse(value)) || new Date(value).toISOString().slice(0, 19) + "Z" !== value) throw new Error("Invalid UTC timestamp.");
        return value;
    }
    let serial = 0;

    function node(tag, text, className) {
        const element = document.createElement(tag);
        if (text !== undefined && text !== null) element.textContent = text;
        if (className) element.className = className;
        return element;
    }
    function button(text, action, className) {
        const result = node("button", text, className);
        result.type = "button";
        result.addEventListener("click", action);
        return result;
    }
    function text(value, max, empty) {
        if (typeof value !== "string" || value.length > max || (!empty && !value.trim()) ||
            /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(value)) {
            throw new Error("Check the text fields and their length limits.");
        }
        return value.trim();
    }
    function identifier(value) {
        if (typeof value !== "string" || (!ID.test(value) || value.trim() !== value)) throw new Error("Invalid record identifier.");
        return value;
    }
    function singleLine(value, max, empty) {
        const result = text(value, max, empty);
        if (/[\r\n\t]/.test(result)) throw new Error("Titles and identifiers must use a single line.");
        return result;
    }
    function ids(values, maximum) {
        if (!Array.isArray(values) || values.length > maximum) throw new Error("Invalid record list.");
        const result = values.map(identifier);
        if (new Set(result).size !== result.length) throw new Error("Duplicate record identifiers.");
        return result;
    }
    function https(value) {
        const result = singleLine(value, 2048, false);
        const url = new URL(result);
        if (url.protocol !== "https:" || !url.hostname || url.username || url.password ||
            (url.port && url.port !== "443") || /[\s\\]/.test(result)) {
            throw new Error("Use an HTTPS source URL without credentials.");
        }
        return result;
    }
    function object(value, fields) {
        if (!value || typeof value !== "object" || Array.isArray(value) ||
            Object.keys(value).some(key => !fields.includes(key))) {
            throw new Error("Unsupported draft fields.");
        }
    }
    function validateDraft(value) {
        object(value, ["contract", "draft_id", "base_revision", "subject", "section", "authorship", "interactions"]);
        if (value.contract !== CONTRACT) throw new Error("Unsupported draft version.");
        const s = value.subject, c = value.section;
        object(s, ["id", "kind", "title", "summary", "domains", "related", "sections", "url", "created_at"]);
        object(c, ["id", "title", "body", "citations"]);
        if (!DRAFT_KINDS.includes(s.kind) ||
            !["human", "ai-assisted"].includes(value.authorship)) throw new Error("Unsupported draft type.");
        if (!Array.isArray(s.sections) || s.sections.length) throw new Error("Expected one section revision.");
        const subject = {id: identifier(s.id), kind: s.kind, title: singleLine(s.title, 200),
            summary: text(s.summary, 2000, true), domains: ids(s.domains, 20),
            related: ids(s.related, 100), sections: []};
        if (s.created_at !== undefined) subject.created_at = timestamp(s.created_at);
        if (RESOURCE_KINDS.includes(s.kind)) subject.url = https(s.url);
        else if ("url" in s) throw new Error("Only source, whiteboard, and video records have a resource URL.");
        if (!Array.isArray(c.citations) || c.citations.length > 32) throw new Error("Too many citations.");
        const citations = c.citations.map(row => {
            object(row, ["source_id", "locator"]);
            return {source_id: identifier(row.source_id), locator: singleLine(row.locator, 500)};
        });
        const result = {contract: CONTRACT, draft_id: identifier(value.draft_id),
            base_revision: singleLine(value.base_revision, 128), subject,
            section: {id: identifier(c.id), title: singleLine(c.title, 200),
                body: text(c.body, 50000, true), citations}, authorship: value.authorship};
        if (value.interactions !== undefined) {
            if (!Array.isArray(value.interactions) || value.interactions.length > 32) throw new Error("Too many interactions.");
            result.interactions = value.interactions.map(event => {
                object(event, ["id", "at", "action", "section_id", "request_id", "workflow_id"]);
                if (!["prompt-copied", "ai-result-selected"].includes(event.action) || event.workflow_id !== "learn.explanation.v1" || event.section_id !== c.id) throw new Error("Invalid interaction context.");
                return {id: identifier(event.id), at: timestamp(event.at), action: event.action,
                    section_id: identifier(event.section_id), request_id: identifier(event.request_id), workflow_id: event.workflow_id};
            });
            if (new Set(result.interactions.map(e => e.id)).size !== result.interactions.length) throw new Error("Duplicate interactions.");
        }
        return result;
    }
    function unique(prefix) {
        if (!globalThis.crypto || !globalThis.crypto.randomUUID) {
            throw new Error("This browser needs a secure context to create draft identifiers.");
        }
        return prefix + "-" + crypto.randomUUID();
    }
    function download(draft) {
        const blob = new Blob([JSON.stringify(validateDraft(draft), null, 2) + "\n"], {type: "application/json"});
        const url = URL.createObjectURL(blob);
        const link = node("a");
        link.href = url;
        link.download = draft.draft_id + ".json";
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    function mount(root) {
        if (mounts.has(root)) return;
        const dataNode = root.querySelector(".la-data");
        if (!dataNode) return;
        let config;
        try {
            config = JSON.parse(dataNode.textContent);
            if (config.catalog.contract !== "learn.catalog.v3" || !ID.test(config.site_id)) return;
            if (!Array.isArray(config.catalog.subjects) || config.catalog.subjects.some(s => !s || !ID.test(s.id) || !CATALOG_KINDS.includes(s.kind))) return;
        } catch (_) { return; }
        const fallback = root.querySelector(".la-static");
        const catalog = config.catalog;
        const byId = new Map(catalog.subjects.map(s => [s.id, s]));
        const prefix = "skplt-learn-ai:v1:" + config.site_id + ":";
        const instance = "la-ui-" + (++serial);
        const ui = node("div", null, "la-app");
        const status = node("p", "", "la-status");
        status.setAttribute("role", "status");
        status.setAttribute("aria-live", "polite");
        const header = node("header", null, "la-header");
        header.append(node("p", "LEARN · CONTRIBUTE · CONNECT", "la-eyebrow"),
            node("h2", "Build understanding, one contribution at a time"),
            node("p", "Explore topics, investigate open problems, and add an explanation in your own words."));
        const toolbar = node("div", null, "la-toolbar");
        const search = node("input");
        search.type = "search"; search.placeholder = "Search topics, references, or explanations";
        search.setAttribute("aria-label", "Search learning content");
        const domain = node("select");
        domain.setAttribute("aria-label", "Filter by domain");
        domain.append(new Option("All domains", ""));
        Array.from(new Set([...config.domains, ...catalog.subjects.flatMap(s => s.domains)])).sort()
            .forEach(d => domain.append(new Option(d.replaceAll("-", " "), d)));
        toolbar.append(search, domain, button("Create topic", () => edit(null), "la-primary"));
        const navigation = node("nav", null, "la-nav");
        navigation.setAttribute("aria-label", "Learning views");
        const content = node("div", null, "la-content");
        const editorHost = node("div", null, "la-editor-host");
        ui.append(header, toolbar, navigation, status, content, editorHost);
        let filter = config.initial_kind || "topic", selected = "", editor = null, controller = null, activeGenerate = null, epoch = 0, destroyed = false;
        const views = [["all", "All"], ["topic", "Topics"], ["problem", "Open problems"], ["source", "Sources"], ["skill", "Skills"], ["whiteboard", "Whiteboards"], ["video", "Videos"], ["audio", "Audio"], ["document", "Documents"], ["draft", "My drafts"]];
        const navButtons = new Map();
        function announce(message) { status.textContent = message; }
        function cancelTask() { epoch += 1; if (controller) controller.abort(); controller = null; if (activeGenerate) activeGenerate.disabled = false; activeGenerate = null; }
        function savedRows() {
            const rows = [];
            try {
                const storage = window.localStorage;
                for (let i = 0; i < storage.length; i++) {
                    const key = storage.key(i);
                    if (!key || !key.startsWith(prefix)) continue;
                    if (rows.length >= MAX_SAVED) break;
                    try {
                        const raw = storage.getItem(key);
                        if (raw.length > 150000) continue;
                        const row = JSON.parse(raw);
                        row.draft = validateDraft(row.draft);
                        if (key !== prefix + row.draft.draft_id || typeof row.version !== "string") continue;
                        rows.push(row);
                    } catch (_) { /* Do not trust or rewrite damaged browser records. */ }
                }
            } catch (_) { announce("Browser storage is unavailable. You can still export your draft."); }
            return rows;
        }
        function linkSubject(id, label) {
            const a = node("a", label || byId.get(id)?.title || id);
            const route = config.routes && config.routes[id];
            if (route) { a.href = route; return a; }
            const url = new URL(location.href);
            url.searchParams.set("subject", id); url.hash = "";
            a.href = url.href;
            a.addEventListener("click", event => {
                if (event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
                event.preventDefault(); showSubject(id, true);
            });
            return a;
        }
        function sectionView(s, section) {
            const block = node("section", null, "la-section");
            block.id = instance + "-" + s.id + "-" + section.id;
            block.append(node("h3", section.title), node("div", section.body, "la-body"));
            section.citations.forEach(c => {
                const ref = node("p", null, "la-citation");
                ref.append(linkSubject(c.source_id), document.createTextNode(" — " + c.locator));
                block.append(ref);
            });
            block.append(button("Improve this section", () => edit(s, section)));
            return block;
        }
        function showSubject(id, push) {
            const subject = byId.get(id);
            if (!subject) { announce("This subject is not in the published snapshot."); return; }
            cancelTask(); selected = id; content.replaceChildren();
            filter = subject.kind;
            navButtons.forEach((b, kind) => b.setAttribute("aria-pressed", String(kind === filter)));
            const top = node("div", null, "la-subject-head");
            top.append(button("Back to catalog", () => showList(true)), node("p", subject.kind, "la-eyebrow"),
                node("h2", subject.title), node("p", subject.summary));
            content.append(top);
            if (subject.url) {
                const a = node("a", subject.kind === "source" ? "Original source" : "Open " + subject.kind);
                a.href = subject.url; a.rel = "noopener noreferrer";
                content.append(a);
            }
            if (subject.kind === "whiteboard") {
                const preview = button("Load external whiteboard image", () => {
                    const img = node("img", null, "la-media");
                    img.alt = subject.title; img.referrerPolicy = "no-referrer";
                    img.src = subject.url;
                    img.addEventListener("error", () => announce("The image could not be loaded. Use the resource link."), {once: true});
                    preview.replaceWith(img);
                });
                content.append(preview, node("p", "Loading contacts the image host.", "la-note"));
            }
            if (subject.kind === "video") {
                const url = new URL(subject.url);
                const videoId = url.hostname === "youtu.be" ? url.pathname.slice(1) :
                    (["youtube.com", "www.youtube.com"].includes(url.hostname) && url.pathname === "/watch" ? url.searchParams.get("v") : "");
                if (videoId && /^[a-zA-Z0-9_-]{11}$/.test(videoId)) {
                    const play = button("Load YouTube player", () => {
                        const frame = node("iframe", null, "la-media");
                        frame.title = subject.title; frame.referrerPolicy = "no-referrer";
                        frame.allow = "encrypted-media; picture-in-picture; fullscreen";
                        frame.src = "https://www.youtube-nocookie.com/embed/" + videoId;
                        play.replaceWith(frame);
                    });
                    content.append(play, node("p", "Loading contacts YouTube. Playback availability depends on the source.", "la-note"));
                }
            }
            if (subject.kind === "audio" && subject.media && subject.media.type === "audio") {
                const audio = document.createElement("audio");
                audio.controls = true; audio.preload = "metadata"; audio.src = subject.media.src;
                content.append(audio);
            }
            if (subject.kind === "document" && subject.media && subject.media.type === "document") {
                const open = node("a", "Open document");
                open.href = subject.media.src; open.target = "_blank"; open.rel = "noopener";
                content.append(open);
            }
            if (subject.sections.length) {
                const toc = node("nav", null, "la-section-nav");
                toc.setAttribute("aria-label", "On this subject");
                subject.sections.forEach(section => {
                    const a = node("a", section.title); a.href = "#" + instance + "-" + id + "-" + section.id;
                    toc.append(a);
                });
                content.append(toc);
                subject.sections.forEach(section => content.append(sectionView(subject, section)));
            } else content.append(node("p", "This subject has no published explanations yet."));
            if (subject.related.length) {
                const related = node("div", null, "la-related");
                related.append(node("h3", "Related topics and resources"));
                subject.related.forEach(target => related.append(linkSubject(target)));
                content.append(related);
            }
            if (!['audio','document'].includes(subject.kind)) content.append(button("Add a section", () => edit(subject), "la-primary"));
            if (push) {
                const url = new URL(location.href); url.searchParams.set("subject", id);
                history.pushState({}, "", url);
            }
        }
        function showList(push) {
            cancelTask(); selected = "";
            content.replaceChildren();
            navButtons.forEach((b, kind) => b.setAttribute("aria-pressed", String(kind === filter)));
            const term = search.value.toLocaleLowerCase();
            const rows = filter === "draft" ? savedRows().map(row => ({...row.draft.subject, row})) : catalog.subjects;
            const matches = rows.filter(s => (filter === "draft" || filter === "all" || s.kind === filter) &&
                (!domain.value || s.domains.includes(domain.value)) &&
                [s.id, s.title, s.summary, s.url || "", ...s.domains,
                    ...s.sections.map(section => section.title + " " + section.body + " " + section.citations.map(c => (byId.get(c.source_id)?.title || c.source_id) + " " + c.locator).join(" ")),
                    ...s.related.map(id => byId.get(id)?.title || id)].join(" ").toLocaleLowerCase().includes(term));
            const label = filter === "draft" ? "local draft(s) on this browser" : "subject(s) in this snapshot";
            content.append(node("p", matches.length + " " + label, "la-count"));
            const cards = node("div", null, "la-cards");
            matches.forEach(s => {
                const card = node("article", null, "la-card");
                card.append(node("p", s.row ? "LOCAL DRAFT · NOT SUBMITTED" : s.kind.toUpperCase(), "la-eyebrow"),
                    node("h3", s.title), node("p", s.summary));
                if (s.row) {
                    card.append(button("Continue editing", () => edit(s.row.draft.subject, s.row.draft.section, s.row)),
                        button("Delete saved draft", () => {
                            try {
                                // Do not delete another tab's newer revision.
                                const raw = JSON.parse(localStorage.getItem(prefix + s.row.draft.draft_id));
                                if (raw.version !== s.row.version) throw new Error("Draft changed in another tab. Refresh the list first.");
                                localStorage.removeItem(prefix + s.row.draft.draft_id); showList(false);
                                announce("Saved draft deleted from this browser.");
                            } catch (error) { announce(error.message); }
                        }));
                } else card.append(linkSubject(s.id, "Explore"));
                cards.append(card);
            });
            content.append(cards);
            if (!matches.length) content.append(node("p", filter === "draft" ?
                "Save a draft explicitly to keep it on this browser." : "No matching content yet. Create a topic to start a local draft.", "la-empty"));
            const listUrl = new URL(location.href); listUrl.searchParams.delete("subject"); listUrl.searchParams.set("view", filter); listUrl.hash = "";
            if (push) history.pushState({}, "", listUrl); else history.replaceState({}, "", listUrl);
        }
        function field(form, label, tag, value, maximum) {
            const wrap = node("label", null, "la-field");
            wrap.append(node("span", label));
            const input = node(tag);
            input.setAttribute("aria-label", label);
            input.value = value || "";
            if (maximum) input.maxLength = maximum;
            wrap.append(input); form.append(wrap);
            return input;
        }
        function edit(subject, section, saved) {
            cancelTask(); editorHost.replaceChildren();
            const original = subject || {id: unique("topic"), kind: "topic", created_at: utcNow(), title: "", summary: "", domains: [], related: [], sections: []};
            const published = byId.has(original.id);
            const draftId = saved ? saved.draft.draft_id : unique("draft");
            let savedVersion = saved ? saved.version : null;
            const initialAuthorship = saved ? saved.draft.authorship : "human";
            const activity = saved ? (saved.draft.interactions || []).slice() : [];
            const customId = section ? section.id : unique("section");
            const form = node("form", null, "la-editor");
            editor = form;
            form.addEventListener("submit", event => event.preventDefault());
            const heading = node("h2", saved ? "Continue your draft" : "Prepare a contribution");
            heading.tabIndex = -1;
            form.append(heading, node("p", "Your draft stays on this page until you save or export it. Saving uses this browser only; nothing is submitted."));
            const kind = field(form, "Subject type", "select");
            [["topic", "Topic"], ["problem", "Open problem"], ["source", "Source"], ["skill", "Skill"], ["whiteboard", "Whiteboard"], ["video", "Video"]]
                .forEach(([value, label]) => kind.append(new Option(label, value)));
            kind.value = original.kind; kind.disabled = published;
            const title = field(form, "Subject title", "input", original.title, 200);
            title.required = true; title.readOnly = published;
            const summary = field(form, "Short description", "textarea", original.summary, 2000);
            summary.readOnly = published;
            const sourceUrl = field(form, "Resource URL (source, whiteboard, or video)", "input", original.url, 2048);
            sourceUrl.type = "url"; sourceUrl.parentElement.hidden = !RESOURCE_KINDS.includes(kind.value); sourceUrl.readOnly = published;
            kind.addEventListener("change", () => { sourceUrl.parentElement.hidden = !RESOURCE_KINDS.includes(kind.value); });
            const domainField = field(form, "Domains (comma-separated IDs)", "input", original.domains.join(", "), 1000);
            domainField.readOnly = published;
            domainField.placeholder = "machine-learning, statistics";
            const preset = field(form, "Section", "select");
            const presets = original.kind === "problem" ?
                [["statement", "Statement"], ["background", "Background"], ["references", "References"], ["related-problems", "Related problems"]] : config.sections;
            presets.forEach(([id, label]) => preset.append(new Option(label, id)));
            preset.append(new Option("Custom section", "custom"));
            preset.disabled = false;
            preset.value = section && presets.some(([id]) => id === section.id) ? section.id : (section ? "custom" : presets[0][0]);
            const sectionTitle = field(form, "Section title", "input", section ? section.title : presets[0][1], 200);
            kind.addEventListener("change", () => {
                if (section) return;
                const choices = kind.value === "problem" ?
                    [["statement", "Statement"], ["background", "Background"], ["references", "References"], ["related-problems", "Related problems"]] : config.sections;
                preset.replaceChildren();
                choices.forEach(([id, label]) => preset.append(new Option(label, id)));
                preset.append(new Option("Custom section", "custom"));
                sectionTitle.value = choices[0][1];
            });
            preset.addEventListener("change", () => {
                cancelTask();
                if (preset.value !== "custom") sectionTitle.value = preset.selectedOptions[0].text;
            });
            const body = field(form, "Explanation (plain text)", "textarea", section ? section.body : "", 50000);
            body.className = "la-draft-body";
            const authorshipField = field(form, "Authorship", "select");
            authorshipField.append(new Option("Written by me", "human"), new Option("AI assisted (including pasted AI text)", "ai-assisted"));
            authorshipField.value = initialAuthorship;
            const citations = section ? section.citations.slice() : [];
            const source = field(form, "Add a source citation", "select");
            source.append(new Option("Choose a published source", ""));
            catalog.subjects.filter(s => s.kind === "source").forEach(s => source.append(new Option(s.title, s.id)));
            const locator = field(form, "Source location (section, page, or timestamp)", "input", "", 500);
            const citationList = node("div", null, "la-citations");
            form.append(citationList);
            function renderCitations() {
                citationList.replaceChildren();
                citations.forEach((c, index) => {
                    const row = node("p");
                    row.append(document.createTextNode((byId.get(c.source_id)?.title || c.source_id) + " — " + c.locator + " "),
                        button("Remove citation", () => { citations.splice(index, 1); renderCitations(); }));
                    citationList.append(row);
                });
            }
            renderCitations();
            form.append(button("Add citation", () => {
                try {
                    if (!source.value || !byId.has(source.value)) throw new Error("Choose a published source.");
                    if (citations.length >= 32) throw new Error("The draft already has 32 citations.");
                    citations.push({source_id: source.value, locator: singleLine(locator.value, 500)});
                    locator.value = ""; renderCitations();
                } catch (error) { announce(error.message); }
            }));
            const actions = node("div", null, "la-actions");
            function collect() {
                const target = {id: original.id, kind: kind.value, title: title.value, summary: summary.value,
                    domains: domainField.value.split(",").map(x => x.trim()).filter(Boolean),
                    related: original.related.slice(), sections: []};
                if (original.created_at) target.created_at = original.created_at;
                if (RESOURCE_KINDS.includes(kind.value)) target.url = sourceUrl.value;
                return validateDraft({contract: CONTRACT, draft_id: draftId,
                    base_revision: saved ? saved.draft.base_revision : catalog.revision, subject: target,
                    section: {id: preset.value === "custom" ? customId : preset.value,
                        title: sectionTitle.value, body: body.value, citations}, authorship: authorshipField.value, interactions: activity.filter(event => event.section_id === (preset.value === "custom" ? customId : preset.value))});
            }
            actions.append(button("Save on this browser", () => {
                try {
                    const draft = collect();
                    const key = prefix + draftId, previousRaw = localStorage.getItem(key);
                    const previous = previousRaw ? JSON.parse(previousRaw) : null;
                    if ((previous ? previous.version : null) !== savedVersion) {
                        throw new Error("This draft changed in another tab. Export your version before reloading.");
                    }
                    if (!previous && savedRows().length >= MAX_SAVED) throw new Error("Browser draft limit reached. Export or delete an older draft.");
                    const version = unique("revision");
                    localStorage.setItem(key, JSON.stringify({version, draft}));
                    savedVersion = version;
                    announce("Saved on this browser. This draft has not been submitted or published.");
                } catch (error) { announce(error.message || "Could not save. Export the draft instead."); }
            }, "la-primary"), button("Export draft JSON", () => {
                try { download(collect()); announce("Draft exported. It has not been submitted."); }
                catch (error) { announce(error.message); }
            }));
            const activityList = node("ul", null, "la-activity");
            function renderActivity() {
                activityList.replaceChildren();
                activity.forEach(event => activityList.append(node("li", event.at + " · " + event.action.replaceAll("-", " "))));
            }
            function recordActivity(action, request) {
                if (activity.length === 32) activity.shift();
                activity.push({id: unique("interaction"), at: utcNow(), action,
                    section_id: request.section_id, request_id: request.request_id, workflow_id: request.workflow_id});
                // Section selection remains editable; export only its own provenance.
                kind.disabled = true;
                renderActivity();
            }
            if (activity.length) { kind.disabled = true; }
            renderActivity();
            form.append(node("h3", "Selected interaction activity"), node("p", "Local provenance claims only. No full conversation, credentials, or automatic tracking is stored. The latest 32 actions are retained."), activityList);
            function task() {
                const draft = collect();
                const request = {contract: "assistant.task.v1", request_id: unique("request"),
                    workflow_id: "learn.explanation.v1", subject_id: original.id,
                    section_id: draft.section.id, instruction: "Write a source-grounded learning explanation for the selected section. Return plain text. Source URLs are pointers; their content has not been retrieved by this extension. Mark unsupported claims and do not invent citations or claim to have read sources you cannot access.",
                    context: {subject: draft.subject, section: draft.section, catalog_revision: catalog.revision,
                        sources: citations.map(c => {
                            const s = byId.get(c.source_id);
                            if (!s) throw new Error("A cited source is missing from this snapshot.");
                            return {id: s.id, title: s.title, url: s.url, locator: c.locator};
                        })}};
                if (JSON.stringify(request).length > 65536) throw new Error("Reduce the selected context before requesting AI help.");
                return request;
            }
            actions.append(button("Copy AI prompt", async () => {
                try {
                    const prompt = task();
                    await navigator.clipboard.writeText(prompt.instruction + "\n\n" + JSON.stringify(prompt.context, null, 2));
                    if (!destroyed && editor === form && collect().section.id === prompt.section_id) recordActivity("prompt-copied", prompt);
                    announce("Selected context copied. Paste it into your preferred AI assistant.");
                } catch (error) { announce(error.message || "Clipboard access is unavailable."); }
            }));
            const bridge = window.AI_ASSISTANT && window.AI_ASSISTANT.tasks;
            const available = config.runtime === "assistant" && bridge &&
                bridge.contract === "assistant.tasks.v1" && typeof bridge.run === "function";
            const generate = button("Generate with assistant", async () => {
                let current = epoch;
                try {
                    const request = task();
                    cancelTask(); current = epoch;
                    controller = new AbortController(); generate.disabled = true; activeGenerate = generate;
                    const previousBody = body.value;
                    announce("Assistant task opened. Select a result there to use it in this draft.");
                    const result = await bridge.run(request, {signal: controller.signal});
                    if (destroyed || current !== epoch || editor !== form) return;
                    if (!result || result.contract !== "assistant.task-result.v1" ||
                        result.request_id !== request.request_id || result.selected !== true) {
                        throw new Error("The assistant did not return an explicitly selected result.");
                    }
                    if (body.value !== previousBody) throw new Error("Your explanation changed while the task was open. Your edits were kept; request a new result to continue.");
                    body.value = text(result.body, 50000, false); authorshipField.value = "ai-assisted";
                    recordActivity("ai-result-selected", request);
                    announce("Selected AI text added to your draft. Check sources before saving or exporting.");
                } catch (error) {
                    if (current === epoch && editor === form && !destroyed && error.name !== "AbortError") announce(error.message);
                } finally { if (current === epoch && editor === form && !destroyed) { generate.disabled = !available; activeGenerate = null; } }
            });
            generate.disabled = !available;
            if (!available) generate.title = "Connect an assistant to enable generation.";
            actions.append(generate, button("Close editor", () => {
                cancelTask(); editorHost.replaceChildren(); editor = null;
                announce("Editor closed. Only explicitly saved drafts are retained.");
            }));
            form.append(actions, node("p", available ?
                "Publication is not connected. Save or export a draft for now." :
                "Assistant connection and online submission are not available here. You can write, save, export, or copy a prompt.", "la-note"));
            editorHost.append(form);
            heading.focus();
        }
        views.forEach(([kind, label]) => {
            const b = button(label, () => { filter = kind; showList(true); });
            navButtons.set(kind, b); navigation.append(b);
        });
        search.addEventListener("input", () => showList(false));
        domain.addEventListener("change", () => showList(false));
        function navigate() {
            const params = new URL(location.href).searchParams;
            const view = params.get("view");
            if (views.some(([kind]) => kind === view)) filter = view;
            const id = params.get("subject") || (view ? "" : config.initial_subject);
            if (id) showSubject(id, false); else showList(false);
        }
        function storageChange(event) {
            if (event.key && event.key.startsWith(prefix) && filter === "draft" && !editor && !selected) showList(false);
        }
        window.addEventListener("popstate", navigate);
        window.addEventListener("storage", storageChange);
        mounts.set(root, () => {
            destroyed = true; cancelTask(); window.removeEventListener("popstate", navigate);
            window.removeEventListener("storage", storageChange); ui.remove();
            if (fallback) fallback.hidden = false; mounts.delete(root);
        });
        root.append(ui);
        if (fallback) fallback.hidden = true;
        navigate();
    }
    const api = {
        contract: "learn.ui.v1",
        mountAll(scope) { (scope || document).querySelectorAll("[data-skplt-learn-ai-mount]").forEach(mount); },
        destroy(root) { if (mounts.has(root)) mounts.get(root)(); },
        validateDraft
    };
    window.SPHINX_AI_LEARN = window.SPHINX_AI_LEARN || api;
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => api.mountAll(), {once: true});
    else api.mountAll();
}());
