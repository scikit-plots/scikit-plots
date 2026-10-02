# scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/tests/test__pages.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Page structure, explicit authoring, review invalidation, and pagination."""
import json
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

import _learn_site
from jinja2 import Environment, FileSystemLoader

from _sphinx_ext._sphinx_ai_learn._materialize import load_content_tree, _media_directive
from _sphinx_ext._sphinx_ai_learn._generation import compact_feedback_count, section_generation_feedback_stats
from _sphinx_ext._sphinx_ai_learn._registry import (
    TOPIC_EMPTY_MESSAGES,
    canonical_detail_section_id,
    detail_sections,
    topic_sections,
)
from _sphinx_ext._sphinx_ai_learn._schema import LearnValidationError, validate_catalog

EXT = _learn_site.STACK_ROOT.parent

def _render_text_generation_actions(scope: str, run_label: str, copy_suffix: str = "") -> str:
    env = Environment(
        loader=FileSystemLoader(str(EXT / "_sphinx_ext/_sphinx_ai_learn/_templates")),
        autoescape=False,
    )
    return env.get_template("learn/text-generation-actions.html").render(
        generation_action_scope=scope,
        generation_action_run_label=run_label,
        generation_action_copy_suffix=copy_suffix,
    )


STUDIO_NAV = [
    {'kind':'topic', 'label':'Topics', 'href':'../topics/new.html'},
    {'kind':'source', 'label':'Sources', 'href':'../sources/new.html'},
    {'kind':'problem', 'label':'Open Problems', 'href':'../open-problems/new.html'},
    {'kind':'whiteboard', 'label':'Whiteboards', 'href':'../whiteboards/new.html'},
    {'kind':'video', 'label':'Videos', 'href':'../videos/new.html'},
    {'kind':'audio', 'label':'Audio', 'href':'../audios/new.html'},
    {'kind':'document', 'label':'Documents', 'href':'../documents/new.html'},
    {'kind':'skill', 'label':'Skills', 'href':'../skills/new.html'},
    {'kind':'topic-prompt', 'label':'Topic Prompts', 'href':'../topic-prompts/new.html'},
]


def topic():
    return {
        'id': 'topic-one',
        'kind': 'topic',
        'title': 'A generic topic',
        'created_at': '2026-09-17T12:00:00Z',
        'sections': [],
    }


def test_scaffold_owns_every_heading_and_nested_prompt():
    rel, record = next(
        (rel, row) for rel, row in _learn_site.content_tree().records.items()
        if row["subject"]["kind"] == "topic"
    )
    rst = (_learn_site.content_root() / rel.with_suffix(".rst")).read_text()
    specs = topic_sections(record["subject"], prompts=_learn_site.content_tree().prompts, skills=_learn_site.content_tree().skills)
    for spec in specs:
        assert f".. _learn-{record['subject']['id']}-{spec['id']}:" in rst
        assert spec["title"] in rst
    assert ".. include:: topic-prompts/index.rst" in rst
    assert "Topic to Video (Beta)" in rst
    assert "Audio Explanation" in rst
    assert len(specs) == len(_learn_site.content_tree().prompts) + len(_learn_site.content_tree().skills) + 12
    assert rst.index("Explain it Like I'm 14") < rst.index("Knowledge Gaps")
    assert rst.index("Tweets") < rst.index("HackerNews")
    assert ".. ai-learn::" not in rst
    assert "{{" not in rst

def test_topic_empty_states_match_topic_language():
    assert TOPIC_EMPTY_MESSAGES["summary"] == "No one has generated a summary of this topic yet."
    assert TOPIC_EMPTY_MESSAGES["video"] == "No one has generated a video about this topic yet."
    assert TOPIC_EMPTY_MESSAGES["audio"] == "No one has generated an audio explanation for this topic yet."
    assert TOPIC_EMPTY_MESSAGES["document"] == "No one has generated a document about this topic yet."
    assert TOPIC_EMPTY_MESSAGES["whiteboard"] == "No one has generated a whiteboard explanation for this topic yet."
    assert TOPIC_EMPTY_MESSAGES["open-problems"] == "We haven't generated a list of open problems mentioned in this topic yet."
    assert TOPIC_EMPTY_MESSAGES["continue-learning"] == "We haven't generated follow-up questions for this topic yet."



def test_context_aware_detail_sections_are_canonical_without_legacy_aliases():
    problem = {
        "kind": "problem",
        "sections": [
            {"id": "statement", "title": "Problem"},
            {"id": "custom-analysis", "title": "Custom Analysis"},
        ],
    }
    assert [(row["id"], row["title"]) for row in detail_sections(problem)] == [
        ("statement", "Statement"),
        ("background", "Background"),
        ("references", "References"),
        ("related", "Related Topics"),
        ("custom-analysis", "Custom Analysis"),
    ]
    assert canonical_detail_section_id("problem", "reason") == "reason"
    assert canonical_detail_section_id("source", "description") == "description"

def test_inline_ai_section_generation_policy_is_explicit_and_kind_safe():
    topic_specs = {row["id"]: row for row in topic_sections(prompts=_learn_site.content_tree().prompts, skills=_learn_site.content_tree().skills)}
    assert topic_specs["summary"]["generation"]["mode"] == "chat"
    assert topic_specs["knowledge-gaps"]["generation"]["skill"] == "topic-prompt:knowledge-gaps"
    assert topic_specs["knowledge-gaps"]["generation"]["agent"] == "learning-section-agent"
    assert "Generation contract:" in topic_specs["knowledge-gaps"]["generation"]["instruction"]
    assert topic_specs["skill-check-reference"]["generation"]["skill"] == "skill:skill-check-reference"
    assert topic_specs["skill-check-reference"]["generation"]["agent"] == "learning-skill-agent"
    for section_id in ("video", "audio", "document", "whiteboard", "open-problems", "tweets", "hackernews"):
        assert topic_specs[section_id]["generation"]["mode"] == "none"

    expected = {
        "problem": ({"statement", "background"}, {"references", "related"}),
        "video": ({"description", "script"}, {"transcript", "evidence"}),
        "audio": ({"description"}, {"transcript", "evidence"}),
        "document": ({"description", "content"}, {"evidence"}),
        "whiteboard": ({"description"}, {"evidence"}),
        "source": ({"overview", "scope", "key-points"}, {"related"}),
    }
    for kind, (generated, owned) in expected.items():
        specs = {row["id"]: row for row in detail_sections({"kind": kind, "sections": []})}
        assert {key for key, row in specs.items() if row["generation"]["mode"] == "chat"} == generated
        assert owned <= {key for key, row in specs.items() if row["generation"]["mode"] == "none"}
    assert "must not be hallucinated" in detail_sections({"kind": "video", "sections": []})[2]["generation"]["reason"]



def test_site_custom_css_is_backed_by_a_configured_static_source():
    conf = (_learn_site.docs_source() / "conf.py").read_text(encoding="utf-8")
    assert "html_static_path = ['_static', 'css']" in conf
    assert 'html_css_files = ["styles/custom.css"]' in conf
    assert (_learn_site.docs_source() / "css/styles/custom.css").is_file()


def test_publication_credit_is_explicit_public_metadata_not_generation_context():
    templates = EXT / '_sphinx_ext/_sphinx_ai_learn/_templates/learn'
    credit = (templates / 'publication-credit.html').read_text(encoding='utf-8')
    section = (templates / 'section-ai-generation.html').read_text(encoding='utf-8')
    record = (templates / 'record-generation.html').read_text(encoding='utf-8')
    overview = (templates / 'overview-actions.html').read_text(encoding='utf-8')
    media = {kind: (templates / f'{kind}-generation.html').read_text(encoding='utf-8') for kind in ('video', 'audio', 'document', 'whiteboard')}
    shared = (EXT / '_sphinx_ext/_sphinx_ai_learn/_static/generation-ui.js').read_text(encoding='utf-8')
    text_workflow = (EXT / '_sphinx_ext/_sphinx_ai_learn/_static/text-generation-ui.js').read_text(encoding='utf-8')
    section_js = (EXT / '_sphinx_ext/_sphinx_ai_learn/_static/section-generation.js').read_text(encoding='utf-8')
    overview_js = (EXT / '_sphinx_ext/_sphinx_ai_learn/_static/overview-generation.js').read_text(encoding='utf-8')
    record_js = (EXT / '_sphinx_ext/_sphinx_ai_learn/_static/record-generation.js').read_text(encoding='utf-8')
    media_js = {kind: (EXT / f'_sphinx_ext/_sphinx_ai_learn/_static/{kind}-generation.js').read_text(encoding='utf-8') for kind in ('video', 'audio', 'document', 'whiteboard')}
    assert 'data-publication-credit' in credit and 'maxlength="80"' in credit
    assert 'autocomplete="off"' in credit
    assert 'do not enter email addresses or other private contact details' in credit
    assert 'never sent to the AI model' in credit
    assert 'learn/publication-credit.html' in section
    assert 'learn/publication-credit.html' in record
    assert 'learn/publication-credit.html' in overview
    assert all('learn/publication-credit.html' in template for template in media.values())
    assert overview.index('learn/publication-credit.html') < overview.index('learn/text-generation-actions.html')
    assert 'data-overview-run' in _render_text_generation_actions('overview', 'Generate AI Overview', '-request')
    assert record.index('learn/publication-credit.html') < record.index('learn/generation-actions.html')
    assert all(template.index('learn/publication-credit.html') < template.index('learn/generation-actions.html') for template in media.values())
    assert 'normalizePublicationCredit' in shared and 'bindPublicationCredit' in shared and 'publicationContributor' in shared
    assert shared.index('/[\\u0000-\\u001f\\u007f]/.test(value)') < shared.index("value = value.replace(/\\s+/g, ' ').trim()")
    assert 'localStorage.setItem(storageKey, value)' in shared
    assert "publicationContributor?.(options.root)" in text_workflow
    assert "buildRequest:(current,contributor)" in section_js
    assert "publicationContributor?.(root)" in record_js
    assert 'bindPublicationCredit?.(panel' in overview_js
    assert "buildRequest:(current,contributor)" in overview_js
    assert 'publicationContributor' not in section_js and 'publicationContributor' not in overview_js
    assert all('bindPublicationCredit' in script for script in media_js.values())
    assert all('publicationContributor' not in script for script in media_js.values())
    assert "contextFor(data,spec,section,originalBody)" in section_js
    assert 'Contributor credit above is used only if you open a reviewed pull request.' in overview
    assert 'credit stays outside the AI generation request' in record


def test_compact_feedback_count_scale_and_rollover():
    cases = {
        0: "0",
        999: "999",
        1000: "1K",
        1500: "1.5K",
        2000: "2K",
        10000: "10K",
        100000: "100K",
        1000000: "1M",
        2300000: "2.3M",
        1000000000: "1B",
        2000000000: "2B",
        1000000000000: "1T",
        999500: "1M",
    }
    assert {value: compact_feedback_count(value) for value in cases} == cases


def test_generation_feedback_stats_keep_sign_distribution_explicit():
    section = {
        "active_generation_id": "generation-current",
        "generations": [{"id": "generation-current"}],
    }
    stats = section_generation_feedback_stats(
        section,
        [
            {"rating": 5},
            {"rating": 1},
            {"rating": -4},
            {"rating": 0},
            {"rating": -1},
            {"rating": 2},
        ],
    )
    assert stats == {
        "score": 3,
        "count": 6,
        "positive_count": 3,
        "negative_count": 2,
        "neutral_count": 1,
    }


def test_generation_feedback_reuses_quick_and_eleven_point_reviewed_publication_flow():
    templates = EXT / '_sphinx_ext/_sphinx_ai_learn/_templates/learn'
    feedback = (templates / 'generation-feedback.html').read_text(encoding='utf-8')
    end = (templates / 'section-end.html').read_text(encoding='utf-8')
    js = (EXT / '_sphinx_ext/_sphinx_ai_learn/_static/generation-feedback.js').read_text(encoding='utf-8')
    pages = (EXT / '_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text(encoding='utf-8')
    css = (EXT / '_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text(encoding='utf-8')
    assert 'learn/generation-feedback.html' in end
    assert '>Was This Helpful?</span>' in feedback
    assert '<span class="learn-generation-feedback-label">Community feedback</span>' not in feedback
    assert 'data-learn-feedback-quick="-1"' in feedback
    assert 'data-learn-feedback-quick="1"' in feedback
    assert 'data-learn-feedback-quick-count="-1"' in feedback
    assert 'data-learn-feedback-quick-count="1"' in feedback
    assert 'feedback_negative_count|default(0)' in feedback
    assert 'feedback_positive_count|default(0)' in feedback
    assert 'learn-generation-feedback-rating-emoji' in feedback
    assert 'learn-generation-feedback-rating-value' in feedback
    assert 'data-feedback-tone="negative"' in feedback
    assert 'data-feedback-tone="positive"' in feedback
    assert "else 'neutral'" in feedback
    assert 'data-learn-feedback-rating="{{ value }}"' in feedback
    assert "(-5,'Terrible','😡')" in feedback and "(5,'Excellent!','🤩')" in feedback
    assert 'data-learn-feedback-comment' in feedback and 'maxlength="2000"' in feedback
    assert 'data-learn-feedback-contributor' in feedback and 'maxlength="80"' in feedback
    assert 'data-learn-generation-order' in feedback
    assert 'data-feedback-section-id=' in feedback
    assert 'Published first' in feedback and 'Highest rated' in feedback and 'Newest generated' in feedback
    assert 'become public canonical metadata' in feedback
    assert 'learn-generation-feedback-score learn-sr-only' in feedback
    assert 'Browser, device, account, IP/network, and telemetry fields are not written' in feedback
    assert 'public page/generation routing identifiers' in feedback
    assert 'transient network information only for abuse-rate limiting' in feedback
    assert "root.dataset.feedbackSectionId||section?.dataset.section" in js
    assert "action:'feedback'" in js
    assert "feedback_mode:mode" in js
    assert "crypto.randomUUID" not in js
    assert "crypto.getRandomValues" in js
    assert "new Uint8Array(24)" in js
    assert "Math.random" not in js
    assert "navigator." not in js and "userAgent" not in js and "screen." not in js
    assert "Secure feedback event nonce is unavailable" in js
    assert "rating < -5||rating > 5" in js
    assert "submitPublication(pending.request)" in js
    assert "button.getAttribute('aria-pressed')==='true'" in js
    assert "mode==='quick'" in js and "setAttribute('aria-pressed'" in js
    assert "feedbackId()" in js
    assert "pending={fingerprint,request};writePending(pending)" in js
    assert "pendingMatches(pending,fingerprint,rating,clean,displayName,mode)" in js
    assert "A pending feedback retry may outlive an unrelated catalog rebuild" in js
    assert "request.base_revision===String(data.revision||'')" not in js
    assert "normalizeComment(textValue)" in js
    assert "Optional details must be at most 2000 characters." in js
    assert "sessionStorage.setItem(pendingKey" in js
    assert "sessionStorage.removeItem(pendingKey)" in js
    assert "pending=readPending()" in js
    assert "learn-ai-feedback-quick:v1:" in js
    assert "const priorQuick=readQuick()" in js
    assert "writeQuick(rating)" in js
    assert "Retry will reuse the same feedback id." in js
    assert "Community ratings update only after merge and rebuild." in js
    assert "function formatCompactCount" in js
    assert "1,000,000,000,000" in js
    assert "rounded>=1000" in js
    assert 'normalizePublicationCredit' in js
    assert "function orderHistory(mode)" in js
    assert "mode==='rating'" in js and "mode==='newest'" in js
    assert 'app.add_js_file("generation-feedback.js", defer="defer", priority=612)' in pages
    assert '.learn-generation-feedback-options' in css
    assert 'grid-template-areas:' in css and '"prompt quick spacer summary"' in css
    assert 'row-gap:0' in css
    assert '@media (max-width:720px)' in css and '@media (max-width:430px)' in css
    assert '[data-feedback-tone="negative"][aria-pressed="true"]' in css
    assert '[data-feedback-tone="positive"][aria-pressed="true"]' in css
    assert '[data-feedback-tone="neutral"][aria-pressed="true"]' in css
    assert '.learn-generation-feedback-rating-emoji' in css
    assert '.learn-generation-feedback-rating-value' in css
    assert '.learn-generation-feedback-quick-count' in css
    assert 'border-inline-start' in css
    assert '[data-learn-feedback-rating="-5"]' in css and '--learn-rating-bg:#7f1d1d' in css
    assert '[data-learn-feedback-rating="0"]' in css and '--learn-rating-bg:#6b7280' in css
    assert '[data-learn-feedback-rating="5"]' in css and '--learn-rating-bg:#15803d' in css
    assert '[aria-pressed="true"] .learn-generation-feedback-rating-value { display:inline; }' in css
    assert '--learn-feedback-bg:var(--learn-gen-bg,var(--pst-color-background,#fff));' in css
    assert '--learn-feedback-accent-line:var(--learn-gen-accent-line,color-mix(in srgb,var(--learn-accent) 48%,var(--learn-line)));' in css
    assert 'html[data-theme="dark"] .learn-generation-feedback,body[data-theme="dark"] .learn-generation-feedback' in css
    assert 'var(--pst-color-background,#15191f)' in css
    assert '.learn-generation-feedback-field textarea,.learn-generation-feedback-field input {' in css
    assert 'background:transparent; color:inherit; font:inherit; line-height:1.45;' in css
    assert 'border:1px solid var(--learn-line); border-radius:.5rem;' in css
    assert '.learn-generation-feedback-field textarea { resize:vertical; }' in css
    assert '.learn-generation-feedback-field :is(textarea,input):hover { border-color:var(--learn-feedback-accent-line); }' in css
    assert '.learn-generation-feedback-field :is(textarea,input):focus-visible' in css
    assert '@media (forced-colors:active)' in css and 'background:Canvas; color:CanvasText; border-color:CanvasText;' in css

def test_inline_ai_section_drafts_use_chat_authority_and_preserve_published_text():
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    topic_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.js').read_text()
    section_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/section-generation.js').read_text()
    start = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/section-start.html').read_text()
    end = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/section-end.html').read_text()
    panel = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/section-ai-generation.html').read_text()
    env = Environment(loader=FileSystemLoader(str(EXT/'_sphinx_ext/_sphinx_ai_learn/_templates')), autoescape=False)
    lenses = env.get_template('learn/ai-lens-profile.html').render(lens_scope='section-ai', lens_variant='section')
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    assert 'app.add_js_file("section-generation.js", defer="defer", priority=609)' in pages
    assert 'publication_state = "published" if filled else "unpublished"' in pages and 'AI draft not generated' in pages
    assert 'generation_id = (' in pages
    assert 'section_generation_id(subject, content)' in pages
    assert 'if content.get("body", "").strip()' in pages
    assert 'data-generation-mode' in start and 'data-discard-ai' in start
    assert 'data-publication-state' in start and 'data-evidence-review-state' in start
    assert 'data-edit hidden' in start
    assert "spec.generation.mode|default('none') == 'chat'" in end
    assert '>Generate Now</button>' in end and '>Regenerate</button>' not in end
    assert 'data-ai-edit' in end and 'AI-assisted' in end
    assert 'data-section-ai-panel' in panel
    assert 'learn/publication-credit.html' in panel
    assert 'data-contributors' in end
    assert 'No references attached · evidence review not applicable.' in end
    assert "{'label':'Agent role'" in panel and 'learn/generation-request-meta.html' in panel
    assert 'data-section-legacy-note' in end and 'data-copy-legacy-edit' in end
    for value in ('young-learner', 'beginner', 'student', 'practitioner', 'decision-maker', 'educator', 'researcher', 'expert'):
        assert f'data-section-ai-audience="{value}"' in lenses
    for value in ('understand', 'apply', 'teach', 'compare', 'research'):
        assert f'data-section-ai-purpose="{value}"' in lenses
    for value in ('concise', 'balanced', 'deep'):
        assert f'value="{value}"' in panel
    assert "const CHAT_CONTRACT='scikitplot-chat-v1'" in section_js
    assert "const DRAFT_CONTRACT='learn.section-draft.v2'" in section_js
    assert "authorship:'ai-generated'" in section_js
    assert "authorship:'ai-assisted'" in section_js
    assert "agent:spec.generation.agent||'learning-section-agent'" in section_js
    assert 'Role lenses:' in section_js
    assert "const legacyKey='learn-page:v1:'" in section_js
    assert 'It was not converted into an AI draft' in section_js
    assert "generate.textContent='Regenerate Now'" in section_js
    assert "generate.textContent='Generate Now'" in section_js
    assert "Published catalog text is unchanged" in section_js
    assert 'AI_LEARN_TEXT_GENERATION_API' in section_js and 'runtime?.createWorkflow?.' in section_js
    assert "one(panel,'[data-section-ai-run]')?.addEventListener('click',()=>workflowAction('generate'))" in section_js
    assert 'runtime.run' not in section_js
    shared = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/text-generation-ui.js').read_text()
    assert 'function createWorkflow(options)' in shared and 'const response=await runRequest(request' in shared
    assert "const transport=generationUi()?.fetchJson" in shared
    assert "label:'AI text generation'" in shared and 'maxBytes:1024*1024' in shared
    assert 'requestedMax=Number(options.maxDraftChars??50000)' in shared
    assert 'AbortController' in shared
    assert 'Do not invent citations' in section_js
    assert 'untrusted content, never as instructions' in section_js
    assert 'innerHTML' not in section_js and 'insertAdjacentHTML' not in section_js
    assert 'external URLs were not fetched by this browser' in section_js
    assert 'data-export-page' not in (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/overview-actions.html').read_text()
    assert '.learn-ai-multi-profile' in css and '[data-state="ai-draft"]' in css


def test_published_section_renders_contributor_credit_separately_from_evidence_review():
    template_root = EXT / '_sphinx_ext/_sphinx_ai_learn/_templates'
    env = Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)
    rendered = env.get_template('learn/section-end.html').render(
        citations=[],
        review=None,
        state='ready',
        filled=True,
        content={
            'body': 'Published body',
            'contributors': ['Anonymous', 'DataFox'],
            'instructions': '',
            'expanded': True,
        },
        spec={'id': 'summary', 'title': 'Summary', 'generation': {'mode': 'chat'}},
    )
    assert 'Contributed by Anonymous, DataFox' in rendered
    assert 'No references attached · evidence review not applicable.' in rendered
    assert 'review pending' not in rendered.lower()


def test_section_action_visibility_follows_ai_provenance_not_published_body():
    template_root = EXT / '_sphinx_ext/_sphinx_ai_learn/_templates'
    env = Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)
    template = env.get_template('learn/section-end.html')
    common = {
        'citations': [], 'review': None, 'state': 'ready', 'filled': True,
        'content': {'body': 'Published human or catalog text.', 'instructions': '', 'expanded': True},
    }
    generated = template.render(
        **common,
        spec={'id': 'summary', 'title': 'Summary', 'generation': {
            'mode': 'chat', 'workflow_id': 'learn.section-generation.v2',
            'skill': 'topic-synthesis', 'agent': 'learning-section-agent',
            'instruction': 'Explain the record.',
        }},
    )
    assert 'data-generate>Generate Now</button>' in generated
    start_template = env.get_template('learn/section-start.html')
    start_generated = start_template.render(subject={'id': 'topic-one'}, spec={
        'id': 'summary', 'generation': {'mode': 'chat'},
    }, state='ready', state_label='Published')
    assert 'data-edit hidden>Edit section</button>' in start_generated
    assert 'data-discard-ai hidden>Discard AI draft</button>' in start_generated
    assert '>Regenerate</button>' not in generated
    assert 'data-section-ai-panel hidden' in generated
    assert '<strong>Agent role</strong>' in generated

    source_owned = template.render(
        **common,
        spec={'id': 'transcript', 'title': 'Transcript', 'generation': {
            'mode': 'none', 'reason': 'Transcripts are media-derived and must not be hallucinated.',
        }},
    )
    assert 'data-generate' not in source_owned
    assert 'data-edit' not in source_owned
    assert 'data-section-ai-panel' not in source_owned
    assert 'must not be hallucinated' in source_owned


def test_prompts_share_idempotent_source_grounded_generation_contract():
    assert all("Generation contract:" not in prompt["instruction"] for prompt in _learn_site.content_tree().prompts)
    specs = {row["id"]: row for row in topic_sections(prompts=_learn_site.content_tree().prompts)}
    for prompt in _learn_site.content_tree().prompts:
        instruction = specs[prompt["id"]]["generation"]["instruction"]
        assert "Rebuild this section from the canonical topic record" in instruction
        assert "do not append" in instruction
        assert "never invent citations" in instruction
        assert "structurally equivalent output" in instruction

def test_production_knowledge_catalog_is_sklearn_19_and_arxiv_free():
    catalog = _learn_site.content_tree().catalog
    topics = [row for row in catalog["subjects"] if row["kind"] == "topic"]
    sources = [row for row in catalog["subjects"] if row["kind"] == "source"]
    problems = [row for row in catalog["subjects"] if row["kind"] == "problem"]
    assert {row["title"] for row in topics} == {
        "Lasso", "Elastic-Net", "Logistic regression", "Bayesian Regression",
        "Generalized Linear Models", "Linear and Quadratic Discriminant Analysis",
        "Support Vector Machines", "Stochastic Gradient Descent", "Nearest Neighbors",
        "Gaussian Processes", "Cross decomposition", "Naive Bayes", "Decision Trees",
        "Feature selection", "Probability calibration", "Isotonic regression",
    }
    assert len(sources) == 12
    assert len(problems) == 8
    assert all(row["url"].startswith("https://scikit-learn.org/1.9/") for row in sources)
    assert "arxiv" not in json.dumps(topics + sources + problems).lower()

@pytest.mark.parametrize(
    'media',
    [
        {
            'image': '/../../private.svg',
            'alt': 'bad',
        },
        {
            'youtube_id': '<script>bad',
        },
        {
            'image': '/_static/example.svg',
        },
        {
            'images': [
                {
                    'image': 'https://example.org/remote.png',
                    'alt': 'remote',
                },
            ],
        },
        {
            'image': '/_static/a.svg',
            'alt': 'A',
            'images': [
                {
                    'image': '/_static/a.svg',
                    'alt': 'duplicate',
                },
            ],
        },
    ],
)
def test_media_fields_do_not_accept_markup_or_arbitrary_paths(media):
    record = topic();
    record.update(kind='whiteboard', url='https://example.org/image', media=media)
    with pytest.raises(LearnValidationError):
        validate_catalog(
            {
                'contract': 'learn.catalog.v3',
                'revision': 'r1',
                'subjects': [record],
            },
        )


def test_whiteboard_gallery_is_ordered_local_media_and_uses_gallery_directive():
    record = topic()
    record.update(
        kind="whiteboard",
        url="https://example.org/image",
        media={
            "image": "/_static/primary.svg", "alt": "Primary diagram", "caption": "Primary",
            "images": [
                {"image": "/_static/detail-1.png", "alt": "Detail one", "caption": "First detail"},
                {"image": "/_static/detail-2.webp", "alt": "Detail two"},
            ],
        },
    )
    normalized = validate_catalog({"contract": "learn.catalog.v3", "revision": "r1", "subjects": [record]})["subjects"][0]
    assert normalized["media"]["image"] == "/_static/primary.svg"
    assert [row["image"] for row in normalized["media"]["images"]] == [
        "/_static/detail-1.png", "/_static/detail-2.webp"
    ]
    assert ".. ai-whiteboard-gallery:: topic-one" in _media_directive(normalized)

def test_trending_topics_template_is_width_responsive_without_forced_horizontal_scroll():
    template_root = EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn'
    template = (template_root/'topic-explorer.html').read_text()
    controls = (template_root/'explorer-controls.html').read_text()
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    assert 'class="learn-trending-table"' in template
    assert 'class="learn-signal-grid"' in template
    assert template.count('data-signal="') == 6
    assert 'learn/explorer-controls.html' in template
    assert 'name="sort"' in controls
    assert 'data-explorer-direction' in controls
    assert 'min-width:980px' not in css
    assert 'container-type:inline-size' in css
    assert '@container learn-topic-explorer (max-width:960px)' in css
    assert '@container learn-topic-explorer (max-width:640px)' in css
    assert 'grid-template-areas:"topic topic" "category added" "signals signals"' in css


def test_problem_source_skill_explorers_share_responsive_table_contract():
    template_root = EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn'
    template = (template_root/'catalog-explorer.html').read_text()
    controls = (template_root/'explorer-controls.html').read_text()
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.js').read_text()
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    assert 'data-catalog-table' in template
    assert 'class="learn-trending-table learn-catalog-table"' in template
    assert 'learn/explorer-controls.html' in template
    assert 'name="timeframe"' in controls and 'name="sort"' in controls
    assert "kind == 'problem'" in template
    assert "kind == 'source'" in template
    assert "kind == 'skill'" in template
    assert 'Open original source ↗' in template
    assert "all(document,'[data-learn-explorer]')" in js
    assert "const tbody=one(explorer,'tbody')" in js
    assert '.learn-topic-explorer,.learn-catalog-explorer,.learn-card-explorer' in css
    assert 'min-width:980px' not in css



def test_prompt_and_skill_switches_use_native_accessible_names_without_theme_hidden_helpers():
    template_root = EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn'
    names = (
        'prompt-group.html', 'prompt-card-start.html', 'prompt-detail-start.html',
        'skill-group.html', 'skill-card-start.html', 'skill-detail-start.html',
    )
    rendered_sources = '\n'.join((template_root/name).read_text() for name in names)
    assert 'class="sr-only"' not in rendered_sources
    assert 'aria-label="Show {{ prompt.title|e }} on topic pages"' in rendered_sources
    assert 'aria-label="Show {{ skill.title|e }} on topic pages"' in rendered_sources
    assert rendered_sources.count('aria-label="Show ') == 6

    env = Environment(loader=FileSystemLoader(str(template_root.parent)), autoescape=False)
    rendered = env.get_template('learn/prompt-card-start.html').render(prompt={
        'id': 'quoted', 'title': 'A & "quoted" <prompt> it\'s', 'author': 'tester',
        'href': '#', 'default_enabled': True,
    })
    assert 'aria-label="Show A &amp; &#34;quoted&#34; &lt;prompt&gt; it&#39;s on topic pages"' in rendered

    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    assert 'label.learn-switch { position:relative;' in css
    assert 'grid-template-columns:1fr 1fr' in css
    assert 'width:76px; min-width:76px; max-width:76px' in css and 'flex:0 0 76px' in css
    assert 'line-height:1.15' in css


def test_skill_library_reuses_topic_prompt_card_visual_contract():
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    skill_start = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/skill-card-start.html').read_text()
    skill_end = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/skill-card-end.html').read_text()
    assert '.learn-prompt-library,.learn-skill-library' in css
    assert 'class="learn-prompt-card"' in skill_start
    assert 'class="learn-switch"' in skill_start
    assert 'View skill' in skill_end
    assert '.learn-prompt-card:hover' in css and '.learn-prompt-card:focus-within' in css
    detail_start = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/skill-detail-start.html').read_text()
    assert 'class="learn-switch"' in detail_start
    assert '.learn-prompt-detail-page { --learn-line:' in css and '--learn-accent:var(--pst-color-primary' in css


def test_media_gallery_cards_have_bounded_surfaces_and_video_embeds():
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    video_card = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/media-card-start.html').read_text()
    video_index = (_learn_site.content_root()/'videos/index.rst').read_text()
    whiteboard_index = (_learn_site.content_root()/'whiteboards/index.rst').read_text()
    assert '.learn-explorer[data-kind=video] .learn-card' in css
    assert '.learn-explorer[data-kind=whiteboard] .learn-card' in css
    assert 'border:1px solid var(--learn-line)' in css
    assert 'media-card-start.html' in pages and 'media-card-end.html' in pages
    assert 'record.get("kind") == "video" and media.get("youtube_id")' in pages
    assert 'record.get("kind") == "whiteboard" and whiteboard_images(record)' in pages
    assert 'image = nodes.image(' in pages
    assert 'uri=preview["image"]' in pages
    assert 'alt=preview["alt"]' in pages
    assert 'directive.state.nested_parse(StringList(lines), 0, card)' in pages  # video still uses native youtube directive parsing
    assert '_rst_text(preview' not in pages
    assert '.learn-whiteboard-card .learn-media-card-content img' in css
    assert ".. youtube:: {media['youtube_id']}" in pages
    assert video_card.index('learn-meta') < video_card.index('<h3>') < video_card.index('learn-media-card-content')
    assert '.. container:: learn-index-actions learn-video-index-actions' in video_index
    assert ':doc:`Create a Video <new>`' in video_index
    topic_index = (_learn_site.content_root()/'topics/index.rst').read_text()
    assert '.. container:: learn-index-actions learn-topic-index-actions' in topic_index
    assert '.learn-index-actions { margin:0 0 1rem; }' in css
    assert '.learn-media-index-actions a' not in css
    assert '.. container:: learn-index-actions learn-whiteboard-index-actions' in whiteboard_index

def test_media_indexes_use_materialized_explorer_and_whiteboard_detail_targets():
    index = (_learn_site.content_root() / "whiteboards/index.rst").read_text()
    assert ".. ai-topic-explorer:: whiteboard" in index
    assert ".. toctree::" in index and ":hidden:" in index
    rel, record = next(
        (rel, row) for rel, row in _learn_site.content_tree().records.items()
        if row["subject"]["kind"] == "whiteboard"
    )
    detail = (_learn_site.content_root() / rel.with_suffix(".rst")).read_text()
    subject_id = record["subject"]["id"]
    assert detail.index(f".. ai-whiteboard-gallery:: {subject_id}") < detail.index(
        f".. ai-media-actions:: {subject_id}"
    )

def test_media_actions_and_gallery_viewer_contract_are_progressive_and_contextual():
    template = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/media-actions.html').read_text()
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.js').read_text()
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    assert 'data-copy-url' in template
    assert 'Generate Variant' in template and 'video_generation_href' in template
    assert 'audio_generation_href' in template and 'document_generation_href' in template
    assert 'whiteboard_generation_href' in template
    assert "subject.kind == 'whiteboard'" in template
    shared = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/overview-actions.html').read_text()
    assert 'context_target' in template and 'learn/overview-actions.html' in template
    assert 'data-bookmark' in shared and 'data-reading' in shared
    assert "contract:'learn.media-request.v1'" in js
    assert '.learn-explorer[data-kind=\"whiteboard\"] .learn-items img' in js
    assert '.learn-whiteboard-index-gallery' not in js
    assert "const whiteboardSelector=" in js
    assert "event.stopPropagation();" in js
    assert "},true);" in js
    assert "all(document,'[data-toc-section]')" in js
    assert 'app.add_css_file("topic.css")' in pages
    assert 'app.add_js_file("topic.js", defer="defer", priority=611)' in pages
    assert 'media_index' not in pages
    assert 'add_inline_videos' in pages and '.. youtube:: {youtube_id}' in pages
    assert '.learn-media-index-gallery' not in css
    assert '.learn-explorer[data-kind=whiteboard] .learn-items' in css
    assert '.learn-whiteboard-openable{position:relative;z-index:3;pointer-events:auto}' in css


def test_video_generation_lifecycle_is_contextual_capability_gated_and_provider_neutral():
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    generated_new = (_learn_site.content_root()/'videos/new.rst').read_text()
    template_root = EXT/'_sphinx_ext/_sphinx_ai_learn/_templates'
    template = (template_root/'learn/video-generation.html').read_text()
    rendered_template = Environment(loader=FileSystemLoader(str(template_root)), autoescape=False).get_template('learn/video-generation.html').render(
        payload='{}', topics=[], sources=[], videos=[], audio=[], documents=[], whiteboards=[], studio_nav=STUDIO_NAV
    )
    overview = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/overview-actions.html').read_text()
    media = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/media-actions.html').read_text()
    section = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/section-end.html').read_text()
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/video-generation.js').read_text()
    assert '.. ai-video-generation::' in generated_new
    assert 'app.add_directive("ai-video-generation", VideoGenerationDirective)' in pages
    assert 'app.add_js_file("video-generation.js", defer="defer", priority=615)' in pages
    assert '"related": row.get("related", [])' in pages
    assert 'Generate Video' in overview and 'video_generation_href' in overview
    assert 'Generate Variant' in media
    assert "spec.id == 'video'" in section and 'video_generation_href' in section
    assert 'data-video-generator' in template
    assert 'data-video-ready-endpoint' in rendered_template
    assert 'data-video-ready-generation' in rendered_template
    assert 'data-video-ready-publishing' in rendered_template
    context_template = (template_root/'learn/generation-context.html').read_text()
    assert 'learn/generation-context.html' in template
    for mode in ('topic', 'source', 'url', 'prompt'):
        assert f'data-generation-context-tab="{{{{ key }}}}"' in context_template or f"('\"{mode}\"')" not in context_template
    assert 'data-generation-context-policy="{{ context_policy|e }}"' in context_template
    assert "const REQUEST_CONTRACT='learn.video-generation-request.v1'" in js
    assert "const JOB_CONTRACT='learn.video-generation-job.v1'" in js
    assert "profile?.video" in js
    assert "cap?.enabled===true" in js
    assert "String(cap?.contract||'')===REQUEST_CONTRACT" in js
    assert "String(cap?.job_contract||'')===JOB_CONTRACT" in js
    assert "runtimeActions=new Set" in js
    assert "runtimeTestMode=cap.test_mode===true" in js
    assert "submit.textContent='Generate Now'" in js
    assert 'Run Test Generation' not in js
    assert "result.test_mode" in js
    assert "runtimeJson(endpoint" in js
    assert "fetch(" not in js
    assert "'Idempotency-Key':idempotencyKey" in js
    assert "provider:String(raw.result.provider||'')" in js
    assert 'Archive this video?' in js
    assert 'onProfileChange' in js
    assert 'learn/generation-authority.html' in template
    assert 'data-generation-authority-picker' in rendered_template
    assert 'data-generation-authority-open' in rendered_template and 'data-generation-authority-more' in rendered_template
    assert 'data-video-model-' not in rendered_template
    assert 'AI_ASSISTANT_MODEL_API' not in js
    assert 'model_selection:modelSnapshot()' in js
    assert 'assistantModelSnapshot' in js
    assert "function signature(){return JSON.stringify({form:formState(),model:modelSnapshot()});}" in js


def test_multimodal_generation_surfaces_share_one_accessible_studio_ux():
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    ui_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/generation-ui.js').read_text()
    template_root = EXT / '_sphinx_ext/_sphinx_ai_learn/_templates'
    env = Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)
    kinds = ('video', 'audio', 'document', 'whiteboard')
    for kind in kinds:
        source = (template_root/f'learn/{kind}-generation.html').read_text()
        assert 'learn-generation-shell' in source
        assert f'data-generation-kind="{kind}"' in source
        assert 'learn/generation-studio-header.html' in source
        assert 'learn/generation-shaping.html' in source
        assert 'learn/generation-advanced.html' in source
        assert 'learn/generation-context.html' in source
        assert "context_policy = 'exclusive'" in source
        assert 'learn-generation-options' in source
        assert 'learn/generation-actions.html' in source
        assert 'learn/generation-status.html' in source
        assert 'learn/generation-library.html' in source
        rendered = env.get_template(f'learn/{kind}-generation.html').render(
            payload='{}', topics=[], sources=[], videos=[], audio=[], documents=[], whiteboards=[], studio_nav=STUDIO_NAV
        )
        assert rendered.count('aria-current="page"') == 1
        assert f'data-{kind}-runtime-status' in rendered
        assert f'data-{kind}-ready-endpoint' in rendered
        assert f'data-{kind}-ready-generation' in rendered
        assert f'data-{kind}-ready-publishing' in rendered
        for target in kinds:
            plural = 'audios' if target == 'audio' else target + 's'
            assert f'../{plural}/new.html' in rendered
    assert 'app.add_js_file("generation-context.js", defer="defer", priority=613)' in pages
    assert 'app.add_js_file("generation-ui.js", defer="defer", priority=606)' in pages
    assert '--learn-gen-bg:var(--pst-color-background,#fff)' in css
    assert 'html[data-theme="dark"] .learn-generation-shell' in css
    assert 'grid-template-columns:repeat(auto-fit,minmax(min(100%,12rem),1fr))' in css
    assert '.learn-generation-context-tabs>button[aria-selected="true"]' in css
    assert '.learn-generation-studio-nav a[aria-current="page"]' in css
    assert '.learn-generation-runtime-readiness>span[data-state="ready"]' in css
    assert "root.querySelectorAll('[data-generation-chip]')" in ui_js
    assert "root.querySelectorAll('[data-generation-countable]')" in ui_js
    assert "field.dispatchEvent(new Event('input', {bubbles:true}))" in ui_js
    assert "bindAssistantAuthorities(document)" in ui_js
    assert 'cleanPublicReferenceUrl' in ui_js
    assert 'readLensProfile' in ui_js and 'restoreLensProfile' in ui_js
    assert 'validateLensProfile' in ui_js and 'withLensGuidance' in ui_js
    assert 'setReadiness' in ui_js
    assert "current.openPicker(open)" in ui_js
    assert "current.selectModel(id)" in ui_js
    assert "learn-generation-authority-menu-item" in ui_js
    assert 'bindGenerationStatus' in ui_js
    assert "root.querySelector('[data-generation-status]')" in ui_js
    assert "idle:'Ready', working:'Working', success:'Complete', warning:'Attention', error:'Error'" in ui_js
    assert 'bindRequestActions' in ui_js
    assert 'bindPrivateLibrary' in ui_js
    assert "root.querySelector('[data-generation-library]')" in ui_js
    assert "section.querySelector('[data-generation-library-filter]')" in ui_js
    assert "section.querySelector('[data-generation-library-refresh]')" in ui_js
    assert 'learn-generation-library-card' in ui_js
    assert 'Archive is available when generation finishes.' in ui_js
    assert "root.querySelector('[data-generation-save-draft]')" in ui_js
    assert "root.querySelector('[data-generation-copy-request]')" in ui_js
    assert "Draft saved in this browser only." in ui_js
    assert "Generation request copied. No network request was sent." in ui_js
    actions = (template_root/'learn/generation-actions.html').read_text()
    assert 'learn-generation-action-bar' in actions
    assert '>Generate Now</button>' in actions
    assert '>Save draft</button>' in actions
    assert '>Copy request</button>' in actions
    assert 'data-generation-save-draft' in actions and 'data-generation-copy-request' in actions
    assert 'data-generation-submit data-{{ generation_kind|e }}-submit disabled' not in actions
    status_template = (template_root/'learn/generation-status.html').read_text()
    assert 'learn-generation-activity' in status_template
    assert 'data-generation-status-label' in status_template
    assert 'data-generation-status-message' in status_template
    assert 'aria-atomic="true"' in status_template
    shaping_template = (template_root/'learn/generation-shaping.html').read_text()
    assert 'learn-generation-shaping' in shaping_template
    assert 'data-generation-chip' in shaping_template
    assert "('Compare'," in shaping_template and "('Limitations'," in shaping_template
    advanced_template = (template_root/'learn/generation-advanced.html').read_text()
    assert 'learn-generation-advanced-body' in advanced_template
    assert 'learn-generation-check' in advanced_template
    assert 'data-audio-pacing' in advanced_template
    assert 'data-document-structure' in advanced_template
    assert 'data-whiteboard-quality' in advanced_template
    library_template = (template_root/'learn/generation-library.html').read_text()
    assert 'Private runtime library' in library_template
    assert 'data-generation-library-filter' in library_template
    assert 'data-generation-library-refresh' in library_template
    assert 'data-generation-library-grid' in library_template
    assert 'data-generation-library-empty' in library_template
    assert 'data-generation-library-status' in library_template
    assert '>Active</option>' in library_template and '>Archived</option>' in library_template and '>All</option>' in library_template
    assert '.learn-generation-shaping textarea' in css
    assert '.learn-generation-advanced-body' in css
    assert '.learn-generation-advanced-grid' in css
    assert '.learn-generation-check' in css
    assert '.learn-video-check' not in css
    assert '.learn-video-advanced' not in css
    assert '.learn-generation-action-bar' in css
    assert '.learn-generation-activity' in css
    assert '.learn-generation-activity[data-state="working"]' in css
    assert '.learn-generation-activity[data-state="success"]' in css
    assert '.learn-generation-activity[data-state="warning"]' in css
    assert '.learn-generation-activity[data-state="error"]' in css
    assert '@media (prefers-reduced-motion:reduce)' in css
    assert '.learn-generation-library' in css
    assert '.learn-generation-library-grid' in css
    assert '.learn-generation-library-card' in css
    assert '.learn-generation-library-menu' in css
    assert '.learn-generation-secondary' in css
    assert 'background:var(--learn-gen-surface)' in css
    authority = (template_root/'learn/generation-authority.html').read_text()
    assert 'learn-generation-authority-picker' in authority
    assert 'learn-generation-authority-main' in authority
    assert 'learn-generation-authority-more' in authority
    assert 'learn-generation-authority-menu' in authority
    assert 'data-generation-runtime-authority' in authority
    assert 'data-generation-authority-context' in authority
    assert 'authorityContext' in ui_js
    assert '.learn-generation-authority-picker' in css
    assert '.learn-generation-authority-main' in css
    assert '.learn-generation-authority-menu-item' in css
    rendered_by_kind = {}
    for kind in kinds:
        source = (template_root/f'learn/{kind}-generation.html').read_text()
        assert 'learn/generation-authority.html' in source
        rendered = env.get_template(f'learn/{kind}-generation.html').render(
            payload='{}', topics=[], sources=[], videos=[], audio=[], documents=[], whiteboards=[], studio_nav=STUDIO_NAV
        )
        rendered_by_kind[kind] = rendered
        assert rendered.count('learn-generation-step-index') == 4
        assert rendered.count('data-generation-shaping=') == 1
        assert rendered.count('data-generation-advanced=') == 1
        advanced_html = rendered.split('data-generation-advanced=', 1)[1].split('</details>', 1)[0]
        assert advanced_html.count('<select') == 1
        assert advanced_html.count('type="checkbox"') == 2
        assert rendered.count('learn-generation-prompt-chips') == 1
        assert rendered.count('Shape the explanation') == 1
        assert rendered.count('Choose AI lenses') == 1
        assert rendered.count('data-ai-lens-group=') == 4
        assert rendered.count('data-ai-lens-required="true"') == 2
        assert 'learn-generation-step-index">4</span><span>Output settings' in rendered
        for label in ('Beginner-friendly', 'Technical', 'Examples', 'Intuition', 'Compare', 'Limitations'):
            assert f'>{label}</button>' in rendered
        assert rendered.count('data-generation-counter') == 1
        assert rendered.count('learn-generation-action-bar') == 1
        assert rendered.count('>Generate Now</button>') == 1
        assert rendered.count('>Save draft</button>') == 1
        assert rendered.count('>Copy request</button>') == 1
        assert rendered.count('     data-generation-status\n') == 1
        assert rendered.count('data-generation-status-label') == 1
        assert rendered.count('data-generation-status-message') == 1
        assert f'data-{kind}-status' in rendered
        assert 'data-state="idle"' in rendered
        assert f'data-{kind}-submit' in rendered
        assert f'data-{kind}-save-draft' in rendered
        assert f'data-{kind}-copy-request' in rendered
        assert rendered.count('learn-generation-authority-picker') == 1
        order_tokens = ['learn-generation-step-index">1', 'data-generation-shaping=', 'data-generation-lenses=', 'learn-generation-step-index">4', 'data-generation-advanced=', 'learn-generation-authority-picker', 'learn-generation-action-bar']
        positions = [rendered.index(token) for token in order_tokens]
        assert positions == sorted(positions)
        action_html = rendered.split('learn-generation-action-bar', 1)[1].split('</div>', 1)[0]
        assert 'Generate Now</button>' in action_html and 'disabled' not in action_html
        assert rendered.count('learn-generation-library"') == 1
        assert rendered.count('data-generation-library-filter') == 1
        assert rendered.count('data-generation-library-refresh') == 1
        assert rendered.count('data-generation-library-grid') == 1
        assert rendered.count('data-generation-library-empty') == 1
        assert rendered.count('data-generation-library-status') == 1
        assert '>Active</option>' in rendered and '>Archived</option>' in rendered and '>All</option>' in rendered
        assert rendered.count('learn-generation-authority-main') == 1
        assert rendered.count('learn-generation-authority-more') == 1
        assert rendered.count('learn-generation-authority-menu') == 1
        assert rendered.count('learn-generation-model-readout') == 1
        assert rendered.count('Generation authority') == 1
        assert 'data-generation-authority-open' in rendered
        assert 'data-generation-authority-more' in rendered
        assert 'data-generation-authority-menu' in rendered
        assert 'data-generation-runtime-authority' in rendered
    assert 'data-generation-authority-label' in rendered_by_kind['video'] and 'data-video-runtime-authority' in rendered_by_kind['video']
    assert 'data-generation-authority-label' in rendered_by_kind['audio'] and 'data-audio-generator-label' in rendered_by_kind['audio']
    assert 'data-generation-authority-label' in rendered_by_kind['document'] and 'data-document-runtime-authority' in rendered_by_kind['document']
    assert 'data-generation-authority-label' in rendered_by_kind['whiteboard'] and 'data-whiteboard-runtime-authority' in rendered_by_kind['whiteboard']
    assert 'data-generation-authority-kind="assistant"' in rendered_by_kind['whiteboard']
    assert 'data-generation-authority-managed="shared"' in rendered_by_kind['whiteboard']
    assert 'data-whiteboard-generator-menu' not in rendered_by_kind['whiteboard']
    for kind in ('audio', 'document', 'whiteboard'):
        source = (template_root/f'learn/{kind}-generation.html').read_text()
        assert 'learn/generation-context.html' in source
        assert "context_policy = 'exclusive'" in source
        assert 'context_prompt_placeholder' in source
    for kind in kinds:
        js = (EXT/f'_sphinx_ext/_sphinx_ai_learn/_static/{kind}-generation.js').read_text()
        assert 'discoverySerial' in js
        assert 'bindGenerationStatus' in js
        assert 'bindRequestActions' in js
        assert 'requestActions' in js
        assert 'submit.disabled = !runtimeEnabled' not in js
        assert 'submit.disabled=!runtimeEnabled' not in js
        assert 'readLensProfile' in js
        assert 'restoreLensProfile' in js
        assert 'validateLensProfile' in js
        assert 'lenses:' in js
        if kind == 'video':
            assert '[data-generation-library-refresh]' in js
            assert '[data-generation-library-grid]' in js
            assert '[data-generation-library-filter]' in js
        else:
            assert 'bindPrivateLibrary' in js
            assert 'library.upsert' in js
    video_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/video-generation.js').read_text()
    assert 'withLensGuidance?.(state.instructions,state.lenses,4000)' in video_js
    assert '[data-video-chip]' not in video_js
    audio_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/audio-generation.js').read_text()
    assert 'function advancedGuidance()' in audio_js
    assert 'data-audio-pacing' in audio_js and 'expand_acronyms' in audio_js and 'verbalize_uncertainty' in audio_js
    whiteboard_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/whiteboard-generation.js').read_text()
    assert 'selectedGeneratorId' not in whiteboard_js
    assert 'data-whiteboard-generator-menu' not in whiteboard_js
    assert "generator_id:generator ? String(generator.id || '') : ''" in whiteboard_js
    assert 'compatibleGenerators.find' in whiteboard_js
    assert 'server-owned Image renderer' in whiteboard_js
    assert "quality:String(quality && quality.value || 'auto')" in whiteboard_js
    assert 'include_legend' in whiteboard_js and 'number_flow' in whiteboard_js
    document_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/document-generation.js').read_text()
    assert 'function advancedGuidance()' in document_js
    assert 'include_glossary' in document_js and 'include_applications' in document_js
    assert "authority('Chat-backed · Ready')" in document_js


def test_shared_detail_actions_and_collections_are_kind_agnostic():
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.js').read_text()
    template = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/overview.html').read_text()
    shared = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/overview-actions.html').read_text()
    media = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/media-actions.html').read_text()
    assert 'learn/overview-actions.html' in template and 'learn/overview-actions.html' in media
    assert 'data-reading' in shared and 'data-bookmark' in shared
    assert 'data-overview-generate' in shared and 'Generate AI Overview' in shared and 'data-export-page' not in shared
    section_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/section-generation.js').read_text()
    overview_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/overview-generation.js').read_text()
    assert 'AI Learn page overview' in overview_js and "contract:'scikitplot-chat-v1'" in overview_js
    assert "page_descriptor:`AI Learn ${subject.kind||'record'} section draft" in section_js
    assert "context:{page_text:contextFor" in section_js
    assert "record.kind==='topic'&&getState(record).reading" not in js
    assert "getState(record).reading===data.collection_name" in js


def test_optional_detail_roots_cannot_abort_shared_actions_or_whiteboard_viewer():
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.js').read_text()
    assert "const one = (root, selector) => root?.querySelector?.(selector) || null;" in js
    assert "const all = (root, selector) => root?.querySelectorAll ? [...root.querySelectorAll(selector)] : [];" in js
    assert "AI Learn: detail-page interaction initialization failed." in js
    assert "const mediaActions=one(page,'[data-media-actions]')" in js
    assert "const overview=one(page,'[data-overview]')" not in js


def test_whiteboard_viewer_uses_single_click_focal_zoom_and_drag_does_not_toggle():
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.js').read_text()
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    assert "function setZoomAt(value,clientX,clientY)" in js
    assert "image.addEventListener('click',event=>{" in js
    assert "scale===1?2:1" in js
    assert "image.addEventListener('dblclick'" not in js
    assert "if(moved){moved=false;return;}" in js
    assert "setZoomAt(scale+(event.deltaY<0?.25:-.25),event.clientX,event.clientY)" in js
    assert "cursor:zoom-in" in css


def test_learn_page_assets_are_registered_before_html_page_context():
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    setup = pages[pages.index('def setup_pages(app):'):]
    context = pages[pages.index('def page_context('):pages.index('def setup_pages(app):')]
    assert 'app.add_css_file("topic.css")' in setup
    assert 'app.add_js_file("section-generation.js", defer="defer", priority=609)' in setup
    assert 'app.add_js_file("topic.js", defer="defer", priority=611)' in setup
    assert 'app.add_js_file("generation-ui.js", defer="defer", priority=606)' in setup
    assert 'app.add_css_file("topic.css")' not in context
    assert 'app.add_js_file("topic.js"' not in context


def test_user_library_template_resolves_namespaced_grid_partial_like_sphinx():
    template_root = EXT / '_sphinx_ext/_sphinx_ai_learn/_templates'
    env = Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)
    template = env.get_template('learn/user-library.html')
    rendered = template.render(
        mode='bookmarks',
        site_id='test-site',
        payload='{}',
        collections_href='../collections/index.html',
    )
    assert 'data-library-grid-control' in rendered
    assert rendered.count('data-library-grid=') == 4


def test_saved_library_grid_density_is_shared_persisted_and_responsive():
    template = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/user-library.html').read_text()
    control = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/user-library-grid-control.html').read_text()
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.js').read_text()
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    assert template.count('data-grid-columns="2"') == 2
    assert 'data-library-grid-control' in control
    assert 'for columns in (2, 3, 4, 5)' in control
    assert 'data-library-grid="{{ columns }}"' in control
    assert "const gridKey=prefix+'library-grid-columns'" in js
    assert "localStorage.setItem(gridKey,columns)" in js
    assert "validGridColumns=new Set(['2','3','4','5'])" in js
    assert 'container-name:learn-user-library' in css
    assert '@container learn-user-library (min-width:48rem)' in css
    assert '@container learn-user-library (min-width:64rem)' in css
    assert '@container learn-user-library (min-width:78rem)' in css
    assert '@container learn-user-library (max-width:35.99rem)' in css
    library_css = css[css.index('/* Browser-local bookmarks and reading collections. */'):css.index('/* Static-first responsive catalog tables.') ]
    assert 'repeat(auto-fit' not in library_css
    assert '.learn-prompt-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr));' in css


def test_all_catalog_explorers_use_live_compact_shared_controls():
    template_root = EXT / '_sphinx_ext/_sphinx_ai_learn/_templates'
    topic = (template_root/'learn/topic-explorer.html').read_text()
    catalog = (template_root/'learn/catalog-explorer.html').read_text()
    cards = (template_root/'learn/explorer-start.html').read_text()
    controls = (template_root/'learn/explorer-controls.html').read_text()
    primary = (template_root/'learn/explorer-search-primary.html').read_text()
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.js').read_text()
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    for template in (topic, catalog, cards):
        assert 'learn/explorer-controls.html' in template
    assert 'learn/explorer-search-primary.html' in controls
    assert 'data-explorer-filter-options' in controls
    assert 'name="timeframe"' in controls
    assert 'data-explorer-sort' in controls
    assert 'data-explorer-direction' in controls
    assert 'data-explorer-reset' in controls
    assert 'data-explorer-search' in primary
    assert 'learn-search-primary-row' in primary
    assert 'learn-search-field' in primary
    assert 'learn-search-input' in primary
    assert 'class="learn-visually-hidden"' in primary
    assert 'class="sr-only"' not in primary
    assert 'aria-label="Search"' in primary
    assert 'maxlength="1024"' in primary
    assert 'aria-haspopup' not in primary
    assert 'M10 3a7 7 0 1 0 0 14' in primary
    assert 'data-explorer-filter-toggle' in primary
    assert 'aria-expanded="false"' in primary
    assert 'data-explorer-search-variant' in primary
    assert 'learn-search-primary-row--pill-overflow' in primary
    assert 'learn-filter-disclosure--overflow' in primary
    assert '<circle cx="12" cy="5" r="1.8"></circle>' in primary
    assert 'learn-filter-disclosure--chevron' in primary
    assert '<polyline points="6 9 12 15 18 9"></polyline>' in primary
    assert "function setFilterOptions(expanded)" in js
    assert "filterOptions.hidden=!expanded" in js
    assert "hasAdvancedState(params)" in js
    assert "form.addEventListener('submit'" in js
    assert "form.addEventListener('input',event=>{if(event.target===form.elements.sort||event.isComposing)return;apply(true);})" in js
    assert "form.elements.q?.addEventListener('compositionend',()=>apply(true))" in js
    assert "if(!tbody){" in js
    assert '.learn-search-primary-row { display:grid; grid-template-columns:minmax(0,1fr) 2.75rem; gap:.42rem; align-items:end;' in css
    assert '.learn-trending-controls .learn-visually-hidden { position:absolute!important;' in css
    assert '.learn-search-primary-row--pill-overflow { grid-template-columns:minmax(0,1fr) 2.5rem; }' in css
    assert '.learn-search-field { display:grid; grid-template-columns:minmax(0,1fr) 2.7rem;' in css
    assert '.learn-search-field--pill { border-radius:999px; }' in css
    assert '.learn-trending-controls .learn-search-submit svg { width:1.12rem; height:1.12rem; fill:currentColor; }' in css
    assert 'height:2.5rem; min-height:2.5rem; box-sizing:border-box;' in css
    assert 'align-self:end; display:inline-flex;' in css
    assert '.learn-trending-controls .learn-filter-disclosure--overflow { width:2.5rem; min-width:2.5rem; border-radius:999px; }' in css
    assert '.learn-trending-controls .learn-filter-disclosure--chevron[aria-expanded="true"] svg { transform:rotate(180deg); }' in css
    assert '.learn-filter-options[hidden] { display:none!important; }' in css
    assert '.learn-filters {' not in css

def test_explorer_search_variant_is_shared_config_not_a_second_controller():
    sphinx = (EXT/'_sphinx_ext/_sphinx_ai_learn/_sphinx.py').read_text()
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    readme = (EXT/'_sphinx_ext/_sphinx_ai_learn/README.md').read_text()
    conf = (_learn_site.docs_source()/'conf.py').read_text()
    assert 'app.add_config_value("ai_learn_explorer_search_variant", "pill-overflow", "env")' in sphinx
    assert "ai_learn_explorer_search_variant must be 'pill-overflow' or 'classic'" in sphinx
    assert '"search_control_variant": search_control_variant' in pages
    assert 'resolve_search_variant(' in pages
    assert '"search-variant": search_variant_option' in pages and '"search_variant": search_variant_option' in pages
    assert 'activation_keys=()' in pages
    assert 'ai_learn_explorer_search_variant = "pill-overflow"  # alternative: "classic"' in conf
    assert '`pill-overflow` (default)' in readme
    assert '`classic` retains the rounded-rectangle field' in readme


def test_explorer_search_partial_resolves_from_sphinx_template_root():
    template_root = EXT / '_sphinx_ext/_sphinx_ai_learn/_templates'
    env = Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)
    template = env.get_template('learn/explorer-search-primary.html')
    rendered = template.render(search_placeholder='Topic or keyword', filter_panel_id='learn-topic-filter-options', search_input_id='learn-topic-search')
    assert 'placeholder="Topic or keyword"' in rendered
    assert 'aria-controls="learn-topic-filter-options"' in rendered
    assert 'id="learn-topic-search"' in rendered
    assert 'for="learn-topic-search"' in rendered
    assert 'aria-label="Search"' in rendered
    assert 'data-explorer-search-variant="pill-overflow"' in rendered
    assert 'learn-search-field--pill' in rendered
    assert 'learn-filter-disclosure--overflow' in rendered
    assert 'learn-filter-overflow-icon' in rendered
    assert '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' in rendered

    classic = template.render(search_placeholder='Topic or keyword', filter_panel_id='learn-topic-filter-options', search_input_id='learn-topic-search', search_control_variant='classic')
    assert 'data-explorer-search-variant="classic"' in classic
    assert 'learn-search-primary-row--classic' in classic
    assert 'learn-search-field--pill' not in classic
    assert 'learn-filter-disclosure--chevron' in classic
    assert '<polyline points="6 9 12 15 18 9"></polyline>' in classic


def test_audio_topic_and_detail_contracts_are_materialized_without_autoplay():
    audio = {
        "id": "audio-one",
        "kind": "audio",
        "title": "Audio One",
        "created_at": "2026-09-22T00:00:00Z",
        "media": {"type": "audio", "src": "/_static/learn/audio/one.mp3", "mime_type": "audio/mpeg"},
        "related": [],
        "sections": [],
    }
    normalized = validate_catalog({"contract": "learn.catalog.v3", "revision": "a", "subjects": [audio]})["subjects"][0]
    assert ".. ai-audio-player:: audio-one" in _media_directive(normalized)
    topic_rel, _ = next((rel, row) for rel, row in _learn_site.content_tree().records.items() if row["subject"]["kind"] == "topic")
    assert "Audio Explanation" in (_learn_site.content_root() / topic_rel.with_suffix(".rst")).read_text()
    template = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/audio-player.html').read_text()
    assert '<audio controls preload="metadata"' in template
    assert 'autoplay' not in template
    generation = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/audio-generation.js').read_text()
    assert 'X-Artifact-Capability' in generation
    assert 'X-Generation-Capability' in generation
    assert 'URL.createObjectURL' in generation

def test_document_and_whiteboard_generation_surfaces_are_first_class_and_fail_closed():
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    document_new = (_learn_site.content_root()/'documents/new.rst').read_text()
    whiteboard_new = (_learn_site.content_root()/'whiteboards/new.rst').read_text()
    overview = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/overview-actions.html').read_text()
    section = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/section-end.html').read_text()
    document_template = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/document-generation.html').read_text()
    whiteboard_template = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/whiteboard-generation.html').read_text()
    document_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/document-generation.js').read_text()
    whiteboard_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/whiteboard-generation.js').read_text()
    assert '.. ai-document-generation::' in document_new
    assert '.. ai-whiteboard-generation::' in whiteboard_new
    assert 'app.add_directive("ai-document-generation", DocumentGenerationDirective)' in pages
    assert 'app.add_directive("ai-whiteboard-generation", WhiteboardGenerationDirective)' in pages
    assert 'app.add_directive("ai-document-viewer", DocumentViewerDirective)' in pages
    assert 'Generate Document' in overview and 'Generate Whiteboard' in overview
    assert "spec.id == 'document'" in section and 'document_generation_href' in section
    assert "spec.id == 'whiteboard'" in section and 'whiteboard_generation_href' in section
    assert 'data-document-generation' in document_template
    assert 'assistant.document-generation-request.v1' in document_js
    assert "base + '/v1/document'" in document_js
    assert 'data-whiteboard-generation' in whiteboard_template
    assert 'provider_artifact_output' in whiteboard_js
    assert "g.kind === 'image'" in whiteboard_js
    assert "base + '/v1/image'" in whiteboard_js
    assert 'generator_id' in whiteboard_js


def test_page_ai_overview_is_record_aware_chat_backed_and_not_topic_ambiguous():
    template = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/overview-actions.html').read_text()
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/overview-generation.js').read_text()
    shared = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/text-generation-ui.js').read_text()
    env = Environment(loader=FileSystemLoader(str(EXT/'_sphinx_ext/_sphinx_ai_learn/_templates')), autoescape=False)
    lenses = env.get_template('learn/ai-lens-profile.html').render(lens_scope='overview', lens_variant='overview')
    assert 'Generate AI Overview' in template
    assert 'Generate Topic Overview' not in template
    assert 'not connected yet' not in template
    assert 'data-overview-source=' in template
    assert lenses.count('data-overview-audience=') >= 9 and 'data-overview-purpose=' in lenses
    assert 'data-overview-audience="young-learner"' in lenses
    assert 'data-overview-audience="decision-maker"' in lenses
    assert 'data-overview-skill=' in lenses and 'data-overview-role=' in lenses
    assert 'data-overview-depth' in template
    assert '<option value="concise">Concise</option><option value="balanced" selected>Balanced</option><option value="deep">Deep</option>' in template
    assert 'one model · bounded lenses'.lower() in template.lower()
    assert "learn.page-overview-draft.v1" in js
    assert "runtime?.createWorkflow?." in js and "AI Learn page overview" in js
    assert "run?.addEventListener('click',()=>workflowAction('generate'))" in js
    assert 'runtime.run' not in js and 'const response=await runRequest(request' in shared
    assert 'External URLs are references only' in template
    assert "const transport=generationUi()?.fetchJson" in shared
    assert "label:'AI text generation'" in shared and 'maxBytes:1024*1024' in shared
    assert 'data-export-page' not in template
    assert "openButton.textContent='Review AI Overview'" in js
    assert "run.textContent='Regenerate AI Overview'" in js
    assert "restoreLensProfile?.(panel,'overview'" in js
    assert "[data-overview-copy-result]" in js and "[data-overview-download]" in js
    assert "new Blob([JSON.stringify(draft,null,2)]" in js
    assert 'contexts:contextState.contexts' in js and 'depth:contextState.depth' in js and 'guidance:contextState.guidance' in js
    assert '`Depth: ${depth}.`' in js
    assert "function maxTokens(depth){return depth==='concise'?900:depth==='deep'?2800:1800;}" in js


def test_section_and_overview_share_one_text_generation_workflow_kernel():
    shared = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/text-generation-ui.js').read_text()
    section = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/section-generation.js').read_text()
    overview = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/overview-generation.js').read_text()
    panel = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/section-ai-generation.html').read_text()

    assert 'function createWorkflow(options)' in shared
    assert 'function canonicalRequest(input)' in shared and 'async function runRequest(input' in shared
    assert "max_tokens must be an integer from 256 to 32000" in shared
    assert "typeof rawMaxTokens!=='number'||!Number.isInteger(rawMaxTokens)||rawMaxTokens<256||rawMaxTokens>32000" in shared
    assert 'return canonicalRequest(request)' in shared
    assert 'const response=await runRequest(request' in shared
    for primitive in ('copyRequest','generate','cancel','publish','sync'):
        assert primitive in shared
    assert 'validateLensProfile' in shared
    assert 'submitPublication' in shared
    assert 'captureRunState' in shared  # section multi-tab compare-and-save remains adapter state
    assert 'runtime?.createWorkflow?.({' in section
    assert 'runtime?.createWorkflow?.({' in overview
    assert "buildRequest:requestBody" in section and "buildRequest:request" in overview
    assert "createDraft:({response,request,profile,startState})" in section
    assert "createDraft:({response,profile,startState})" in overview
    assert 'fetch(' not in section and 'fetch(' not in overview
    assert 'runtime.run' not in section and 'runtime.run' not in overview
    assert "readProfile:draftProfile" in section
    assert "captureRunState:()=>({storageRaw:loadedRaw,instructions:instructions()})" in section
    assert "captureRunState:()=>({...selectedContextState(),storageRaw:loadedRaw})" in overview
    assert "request=requestFor(profile,startState)" in shared
    assert "options.buildRequest(profile,startState)" in shared
    assert "function requestBody(profile=draftProfile(),startState)" in section
    assert "function request(profile=selectedProfile(),contextState=selectedContextState())" in overview
    assert "buildMessage(profile,contextState)" in overview and "buildContext(contextState)" in overview
    assert "save(next,context.startState?.storageRaw)" in section
    assert "profile:startState?.profile" not in section
    assert "const payload=response.payload||{},now=new Date().toISOString()" in section
    assert "const contextState=startState||selectedContextState()" in overview
    assert "profile,contexts:contextState.contexts" in overview
    assert "function safeDraft(value)" in overview
    assert "Browser storage is unavailable; the generated overview was not saved." in overview
    assert "This AI overview draft changed in another tab. Reload before replacing it." in overview
    assert "This AI overview draft changed in another tab. Reload before discarding it." in overview
    assert "saveDraft:(next,context)=>saveDraft(next,context.startState?.storageRaw)" in overview
    assert "Browser storage is unavailable; the AI section draft was not saved." in section
    assert "draft=safeDraft(JSON.parse(loadedRaw))" in overview
    assert "const transport=generationUi()?.fetchJson" in shared
    assert "label:'AI text generation'" in shared and 'maxBytes:1024*1024' in shared
    assert "if(runButton)runButton.disabled=running||publishing" in shared
    assert "button.disabled=running||publishing||unavailable" in shared
    assert "if(running||publishing)return false" in shared
    assert "if(publishing||running)return false" in shared
    assert "await options.saveDraft(draft,context)" in shared
    assert 'learn/text-generation-actions.html' in panel
    assert 'data-section-ai-publish disabled' in _render_text_generation_actions('section-ai', 'Generate Now')


def test_evidence_review_controls_are_local_non_verifying_and_feed_ai_context():
    template = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/evidence-review.html').read_text()
    end = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/section-end.html').read_text()
    evidence_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/evidence-review.js').read_text()
    section_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/section-generation.js').read_text()
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    assert 'Use in AI context' in template
    for value in (
        'not-reviewed',
        'supports',
        'partial',
        'challenges',
        'unclear',
    ):
        assert f'value="{value}"' in template
    assert 'not verification' in template.lower() or 'do not verify' in template.lower()
    assert 'learn/evidence-review.html' in end and 'data-verify>Review sources' in end
    assert 'Add evidence' in end and 'source_creation_href' in end
    assert "learn-evidence-review:v1:" in evidence_js
    assert 'contextFor(page,sectionId)' in evidence_js
    assert 'AI_LEARN_EVIDENCE_API' in section_js
    assert 'Reviewer-selected catalog evidence' in section_js
    assert "Array.isArray(raw?.citations)?raw.citations:[]" in section_js
    assert "for(const row of rawSections)" not in section_js
    assert 'evidence_sources' in pages
    assert '"id": source["id"]' in pages and '"title": source["title"]' in pages
    assert "const verify=one(section,'[data-verify]')" not in section_js
    assert "verify?.addEventListener" not in section_js
    assert "spec.generation.mode|default('none') == 'chat' or spec.id in ['evidence', 'references']" in end




def test_shared_generation_context_component_supports_composable_and_exclusive_policies():
    template_root = EXT/'_sphinx_ext/_sphinx_ai_learn/_templates'
    source = (template_root/'learn/generation-context.html').read_text()
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/generation-context.js').read_text()
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    assert 'data-generation-context-picker' in source
    assert 'Topic' in source and 'Source' in source and 'URL' in source and 'Prompt' in source
    assert "context_input_type = 'checkbox' if context_policy == 'composable' else 'radio'" in source
    assert 'data-generation-context-summary' in source and 'data-generation-context-tab-count' in source
    assert 'aria-controls="learn-context-{{ context_scope|e }}-{{ key }}-panel"' in source
    assert 'aria-labelledby="learn-context-{{ context_scope|e }}-topic-tab"' in source
    assert 'aria-required="true"' in source
    assert '{% if context_url_required|default(false) %}required{% endif %}' not in source
    assert '{% if context_prompt_required|default(false) %}required{% endif %}' not in source
    assert "policy=root.dataset.generationContextPolicy==='composable'?'composable':'exclusive'" in js
    assert 'selectedIds' in js and 'selectId' in js and 'applyQuery' in js
    assert '.learn-generation-context-tabs' in css and '.learn-generation-context-list' in css
    assert 'app.add_js_file("generation-context.js", defer="defer", priority=613)' in pages

def test_record_creation_studios_are_shared_private_and_source_safe():
    template = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/record-generation.html').read_text()
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/record-generation.js').read_text()
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    assert 'data-record-generation' in template
    assert 'Choose context' in template and 'Shape the draft' in template and 'Choose AI lenses' in template
    assert 'learn/generation-context.html' in template and "context_policy = 'composable'" in template
    assert 'data-record-generation-form novalidate' in template
    assert 'learn/generation-authority.html' in template
    assert "authority_kind = 'assistant'" in template and "authority_managed = 'shared'" in template
    assert "authority_runtime_label = 'Draft runtime'" in template
    assert 'data-record-model-open' not in template and 'data-record-model-label' not in template
    env = Environment(loader=FileSystemLoader(str(EXT/'_sphinx_ext/_sphinx_ai_learn/_templates')), autoescape=False)
    for creation_kind in (
        'topic',
        'source',
        'problem',
        'topic-prompt',
        'skill',
    ):
        rendered = env.get_template('learn/record-generation.html').render(
            creation_kind=creation_kind, payload='{}', site_id='test-site', revision='test-revision', topics=[], sources=[], studio_nav=STUDIO_NAV
        )
        assert rendered.count('learn-generation-authority-picker') == 1
        assert rendered.count('data-generation-authority-open') == 1
        assert rendered.count('data-generation-authority-more') == 1
        assert rendered.count('data-generation-authority-menu') == 1
        assert 'data-generation-authority-kind="assistant"' in rendered
        assert 'data-generation-authority-managed="shared"' in rendered
        assert '>Draft runtime</span>' in rendered
        assert 'data-generation-runtime-authority' in rendered
        assert 'Try a different model' in rendered
        assert 'data-generation-authority-context="Drafts stay private in this studio; publication remains a separate reviewed handoff."' in rendered
        assert rendered.count('data-generation-advanced="record"') == 1
        advanced_html = rendered.split('data-generation-advanced="record"', 1)[1].split('</details>', 1)[0]
        assert advanced_html.count('<select') == 1
        assert advanced_html.count('type="checkbox"') == 2
        assert f'Advanced {({"topic":"Topic","source":"Source","problem":"Open Problem","topic-prompt":"Topic Prompt","skill":"Skill"})[creation_kind]} options' in rendered
        order_tokens = ['Choose AI lenses', 'data-generation-advanced="record"', 'learn-generation-authority-picker', 'learn-generation-action-bar']
        positions = [rendered.index(token) for token in order_tokens]
        assert positions == sorted(positions)
        action_html = rendered.split('learn-generation-action-bar', 1)[1].split('</div>', 1)[0]
        assert 'Generate Now</button>' in action_html and 'disabled' not in action_html
        assert 'class="learn-generation-form learn-record-generation-form"' in rendered
        assert rendered.count('data-generation-library ') == 1
        assert rendered.count('data-generation-library-filter') == 1
        assert rendered.count('data-generation-library-refresh') == 1
        assert rendered.count('data-generation-library-grid') == 1
        assert rendered.count('data-generation-library-empty') == 1
        assert rendered.count('data-generation-library-status') == 1
        expected_title = {
            'topic': 'Your Topics',
            'source': 'Your Sources',
            'problem': 'Your Open Problems',
            'topic-prompt': 'Your Topic Prompts',
            'skill': 'Your Skills',
        }[creation_kind]
        assert f'>{expected_title}</h2>' in rendered
        assert 'Private AI draft library' in rendered
        assert rendered.index('learn-generation-action-bar') < rendered.index('data-generation-library')
    assert 'learn/generation-library.html' in template
    assert "bindPrivateLibrary?.(root" in js
    assert "learn-record-studio:v2:${data.site_id||'default'}:${kind}:library" in js
    assert 'rememberGeneratedDraft(draft,response)' in js
    assert 'Metadata-only receipt.' in js
    context_template = (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/generation-context.html').read_text()
    assert 'data-generation-context-record' in context_template
    assert "context_policy = 'composable'" in template
    assert rendered.count('data-record-audience=') >= 9 and 'data-record-purpose=' in rendered
    assert 'data-record-audience="young-learner"' in rendered
    assert 'data-record-audience="decision-maker"' in rendered
    assert 'data-record-skill=' in rendered and 'data-record-role=' in rendered
    assert 'learn/generation-advanced.html' in template
    assert 'data-record-advanced-strategy' in (EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn/generation-advanced.html').read_text()
    assert 'surface_evidence_gaps' in js and 'include_review_questions' in js
    assert 'Advanced drafting preferences:' in js
    assert 'The Chat generation runtime is not configured for the active endpoint profile.' in js
    assert 'submit.disabled=!!controller||!current.model' not in js
    assert 'Nothing here writes to the published catalog' in template
    source_rendered = env.get_template('learn/record-generation.html').render(creation_kind='source', payload='{}', site_id='test-site', revision='test-revision', topics=[], sources=[], studio_nav=STUDIO_NAV)
    assert 'must not invent source metadata' in source_rendered
    assert "kind==='source'&&!sourceUrl()" in js
    assert "kind==='source'&&!prompt" in js
    assert 'The browser does not fetch the source URL for the model.' in js
    assert 'A URL alone is reference metadata and does not count as generation content.' in js
    assert 'Source notes / excerpt' in template
    assert 'Do NOT return or invent a URL' in js
    assert 'requires_metadata_review=true' in js
    assert 'suggested_attachment' in js and 'review_required:true' in js
    assert 'learn.record-creation-draft.v1' in js and 'learn.topic-prompt-draft.v1' in js
    assert "url.port&&url.port!=='443'" in js
    assert "host.startsWith('fc')" in js and "host.startsWith('fd')" in js
    assert 'CSS.escape' not in js
    assert 'app.add_directive("ai-record-generation", RecordGenerationDirective)' in pages
    assert 'app.add_js_file("record-generation.js", defer="defer", priority=619)' in pages
    assert 'data-record-cancel' in template
    assert 'learn-record-result:v1:' in js
    assert 'persistGenerated(draft)' in js and 'renderGenerated()' in js
    assert "controller?.abort()" in js
    assert "one(root,'[data-generation-runtime-authority]')" in js
    assert "label?'Chat · '+label:'Chat-backed · Ready'" in js and "'Unavailable'" in js
    assert 'data-record-model-open' not in js and 'data-record-model-label' not in js
    assert 'endpointApi().onProfileChange' not in js  # subscription is guarded through the local api variable
    assert "typeof api.onProfileChange==='function'" in js


def test_ai_lens_selection_summary_is_shared_live_and_mobile_compact():
    template_root = EXT/'_sphinx_ext/_sphinx_ai_learn/_templates'
    env = Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)
    partial = (template_root/'learn/ai-lens-profile.html').read_text()
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/lens-selection.js').read_text()
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    for scope, variant in [
        ('record', 'record'),
        ('overview', 'overview'),
        ('section-ai', 'section'),
        ('video', 'media'),
    ]:
        rendered = env.get_template('learn/ai-lens-profile.html').render(lens_scope=scope, lens_variant=variant)
        assert rendered.count('data-ai-lens-group=') == 4
        assert rendered.count('data-ai-choice-summary') == 4
        assert rendered.count('data-ai-choice-count') == 4
        assert rendered.count('data-ai-choice-preview') == 4
        assert rendered.count('data-ai-lens-required="true"') == 2
        assert 'data-ai-lens-profile-summary' in rendered
    assert 'data-{{ lens_scope|e }}-audience' in partial
    assert 'Combined into the media instructions' in env.get_template('learn/ai-lens-profile.html').render(lens_scope='video', lens_variant='media')
    assert "slice(0, 2)" in js and "+ ' more'" in js
    assert "data-ai-lens-required" in js and "required-empty" in js
    assert "summary.setAttribute('aria-label'" in js and "labels.join(', ')" in js
    assert "profile.addEventListener('change'" in js
    assert '.learn-ai-choice-summary' in css and '.learn-ai-choice-preview' in css
    assert '@media(max-width:520px)' in css
    assert 'app.add_js_file("lens-selection.js", defer="defer", priority=620)' in pages


def test_inline_generation_supports_multi_audience_purpose_skill_and_role_lenses():
    env = Environment(loader=FileSystemLoader(str(EXT/'_sphinx_ext/_sphinx_ai_learn/_templates')), autoescape=False)
    template = env.get_template('learn/ai-lens-profile.html').render(lens_scope='section-ai', lens_variant='section')
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/section-generation.js').read_text()
    assert template.count('data-section-ai-audience=') >= 9
    assert template.count('data-section-ai-purpose=') >= 5
    assert template.count('data-section-ai-skill=') >= 5
    assert template.count('data-section-ai-role=') >= 6
    assert 'One model request, not separate autonomous agents' in template
    assert "audiences:profile.audiences" in js and "purposes:profile.purposes" in js
    assert "skills:profile.skills" in js and "roles:profile.roles" in js
    assert 'Skill mix:' in js and 'combined instructional lenses in one model request' in js
    assert 'AI_LEARN_TEXT_GENERATION_API' in js and 'runtime?.createWorkflow?.' in js
    assert 'runtime.run' not in js
    assert 'fetch(' not in js and 'chatEndpoint' not in js and 'extractReply' not in js


def test_all_nine_creation_studios_share_one_registry_header_and_responsive_navigation():
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    template_root = EXT/'_sphinx_ext/_sphinx_ai_learn/_templates'
    nav_source = (template_root/'learn/generation-studio-nav.html').read_text()
    header_source = (template_root/'learn/generation-studio-header.html').read_text()
    record_source = (template_root/'learn/record-generation.html').read_text()
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    env = Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)

    assert 'STUDIO_DEFINITIONS = (' in pages
    assert '_STUDIO_LEAF_BY_KIND = {kind: leaf' in pages
    assert 'for kind, label, leaf in STUDIO_DEFINITIONS' in pages
    assert 'merged = {**context, "studio_nav": _studio_navigation()}' in pages
    assert '_STUDIO_LEAF_BY_KIND.get(kind)' in pages
    assert 'learn-generation-modality-nav' not in nav_source
    assert 'learn-generation-studio-nav' in nav_source
    assert 'for studio in studio_nav' in nav_source
    assert 'learn/generation-studio-nav.html' in header_source
    assert 'learn/generation-studio-header.html' in record_source
    for kind in (
        'video',
        'audio',
        'document',
        'whiteboard',
    ):
        assert 'learn/generation-studio-header.html' in (template_root/f'learn/{kind}-generation.html').read_text()

    expected_labels = [row['label'] for row in STUDIO_NAV]
    expected_hrefs = [row['href'] for row in STUDIO_NAV]
    nav = env.get_template('learn/generation-studio-nav.html')
    for current in [row['kind'] for row in STUDIO_NAV]:
        rendered = nav.render(studio_nav=STUDIO_NAV, generation_kind=current)
        assert rendered.count('<a ') == 9
        assert rendered.count('aria-current="page"') == 1
        for label, href in zip(expected_labels, expected_hrefs):
            assert f'href="{href}"' in rendered
            assert f'>{label}</a>' in rendered

    assert 'grid-template-columns:repeat(5,minmax(0,1fr))' in css
    assert 'box-sizing:border-box' in css
    assert '@container learn-generation-studio (max-width:42rem)' in css
    assert 'grid-template-columns:repeat(3,minmax(0,1fr))' in css
    assert '@container learn-generation-studio (max-width:28rem)' in css
    assert 'grid-template-columns:repeat(2,minmax(0,1fr))' in css
    assert '@container learn-generation-studio (max-width:20rem)' in css
    assert '.learn-generation-studio-nav { grid-template-columns:1fr; }' in css
    assert '.learn-generation-modality-nav' not in css


def test_nested_and_studio_generation_share_authority_lenses_and_private_flow_primitives():
    template_root = EXT/'_sphinx_ext/_sphinx_ai_learn/_templates'
    section = (template_root/'learn/section-ai-generation.html').read_text()
    overview = (template_root/'learn/overview-actions.html').read_text()
    lifecycle = (template_root/'learn/generation-private-flow.html').read_text()
    ui = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/generation-ui.js').read_text()
    section_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/section-generation.js').read_text()
    overview_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/overview-generation.js').read_text()
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()

    for source in (section, overview):
        assert 'learn/generation-authority.html' in source
        assert "authority_kind = 'assistant'" in source
        assert "authority_managed = 'shared'" in source
        assert 'learn/generation-private-flow.html' in source
        assert 'learn/ai-lens-profile.html' in source
    assert 'Choose context' in lifecycle
    assert 'Generate draft' in lifecycle
    assert '>Review<' in lifecycle
    assert '>Handoff<' in lifecycle
    assert 'function setFlowStage(' in ui and 'setFlowStage: setFlowStage' in ui
    assert "setFlowStage?.(panel,'section-ai'" in section_js
    assert "setFlowStage?.(panel,'overview'" in overview_js
    assert "assistantAuthorityUnsubscribe = current.onChange(syncAssistantAuthorities)" in ui
    assert ui.count('current.onChange(syncAssistantAuthorities)') == 1
    assert 'syncAssistantAuthorities(typeof current.getState' in ui
    assert '.learn-section-ai-authority' not in css
    assert '@container learn-user-library (max-width:24rem) { .learn-section-ai-flow { grid-template-columns:1fr; } }' in css

    for filename in (
        'video-generation.js',
        'audio-generation.js',
        'document-generation.js',
        'whiteboard-generation.js',
        'record-generation.js',
        'overview-generation.js',
        'section-generation.js',
    ):
        controller = (EXT/f'_sphinx_ext/_sphinx_ai_learn/_static/{filename}').read_text()
        assert 'AI_ASSISTANT_MODEL_API' not in controller
    assert 'assistantModelSnapshot' in (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/video-generation.js').read_text()
    assert 'assistantModelSnapshot' in (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/audio-generation.js').read_text()
    assert 'assistantModelSnapshot' in (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/document-generation.js').read_text()


def test_reviewed_publication_ui_is_explicit_and_shared_across_private_drafts():
    template_root = EXT / '_sphinx_ext/_sphinx_ai_learn/_templates'
    record = (template_root/'learn/record-generation.html').read_text()
    overview = (template_root/'learn/overview-actions.html').read_text()
    section = (template_root/'learn/section-ai-generation.html').read_text()
    shared = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/generation-ui.js').read_text()
    record_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/record-generation.js').read_text()
    overview_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/overview-generation.js').read_text()
    section_js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/section-generation.js').read_text()

    assert 'data-record-publish' in record
    assert 'data-record-metadata-reviewed' in record
    overview_actions = _render_text_generation_actions('overview', 'Generate AI Overview', '-request')
    assert overview.count('data-overview-publish') + overview_actions.count('data-overview-publish') == 2
    assert 'data-overview-publish disabled>Open pull request' in overview_actions
    section_actions = _render_text_generation_actions('section-ai', 'Generate Now')
    assert 'data-section-ai-publish' in section_actions
    assert 'function submitPublication' in shared
    assert "resolveEndpoint('publication')" in shared
    assert "credentials:'omit'" in shared
    assert "redirect:'error'" in shared
    assert "url.port && url.port !== '443'" in shared
    assert "action:'publish'" in record_js
    assert "metadata_reviewed=true" in record_js
    assert "section_id:'summary'" in overview_js
    assert "publishButtons=all(panel,'[data-overview-publish]')" in overview_js
    text_workflow = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/text-generation-ui.js').read_text()
    assert "publishButtons.forEach(button=>button.addEventListener('click',()=>workflowAction('publish')))" in overview_js
    assert 'async function publish()' in text_workflow and 'ui?.submitPublication?.(publicationRequest)' in text_workflow
    assert 'syncPublishButtons' not in overview_js
    assert "section_id:String(id||'')" in section_js
    assert 'Generate Now' in section  # generation remains distinct from publication



def test_multimodal_generation_uses_one_bounded_redirect_safe_transport_and_retry_identity():
    static = EXT/'_sphinx_ext/_sphinx_ai_learn/_static'
    shared = (static/'generation-ui.js').read_text()
    media = {
        name: (static/name).read_text()
        for name in ('video-generation.js','audio-generation.js','document-generation.js','whiteboard-generation.js')
    }

    # Raw browser transport is centralized. Modality controllers retain only
    # contract-specific request/job logic.
    assert shared.count('fetch(requestUrl.href, requestInit)') == 1
    assert "credentials:'omit', cache:'no-store', redirect:'error', referrerPolicy:'no-referrer'" in shared
    assert 'Runtime request URL must use HTTPS (or loopback HTTP for local development) without credentials or fragments.' in shared
    assert "requestUrl.protocol === 'http:' && !loopback" in shared
    assert 'This browser does not support bounded runtime requests.' in shared
    assert 'readBoundedResponse(response' in shared
    assert 'timed out after ' in shared
    assert 'fetchJson: fetchJson' in shared and 'fetchBlob: fetchBlob' in shared
    assert 'function mimeMatches(expected, actual)' in shared
    assert "response.ok && expectedMime && !mimeMatches(expectedMime, actualMime)" in shared
    assert "returned an unexpected content type" in shared
    for source in media.values():
        assert 'fetch(' not in source
        assert 'Shared AI Learn runtime transport is unavailable.' in source

    # Discovery, job/status, artifacts and synchronous outputs all declare
    # bounded waits and byte ceilings rather than inheriting an unbounded read.
    assert "label:'Video capability discovery'" in media['video-generation.js']
    assert "label:'Video generation'" in media['video-generation.js']
    assert "label:'Audio artifact'" in media['audio-generation.js']
    assert '64*1024*1024' in media['audio-generation.js']
    assert "label:'Document generation'" in media['document-generation.js']
    assert '8*1024*1024' in media['document-generation.js']
    assert "label:'Whiteboard image'" in media['whiteboard-generation.js']
    assert '64*1024*1024' in media['whiteboard-generation.js']

    # Ambiguous audio retries now reuse one key for the exact same request,
    # matching the video/server idempotency contract.
    audio = media['audio-generation.js']
    assert 'function ensureIdempotency(request)' in audio
    assert 'idempotencySignature !== signature' in audio
    assert "'Idempotency-Key':ensureIdempotency(request)" in audio
    assert "'Idempotency-Key':randomKey()" not in audio
    assert "['localhost','127.0.0.1','::1'].includes(url.hostname)" in audio
    assert "url.protocol === 'https:' && (!url.port || url.port === '443')" in audio

    # Public runtime video links are HTTPS; loopback HTTP remains available for
    # local development without permitting arbitrary insecure artifact links.
    video = media['video-generation.js']
    assert "u.protocol==='https:'" in video
    assert "['localhost','127.0.0.1','::1'].includes(u.hostname)" in video


def test_private_generation_library_fails_closed_on_storage_errors_and_tracks_other_tabs():
    shared = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/generation-ui.js').read_text()
    assert 'That \' + singular + \' is no longer available in this browser.' in shared
    assert "for this tab. Browser storage is unavailable, so the change will not survive reload." in shared
    assert "receipt is kept only in this tab and will not survive reload." in shared
    assert "window.addEventListener('storage', onStorage)" in shared
    assert "window.removeEventListener('storage', onStorage)" in shared
    assert "var persisted = write(rows);" in shared
    assert 'boundedInteger(options.maxRows, 50, 1, 100)' in shared
    assert 'boundedInteger(options.maxStorage, 250000, 10000, 1000000)' in shared



def test_video_receipts_survive_storage_failure_in_the_current_tab_and_sync_other_tabs():
    video = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/video-generation.js').read_text()
    assert 'let volatileJobs=[]' in video
    assert 'function boundedJobs(rows)' in video
    assert 'rows.map(row=>normalizeJob(row)).filter(Boolean).slice(0,50)' in video
    assert 'catch{return volatileJobs.slice();}' in video
    assert 'volatileJobs=normalized' in video
    assert 'return true;}catch{return false;}' in video
    assert 'const persisted=mergeJob(job)' in video
    assert 'receipt is kept only in this tab because browser storage is unavailable' in video
    assert "window.addEventListener('storage',handleJobsStorage)" in video
    assert "window.removeEventListener('storage',handleJobsStorage)" in video
    assert 'if(changed&&!writeJobs(jobs))' in video


def test_record_generation_freezes_request_provenance_and_requires_durable_local_result():
    js = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/record-generation.js').read_text()
    assert 'runProfile=profile()' in js and 'runAdvanced=advanced()' in js
    assert 'runContextIds=selectedContext().map(row=>row.id)' in js
    assert 'runSourceUrl=sourceUrl()' in js
    assert 'profile:runProfile,advanced:runAdvanced,context_ids:runContextIds' in js
    assert 'sanitizeDraft(kind,parsed,runSourceUrl)' in js
    assert 'current!==loadedResultRaw' in js
    assert 'localStorage.setItem(resultKey,encoded)' in js
    assert 'loadedResultRaw=encoded;generated=value;renderGenerated()' in js
    assert 'Browser storage is unavailable; the generated draft was not saved.' in js
    assert 'if(controller||publishing)' in js
    assert 'submit.disabled=!!controller||publishing' in js
    assert 'publishing=false;publish.disabled=false;syncAuthority()' in js


def test_generation_cleanup_is_bfcache_safe_and_private_libraries_keep_volatile_receipts():
    static = EXT/'_sphinx_ext/_sphinx_ai_learn/_static'
    shared = (static/'generation-ui.js').read_text()
    assert 'function onPageDispose(callback)' in shared
    assert 'event && event.persisted === true' in shared
    assert 'onPageDispose: onPageDispose' in shared
    assert 'var volatileRows = [];' in shared
    assert 'function safeLibraryRow(row)' in shared
    assert 'Persist display metadata only.' in shared
    assert 'request bodies and any future unknown fields are intentionally dropped.' in shared
    assert 'return volatileRows.slice();' in shared
    assert 'volatileRows = normalized;' in shared
    assert 'receipt is kept only in this tab and will not survive reload.' in shared
    assert 'volatileRows = [];' in shared

    files = (
        'generation-ui.js', 'video-generation.js', 'audio-generation.js',
        'document-generation.js', 'whiteboard-generation.js', 'record-generation.js'
    )
    for name in files:
        source = (static/name).read_text()
        assert "pagehide'," in source
        import re
        assert not re.search(r"pagehide[^\n]{0,160}once\s*:\s*true", source)
    for name in ('video-generation.js','audio-generation.js','document-generation.js','whiteboard-generation.js','record-generation.js'):
        source = (static/name).read_text()
        assert 'onPageDispose' in source or 'event?.persisted===true' in source or 'event.persisted === true' in source


def test_index_explorer_header_is_shared_semantic_and_link_safe():
    template_root = EXT/'_sphinx_ext/_sphinx_ai_learn/_templates/learn'
    shared = (template_root/'index-explorer-header.html').read_text()
    topic = (template_root/'topic-explorer.html').read_text()
    catalog = (template_root/'catalog-explorer.html').read_text()
    pages = (EXT/'_sphinx_ext/_sphinx_ai_learn/_pages.py').read_text()
    assert 'learn-index-explorer-head' in shared
    assert '<nav class="learn-trending-links"' in shared
    assert 'aria-label="{{ explorer_title|e }} shortcuts"' in shared
    assert '{{ bookmarks_href|e }}' in shared and '{{ collections_href|e }}' in shared
    assert 'Research explorer' not in topic
    assert 'learn-trending-head' not in topic
    assert 'learn-trending-head' not in catalog
    assert 'class IndexExplorerHeaderDirective' in pages
    assert 'root["index_explorer_header"] = True' in pages
    assert 'app.add_directive("ai-index-explorer-header", IndexExplorerHeaderDirective)' in pages
    css = (EXT/'_sphinx_ext/_sphinx_ai_learn/_static/topic.css').read_text()
    assert 'container-name:learn-index-explorer-masthead' in css
    assert '@container learn-index-explorer-masthead (max-width:640px)' in css


def test_index_explorer_header_template_escapes_dynamic_copy_and_links():
    template_root = EXT/'_sphinx_ext/_sphinx_ai_learn/_templates'
    env = Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)
    rendered = env.get_template('learn/index-explorer-header.html').render(
        kicker='Topic <Source>',
        explorer_title='Exploring "Sources" & more',
        bookmarks_href='../bookmarks/index.html?x=1&y=2',
        collections_href='../collections/index.html',
    )
    assert 'Topic &lt;Source&gt;' in rendered
    assert 'Exploring &#34;Sources&#34; &amp; more' in rendered
    assert '?x=1&amp;y=2' in rendered
    assert rendered.count('class="learn-button"') == 2
