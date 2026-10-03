# scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/_registry.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""AI Learn section structure and generation policy.

Canonical content and reusable interaction definitions live in JSON.  This module
owns only renderer policy that must remain Python code: fixed structural sections,
generation capabilities, and detail-page section shapes.
"""

from __future__ import annotations

AI_SECTION_WORKFLOW = "learn.section-generation.v2"


PROMPT_GENERATION_CONTRACT = """

Generation contract:
- Rebuild this section from the canonical topic record and the currently available source records on every run; do not append to or repair a previous generated answer.
- Keep the requested heading/list structure stable for equivalent inputs so repeated runs remain easy to compare.
- Use source-grounded facts first, label synthesis or inference explicitly, and never invent citations, quotations, metrics, experiments, or missing source detail.
- If the available sources cannot support a requested point, state that the source evidence is insufficient instead of filling the gap with a guess.
- Treat generated text as a replaceable draft: the same canonical inputs should yield structurally equivalent output, and newer evidence should replace stale claims rather than accumulate beside them.
"""

# Prompt-specific sections are injected from canonical topic-prompt JSON.  Keeping
# only fixed sections here prevents a second prompt registry from drifting away
# from the repository content tree.
TOPIC_SECTIONS_BEFORE_PROMPTS = (
    ("topic", "Topic", 2, "overview", "Define the question and its scope."),
    (
        "summary",
        "Summary",
        2,
        "text",
        "Bring the main ideas and their evidence together.",
    ),
    (
        "video",
        "Topic to Video (Beta)",
        2,
        "media",
        "Turn the topic into a narrated explanation.",
    ),
    (
        "audio",
        "Audio Explanation",
        2,
        "media",
        "Listen to a concise spoken explanation of the topic.",
    ),
    (
        "document",
        "Document",
        2,
        "media",
        "Turn the topic into a durable written learning artifact.",
    ),
    ("whiteboard", "Whiteboard", 2, "media", "Make the relationships visible."),
)

TOPIC_PROMPT_GROUP = (
    "topic-prompts",
    "Topic Prompts",
    2,
    "prompt-group",
    "Explore the topic from different angles.",
)

TOPIC_SKILL_GROUP = (
    "skills",
    "Skills",
    2,
    "skill-group",
    "Apply reusable learning and evidence workflows to this topic.",
)

TOPIC_SECTIONS_AFTER_INTERACTIONS = (
    (
        "open-problems",
        "Open Problems",
        2,
        "problems",
        "Explore questions that still need evidence or a solution.",
    ),
    (
        "continue-learning",
        "Continue Learning",
        2,
        "text",
        "Find the next questions and sources to explore.",
    ),
    ("tweets", "Tweets", 2, "social", "Collect relevant posts with links and context."),
    (
        "hackernews",
        "HackerNews",
        2,
        "social",
        "Collect relevant Hacker News discussions with links and context.",
    ),
)

TOPIC_EMPTY_MESSAGES = {
    "summary": "No one has generated a summary of this topic yet.",
    "video": "No one has generated a video about this topic yet.",
    "audio": "No one has generated an audio explanation for this topic yet.",
    "document": "No one has generated a document about this topic yet.",
    "whiteboard": "No one has generated a whiteboard explanation for this topic yet.",
    "open-problems": (
        "We haven't generated a list of open problems mentioned in this topic yet."
    ),
    "continue-learning": "We haven't generated follow-up questions for this topic yet.",
    "tweets": "No relevant tweets have been added for this topic yet.",
    "hackernews": (
        "No relevant Hacker News discussions have been added for this topic yet."
    ),
}


def _generation_policy(
    subject_kind,
    section_id,
    section_kind,
    title,
    description,
    *,
    prompts=(),
    skills=(),
):
    """Declare which sections may receive an AI-authored private draft."""
    if subject_kind == "topic":
        if section_kind != "text":
            reasons = {
                "media": "Media generation belongs to its dedicated generation studio.",
                "problems": (
                    "Open-problem cards are catalog records, not generated prose."
                ),
                "social": (
                    "Social/feed sections must come from real linked records, not synthetic posts."
                ),
                "prompt-group": (
                    "Prompt groups are navigation controls, not generated content."
                ),
                "skill-group": (
                    "Skill groups are navigation controls, not generated content."
                ),
                "overview": "The topic overview has its own page-level workflow.",
            }
            return {
                "mode": "none",
                "reason": reasons.get(
                    section_kind,
                    "This section is not AI-draftable.",
                ),
            }
        prompt = next((row for row in prompts if row["id"] == section_id), None)
        skill_definition = next(
            (row for row in skills if row["id"] == section_id), None
        )
        if prompt is not None and skill_definition is not None:
            raise ValueError(f"duplicate topic interaction identifier: {section_id}")
        interaction = prompt or skill_definition
        return {
            "mode": "chat",
            "workflow_id": AI_SECTION_WORKFLOW,
            "skill": (
                "topic-prompt:" + section_id
                if prompt
                else ("skill:" + section_id if skill_definition else "topic-synthesis")
            ),
            "agent": (
                "learning-skill-agent" if skill_definition else "learning-section-agent"
            ),
            "instruction": (
                interaction["instruction"].rstrip() + PROMPT_GENERATION_CONTRACT
                if interaction
                else description
            ),
        }

    blocked = {
        "problem": {
            "references": (
                "References are evidence-owned and must point to real Source records."
            ),
            "related": (
                "Related topics come from the catalog graph, not generated prose."
            ),
        },
        "video": {
            "transcript": (
                "Transcripts are media-derived and must not be hallucinated by a text model."
            ),
            "evidence": (
                "Sources are evidence-owned and must point to real Source records."
            ),
        },
        "audio": {
            "transcript": (
                "Transcripts are media-derived and must not be hallucinated by a text model."
            ),
            "evidence": (
                "Sources are evidence-owned and must point to real Source records."
            ),
        },
        "document": {
            "evidence": (
                "Sources are evidence-owned and must point to real Source records."
            ),
        },
        "whiteboard": {
            "evidence": (
                "Sources are evidence-owned and must point to real Source records."
            ),
        },
        "source": {
            "related": (
                "Related topics come from the catalog graph, not generated prose."
            ),
        },
    }
    reason = blocked.get(subject_kind, {}).get(section_id)
    if reason:
        return {"mode": "none", "reason": reason}
    skill = {
        "problem": "problem-analysis",
        "video": (
            "video-explanation" if section_id == "description" else "video-script-draft"
        ),
        "audio": "audio-explanation",
        "document": "document-synthesis",
        "whiteboard": "visual-explanation",
        "source": "source-synthesis",
    }.get(subject_kind, "record-synthesis")
    agent = {
        "problem": "problem-analysis-agent",
        "video": (
            "media-explanation-agent"
            if section_id == "description"
            else "script-drafting-agent"
        ),
        "audio": "media-explanation-agent",
        "document": "document-analysis-agent",
        "whiteboard": "visual-explanation-agent",
        "source": "source-analysis-agent",
    }.get(subject_kind, "learning-section-agent")
    return {
        "mode": "chat",
        "workflow_id": AI_SECTION_WORKFLOW,
        "skill": skill,
        "agent": agent,
        "instruction": description,
    }


def _section_spec(row, *, prompts=(), skills=()):
    item = dict(zip(("id", "title", "level", "kind", "description"), row))
    item["generation"] = _generation_policy(
        "topic",
        item["id"],
        item["kind"],
        item["title"],
        item["description"],
        prompts=prompts,
        skills=skills,
    )
    item["empty_message"] = TOPIC_EMPTY_MESSAGES.get(item["id"], "")
    return item


def topic_sections(subject=None, *, prompts=(), skills=()):
    """Return canonical Topic structure with JSON-defined interactions inserted."""
    prompts = tuple(prompts or ())
    skills = tuple(skills or ())
    result = [
        _section_spec(row, prompts=prompts, skills=skills)
        for row in TOPIC_SECTIONS_BEFORE_PROMPTS
    ]
    result.append(_section_spec(TOPIC_PROMPT_GROUP, prompts=prompts, skills=skills))
    result.extend(
        {
            "id": prompt["id"],
            "title": prompt["title"],
            "level": 3,
            "kind": "text",
            "description": prompt["description"],
            "empty_message": prompt["empty_message"],
            "generation": _generation_policy(
                "topic",
                prompt["id"],
                "text",
                prompt["title"],
                prompt["description"],
                prompts=prompts,
                skills=skills,
            ),
        }
        for prompt in prompts
    )
    result.append(_section_spec(TOPIC_SKILL_GROUP, prompts=prompts, skills=skills))
    result.extend(
        {
            "id": skill["id"],
            "title": skill["title"],
            "level": 3,
            "kind": "text",
            "description": skill["description"],
            "empty_message": skill["empty_message"],
            "generation": _generation_policy(
                "topic",
                skill["id"],
                "text",
                skill["title"],
                skill["description"],
                prompts=prompts,
                skills=skills,
            ),
        }
        for skill in skills
    )
    result.extend(
        _section_spec(row, prompts=prompts, skills=skills)
        for row in TOPIC_SECTIONS_AFTER_INTERACTIONS
    )

    content = {s["id"]: s for s in (subject or {}).get("sections", [])}
    known = {s["id"] for s in result}
    result.extend(
        {
            "id": section["id"],
            "title": section["title"],
            "level": 2,
            "kind": "text",
            "description": "Explore this part of the topic.",
            "empty_message": (
                "Explore this part of the topic. This section has not been generated yet."
            ),
            "generation": _generation_policy(
                "topic",
                section["id"],
                "text",
                section["title"],
                "Explore this part of the topic.",
                prompts=prompts,
                skills=skills,
            ),
        }
        for section in content.values()
        if section["id"] not in known
    )
    return result


DETAIL_SECTIONS = {
    "problem": (
        ("statement", "Statement"),
        ("background", "Background"),
        ("references", "References"),
        ("related", "Related Topics"),
    ),
    "video": (
        ("description", "About this video"),
        ("script", "Script"),
        ("transcript", "Transcript"),
        ("evidence", "Sources"),
    ),
    "audio": (
        ("description", "About this audio"),
        ("transcript", "Transcript"),
        ("evidence", "Sources"),
    ),
    "document": (
        ("description", "About this document"),
        ("content", "Content overview"),
        ("evidence", "Sources"),
    ),
    "whiteboard": (("description", "Visual explanation"), ("evidence", "Sources")),
    "source": (
        ("overview", "Overview"),
        ("scope", "Scope"),
        ("key-points", "Key Points"),
        ("related", "Related Topics"),
    ),
}


def canonical_detail_section_id(kind, section_id):
    """Return a section ID unchanged; the JSON-first tree has no legacy aliases."""
    return section_id


def detail_sections(subject):
    """Return stable context-aware detail sections plus explicit custom sections."""
    result = []
    for key, title in DETAIL_SECTIONS[subject["kind"]]:
        description = "Add source-grounded notes."
        result.append(
            {
                "id": key,
                "title": title,
                "level": 2,
                "kind": "text",
                "description": description,
                "generation": _generation_policy(
                    subject["kind"], key, "text", title, description
                ),
            }
        )
    known = {section["id"] for section in result}
    for section in subject.get("sections", []):
        if section["id"] in known:
            continue
        result.append(
            {
                "id": section["id"],
                "title": section["title"],
                "level": 2,
                "kind": "text",
                "description": "Explore this part of the record.",
                "generation": _generation_policy(
                    subject["kind"],
                    section["id"],
                    "text",
                    section["title"],
                    "Explore this part of the record.",
                ),
            }
        )
        known.add(section["id"])
    return result
