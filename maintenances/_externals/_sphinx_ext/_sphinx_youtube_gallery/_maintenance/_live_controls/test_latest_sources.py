"""Static regression checks for strict cards plus resolve-before-render latest sources."""
from pathlib import Path
import sys
import ast

_MAINTENANCE_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "_maintenance")
if str(_MAINTENANCE_DIR) not in sys.path:
    sys.path.insert(0, str(_MAINTENANCE_DIR))
from bootstrap import activate
ROOT = activate(__file__)
source = (ROOT / "_sphinx_collection" / "assets.py").read_text(encoding="utf-8")
tree = ast.parse(source)
assets = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in {"ASSET_JS", "ASSET_CSS"}:
            assets[node.targets[0].id] = ast.literal_eval(node.value)
js = assets["ASSET_JS"]
css = assets["ASSET_CSS"]

for required in [
    "['videos','shorts','streams','courses']",
    "Latest channel section",
    "var channelAddingMode=",
    "var videoAddingMode=",
    "if(videoAddingMode)",
    "sd-stretched-link sd-hide-link-text reference external",
    "channelLink.href=ref.url",
    "sk-collection-source-section-options",
    "var isVideo=ref.kind==='video'",
    "var isChannel=ref.kind==='channel'",
    "function normalizeHandle(token)",
    "decodeURIComponent(text)",
    "encodeURIComponent(normalizedHandle)",
    "title=suppliedTitle||ref.handle||ref.name||ref.channelId||'Channel'",
    "element('div','video_wrapper')",
    "https://www.youtube.com/embed/",
    "frame.title='youtube video player'",
    "aspect-ratio: 16 / 9; max-width: 100%; position: relative; width: 560px",
    "sd-card sd-sphinx-override sd-w-100 sd-shadow-sm sk-collection-item-",
    # Latest-source architecture: resolve to one exact ID before calling the video renderer.
    "function resolveLatest(source)",
    "function resolveAndInsert(source,title,generation)",
    "window.skCollectionResolveYouTubeLatest",
    "window.skCollectionYouTubeDataApiKey",
    "forHandle:source.handle",
    "contentDetails.videoPublishedAt",
    "requestedTitle:suppliedTitle",
    "title:c.requestedTitle||''",
    "if(key)return latestPlaylistFromApi",
    "videos:'UULF'",
    "shorts:'UUSH'",
    "streams:'UULV'",
    "Resolve this source to one video before rendering it.",
    "sourceRef=sourceRef||ref",
    "url:sourceRef.url",
    "Finding the latest playable video…",
    "Latest-video resolver did not return a valid YouTube video ID.",
    "YouTube does not expose a stable public Courses-tab resolver.",
    "kind:'post'",
    "postId:source.postId||null",
    "HTTP YouTube links are upgraded automatically.",
    "A YouTube post URL does not expose an attached video ID.",
    "sk-collection-add-guide",
    "Prefill a YouTube source example",
    "Works without site setup: exact video IDs and video, Short, live, or watch URLs.",
    "sk-collection-source-preview",
    "updateSourcePreview",
    "Ready · exact video ID known · plays inline with no API key.",
    "Post recognized · needs optional site resolver",
    "Site setup for latest/post sources (optional)",
    "keep provider credentials server-side",
    "fetch('/api/youtube/resolve'",
    "['Live','https://www.youtube.com/live/VIDEO_ID'",
    "['Watch + list','https://www.youtube.com/watch?v=VIDEO_ID&list=PLAYLIST_ID'",
    "function withTimeout(value,ms,message,onTimeout)",
    "YouTube Data API request timed out.",
    "Site resolver timed out.",
    "sk-collection-resolver-player",
]:
    assert required in js, required

for forbidden in [
    "Playable latest (recommended)",
    "Source link only",
    "skCollectionResolveYouTubeChannel",
    "sk-collection-source-kind",
    "sk-collection-added-player",
    "sk-collection-play-copy",
    "Watch on YouTube",
    "Open videos on YouTube",
    "https://www.youtube.com/embed/videoseries?list=",
    "^@[A-Za-z0-9_.-]+$",
]:
    assert forbidden not in js, forbidden

assert "grid-template-columns:minmax(0,1fr) 3rem" in css
assert ".sk-collection-add-guide" in css
assert ".sk-collection-add-example" in css
assert ".sk-collection-source-preview" in css
assert ".sk-collection-resolver-stub" in css
assert ".sk-collection-resolver-player" in css
assert ".sk-collection-source-kind" not in css
assert ".sk-collection-added-player" not in css
assert ".sk-collection-play" not in css

readme = (ROOT / "_sphinx_youtube_gallery" / "README.md").read_text(encoding="utf-8")
for required in [
    "Channel additions",
    "only the title visible",
    "Direct video additions",
    "video_wrapper",
    "@claude/videos",
    "/shorts",
    "/streams",
    "/courses",
    "PLHfy8mSAC18s",
    "resolve first",
    "skCollectionResolveYouTubeLatest",
    "skCollectionYouTubeDataApiKey",
    "YouTube post URL",
    "prefill chips",
    "postId",
    "scheme-less YouTube links",
]:
    assert required in readme, required
for forbidden in ["Playable latest (recommended)", "Source link only"]:
    assert forbidden not in readme, forbidden

print("Strict cards and latest-source resolve-before-render checks passed")
