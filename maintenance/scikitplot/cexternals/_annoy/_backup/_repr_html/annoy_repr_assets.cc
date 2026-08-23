// scikitplot/cexternals/_annoy/_repr_html/annoy_repr_assets.cc

// <module_dir>/_repr_html/{estimator.css,params.css,estimator.js}
// After module object is created successfully:
// annoy_repr_assets_init(module);
// html.append("<style>");
// annoy_repr_assets_append_css(html, idbuf);
// html.append("</style>");
// html.append("<script>");
// annoy_repr_assets_append_js(html, idbuf);
// html.append("</script>");

#include "annoy_repr_assets.h"
#include <cstdio>
#include <cstring>

// Placeholder must match the asset files exactly.
static const char kAnnoyReprIdPlaceholder[] = "__ANNOY_REPR_ID__";

// Embedded fallbacks: used if on-disk assets cannot be loaded.
extern const char kAnnoyReprCssFallback[];
extern const char kAnnoyReprJsFallback[];

// Loaded templates (either from disk or fallback), stored with placeholders intact.
static std::string g_css_tpl;
static std::string g_js_tpl;
static bool g_assets_inited = false;

// Directory containing the extension module file (derived from module.__file__).
// We build asset paths relative to this directory (no imports, no guessing beyond fixed relatives).
static std::string g_module_dir;
static char g_path_sep = '/';

// ---------- small, deterministic helpers ----------

static size_t annoy_find_last_sep(const std::string& s, char* out_sep) {
  const size_t pos_slash = s.find_last_of('/');
  const size_t pos_back  = s.find_last_of('\\');
  size_t pos = std::string::npos;

  if (pos_slash == std::string::npos) pos = pos_back;
  else if (pos_back == std::string::npos) pos = pos_slash;
  else pos = (pos_slash > pos_back) ? pos_slash : pos_back;

  if (pos != std::string::npos && out_sep) {
    *out_sep = (pos_slash > pos_back) ? '/' : '\\';
  }
  return pos;
}

static bool annoy_read_text_file(const std::string& path, std::string& out) {
  // Deterministic read:
  // - Open in binary mode.
  // - Read full file.
  // - If any failure, return false.
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) return false;

  if (std::fseek(fp, 0, SEEK_END) != 0) { std::fclose(fp); return false; }
  const long sz = std::ftell(fp);
  if (sz < 0) { std::fclose(fp); return false; }
  if (std::fseek(fp, 0, SEEK_SET) != 0) { std::fclose(fp); return false; }

  // Explicit maximum size (policy, not heuristic): prevents pathological files.
  // Keep this conservative; repr assets should be small.
  const long kMaxBytes = 1024L * 1024L;  // 1 MiB
  if (sz > kMaxBytes) { std::fclose(fp); return false; }

  out.assign((size_t)sz, '\0');
  if (sz > 0) {
    const size_t n = std::fread(&out[0], 1u, (size_t)sz, fp);
    if (n != (size_t)sz) { std::fclose(fp); return false; }
  }
  std::fclose(fp);
  return true;
}

static void annoy_append_with_id(std::string& out, const std::string& tpl, const char* idbuf) {
  // Replace all occurrences of placeholder with idbuf.
  // No parsing/rewriting beyond strict substring replacement.
  const std::string ph(kAnnoyReprIdPlaceholder);
  size_t i = 0;
  while (true) {
    const size_t hit = tpl.find(ph, i);
    if (hit == std::string::npos) {
      out.append(tpl, i, std::string::npos);
      return;
    }
    out.append(tpl, i, hit - i);
    out.append(idbuf);
    i = hit + ph.size();
  }
}

static std::string annoy_join(const std::string& a, const char* b) {
  std::string r = a;
  if (!r.empty() && r.back() != g_path_sep) r.push_back(g_path_sep);
  r.append(b);
  return r;
}

static void annoy_init_module_dir(PyObject* module) {
  // Derive base directory from module filename deterministically (no imports).
  // If anything fails, leave g_module_dir empty and rely on fallbacks.
  PyObject* fileobj = PyModule_GetFilenameObject(module);  // new ref or NULL
  if (!fileobj) return;

  const char* filename = PyUnicode_AsUTF8(fileobj);
  if (!filename || !*filename) { Py_DECREF(fileobj); return; }

  std::string full(filename);
  char sep = '/';
  const size_t pos = annoy_find_last_sep(full, &sep);
  if (pos == std::string::npos) { Py_DECREF(fileobj); return; }

  g_path_sep = sep;
  g_module_dir.assign(full.c_str(), pos);  // directory only
  Py_DECREF(fileobj);
}

// ---------- public API ----------

void annoy_repr_assets_init(PyObject* module) {
  if (g_assets_inited) return;
  g_assets_inited = true;

  annoy_init_module_dir(module);

  // Default to embedded fallbacks (guaranteed available).
  g_css_tpl = kAnnoyReprCssFallback;
  g_js_tpl  = kAnnoyReprJsFallback;

  // If we can locate module dir, attempt to load disk assets.
  if (g_module_dir.empty()) return;

  // Fixed, explicit relative directory (no discovery / no guessing).
  const std::string asset_dir = annoy_join(g_module_dir, "_repr_html");

  // Load CSS: require BOTH css files for a complete external style (deterministic rule).
  std::string css_est, css_params;
  const bool have_est    = annoy_read_text_file(annoy_join(asset_dir, "estimator.css"), css_est);
  const bool have_params = annoy_read_text_file(annoy_join(asset_dir, "params.css"), css_params);
  if (have_est && have_params) {
    g_css_tpl.clear();
    g_css_tpl.reserve(css_est.size() + 1 + css_params.size());
    g_css_tpl.append(css_est);
    g_css_tpl.push_back('\n');
    g_css_tpl.append(css_params);
  }

  // Load JS: require estimator.js.
  std::string js_est;
  if (annoy_read_text_file(annoy_join(asset_dir, "estimator.js"), js_est)) {
    g_js_tpl = js_est;
  }
}

void annoy_repr_assets_append_css(std::string& out, const char* idbuf) {
  if (!idbuf) return;
  annoy_append_with_id(out, g_css_tpl, idbuf);
}

void annoy_repr_assets_append_js(std::string& out, const char* idbuf) {
  if (!idbuf) return;
  annoy_append_with_id(out, g_js_tpl, idbuf);
}
