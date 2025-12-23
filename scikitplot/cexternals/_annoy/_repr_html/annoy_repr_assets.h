// scikitplot/cexternals/_annoy/_repr_html/annoy_repr_assets.h

#pragma once
#include <string>
#include <Python.h>

// Initializes repr asset templates once, at module init.
// This never fails module import: it always falls back to embedded templates.
void annoy_repr_assets_init(PyObject* module);

// Appends CSS/JS into `out`, with __ANNOY_REPR_ID__ replaced by `idbuf`.
// These functions are deterministic for a given (loaded templates, idbuf).
void annoy_repr_assets_append_css(std::string& out, const char* idbuf);
void annoy_repr_assets_append_js(std::string& out, const char* idbuf);
