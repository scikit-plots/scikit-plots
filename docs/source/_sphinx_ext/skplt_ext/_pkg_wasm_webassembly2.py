try:
    import micropip
    from IPython.display import clear_output

    display(HTML("<h4>Pyodide Environment preparing...</h4>"))
    # await micropip.install("scikit-plots", keep_going=True)
    clear_output(wait=True)
except:
    print(f"Can't find a pure Python 3 wheel for: '{'scikit-plots'}'")
    # await micropip.install("scikit-plots==0.3.9rc3", keep_going=True)
else:
    # Pyodide Environment preparing scikit-plots>=0.4...
    import scikitplot as skplt

    skplt._utils._wasm.pyodide_env()
    skplt._utils._wasm._clear_console()
finally:
    clear_output(wait=True)
