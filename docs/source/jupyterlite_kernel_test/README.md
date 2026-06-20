:orphan:

```bash
docker run --rm -it cimg/python:3.12 bash
```


```bash
# JupyterLab: dark theme (optional)
pip install ipywidgets ipyevents ipympl ipycanvas ipyleaflet
pip install jupyterlab-night jupyterlab_miami_nights jupyterlab-fasta jupyterlab-geojson

# Additional jupyterlite (optional)
# Or with pip (you must install micromamba 2.0.5):
# https://github.com/jupyterlite/xeus#installation
# pip install jupyterlite-javascript-kernel
pip install -U \
    jupyterlite-core \
    jupyterlite-sphinx \
    jupyterlite-pyodide-kernel \
    jupyterlite-xeus \
    jupyterlite-terminal
```


```bash
# sudo apt-get install -y npm  # need npm trigger jupyterlite_terminal @jupyterlite/cockle
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version
npm --version
```

```bash
run:
- jupyter lite build   --debug   --output-dir build/html/stable/lite   --lite-dir source
- sphinx-build -D plot_gallery=0 -b html -d build/doctrees  -T -v  source   -j1     build/html/stable
```
