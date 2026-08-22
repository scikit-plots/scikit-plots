// Related to https://github.com/scikit-learn/scikit-learn/issues/30279
// There an interaction between plotly and bootstrap/pydata-sphinx-theme
// that causes plotly figures to not detect the right-hand sidebar width

// Plotly figures are responsive, this triggers a resize event once the DOM has
// finished loading so that they resize themselves.

document.addEventListener("DOMContentLoaded", () => {
  window.dispatchEvent(new Event("resize"));
});

// function resizePlotlyGraphs() {
//     const plotlyDivs = document.getElementsByClassName("plotly-graph-div");
//     for (const div of plotlyDivs) {
//         Plotly.Plots.resize(div);
//     }
// }
// window.addEventListener("resize", resizePlotlyGraphs);
// document.addEventListener("DOMContentLoaded", resizePlotlyGraphs);
