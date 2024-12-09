.. _citing_scikit_plots:

.. raw:: html

    <div class="citation-container">
        <!-- Citation for latest version -->
        <div class="citation-item">
            <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13367000.svg" alt="DOI Badge for scikit-plots" />
            <a href="https://doi.org/10.5281/zenodo.13367000" target="_blank">DOI: 10.5281/zenodo.13367000</a>
            <div id="citation1">
                [1] scikit-plots, “scikit-plots: vlatest”. Zenodo, Aug. 23, 2024. 
                DOI: <a href="https://doi.org/10.5281/zenodo.13367000" target="_blank">10.5281/zenodo.13367000</a>.
            </div>
            <button class="copy-btn" onclick="copyToClipboard('citation1')">Copy to Clipboard</button>
        </div>

        <!-- Citation for specific version -->
        <div class="citation-item">
            <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.13367001.svg" alt="DOI Badge for scikit-plots v0.3.8dev0" />
            <a href="https://doi.org/10.5281/zenodo.13367001" target="_blank">DOI: 10.5281/zenodo.13367001</a>
            <div id="citation2">
                [2] scikit-plots, “scikit-plots: v0.3.8dev0”. Zenodo, Aug. 23, 2024. 
                DOI: <a href="https://doi.org/10.5281/zenodo.13367001" target="_blank">10.5281/zenodo.13367001</a>.
            </div>
            <button class="copy-btn" onclick="copyToClipboard('citation2')">Copy to Clipboard</button>
        </div>
    </div>

.. raw:: html

    <style>
        .citation-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            margin-top: 20px;
        }
        .citation-item {
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .citation-item img {
            margin-bottom: 10px;
            display: block;
        }
        .citation-item a {
            display: block;
            margin-bottom: 10px;
        }
        .copy-btn {
            padding: 5px 10px;
            border: none;
            background-color: #007bff;
            color: white;
            border-radius: 5px;
            cursor: pointer;
        }
        .copy-btn:hover {
            background-color: #0056b3;
        }
    </style>

.. raw:: html

    <script>
        function copyToClipboard(elementId) {
            var text = document.getElementById(elementId).innerText;
            navigator.clipboard.writeText(text).then(function() {
                alert('Citation copied to clipboard!');
            }, function(err) {
                alert('Failed to copy citation: ', err);
            });
        }
    </script>