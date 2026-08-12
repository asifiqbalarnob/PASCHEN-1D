# Third-Party Data and Software Notices

PASCHEN-1D uses NumPy, SciPy, Matplotlib, tqdm, and optionally Numba and Jupyter. Their licenses remain with their respective authors and distributions.

The bundled electron swarm tables were generated with BOLSIG+ from collision data obtained through LXCat. The bundled positive-ion tables are normalized from LXCat ion-swarm exports. Each normalized ion file and both data manifests retain database names, permalinks, references, retrieval information, source-file checksums, and dataset identity. Users must cite the original datasets and comply with the applicable LXCat database terms.

The BOLSIG+ executable, associated BOLSIG+ files, and verbatim LXCat downloads are not distributed with PASCHEN-1D. Obtain BOLSIG+ directly from <https://www.bolsig.laplace.univ-tlse.fr/download.html> and LXCat source data directly from <https://www.lxcat.net/>. Raw ion downloads created by `tools/download_lxcat_ion_data.py` remain in the local, Git-ignored `ion_swarm_data/raw_lxcat/` workspace.

PASCHEN-1D does not claim ownership of third-party collision, electron-swarm, or ion-swarm measurements. See `electron_swarm_data/manifest.json`, `ion_swarm_data/normalized_lxcat_2026-07-21/manifest.json`, and the metadata headers in each normalized ion table for source-specific provenance.
