# Data sources and terms

All inputs come from one record: the IUTF dataset.

- **IUTF dataset.** Lin, X. et al. IUTF Dataset: enabling cross-border resource for analysing the impact of rainfall on
  urban transportation. *Sci. Data* 13, 22 (2026). Data: <https://doi.org/10.6084/m9.figshare.30022807.v1>, licence
  CC BY 4.0.
- **Traffic data.** The detector flows in IUTF were derived from UTD19 (Loder, A., Ambühl, L., Menendez, M. & Axhausen,
  K. W. Understanding traffic capacity of urban networks. *Sci. Rep.* 9, 16283 (2019); <https://utd19.ethz.ch>).
- **Road networks.** Derived from OpenStreetMap, © OpenStreetMap contributors, Open Database Licence (ODbL 1.0).
- **Rainfall.** Derived from the ERA5 reanalysis (Hersbach, H. et al. *Q. J. R. Meteorol. Soc.* 146, 1999–2049 (2020);
  Copernicus Climate Change Service, <https://doi.org/10.24381/cds.adbb2d47>). Contains modified Copernicus Climate Change
  Service information 2026.

## Processed data and Source Data in this repository

The processed station tables (`out/x0/`), the per-city results of each stage (`out/main34/`, `out/e*/`, `out/checks/`)
and the Source Data (`source_data/`) are processed from the IUTF dataset and are released under the Creative Commons
Attribution 4.0 International licence (CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>). When you use them,
please cite the paper and the IUTF dataset.
