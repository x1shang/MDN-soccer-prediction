# Data sources and split definitions

The packaged NPZ files contain synthetic trajectories generated from the anchor parameters frozen in `simulation/frozen_prior.json`. Source measurements are not redistributed in this bundle: for every fitted anchor, only the parameter vector and a hash of the source record are included. Reuse of the derived parameters should follow the terms of the corresponding publication.

| Source | Role in this package |
|---|---|
| DFL open tracking | Two airborne penalty anchors contribute launch and aerodynamic parameters. Eleven screened match-flight segments contribute aerodynamic parameters only. The dataset is released under CC BY 4.0. [Bassek et al., dataset](https://doi.org/10.6084/m9.figshare.28196177.v1) · [Data paper](https://doi.org/10.1038/s41597-025-04505-y) |
| Asai three-trial set | Three 100 Hz tracked curve kicks from [Asai et al., Proceedings 49, 88 (2020)](https://doi.org/10.3390/proceedings2020049088) (MDPI, CC BY 4.0) contribute fitted launch and aerodynamic parameters. The source observation records are not redistributed. Because these trials enter the fitted prior, replay errors on them are retrospective. |
| ISSIA-3D | Three screened multiview clips contribute fitted aerodynamic parameters. Raw videos and annotations are distributed upstream; consult the release terms. [ISSIA-3D description](https://github.com/mguti97/SoccerNet-v3D#issia-3d) |
| Barber and Carré | One published planar trajectory was digitized from Figure 5 and used for aerodynamic fitting. [Paper](https://eprints.whiterose.ac.uk/id/eprint/98036/) |
| Ground-contact sensitivity | Natural-turf rebound and rolling measurements inform the sensitivity range. The collision friction coefficient is an exploratory input. [Baker turf measurements](https://store.astm.org/stp25348s.html) · [Rezaei et al. oblique impact](https://doi.org/10.1080/02640414.2011.587443) |

`data/v3/data.npz` and `data/v3/bounce/data_bounce.npz` share the same row order, launch vectors, anchor IDs, and split codes. `observations` stores the ball-centre positions at 50 and 100 ms. `endpoint` stores the pair `(Y, Z)` at the 11 m plane, in metres, and `arrival` stores its time in seconds. Status codes are 0 for timeout, 1 for direct airborne crossing, 2 for first ground contact before crossing in the airborne dataset or no recovered crossing in the extension, and 3 for crossing after ground contact. The extension distinguishes bouncing and rolling crossings with `contact_kind` values 1 and 2.

Both index archives contain disjoint train, selection, calibration, and test row IDs. The original airborne rows retain their exact memberships. The extended test indices define the common evaluation set for both training arms. All three Asai anchors are present in this v3 prior. The [source-held-out audit](results/source_holdout/README.md) reports a separate earlier protocol.
