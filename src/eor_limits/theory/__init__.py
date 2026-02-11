"""Module for dealing with theory datasets."""
from pathlib import Path

THEORY_PATH = Path(__file__).parent.resolve()
KNOWN_THEORIES = {
 'Mesinger2016Faint' : THEORY_PATH / "mesinger_2016_faint_galaxies/",
 'Mesinger2016Bright': THEORY_PATH / "mesinger_2016_bright_galaxies/",
 'Munoz2018FDM3': THEORY_PATH / "munoz_2018_fdm3.yaml",
 'Munoz2022AllGalaxies': THEORY_PATH / "munoz_2022_allgalaxies/",
 'Munoz2022Optimistic': THEORY_PATH / "munoz_2022_optimistic/",
 'PaganoLiu2020Beta1.00': THEORY_PATH / "pagano_liu_2020.npz",
 'PaganoLiu2020Beta0.84': THEORY_PATH / "pagano_liu_2020.npz",
 'PaganoLiu2020Beta0.76': THEORY_PATH / "pagano_liu_2020.npz",
 'PaganoLiu2020Beta0.68': THEORY_PATH / "pagano_liu_2020.npz",
 'PaganoLiu2020Beta0.36': THEORY_PATH / "pagano_liu_2020.npz",
 'PaganoLiu2020Beta0.00': THEORY_PATH / "pagano_liu_2020.npz",
 'PaganoLiu2020Beta-0.36': THEORY_PATH / "pagano_liu_2020.npz",
 'PaganoLiu2020Beta-0.68': THEORY_PATH / "pagano_liu_2020.npz",
 'PaganoLiu2020Beta-0.76': THEORY_PATH / "pagano_liu_2020.npz",
 'PaganoLiu2020Beta-0.84': THEORY_PATH / "pagano_liu_2020.npz",
 'PaganoLiu2020Beta-1.00': THEORY_PATH / "pagano_liu_2020.npz",
}
__all_theories__ = {}
