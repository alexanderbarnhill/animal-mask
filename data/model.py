from dataclasses import dataclass


@dataclass
class SpeciesConfiguration:
    sample_rate: int
    hop_length: int
    n_fft: int
    pre_emphasis: float
    minimum_frequency: int
    maximum_frequency: int
    min_level_db: int = -100
    ref_level_db: int = 20
    normalization_method: str = "min_max"


class OrcaConfiguration(SpeciesConfiguration):
    sample_rate = 44100
    hop_length = 441
    n_fft = 4096
    pre_emphasis = 0.98
    minimum_frequency = 500
    maximum_frequency = 10000
    min_level_db = -100
    ref_level_db = 20
    normalization_method = "db"

    def __init__(self):
        super().__init__(self.sample_rate,
                         self.hop_length,
                         self.n_fft,
                         self.pre_emphasis,
                         self.minimum_frequency,
                         self.maximum_frequency,
                         self.min_level_db,
                         self.ref_level_db,
                         self.normalization_method)

