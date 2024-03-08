import os
from torch import Tensor
import data.transforms as T

from data.model import SpeciesConfiguration, OrcaConfiguration
from utilities.viewing import plot_spectrogram


def get_transforms(configuration: SpeciesConfiguration) -> T.Compose:
    transforms = [
        lambda fn: T.load_audio_file(
            file_name=fn,
            sr=configuration.sample_rate),
        T.PreEmphasize(
            factor=configuration.pre_emphasis),
        T.Spectrogram(
            n_fft=configuration.n_fft,
            hop_length=configuration.hop_length,
            center=False),
        T.Amp2Db(
            min_level_db=configuration.min_level_db,
            stype="power")
    ]

    if configuration.normalization_method == "min_max":
        transforms.append(
            T.MinMaxNormalize()
        )
    else:
        transforms.append(
            T.Normalize(min_level_db=configuration.min_level_db,
                        ref_level_db=configuration.ref_level_db)
        )

    transforms.append(
        T.PaddedSubsequenceSampler(
            sequence_length=128,
            random=False,
            dim=1)
    )
    return T.Compose(transforms)


def load_create_spectrogram(audio_file: str, configuration: SpeciesConfiguration) -> Tensor:
    t = get_transforms(configuration=configuration)
    return t(audio_file)


def preprocess(audio_file: str, configuration: SpeciesConfiguration) -> Tensor:
    spectrogram = load_create_spectrogram(audio_file, configuration)
    return spectrogram


if __name__ == '__main__':
    file = os.path.join(os.getcwd(), "samples/orca.wav")
    s_conf = OrcaConfiguration()
    s = preprocess(file, s_conf)
    plot_spectrogram(s, s_conf)
