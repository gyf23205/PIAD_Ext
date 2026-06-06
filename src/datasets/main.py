from .spoofing import SpoofingDataset
from .spoofing_flat import SpoofingDatasetFlat
from .spoofing_physical import SpoofingDatasetPhysical
from .spoofing_state_only import SpoofingDatasetStateOnly
from .spoofing_other_attack import SpoofingDatasetOther
from .spoofing_state_only_other_attack import SpoofingDatasetStateOnlyOther
from .spoofing_unsupervised import SpoofingDatasetUnsupervised
from .spoofing_unsupervised_other_attack import SpoofingDatasetUnsupervisedOther
from .spoofing_multi_anomaly import SpoofingMultiAnomalyPhysical
from .spoof_wind import SpoofingWindPhysical
from .ALFA import ALFA
from .Pegasus import Pegasus


def load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0,
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ("spoofing", "spoofing_flat", "spoofing_physical",
                                "spoofing_state_only", "spoofing_other_attack", "spoofing_state_only_other_attack",
                                "spoofing_unsupervised","spoofing_unsupervised_other_attack",
                                "spoofing_multi_profile", "spoofing_wind", "ALFA", "Pegasus")
    assert dataset_name in implemented_datasets

    dataset = None


    if dataset_name == "spoofing":
        dataset = SpoofingDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    if dataset_name == "spoofing_flat":
        dataset = SpoofingDatasetFlat(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    if dataset_name == "spoofing_physical":
        dataset = SpoofingDatasetPhysical(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)
    if dataset_name == "spoofing_state_only":
        dataset = SpoofingDatasetStateOnly(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    if dataset_name == "spoofing_other_attack":
        dataset = SpoofingDatasetOther(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    if dataset_name == "spoofing_state_only_other_attack":
        dataset = SpoofingDatasetStateOnlyOther(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)
    if dataset_name == "spoofing_unsupervised":
        dataset = SpoofingDatasetUnsupervised(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    if dataset_name == "spoofing_unsupervised_other_attack":
        dataset = SpoofingDatasetUnsupervisedOther(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    if dataset_name == "spoofing_multi_profile":
        dataset = SpoofingMultiAnomalyPhysical(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    if dataset_name == "spoofing_wind":
        dataset = SpoofingWindPhysical(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    if dataset_name == "ALFA":
        dataset = ALFA(root=data_path,
                        known_outlier_class=known_outlier_class,
                        n_known_outlier_classes=n_known_outlier_classes,
                        ratio_known_normal=ratio_known_normal,
                        ratio_known_outlier=ratio_known_outlier,
                        ratio_pollution=ratio_pollution,
                        random_state=random_state)

    if dataset_name == "Pegasus":
        dataset = Pegasus(root=data_path,
                        # dataset_name=dataset_name,
                        known_outlier_class=known_outlier_class,
                        n_known_outlier_classes=n_known_outlier_classes,
                        ratio_known_normal=ratio_known_normal,
                        ratio_known_outlier=ratio_known_outlier,
                        ratio_pollution=ratio_pollution)

    return dataset
