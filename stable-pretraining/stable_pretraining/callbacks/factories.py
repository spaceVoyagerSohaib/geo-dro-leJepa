from .checkpoint_sklearn import SklearnCheckpoint, WandbCheckpoint
from .trainer_info import LoggingCallback, ModuleSummary, TrainerInfo, SLURMInfo
from .env_info import EnvironmentDumpCallback
from .registry import ModuleRegistryCallback
from .run_manifest import RunManifestCallback
from .unused_parameters import LogUnusedParametersOnce


def default():
    """Factory function that returns default callbacks."""
    callbacks = [
        # RichProgressBar(),
        ModuleRegistryCallback(),
        LoggingCallback(),
        EnvironmentDumpCallback(async_dump=True),
        RunManifestCallback(),
        TrainerInfo(),
        SklearnCheckpoint(),
        WandbCheckpoint(),
        ModuleSummary(),
        SLURMInfo(),
        LogUnusedParametersOnce(),
    ]

    return callbacks
