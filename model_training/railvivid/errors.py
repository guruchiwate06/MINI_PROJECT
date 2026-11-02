class PipelineError(RuntimeError):
    """An actionable pipeline failure; original exception is chained."""

class ConfigurationError(PipelineError): pass
class DataIntegrityError(PipelineError): pass
class LeakageError(DataIntegrityError): pass
class ArtifactError(PipelineError): pass
class TrainingError(PipelineError): pass
