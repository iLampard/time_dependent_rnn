import enum


class StrEnum(str, enum.Enum):
    """ Define a string enum class """
    pass


class RunnerPhase(StrEnum):
    """ Model Runner Phase Enum """
    TRAIN = 'train'
    VALIDATE = 'validate'
    PREDICT = 'predict'
