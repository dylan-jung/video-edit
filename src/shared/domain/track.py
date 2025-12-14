from dataclasses import dataclass

@dataclass
class Track:
    src: str
    start: str
    end: str
    duration: str
    trim_in: str
    trim_out: str
