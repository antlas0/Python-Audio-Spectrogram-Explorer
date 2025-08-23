from typing import Optional, Any
from dataclasses import dataclass
import datetime
import pandas as pd

MAX_LENGHT_SEC = 60 * 10


LICENCE_STR="""
CC BY-NC 4.0
<br/><br/>
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
This is a human-readable summary of (and not a substitute for) the license.<br/>
Disclaimer.
<br/><br/>
You are free to:
<br/><br/>
Share — copy and redistribute the material in any medium or format<br/>
Adapt — remix, transform, and build upon the material<br/>
The licensor cannot revoke these freedoms as long as you follow the license terms.<br/>
Under the following terms:<br/>
Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
<br/><br/>
NonCommercial — You may not use the material for commercial purposes.
<br/><br/>
No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.<br/>
Notices:<br/>
You do not have to comply with the license for elements of the material in the public domain or where your use is permitted by an applicable exception or limitation.
No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.
"""


def str2bool(s:str) -> bool:
    res = {
        "true": True,
        "false": False,
    }
    return res.get(s.lower(), False)


@dataclass
class AudioSample:
    x: Any
    fs: int


@dataclass
class FFTSample:
    ssx: Any
    f: Any
    t: Any


@dataclass
class InputData:
    filename: str
    start: int=-1
    end: int=-1
    date: datetime.datetime=datetime.datetime(1970, 1, 1)
    audio_data: Optional[AudioSample]=None
    fft_data: Optional[FFTSample]=None
    annotations: Optional[pd.DataFrame]=None
    annotations_file: Optional[str]=None
    drawing: Optional[pd.DataFrame]=None
