from typing import Set, List

from InquirerPy import inquirer
from InquirerPy.base import BaseSimplePrompt
from prompt_toolkit.completion import FuzzyWordCompleter
from prompt_toolkit.validation import ValidationError

_HF_LICENCES = [
    ("apache-2.0", "Apache license 2.0"),
    ("mit", "MIT"),
    ("openrail", "OpenRAIL license family"),
    ("bigscience-openrail-m", "BigScience OpenRAIL-M"),
    ("creativeml-openrail-m", "CreativeML OpenRAIL-M"),
    ("bigscience-bloom-rail-1.0", "BigScience BLOOM RAIL 1.0"),
    ("bigcode-openrail-m", "BigCode Open RAIL-M v1"),
    ("afl-3.0", "Academic Free License v3.0"),
    ("artistic-2.0", "Artistic license 2.0"),
    ("bsl-1.0", "Boost Software License 1.0"),
    ("bsd", "BSD license family"),
    ("bsd-2-clause", "BSD 2-clause \"Simplified\" license"),
    ("bsd-3-clause", "BSD 3-clause \"New\" or \"Revised\" license"),
    ("bsd-3-clause-clear", "BSD 3-clause Clear license"),
    ("c-uda", "Computational Use of Data Agreement"),
    ("cc", "Creative Commons license family"),
    ("cc0-1.0", "Creative Commons Zero v1.0 Universal"),
    ("cc-by-2.0", "Creative Commons Attribution 2.0"),
    ("cc-by-2.5", "Creative Commons Attribution 2.5"),
    ("cc-by-3.0", "Creative Commons Attribution 3.0"),
    ("cc-by-4.0", "Creative Commons Attribution 4.0"),
    ("cc-by-sa-3.0", "Creative Commons Attribution Share Alike 3.0"),
    ("cc-by-sa-4.0", "Creative Commons Attribution Share Alike 4.0"),
    ("cc-by-nc-2.0", "Creative Commons Attribution Non Commercial 2.0"),
    ("cc-by-nc-3.0", "Creative Commons Attribution Non Commercial 3.0"),
    ("cc-by-nc-4.0", "Creative Commons Attribution Non Commercial 4.0"),
    ("cc-by-nd-4.0", "Creative Commons Attribution No Derivatives 4.0"),
    ("cc-by-nc-nd-3.0", "Creative Commons Attribution Non Commercial No Derivatives 3.0"),
    ("cc-by-nc-nd-4.0", "Creative Commons Attribution Non Commercial No Derivatives 4.0"),
    ("cc-by-nc-sa-2.0", "Creative Commons Attribution Non Commercial Share Alike 2.0"),
    ("cc-by-nc-sa-3.0", "Creative Commons Attribution Non Commercial Share Alike 3.0"),
    ("cc-by-nc-sa-4.0", "Creative Commons Attribution Non Commercial Share Alike 4.0"),
    ("cdla-sharing-1.0", "Community Data License Agreement – Sharing, Version 1.0"),
    ("cdla-permissive-1.0", "Community Data License Agreement – Permissive, Version 1.0"),
    ("cdla-permissive-2.0", "Community Data License Agreement – Permissive, Version 2.0"),
    ("wtfpl", "Do What The F*ck You Want To Public License"),
    ("ecl-2.0", "Educational Community License v2.0"),
    ("epl-1.0", "Eclipse Public License 1.0"),
    ("epl-2.0", "Eclipse Public License 2.0"),
    ("etalab-2.0", "Etalab Open License 2.0"),
    ("eupl-1.1", "European Union Public License 1.1"),
    ("agpl-3.0", "GNU Affero General Public License v3.0"),
    ("gfdl", "GNU Free Documentation License family"),
    ("gpl", "GNU General Public License family"),
    ("gpl-2.0", "GNU General Public License v2.0"),
    ("gpl-3.0", "GNU General Public License v3.0"),
    ("lgpl", "GNU Lesser General Public License family"),
    ("lgpl-2.1", "GNU Lesser General Public License v2.1"),
    ("lgpl-3.0", "GNU Lesser General Public License v3.0"),
    ("isc", "ISC"),
    ("lppl-1.3c", "LaTeX Project Public License v1.3c"),
    ("ms-pl", "Microsoft Public License"),
    ("apple-ascl", "Apple Sample Code license"),
    ("mpl-2.0", "Mozilla Public License 2.0"),
    ("odc-by", "Open Data Commons License Attribution family"),
    ("odbl", "Open Database License family"),
    ("openrail++", "Open Rail++-M License"),
    ("osl-3.0", "Open Software License 3.0"),
    ("postgresql", "PostgreSQL License"),
    ("ofl-1.1", "SIL Open Font License 1.1"),
    ("ncsa", "University of Illinois/NCSA Open Source License"),
    ("unlicense", "The Unlicense"),
    ("zlib", "zLib License"),
    ("pddl", "Open Data Commons Public Domain Dedication and License"),
    ("lgpl-lr", "Lesser General Public License For Linguistic Resources"),
    ("deepfloyd-if-license", "DeepFloyd IF Research License Agreement"),
    ("llama2", "Llama 2 Community License Agreement"),
    ("llama3", "Llama 3 Community License Agreement"),
    ("llama3.1", "Llama 3.1 Community License Agreement"),
    ("gemma", "Gemma Terms of Use"),
    ("unknown", "Unknown"),
    ("other", "Other")
]

_VALID_LICENCES: List[str] = [name for name, _ in _HF_LICENCES]
_VALID_LICENCES_SET: Set[str] = set(_VALID_LICENCES)


def hf_licence(message: str, **kwargs) -> BaseSimplePrompt:
    def _fn_validate(text):
        if text not in _VALID_LICENCES_SET:
            raise ValidationError(
                message=f"Invalid licence - {text!r}",
                cursor_position=len(text)
            )
        return True

    completer = FuzzyWordCompleter(_VALID_LICENCES)

    return inquirer.text(
        message=message,
        validate=_fn_validate,
        completer=completer,
        **kwargs
    )
