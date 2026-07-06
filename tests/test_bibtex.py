from hep_rag.ingest.bibtex import parse_bibtex
from hep_rag.ingest.living_review_sources import arxiv_id_from_entry


def test_parse_multiline_bibtex_entry() -> None:
    text = r'''
@article{Bothmann:2026dar,
    author = "Bothmann, Enrico and Jan{\ss}en, Timo",
    title = "{Monte Carlo Event Generation with Continuous Normalizing Flows}",
    eprint = "2604.03511",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "FERMILAB-PUB-25-0468-T",
    month = "4",
    year = "2026"
}
'''

    entries = parse_bibtex(text)

    assert len(entries) == 1
    assert entries[0].key == "Bothmann:2026dar"
    assert entries[0].fields["eprint"] == "2604.03511"
    assert entries[0].fields["primaryclass"] == "hep-ph"
    assert arxiv_id_from_entry(entries[0]) == "2604.03511"


def test_parse_old_style_arxiv_id() -> None:
    text = r'''
@article{Example:1999abc,
    title = "{Old style arXiv ID}",
    eprint = "hep-ph/9901001",
    archivePrefix = "arXiv"
}
'''

    entry = parse_bibtex(text)[0]

    assert arxiv_id_from_entry(entry) == "hep-ph/9901001"


def test_skip_non_arxiv_entry() -> None:
    text = r'''
@article{Peterson:1993nk,
    title = "{JETNET 3.0: A Versatile artificial neural network package}",
    doi = "10.1016/0010-4655(94)90120-1",
    year = "1994"
}
'''

    entry = parse_bibtex(text)[0]

    assert arxiv_id_from_entry(entry) is None
