from backend.indexing import _chunk_text, _split_by_chapter


def test_split_by_chapter_finds_headings():
    text = "# Chapter 1: Intro\nbody one\n\n# Chapter 2: More\nbody two"
    sections = _split_by_chapter(text)
    assert [title for title, _ in sections] == ["Chapter 1: Intro", "Chapter 2: More"]
    assert sections[0][1] == "body one"
    assert sections[1][1] == "body two"


def test_split_by_chapter_no_headings_returns_whole_text():
    text = "just some plain text with no chapter markers"
    sections = _split_by_chapter(text)
    assert len(sections) == 1
    assert sections[0][1] == text


def test_chunk_text_respects_paragraph_boundaries_when_small():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = _chunk_text(text, size=100, overlap=10)
    # All paragraphs fit comfortably under the size limit, so they merge into one chunk.
    assert len(chunks) == 1


def test_chunk_text_splits_long_paragraph_with_overlap():
    long_paragraph = " ".join(f"word{i}" for i in range(50))
    chunks = _chunk_text(long_paragraph, size=20, overlap=5)
    assert len(chunks) > 1
    # Every chunk should stay within the requested word budget.
    assert all(len(c.split()) <= 20 for c in chunks)
