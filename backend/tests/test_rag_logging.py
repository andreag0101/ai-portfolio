import logging

from app.rag.logging_utils import log_ask


def test_log_ask_emits_one_info_line_with_expected_fields(caplog):
    with caplog.at_level(logging.INFO, logger="app.rag"):
        with log_ask("a test question") as fields:
            fields["mode"] = "generated"
            fields["top_score"] = 0.42
            fields["num_citations"] = 3

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "a test question" in message
    assert "mode=generated" in message
    assert "top_score=0.42" in message
    assert "num_citations=3" in message


def test_log_ask_still_logs_and_reraises_on_exception(caplog):
    with caplog.at_level(logging.INFO, logger="app.rag"):
        try:
            with log_ask("a question that errors") as fields:
                fields["mode"] = "generated"
                raise ValueError("boom")
        except ValueError:
            pass
        else:
            raise AssertionError("expected the ValueError to propagate")

    assert len(caplog.records) == 1
    assert "a question that errors" in caplog.records[0].getMessage()
