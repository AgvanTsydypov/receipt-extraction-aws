from idp.metrics import Counts, aggregate, score_document, score_header_field, score_items
from idp.normalize import norm_amount, norm_text
from idp.schema import LineItem, Receipt


def test_norm_amount_ignores_separators_and_currency():
    assert norm_amount("45,500") == norm_amount("45.500") == norm_amount("Rp 45500") == "45500"
    assert norm_amount("007") == "7"
    assert norm_amount(None) is None
    assert norm_amount("n/a") is None


def test_norm_text():
    assert norm_text("  Nasi  Goreng! ") == "nasi goreng"
    assert norm_text("") is None


def test_header_field_cases():
    assert score_header_field("10,000", "10.000") == Counts(tp=1)
    assert score_header_field(None, None) == Counts()
    assert score_header_field("9,000", "10,000") == Counts(fp=1, fn=1)
    assert score_header_field(None, "10,000") == Counts(fn=1)
    assert score_header_field("10,000", None) == Counts(fp=1)


def test_items_multiset_match():
    gt = [LineItem(name="Tea", price="5,000"), LineItem(name="Tea", price="5,000"), LineItem(name="Rice", price="8,000")]
    pred = [LineItem(name="tea", price="5.000"), LineItem(name="Rice", price="9,000")]
    assert score_items(pred, gt) == Counts(tp=1, fp=1, fn=2)


def test_aggregate_doc_accuracy():
    gt = Receipt(total="10,000", items=[LineItem(name="Tea", price="10,000")])
    perfect = score_document(gt, gt)
    wrong_total = score_document(Receipt(total="1", items=gt.items), gt)
    summary = aggregate([perfect, wrong_total])
    assert summary["header_doc_accuracy"] == 0.5
    assert summary["full_doc_accuracy"] == 0.5
    assert summary["fields"]["items"]["f1"] == 1.0
