from idp.features import build_features, is_correct, textract_signals_from_result


def _receipt(total="16,500", subtotal="15,000", tax="1,500", items=None):
    if items is None:
        items = [{"name": "Tea", "price": "5,000"}, {"name": "Rice", "price": "10,000"}]
    return {"total": total, "subtotal": subtotal, "tax": tax, "items": items}


def test_agreement_and_arithmetic_features():
    f = build_features(_receipt(), _receipt(), {})
    assert f["n_agree_header"] == 3
    assert f["items_agreement_f1"] == 1.0
    assert f["items_sum_eq_subtotal"] == 1.0
    assert f["subtotal_plus_tax_eq_total"] == 1.0


def test_disagreement_is_visible():
    f = build_features(_receipt(total="16,500"), _receipt(total="61,500", tax=None), {})
    assert f["agree_total"] == 0.0
    assert f["one_missing_tax"] == 1.0
    assert f["n_agree_header"] == 1


def test_textract_signals_handle_missing_fields():
    signals = textract_signals_from_result({})
    assert signals["tx_conf_total"] == 0.0
    assert signals["ocr_low_conf_share"] == 1.0

    result = {"ExpenseDocuments": [{
        "SummaryFields": [{"Type": {"Text": "TOTAL"}, "ValueDetection": {"Confidence": 90.0}}],
        "Blocks": [{"BlockType": "WORD", "Confidence": 70.0}, {"BlockType": "WORD", "Confidence": 99.0}],
    }]}
    signals = textract_signals_from_result(result)
    assert signals["tx_conf_total"] == 0.9
    assert signals["ocr_low_conf_share"] == 0.5


def test_is_correct_targets():
    ok = {"tp": 1, "fp": 0, "fn": 0}
    bad = {"tp": 0, "fp": 1, "fn": 1}
    record = {"scores": {"subtotal": ok, "tax": ok, "total": ok, "items": bad}}
    assert is_correct(record, "header") is True
    assert is_correct(record, "full") is False
