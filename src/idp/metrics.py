"""Field-level metrics: precision, recall and F1 per field, plus document-level accuracy."""

from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from idp.normalize import norm_amount, norm_text

HEADER_FIELDS = ("subtotal", "tax", "total")


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __iadd__(self, other: "Counts") -> "Counts":
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def is_perfect(self) -> bool:
        return self.fp == 0 and self.fn == 0


def score_header_field(pred, gt) -> Counts:
    """A wrong value counts as both a false positive and a false negative."""
    p, g = norm_amount(pred), norm_amount(gt)
    if p == g:
        return Counts(tp=1) if g is not None else Counts()
    return Counts(fp=int(p is not None), fn=int(g is not None))


def _item_key(item) -> tuple:
    return norm_text(item.name), norm_amount(item.price)


def score_items(pred_items, gt_items) -> Counts:
    """An item matches only if both normalized name and price match (multiset match)."""
    pred = Counter(_item_key(i) for i in pred_items)
    gt = Counter(_item_key(i) for i in gt_items)
    tp = sum((pred & gt).values())
    return Counts(tp=tp, fp=sum(pred.values()) - tp, fn=sum(gt.values()) - tp)


def names_similar(a: str | None, b: str | None, threshold: float = 0.8) -> bool:
    """Lenient name match: high similarity, or the shorter name is contained in the longer."""
    if a == b:
        return True
    if not a or not b:
        return False
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 4 and short in long_:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


def score_items_fuzzy(pred_items, gt_items) -> Counts:
    """Price must match exactly, name may differ slightly (e.g. '1Prs Sop Sui Jiao')."""
    unused = [(norm_text(p.name), norm_amount(p.price)) for p in pred_items]
    tp = 0
    for gt in gt_items:
        g_name, g_price = norm_text(gt.name), norm_amount(gt.price)
        candidates = [
            (SequenceMatcher(None, g_name or "", name or "").ratio(), idx)
            for idx, (name, price) in enumerate(unused)
            if price == g_price and names_similar(g_name, name)
        ]
        if candidates:
            _, best = max(candidates)
            unused.pop(best)
            tp += 1
    return Counts(tp=tp, fp=len(pred_items) - tp, fn=len(gt_items) - tp)


def score_document(pred, gt) -> dict[str, Counts]:
    scores = {f: score_header_field(getattr(pred, f), getattr(gt, f)) for f in HEADER_FIELDS}
    scores["items"] = score_items(pred.items, gt.items)
    scores["items_fuzzy"] = score_items_fuzzy(pred.items, gt.items)
    return scores


def aggregate(per_doc: list[dict[str, Counts]]) -> dict:
    """Sum counts over documents and compute summary metrics."""
    totals: dict[str, Counts] = {}
    header_perfect = 0
    fully_perfect = 0
    for scores in per_doc:
        for name, counts in scores.items():
            totals.setdefault(name, Counts())
            totals[name] += counts
        if all(scores[f].is_perfect for f in HEADER_FIELDS):
            header_perfect += 1
            if scores["items"].is_perfect:
                fully_perfect += 1

    n = len(per_doc) or 1
    return {
        "fields": {
            name: {
                "precision": round(c.precision, 4),
                "recall": round(c.recall, 4),
                "f1": round(c.f1, 4),
                **asdict(c),
            }
            for name, c in totals.items()
        },
        # Share of documents where every header field is correct
        "header_doc_accuracy": round(header_perfect / n, 4),
        # Share of documents extracted with zero errors, including all line items
        "full_doc_accuracy": round(fully_perfect / n, 4),
    }
