from statistics import mean

from retriever.metrics import evaluate as evaluate_ranking


def evaluate(queries, before, results):
    after = {row["qid"]: row["ranking"] for row in results}
    return {"before": evaluate_ranking(queries, before), "after": evaluate_ranking(queries, after)}


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def best_overlap(predicted, ground_truth):
    best_iou = best_containment = 0.0
    for box in predicted:
        area = box_area(box)
        for truth in ground_truth:
            width = max(0, min(box[2], truth[2]) - max(box[0], truth[0]))
            height = max(0, min(box[3], truth[3]) - max(box[1], truth[1]))
            intersection = width * height
            best_iou = max(best_iou, intersection / (area + box_area(truth) - intersection))
            best_containment = max(best_containment, intersection / area)
    return best_iou, best_containment


def evaluate_localization(queries, results):
    """Score localization on annotated pages, with missed pages counted end to end."""
    results_by_query = {row["qid"]: row for row in results}
    pages = []
    n_gt_pages = n_facts = covered_facts = 0
    for query in queries:
        qid = query["qid"]
        result = results_by_query[qid]
        ranking = set(result["ranking"])
        predictions = {
            row["page_uid"]: row["evidence_boxes"]
            for row in result.get("evidence", []) if row["page_uid"] in ranking
        }
        truth_by_page = {uid: boxes for uid, boxes in query.get("gt_boxes", {}).items() if boxes}
        n_gt_pages += len(truth_by_page)
        for alternatives in query.get("gt_facts", []):
            n_facts += 1
            covered_facts += any(
                best_overlap(predictions.get(uid, []), [box])[0] >= 0.5
                for uid, box in alternatives
            )
        for uid in result["ranking"]:
            truth = truth_by_page.get(uid)
            if not truth:
                continue
            predicted = predictions.get(uid, [])
            iou, containment = best_overlap(predicted, truth)
            pages.append({
                "qid": qid, "page_uid": uid, "n_predicted_boxes": len(predicted),
                "n_gt_boxes": len(truth), "best_iou": iou, "best_containment": containment,
            })

    n = len(pages)
    matched = sum(page["best_iou"] >= 0.5 for page in pages)
    return {
        "n_pages": n,
        "n_gt_pages": n_gt_pages,
        "n_pages_with_predictions": sum(page["n_predicted_boxes"] > 0 for page in pages),
        "mean_iou": mean(page["best_iou"] for page in pages) if n else None,
        "IoU@0.5": matched / n if n else None,
        "end_to_end_IoU@0.5": matched / n_gt_pages if n_gt_pages else None,
        "n_gt_facts": n_facts,
        "fact_coverage@0.5": covered_facts / n_facts if n_facts else None,
        "mean_containment": mean(page["best_containment"] for page in pages) if n else None,
        "containment@0.5": mean(page["best_containment"] >= 0.5 for page in pages) if n else None,
        "per_page": pages,
    }
