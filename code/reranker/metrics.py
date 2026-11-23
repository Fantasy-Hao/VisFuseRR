from statistics import mean, median

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
    """Score each retrieved, relevant query-page pair with valid ground-truth boxes."""
    results_by_query = {row["qid"]: row for row in results}
    pages = []
    predicted_areas, gt_areas = [], []
    n_gt_pages = n_facts = covered_facts = 0
    for query in queries:
        qid = query["qid"]
        result = results_by_query[qid]
        predictions = {row["page_uid"]: row["evidence_boxes"] for row in result.get("evidence", [])}
        n_gt_pages += len(query["gt_boxes"])
        for alternatives in query.get("gt_facts", []):
            n_facts += 1
            covered_facts += any(
                best_overlap(predictions.get(uid, []), [box])[0] >= 0.5
                for uid, box in alternatives
            )
        for uid in result["ranking"]:
            truth = query["gt_boxes"].get(uid)
            if not truth:
                continue
            predicted = predictions.get(uid, [])
            iou, containment = best_overlap(predicted, truth)
            pages.append({
                "qid": qid, "page_uid": uid, "n_predicted_boxes": len(predicted),
                "n_gt_boxes": len(truth), "best_iou": iou, "best_containment": containment,
            })
            predicted_areas.extend(box_area(box) / 1_000_000 for box in predicted)
            gt_areas.extend(box_area(box) / 1_000_000 for box in truth)

    n = len(pages)
    with_predictions = sum(page["n_predicted_boxes"] > 0 for page in pages)
    mean_prediction = mean(predicted_areas) if predicted_areas else None
    mean_gt = mean(gt_areas) if gt_areas else None
    matched = sum(page["best_iou"] >= 0.5 for page in pages)
    return {
        "n_pages": n,
        "n_gt_pages": n_gt_pages,
        "n_pages_with_predictions": with_predictions,
        "n_predicted_boxes": len(predicted_areas),
        "n_gt_boxes": len(gt_areas),
        "mean_iou": mean(page["best_iou"] for page in pages) if n else None,
        "IoU@0.3": mean(page["best_iou"] >= 0.3 for page in pages) if n else None,
        "IoU@0.5": matched / n if n else None,
        "end_to_end_IoU@0.5": matched / n_gt_pages if n_gt_pages else None,
        "n_gt_facts": n_facts,
        "fact_coverage@0.5": covered_facts / n_facts if n_facts else None,
        "mean_containment": mean(page["best_containment"] for page in pages) if n else None,
        "containment@0.5": mean(page["best_containment"] >= 0.5 for page in pages) if n else None,
        "containment@0.8": mean(page["best_containment"] >= 0.8 for page in pages) if n else None,
        "mean_predicted_box_area": mean_prediction,
        "median_predicted_box_area": median(predicted_areas) if predicted_areas else None,
        "mean_gt_box_area": mean_gt,
        "predicted_to_gt_mean_area_ratio": mean_prediction / mean_gt if mean_prediction is not None else None,
        "predicted_box_area_gt_0.8_rate": mean(area > 0.8 for area in predicted_areas) if predicted_areas else None,
        "mean_boxes_per_predicted_page": len(predicted_areas) / with_predictions if with_predictions else None,
        "per_page": pages,
    }
