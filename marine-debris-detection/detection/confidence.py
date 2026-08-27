def classify_confidence(confidence):

    confidence = float(confidence)

    if confidence >= 0.80:
        return "HIGH"

    if confidence >= 0.50:
        return "MEDIUM"

    if confidence >= 0.25:
        return "LOW"

    return "VERY LOW"


def detection_status(confidence, threshold=0.25):

    confidence = float(confidence)

    if confidence < threshold:
        return "REJECTED"

    if confidence >= 0.80:
        return "CONFIRMED"

    if confidence >= 0.50:
        return "PROBABLE"

    return "LOW CONFIDENCE"


def calculate_confidence_score(confidence):

    confidence = max(
        0.0,
        min(1.0, float(confidence))
    )

    return round(
        confidence * 100,
        2
    )