import argparse
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


IMAGE_NAME = re.compile(r"^grade_(\d+)_.+\.png$", re.IGNORECASE)


def predict(url, image_path):
    boundary = uuid.uuid4().hex
    image_bytes = image_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("utf-8")
    body += image_bytes + f"\r\n--{boundary}--\r\n".encode("ascii")
    request = urllib.request.Request(
        url.rstrip("/") + "/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(
        description="Compare grades in image filenames with predictions from a running API."
    )
    parser.add_argument("image_folder", type=Path, help="Folder of grade_<n>_*.png images")
    parser.add_argument("--url", required=True, help="Base URL of the running API")
    args = parser.parse_args()

    if not args.image_folder.is_dir():
        parser.error(f"not a folder: {args.image_folder}")

    images = []
    for image_path in sorted(args.image_folder.iterdir()):
        match = IMAGE_NAME.fullmatch(image_path.name)
        if image_path.is_file() and match:
            images.append((image_path, int(match.group(1))))
    if not images:
        parser.error(f"no grade_<n>_something.png images found in {args.image_folder}")

    print(f"{'File':<36} {'True':>5} {'Predicted':>10} {'Confidence':>12} {'Low confidence':>15} {'Time (ms)':>11}")
    print("-" * 96)
    correct = 0
    for image_path, true_grade in images:
        started = time.perf_counter()
        try:
            result = predict(args.url, image_path)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"API returned HTTP {error.code} for {image_path.name}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach API for {image_path.name}: {error}") from error
        elapsed_ms = (time.perf_counter() - started) * 1000

        predicted_grade = int(result["grade"])
        confidence = float(result["confidence"])
        low_confidence = bool(result["low_confidence"])
        correct += predicted_grade == true_grade
        print(
            f"{image_path.name:<36.36} {true_grade:>5} {predicted_grade:>10} "
            f"{confidence:>12.4f} {str(low_confidence):>15} {elapsed_ms:>11.1f}"
        )

    accuracy = correct / len(images)
    print("-" * 96)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(images)})")


if __name__ == "__main__":
    main()
