import os
import csv
import random
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = r"D:\Project\Processed_Data_v2"
OUTPUT_DIR = r"D:\Project\splits"

# Tỷ lệ mong muốn
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Seed để kết quả có thể tái lập
RANDOM_SEED = 42

# Số phương án ngẫu nhiên thử nghiệm
NUM_TRIALS = 2000


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

def validate_config():
    """
    Kiểm tra cấu hình tỷ lệ.
    """

    total_ratio = (
        TRAIN_RATIO
        + VAL_RATIO
        + TEST_RATIO
    )

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO "
            "phải bằng 1.0"
        )

    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(
            f"Không tìm thấy DATA_ROOT:\n{DATA_ROOT}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ============================================================
# UNION FIND
# ============================================================

class UnionFind:
    """
    Disjoint Set Union / Union-Find.

    Dùng để gom các sequence có quan hệ target-source
    vào cùng một connected component.
    """

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, item):
        """
        Thêm node nếu chưa tồn tại.
        """

        if item not in self.parent:

            self.parent[item] = item
            self.rank[item] = 0

    def find(self, item):
        """
        Tìm root của node.
        """

        if item not in self.parent:
            self.add(item)

        if self.parent[item] != item:

            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(self, first, second):
        """
        Gộp hai node vào cùng một component.
        """

        self.add(first)
        self.add(second)

        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        if (
            self.rank[root_first]
            < self.rank[root_second]
        ):

            root_first, root_second = (
                root_second,
                root_first
            )

        self.parent[root_second] = root_first

        if (
            self.rank[root_first]
            == self.rank[root_second]
        ):

            self.rank[root_first] += 1


# ============================================================
# PARSE FILENAME
# ============================================================

def parse_filename(file_name):
    """
    Phân tích filename của FaceForensics++.

    Real:
        008.npy
        -> target = 008
        -> source = ""

    Fake:
        008_990.npy
        -> target = 008
        -> source = 990
    """

    stem = os.path.splitext(
        file_name
    )[0]

    parts = stem.split("_")

    # --------------------------------------------------------
    # REAL
    # --------------------------------------------------------

    if len(parts) == 1:

        sequence_id = parts[0]

        return {
            "target_id": sequence_id,
            "source_id": "",
            "is_fake": False
        }

    # --------------------------------------------------------
    # FAKE
    # --------------------------------------------------------

    if len(parts) == 2:

        target_id = parts[0]
        source_id = parts[1]

        return {
            "target_id": target_id,
            "source_id": source_id,
            "is_fake": True
        }

    raise ValueError(
        f"Filename không đúng định dạng: "
        f"{file_name}"
    )


# ============================================================
# LOAD SAMPLES
# ============================================================

def load_samples():
    """
    Đọc toàn bộ .npy trong:

        Processed_Data_v2/Real
        Processed_Data_v2/Fake/<method>

    Đồng thời xây quan hệ target-source.
    """

    samples = []

    union_find = UnionFind()

    # ========================================================
    # REAL
    # ========================================================

    real_dir = os.path.join(
        DATA_ROOT,
        "Real"
    )

    real_files = sorted([
        file_name

        for file_name
        in os.listdir(real_dir)

        if file_name.lower().endswith(".npy")
    ])

    if not real_files:

        raise RuntimeError(
            "Không tìm thấy file .npy trong thư mục Real."
        )

    for file_name in real_files:

        parsed = parse_filename(
            file_name
        )

        target_id = parsed[
            "target_id"
        ]

        union_find.add(
            target_id
        )

        samples.append({
            "relative_path": os.path.join(
                "Real",
                file_name
            ),

            "file_name": file_name,

            "method": "Real",

            "label": 0,

            "target_id": target_id,

            "source_id": ""
        })

    # ========================================================
    # FAKE
    # ========================================================

    fake_root = os.path.join(
        DATA_ROOT,
        "Fake"
    )

    fake_methods = sorted([
        method

        for method
        in os.listdir(fake_root)

        if os.path.isdir(
            os.path.join(
                fake_root,
                method
            )
        )
    ])

    if not fake_methods:

        raise RuntimeError(
            "Không tìm thấy thư mục Fake."
        )

    for method in fake_methods:

        method_dir = os.path.join(
            fake_root,
            method
        )

        fake_files = sorted([
            file_name

            for file_name
            in os.listdir(method_dir)

            if file_name.lower().endswith(".npy")
        ])

        for file_name in fake_files:

            parsed = parse_filename(
                file_name
            )

            target_id = parsed[
                "target_id"
            ]

            source_id = parsed[
                "source_id"
            ]

            # ------------------------------------------------
            # Gộp target và source
            # ------------------------------------------------

            union_find.union(
                target_id,
                source_id
            )

            samples.append({
                "relative_path": os.path.join(
                    "Fake",
                    method,
                    file_name
                ),

                "file_name": file_name,

                "method": method,

                "label": 1,

                "target_id": target_id,

                "source_id": source_id
            })

    return samples, union_find


# ============================================================
# BUILD CONNECTED GROUPS
# ============================================================

def assign_group_ids(
    samples,
    union_find
):
    """
    Gán group_id cho từng sample.

    Ví dụ:

        008_990
        990_008

    sẽ thuộc cùng group.
    """

    root_to_group = {}

    next_group_id = 1

    for sample in samples:

        target_id = sample[
            "target_id"
        ]

        root = union_find.find(
            target_id
        )

        if root not in root_to_group:

            root_to_group[root] = (
                f"G{next_group_id:03d}"
            )

            next_group_id += 1

        sample["group_id"] = (
            root_to_group[root]
        )

    return samples


# ============================================================
# CREATE GROUP STATISTICS
# ============================================================

def create_group_statistics(samples):
    """
    Tạo thống kê cho từng connected component.
    """

    grouped_samples = defaultdict(list)

    for sample in samples:

        grouped_samples[
            sample["group_id"]
        ].append(sample)

    group_stats = []

    for group_id, group_samples in (
        grouped_samples.items()
    ):

        real_count = sum(
            1

            for sample
            in group_samples

            if sample["label"] == 0
        )

        fake_count = sum(
            1

            for sample
            in group_samples

            if sample["label"] == 1
        )

        sequence_ids = set()

        for sample in group_samples:

            sequence_ids.add(
                sample["target_id"]
            )

            if sample["source_id"]:

                sequence_ids.add(
                    sample["source_id"]
                )

        methods = sorted(
            set(
                sample["method"]

                for sample
                in group_samples
            )
        )

        group_stats.append({

            "group_id": group_id,

            "sample_count": len(
                group_samples
            ),

            "real_count": real_count,

            "fake_count": fake_count,

            "methods": methods,

            "sequence_ids": sorted(
                sequence_ids
            )
        })

    return group_stats


# ============================================================
# CALCULATE TOTALS
# ============================================================

def calculate_totals(group_stats):
    """
    Tính tổng sample / Real / Fake.
    """

    total_samples = sum(
        group["sample_count"]

        for group
        in group_stats
    )

    total_real = sum(
        group["real_count"]

        for group
        in group_stats
    )

    total_fake = sum(
        group["fake_count"]

        for group
        in group_stats
    )

    return (
        total_samples,
        total_real,
        total_fake
    )


# ============================================================
# TARGET COUNTS
# ============================================================

def calculate_targets(
    total_samples,
    total_real,
    total_fake
):
    """
    Tính số lượng mong muốn cho mỗi split.
    """

    return {

        "train": {
            "samples":
                total_samples * TRAIN_RATIO,

            "real":
                total_real * TRAIN_RATIO,

            "fake":
                total_fake * TRAIN_RATIO
        },

        "val": {
            "samples":
                total_samples * VAL_RATIO,

            "real":
                total_real * VAL_RATIO,

            "fake":
                total_fake * VAL_RATIO
        },

        "test": {
            "samples":
                total_samples * TEST_RATIO,

            "real":
                total_real * TEST_RATIO,

            "fake":
                total_fake * TEST_RATIO
        }
    }


# ============================================================
# SCORE SPLIT
# ============================================================

def calculate_score(
    current,
    targets
):
    """
    Tính điểm chất lượng của phương án split.

    Score càng nhỏ càng tốt.

    Ưu tiên:
        1. Tổng số sample
        2. Real
        3. Fake
    """

    score = 0.0

    for split_name in [
        "train",
        "val",
        "test"
    ]:

        target_samples = targets[
            split_name
        ]["samples"]

        target_real = targets[
            split_name
        ]["real"]

        target_fake = targets[
            split_name
        ]["fake"]

        actual_samples = current[
            split_name
        ]["samples"]

        actual_real = current[
            split_name
        ]["real"]

        actual_fake = current[
            split_name
        ]["fake"]

        # ----------------------------------------------------
        # Sai lệch sample
        # ----------------------------------------------------

        sample_error = (
            abs(
                actual_samples
                - target_samples
            )
            / max(
                target_samples,
                1.0
            )
        )

        # ----------------------------------------------------
        # Sai lệch Real
        # ----------------------------------------------------

        real_error = (
            abs(
                actual_real
                - target_real
            )
            / max(
                target_real,
                1.0
            )
        )

        # ----------------------------------------------------
        # Sai lệch Fake
        # ----------------------------------------------------

        fake_error = (
            abs(
                actual_fake
                - target_fake
            )
            / max(
                target_fake,
                1.0
            )
        )

        score += (
            5.0 * sample_error
            +
            1.5 * real_error
            +
            1.5 * fake_error
        )

    return score


# ============================================================
# CREATE EMPTY SPLIT
# ============================================================

def create_empty_split():
    """
    Tạo cấu trúc thống kê rỗng.
    """

    return {

        "train": {
            "samples": 0,
            "real": 0,
            "fake": 0,
            "groups": 0
        },

        "val": {
            "samples": 0,
            "real": 0,
            "fake": 0,
            "groups": 0
        },

        "test": {
            "samples": 0,
            "real": 0,
            "fake": 0,
            "groups": 0
        }
    }


# ============================================================
# GENERATE ONE CANDIDATE
# ============================================================

def generate_candidate(
    group_stats,
    targets,
    rng
):
    """
    Tạo một phương án split.

    Mỗi group chỉ được đưa vào đúng một split.
    """

    groups = list(group_stats)

    # --------------------------------------------------------
    # Random order
    # --------------------------------------------------------

    rng.shuffle(groups)

    # --------------------------------------------------------
    # Group lớn được xử lý trước
    # --------------------------------------------------------

    groups.sort(
        key=lambda group: (
            group["sample_count"],
            group["fake_count"]
        ),
        reverse=True
    )

    current = create_empty_split()

    assignments = {}

    split_names = [
        "train",
        "val",
        "test"
    ]

    # ========================================================
    # Bắt buộc 3 split đều có ít nhất 1 group
    # ========================================================

    for index in range(3):

        group = groups[index]

        split_name = split_names[index]

        group_id = group[
            "group_id"
        ]

        assignments[
            group_id
        ] = split_name

        current[
            split_name
        ]["samples"] += group[
            "sample_count"
        ]

        current[
            split_name
        ]["real"] += group[
            "real_count"
        ]

        current[
            split_name
        ]["fake"] += group[
            "fake_count"
        ]

        current[
            split_name
        ]["groups"
        ] += 1

    # ========================================================
    # Phần còn lại
    # ========================================================

    remaining_groups = groups[3:]

    rng.shuffle(
        remaining_groups
    )

    for group in remaining_groups:

        best_split = None

        best_score = float("inf")

        for split_name in split_names:

            candidate = create_empty_split()

            # ------------------------------------------------
            # Copy current stats
            # ------------------------------------------------

            for name in split_names:

                candidate[name] = (
                    current[name].copy()
                )

            # ------------------------------------------------
            # Thêm group thử nghiệm
            # ------------------------------------------------

            candidate[
                split_name
            ]["samples"] += group[
                "sample_count"
            ]

            candidate[
                split_name
            ]["real"] += group[
                "real_count"
            ]

            candidate[
                split_name
            ]["fake"] += group[
                "fake_count"
            ]

            candidate[
                split_name
            ]["groups"] += 1

            # ------------------------------------------------
            # Tính score
            # ------------------------------------------------

            score = calculate_score(
                candidate,
                targets
            )

            if score < best_score:

                best_score = score

                best_split = split_name

        # ----------------------------------------------------
        # Gán group
        # ----------------------------------------------------

        group_id = group[
            "group_id"
        ]

        assignments[
            group_id
        ] = best_split

        current[
            best_split
        ]["samples"] += group[
            "sample_count"
        ]

        current[
            best_split
        ]["real"] += group[
            "real_count"
        ]

        current[
            best_split
        ]["fake"] += group[
            "fake_count"
        ]

        current[
            best_split
        ]["groups"] += 1

    return assignments, current


# ============================================================
# FIND BEST SPLIT
# ============================================================

def find_best_split(group_stats):
    """
    Thử nhiều phương án và chọn phương án tốt nhất.
    """

    (
        total_samples,
        total_real,
        total_fake
    ) = calculate_totals(
        group_stats
    )

    targets = calculate_targets(
        total_samples,
        total_real,
        total_fake
    )

    best_assignments = None

    best_statistics = None

    best_score = float("inf")

    print(
        f"[INFO] Đang thử {NUM_TRIALS} "
        "phương án split..."
    )

    for trial in range(
        NUM_TRIALS
    ):

        rng = random.Random(
            RANDOM_SEED + trial
        )

        (
            assignments,
            statistics
        ) = generate_candidate(
            group_stats,
            targets,
            rng
        )

        # ----------------------------------------------------
        # Không chấp nhận split rỗng
        # ----------------------------------------------------

        if any(
            statistics[
                split_name
            ]["groups"] == 0

            for split_name
            in [
                "train",
                "val",
                "test"
            ]
        ):

            continue

        score = calculate_score(
            statistics,
            targets
        )

        if score < best_score:

            best_score = score

            best_assignments = assignments

            best_statistics = statistics

    if best_assignments is None:

        raise RuntimeError(
            "Không thể tạo split hợp lệ."
        )

    return (
        best_assignments,
        best_statistics,
        targets,
        best_score
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_split_csv(
    samples,
    assignments,
    split_name
):
    """
    Lưu một split thành CSV.
    """

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{split_name}.csv"
    )

    selected_samples = [

        sample

        for sample
        in samples

        if assignments[
            sample["group_id"]
        ] == split_name
    ]

    selected_samples.sort(
        key=lambda sample:
        sample["relative_path"]
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "relative_path",
            "file_name",
            "method",
            "label",
            "target_id",
            "source_id",
            "group_id"
        ])

        for sample in selected_samples:

            writer.writerow([
                sample["relative_path"],
                sample["file_name"],
                sample["method"],
                sample["label"],
                sample["target_id"],
                sample["source_id"],
                sample["group_id"]
            ])

    return output_path


# ============================================================
# CHECK GROUP LEAKAGE
# ============================================================

def check_group_leakage(
    assignments
):
    """
    Kiểm tra group xuất hiện ở nhiều split.
    """

    group_to_splits = defaultdict(set)

    for group_id, split_name in (
        assignments.items()
    ):

        group_to_splits[
            group_id
        ].add(
            split_name
        )

    leakage = {

        group_id: sorted(
            split_names
        )

        for group_id, split_names
        in group_to_splits.items()

        if len(split_names) > 1
    }

    return leakage


# ============================================================
# CHECK SEQUENCE OVERLAP
# ============================================================

def check_sequence_overlap(
    samples,
    assignments
):
    """
    Kiểm tra sequence ID có xuất hiện ở nhiều split.

    Đây là kiểm tra bổ sung.
    """

    sequences_by_split = {

        "train": set(),

        "val": set(),

        "test": set()
    }

    for sample in samples:

        split_name = assignments[
            sample["group_id"]
        ]

        sequences_by_split[
            split_name
        ].add(
            sample["target_id"]
        )

        if sample["source_id"]:

            sequences_by_split[
                split_name
            ].add(
                sample["source_id"]
            )

    train_val = (
        sequences_by_split["train"]
        &
        sequences_by_split["val"]
    )

    train_test = (
        sequences_by_split["train"]
        &
        sequences_by_split["test"]
    )

    val_test = (
        sequences_by_split["val"]
        &
        sequences_by_split["test"]
    )

    return {
        "train_val": train_val,
        "train_test": train_test,
        "val_test": val_test
    }


# ============================================================
# METHOD DISTRIBUTION
# ============================================================

def get_method_distribution(
    samples,
    assignments,
    split_name
):
    """
    Thống kê số lượng theo method.
    """

    distribution = defaultdict(int)

    for sample in samples:

        if assignments[
            sample["group_id"]
        ] == split_name:

            distribution[
                sample["method"]
            ] += 1

    return dict(
        sorted(
            distribution.items()
        )
    )


# ============================================================
# CREATE REPORT
# ============================================================

def create_report(
    samples,
    group_stats,
    assignments,
    best_score
):
    """
    Tạo split_report.txt.
    """

    report_path = os.path.join(
        OUTPUT_DIR,
        "split_report.txt"
    )

    group_leakage = (
        check_group_leakage(
            assignments
        )
    )

    sequence_overlap = (
        check_sequence_overlap(
            samples,
            assignments
        )
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "========================================\n"
        )

        file.write(
            "FACEFORENSICS++ GROUP-BASED SPLIT\n"
        )

        file.write(
            "========================================\n\n"
        )

        file.write(
            f"Data root: {DATA_ROOT}\n"
        )

        file.write(
            f"Total samples: {len(samples)}\n"
        )

        file.write(
            f"Total groups: {len(group_stats)}\n"
        )

        file.write(
            f"Random seed: {RANDOM_SEED}\n"
        )

        file.write(
            f"Trials: {NUM_TRIALS}\n"
        )

        file.write(
            f"Best score: {best_score:.8f}\n\n"
        )

        # ----------------------------------------------------
        # Overall
        # ----------------------------------------------------

        total_real = sum(
            1
            for sample in samples
            if sample["label"] == 0
        )

        total_fake = sum(
            1
            for sample in samples
            if sample["label"] == 1
        )

        file.write(
            "========================================\n"
        )

        file.write(
            "OVERALL DISTRIBUTION\n"
        )

        file.write(
            "========================================\n"
        )

        file.write(
            f"Real: {total_real}\n"
        )

        file.write(
            f"Fake: {total_fake}\n\n"
        )

        # ----------------------------------------------------
        # Split statistics
        # ----------------------------------------------------

        for split_name in [
            "train",
            "val",
            "test"
        ]:

            selected = [

                sample

                for sample
                in samples

                if assignments[
                    sample["group_id"]
                ] == split_name
            ]

            real_count = sum(
                1
                for sample
                in selected
                if sample["label"] == 0
            )

            fake_count = sum(
                1
                for sample
                in selected
                if sample["label"] == 1
            )

            groups = len(
                set(
                    sample["group_id"]
                    for sample
                    in selected
                )
            )

            percentage = (
                len(selected)
                / max(len(samples), 1)
                * 100
            )

            real_percentage = (
                real_count
                / max(len(selected), 1)
                * 100
            )

            fake_percentage = (
                fake_count
                / max(len(selected), 1)
                * 100
            )

            file.write(
                f"[{split_name.upper()}]\n"
            )

            file.write(
                f"Samples: {len(selected)}\n"
            )

            file.write(
                f"Real: {real_count}\n"
            )

            file.write(
                f"Fake: {fake_count}\n"
            )

            file.write(
                f"Groups: {groups}\n"
            )

            file.write(
                f"Percentage: "
                f"{percentage:.2f}%\n"
            )

            file.write(
                f"Real percentage: "
                f"{real_percentage:.2f}%\n"
            )

            file.write(
                f"Fake percentage: "
                f"{fake_percentage:.2f}%\n"
            )

            file.write(
                "Methods:\n"
            )

            distribution = (
                get_method_distribution(
                    samples,
                    assignments,
                    split_name
                )
            )

            for method, count in (
                distribution.items()
            ):

                file.write(
                    f"  {method}: {count}\n"
                )

            file.write("\n")

        # ----------------------------------------------------
        # Group leakage
        # ----------------------------------------------------

        file.write(
            "========================================\n"
        )

        file.write(
            "GROUP LEAKAGE CHECK\n"
        )

        file.write(
            "========================================\n"
        )

        if group_leakage:

            file.write(
                "STATUS: FAILED\n"
            )

            file.write(
                f"Leaking groups: "
                f"{len(group_leakage)}\n"
            )

            for group_id, splits in (
                group_leakage.items()
            ):

                file.write(
                    f"{group_id}: "
                    f"{', '.join(splits)}\n"
                )

        else:

            file.write(
                "STATUS: PASSED\n"
            )

            file.write(
                "No group appears in multiple splits.\n"
            )

        # ----------------------------------------------------
        # Sequence overlap
        # ----------------------------------------------------

        file.write(
            "\n========================================\n"
        )

        file.write(
            "SEQUENCE OVERLAP CHECK\n"
        )

        file.write(
            "========================================\n"
        )

        for name, overlap in [

            (
                "Train-Val",
                sequence_overlap["train_val"]
            ),

            (
                "Train-Test",
                sequence_overlap["train_test"]
            ),

            (
                "Val-Test",
                sequence_overlap["val_test"]
            )
        ]:

            file.write(
                f"{name}: "
                f"{len(overlap)} overlapping IDs\n"
            )

        has_sequence_overlap = any(
            sequence_overlap[key]

            for key
            in sequence_overlap
        )

        if has_sequence_overlap:

            file.write(
                "STATUS: FAILED\n"
            )

        else:

            file.write(
                "STATUS: PASSED\n"
            )

        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        split_counts = {

            "train": 0,

            "val": 0,

            "test": 0
        }

        for sample in samples:

            split_name = assignments[
                sample["group_id"]
            ]

            split_counts[
                split_name
            ] += 1

        total_after_split = sum(
            split_counts.values()
        )

        file.write(
            "\n========================================\n"
        )

        file.write(
            "FINAL VALIDATION\n"
        )

        file.write(
            "========================================\n"
        )

        file.write(
            f"Original samples: "
            f"{len(samples)}\n"
        )

        file.write(
            f"Split samples: "
            f"{total_after_split}\n"
        )

        file.write(
            f"Train: "
            f"{split_counts['train']}\n"
        )

        file.write(
            f"Val: "
            f"{split_counts['val']}\n"
        )

        file.write(
            f"Test: "
            f"{split_counts['test']}\n"
        )

        count_check = (
            total_after_split
            == len(samples)
        )

        non_empty_check = all(
            split_counts[name] > 0

            for name
            in [
                "train",
                "val",
                "test"
            ]
        )

        if count_check:

            file.write(
                "Sample count check: PASSED\n"
            )

        else:

            file.write(
                "Sample count check: FAILED\n"
            )

        if non_empty_check:

            file.write(
                "Non-empty split check: PASSED\n"
            )

        else:

            file.write(
                "Non-empty split check: FAILED\n"
            )

        overall_valid = (

            count_check

            and non_empty_check

            and not group_leakage

            and not has_sequence_overlap
        )

        if overall_valid:

            file.write(
                "\nOVERALL STATUS: PASSED\n"
            )

        else:

            file.write(
                "\nOVERALL STATUS: FAILED\n"
            )

    return report_path


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    samples,
    group_stats,
    assignments,
    best_score
):
    """
    In thống kê split ra console.
    """

    print(
        "\n========================================"
    )

    print(
        "SPLIT SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Total samples: {len(samples)}"
    )

    print(
        f"Total groups : {len(group_stats)}"
    )

    print(
        f"Best score   : {best_score:.8f}"
    )

    print()

    for split_name in [
        "train",
        "val",
        "test"
    ]:

        selected = [

            sample

            for sample
            in samples

            if assignments[
                sample["group_id"]
            ] == split_name
        ]

        real_count = sum(
            1
            for sample
            in selected
            if sample["label"] == 0
        )

        fake_count = sum(
            1
            for sample
            in selected
            if sample["label"] == 1
        )

        group_count = len(
            set(
                sample["group_id"]
                for sample
                in selected
            )
        )

        percentage = (
            len(selected)
            / max(len(samples), 1)
            * 100
        )

        print(
            f"{split_name.upper():5s} | "
            f"Samples={len(selected):3d} | "
            f"Real={real_count:3d} | "
            f"Fake={fake_count:3d} | "
            f"Groups={group_count:2d} | "
            f"{percentage:6.2f}%"
        )

    print(
        "========================================"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "FACEFORENSICS++ GROUP-BASED SPLIT"
    )

    print(
        "========================================\n"
    )

    # --------------------------------------------------------
    # 1. Validate
    # --------------------------------------------------------

    validate_config()

    # --------------------------------------------------------
    # 2. Load samples
    # --------------------------------------------------------

    print(
        "[1/6] Đang đọc dataset..."
    )

    (
        samples,
        union_find
    ) = load_samples()

    print(
        f"[INFO] Samples đọc được: "
        f"{len(samples)}"
    )

    # --------------------------------------------------------
    # 3. Build groups
    # --------------------------------------------------------

    print(
        "[2/6] Đang xây dựng connected groups..."
    )

    samples = assign_group_ids(
        samples,
        union_find
    )

    group_stats = (
        create_group_statistics(
            samples
        )
    )

    print(
        f"[INFO] Connected groups: "
        f"{len(group_stats)}"
    )

    # --------------------------------------------------------
    # 4. Find best split
    # --------------------------------------------------------

    print(
        "[3/6] Đang tối ưu Train / Val / Test..."
    )

    (
        assignments,
        statistics,
        targets,
        best_score
    ) = find_best_split(
        group_stats
    )

    # --------------------------------------------------------
    # 5. Print summary
    # --------------------------------------------------------

    print_summary(
        samples,
        group_stats,
        assignments,
        best_score
    )

    # --------------------------------------------------------
    # 6. Save CSV
    # --------------------------------------------------------

    print(
        "\n[4/6] Đang lưu CSV..."
    )

    for split_name in [
        "train",
        "val",
        "test"
    ]:

        output_path = save_split_csv(
            samples,
            assignments,
            split_name
        )

        print(
            f"[OK] {output_path}"
        )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    print(
        "\n[5/6] Đang kiểm tra leakage..."
    )

    group_leakage = (
        check_group_leakage(
            assignments
        )
    )

    sequence_overlap = (
        check_sequence_overlap(
            samples,
            assignments
        )
    )

    if group_leakage:

        print(
            "❌ GROUP LEAKAGE DETECTED"
        )

    else:

        print(
            "✅ GROUP LEAKAGE: NONE"
        )

    has_sequence_overlap = any(
        sequence_overlap[key]

        for key
        in sequence_overlap
    )

    if has_sequence_overlap:

        print(
            "❌ SEQUENCE OVERLAP DETECTED"
        )

    else:

        print(
            "✅ SEQUENCE OVERLAP: NONE"
        )

    # --------------------------------------------------------
    # Create report
    # --------------------------------------------------------

    report_path = create_report(
        samples,
        group_stats,
        assignments,
        best_score
    )

    print(
        f"[OK] Report: {report_path}"
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print(
        "\n[6/6] Final validation..."
    )

    split_counts = {

        "train": 0,

        "val": 0,

        "test": 0
    }

    for sample in samples:

        split_name = assignments[
            sample["group_id"]
        ]

        split_counts[
            split_name
        ] += 1

    total_after_split = sum(
        split_counts.values()
    )

    print(
        f"Train: {split_counts['train']}"
    )

    print(
        f"Val  : {split_counts['val']}"
    )

    print(
        f"Test : {split_counts['test']}"
    )

    print(
        f"Total: {total_after_split}"
    )

    count_check = (
        total_after_split
        == len(samples)
    )

    non_empty_check = all(
        split_counts[name] > 0

        for name
        in [
            "train",
            "val",
            "test"
        ]
    )

    overall_valid = (

        count_check

        and non_empty_check

        and not group_leakage

        and not has_sequence_overlap
    )

    print(
        "\n========================================"
    )

    if overall_valid:

        print(
            "✅ SPLIT VALIDATION PASSED"
        )

        print(
            "✅ NO DATA LEAKAGE"
        )

    else:

        print(
            "❌ SPLIT VALIDATION FAILED"
        )

        raise RuntimeError(
            "Split không đạt yêu cầu. "
            "Không được training."
        )

    print(
        "========================================\n"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()