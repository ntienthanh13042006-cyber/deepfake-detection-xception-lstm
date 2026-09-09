import os
import numpy as np


def check_dataset(data_dir):

    total = 0
    errors = 0

    for root, _, files in os.walk(data_dir):

        for file_name in files:

            if not file_name.endswith(".npy"):
                continue

            file_path = os.path.join(
                root,
                file_name
            )

            try:

                data = np.load(
                    file_path,
                    mmap_mode="r"
                )

                total += 1

                if data.shape != (15, 224, 224, 3):

                    print(
                        f"[ERROR] Shape sai: "
                        f"{file_path} -> {data.shape}"
                    )

                    errors += 1

                if data.dtype != np.uint8:

                    print(
                        f"[WARNING] dtype: "
                        f"{file_path} -> {data.dtype}"
                    )

            except Exception as e:

                print(
                    f"[ERROR] Không đọc được: "
                    f"{file_path}"
                )

                print(e)

                errors += 1

    print("\n============================")
    print(f"Tổng file: {total}")
    print(f"Lỗi: {errors}")
    print("============================")


if __name__ == "__main__":

    check_dataset(
        r"D:\Project\Processed_Data"
    )