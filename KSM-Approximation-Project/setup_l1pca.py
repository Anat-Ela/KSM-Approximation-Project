import os
import sys
import subprocess


def clone_and_configure():
    repo_url = "https://github.com/ktountas/L1-Norm-Algorithms.git"
    repo_path = "external/L1-Norm-Algorithms"

    if not os.path.exists(repo_path):
        print("Cloning L1-Norm-Algorithms repository...")
        subprocess.run(["git", "clone", repo_url])

    lib_path = os.path.join(repo_path, "python", "lib")
    if lib_path not in sys.path:
        sys.path.append(lib_path)
        print(f"Added {lib_path} to sys.path")

    print("Available files in lib/:", os.listdir(lib_path))


if __name__ == "__main__":
    clone_and_configure()
