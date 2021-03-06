#!/usr/bin/env python3
import os
import shutil

def main():
    dst = "/models/stocknet"
    src = "./stocknet"
    # dst = "./B"
    # src = "./A"

    if os.path.isdir(dst):
        print('The model existed. Aaaar!')
    else:
        shutil.copytree(src, dst)
        assert os.path.isdir(dst)
        print("model copied.")


if __name__ == "__main__":
    main()