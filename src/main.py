import sys

def run_hw_test(hw_name: str):
    # NOTE: use relative imports so it works whether you run:
    #   python3 -m src.main test hw1
    # or set PYTHONPATH appropriately.
    if hw_name == "hw1":
        from .test_hw1 import test_hw1
        test_hw1()
    elif hw_name == "hw2":
        from .test_hw2 import test_hw2
        test_hw2()
    elif hw_name == "hw3":
        from .test_hw3 import test_hw3
        test_hw3()
    elif hw_name == "hw4":
        from .test_hw4 import test_hw4
        test_hw4()
    elif hw_name == "hw6":
        from .test_hw6 import test_hw6
        test_hw6()
    else:
        print(f"Unknown homework: {hw_name}")


def main():
    if len(sys.argv) < 3:
        print("usage: python -m src.main test <hw1|hw2|hw3|hw4>")
        return

    cmd = sys.argv[1]
    hw = sys.argv[2]

    if cmd == "test":
        run_hw_test(hw)
    else:
        print("Unknown command")


if __name__ == "__main__":
    main()