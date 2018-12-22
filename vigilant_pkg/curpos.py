#!/usr/bin/env python3

import re
import sys
import termios
import tty


def get_pos():
    matches = None
    while matches is None:
        buf = ""
        stdin = sys.stdin.fileno()
        tattr = termios.tcgetattr(stdin)

        try:
            tty.setcbreak(stdin, termios.TCSANOW)
            sys.stdout.write("\x1b[6n")
            sys.stdout.flush()

            while True:
                buf += sys.stdin.read(1)
                if buf[-1] == "R":
                    break
        finally:
            termios.tcsetattr(stdin, termios.TCSANOW, tattr)

        matches = re.match(r"^\x1b\[(\d*);(\d*)R", buf)

    groups = matches.groups()

    return (int(groups[0]), int(groups[1]))


def set_pos(line, col):
    print("\033[%d;%df" % (line, col), end='')


if __name__ == "__main__":
    x, y = get_pos()
    print("Omae")
    set_pos(x - 1, y)
    print("Aahh!??")
    print(x, y)
