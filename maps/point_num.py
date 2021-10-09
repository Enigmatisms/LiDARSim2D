import sys

if __name__ == "__main__":
    fname = sys.argv[1]
    with open(fname, 'r') as file:
        number = 0
        for l in file:
            num_str = l.strip().split(' ')
            number += int(num_str[0])
        print("Total point number: %d", number + 20)