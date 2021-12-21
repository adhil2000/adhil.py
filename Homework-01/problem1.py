#!/usr/bin/python3
year=2021
# Your code should be below this line
if (year % 4) == 0:
    if (year % 100) == 0:
        if (year % 400) == 0:
            print('True')
        else:
            print('False')
    else:
        print('True')
else:
    print('False')

