def histogram(data, n, b, h):
    # data is a list
    # n is an integer
    # b and h are floats
    # Write your code here
    hist = [0] * n
    if ((type(n) == int) and n > 0) and (h >= b):
        w = (h - b) / n
        for values in data:
            if values <= b or values >= h:
                donothing = 0
            else:
                hist[int((values - b) / w)] += 1
        return hist
    # return the variable storing the histogram
    # Output should be a list
    else:
        print('Check that n is a positive integer and that h >= b')
        return hist
    pass


def addressbook(name_to_phone, name_to_address):
    # name_to_phone and name_to_address are both dictionaries
    # Write your code here
    add_list = []
    address_to_all = {}
    for name in name_to_address:
        address = name_to_address[name]
        phone = name_to_phone[name]
        if address in add_list:
            if phone != address_to_all[address][1]:
                print('Warning: ' + name + ' has a different number for ' + address + ' than ' + address_to_all[address][0][0] + '. Using the number for ' + address_to_all[address][0][0] + '.')
            address_to_all[address][0].append(name)
        else:
            add_list.append(address)
            address_to_all[address] = ([name], phone)

    return address_to_all
    # return the variable storing address_to_all
    # Output should be a dictionary
    pass
