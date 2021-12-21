import re

def problem1(searchstring):
    """
    Match phone numbers.

    :param searchstring: string
    :return: True or False
    """
    p = re.compile(r'((\(\d\d\d\) )|(\d\d\d\-))?\d\d\d\-\d\d\d\d')
    if p.match(searchstring):
        return True
    else:
        return False

def problem2(searchstring):
    """
    Extract street name from address.

    :param searchstring: string
    :return: string
    """
    p = re.compile(r'\d+ (([A-Z][a-z]* )+)(Dr.|Ave.|St.|Rd.)')
    return p.search(searchstring).group(1)[:-1]
    
def problem3(searchstring):
    """
    Garble Street name.

    :param searchstring: string
    :return: string
    """
    p = re.compile(r'(\d+) (([A-Z][a-z]* )+)(Dr.|St.|Ave.|Rd.)')
    return p.sub(lambda match: match.group(1) + " " + (match.group(2)[:-1])[::-1] + " " + match.group(4), searchstring)


if __name__ == '__main__' :
    print(problem1('765-494-4600')) #True
    print(problem1(' 765-494-4600 ')) #False
    print(problem1('(765) 494 4600')) #False
    print(problem1('(765) 494-4600')) #True
    print(problem1('494-4600')) #True
    
    print(problem2('The EE building is at 465 Northwestern Ave.')) #Northwestern
    print(problem2('Meet me at 201 South First St. at noon')) #South First
    
    print(problem3('The EE building is at 465 Northwestern Ave.'))
    print(problem3('Meet me at 201 South First St. at noon'))
