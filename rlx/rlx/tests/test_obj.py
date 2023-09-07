from copy import deepcopy


class A:
    pass


class B:
    pass


m = {
    A(): B(),
    A(): B(),
}

print(m)

mm = deepcopy(m)

print(mm)

s = set()
print(id(s))


def show(s):
    print(id(s))


show(s)
