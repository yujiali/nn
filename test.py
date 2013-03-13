class A:
    def __init__(self, data):
        self.x = data

    def p(self):
        print 'data=' + str(self.x) + self.__class__.__name__

class B(A):
    pass

class C(A):
    pass

x = 10

def f():
    print x

if __name__ == "__main__":
    print 'Hello'

    f()

    b = B(12)
    b.p()

    c = C(10)
    c.p()

