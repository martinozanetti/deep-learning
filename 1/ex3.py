

# c++: #include <iostream>
# python: ---

#int main()
#{
#    int a = 1, b = 1;
#    int target = 48;
#    for(int n = 3; n <= target; ++n)
#    {
#        int fib = a + b;
#        std::cout << "F("<< n << ") = " << fib << std::endl;
#        a = b;
#        b = fib;
#    }

#    return 0;
#}

from tkinter.tix import INTEGER


def main():
    a=1; b=1
    target = 45
    for n in range(3, target+1):
        fib = a+b
        print(f'F({n})={fib}')
        a=b
        b=fib

main()
