#    Write a function which returns the average from a a list. Test the function using the list v=[2,6,3,8,9,11,-5].

def ave(mylist):
    a=0
    for elem in mylist:
        a+=elem
    
    return a/len(mylist)

trylist = [2,6,3,8,9,11,-5]

print(ave(trylist))


#    Write the factorial function: f(0) = 1, f(x) = x * f(x-1).

def factorial(x):
    if x==0:
        return 1
    else:
        return x*(factorial(x-1))

a = int(input())
print(factorial(a))

