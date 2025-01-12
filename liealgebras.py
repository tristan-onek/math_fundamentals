# imports:
import sympy as sp
from sympy.liealgebras.cartan_type import CartanType
# end imports

c = CartanType("A2")
la = c.lie_algebra()

# calculate the Lie bracket from two elements:
x = la.get_basis()[0]
y = la.get_basis()[1]
print(la.bracket(x,y))
