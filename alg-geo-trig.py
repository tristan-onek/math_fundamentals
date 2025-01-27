# Quick practice demo of algebra, geometry, and trigonometry concepts.
import math

def algebra_demo():
    print("\n--- Algebra: Solving Linear Equations ---")
    # Solve 2x + 5 = 15
    a = 2
    b = 5
    c = 15
    x = (c - b) / a
    print(f"Equation: 2x + 5 = 15\nSolution: x = {x}")

def geometry_demo():
    print("\n--- Geometry: Area and Perimeter Calculations ---")
    # Circle
    radius = 5
    circle_area = math.pi * radius**2
    circle_circumference = 2 * math.pi * radius
    print(f"Circle (radius = {radius}):")
    print(f"  Area = {circle_area:.2f}")
    print(f"  Circumference = {circle_circumference:.2f}")

    # Rectangle
    length = 10
    width = 4
    rectangle_area = length * width
    rectangle_perimeter = 2 * (length + width)
    print(f"\nRectangle (length = {length}, width = {width}):")
    print(f"  Area = {rectangle_area}")
    print(f"  Perimeter = {rectangle_perimeter}")

    # Triangle (Heron's Formula)
    a, b, c = 3, 4, 5  # Right triangle
    s = (a + b + c) / 2
    triangle_area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    print(f"\nTriangle (sides = {a}, {b}, {c}):")
    print(f"  Area = {triangle_area:.2f}")
    print(f"  Perimeter = {a + b + c}")

def trigonometry_demo():
    print("\n--- Trigonometry: Sine, Cosine, and Tangent ---")
    angle_degrees = 30
    angle_radians = math.radians(angle_degrees)
    sine = math.sin(angle_radians)
    cosine = math.cos(angle_radians)
    tangent = math.tan(angle_radians)
    print(f"Angle: {angle_degrees}°")
    print(f"  Sine = {sine:.2f}")
    print(f"  Cosine = {cosine:.2f}")
    print(f"  Tangent = {tangent:.2f}")

    # Right triangle: given adjacent = 4, angle = 45°, find opposite and hypotenuse
    adjacent = 4
    angle = 45
    opposite = adjacent * math.tan(math.radians(angle))
    hypotenuse = adjacent / math.cos(math.radians(angle))
    print(f"\nRight Triangle (angle = {angle}°, adjacent = {adjacent}):")
    print(f"  Opposite = {opposite:.2f}")
    print(f"  Hypotenuse = {hypotenuse:.2f}")

if __name__ == "__main__":
    print("Python Demo: Algebra, Geometry, and Trigonometry")
    algebra_demo()
    geometry_demo()
    trigonometry_demo()
