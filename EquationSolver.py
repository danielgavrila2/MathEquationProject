import sympy as sym
from sympy import symbols, sympify, solve, simplify

def solve_equation(equation):
    try:
        # Handle algebraic equations
        if 'x' in equation and '=' in equation:
            x = symbols('x')
            left, right = equation.split("=")
            eq = sympify(left) - sympify(right)
            result = solve(eq, x)
            return result

        elif 'y' in equation and '=' in equation:
            y = symbols('y')
            left, right = equation.split("=")
            eq = sympify(left) - sympify(right)
            result = solve(eq, y)
            return result

        elif 'x' in equation and 'y' in equation and '=' in equation:
            x, y = symbols('x y')
            left, right = equation.split("=")
            eq = sympify(left) - sympify(right)
            result = solve(eq, (x, y))
            return result

        # Handle trigonometric equations
        elif '=' in equation and ('sin' in equation or 'tan' in equation or 'cos' in equation):
            eq_sympy = sympify(equation)
            if 'x' in equation and 'y' in equation:
                x, y = symbols('x y')
                result = solve(eq_sympy, (x, y))
            elif 'x' in equation:
                x = symbols('x')
                result = solve(eq_sympy, x)
            elif 'y' in equation:
                y = symbols('y')
                result = solve(eq_sympy, y)
            else:
                return "Invalid equation"
            return result

        # Handle expressions (not equations)
        elif "=" not in equation:
            if 'sin' in equation or 'tan' in equation or 'cos' in equation:
                return sympify(equation).evalf()
            return simplify(equation)

        # Handle other cases
        else:
            return "Invalid equation"

    except (ValueError, TypeError, AttributeError, RuntimeError, SyntaxError) as e:
        return "Wrong equation prediction"
