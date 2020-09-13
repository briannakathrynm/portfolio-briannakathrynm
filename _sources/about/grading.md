---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Compute the Grade for CSC/DSP 310
* To run with input from user, please run **main()** in a cell.
* To run by entering values into function, please run **compute_grade()** with desired values.

```{code-cell} ipython3
def main():
    # Initializing Variables
    num_level1 = int(input("Enter number of total level 1 achievements earned: "))
    num_level2 = int(input("Enter number of total level 2 achievements earned: "))
    num_level3 = int(input("Enter number of total level 3 achievements earned: "))

    compute_grade(num_level1, num_level2, num_level3)
```

```{code-cell} ipython3
def compute_grade(num_level1, num_level2, num_level3):
    """
    Computes a grade for CSC/DSP 310 from numbers of achievements earned at each level
    :param num_level1: int, number of level 1 achievements earned
    :param num_level2: int, number of level 2 achievements earned
    :param num_level3: int, number of level 3 achievements earned
    :return: letter_grade: string, letter grade with possible modifier (+/-)
    """
    # Initializing Variables
    total_grade = num_level1 + num_level2 + num_level3
    letter_grade = str

    # Error Handling
    if total_grade > 45:
        print("Invalid total. Please re-enter values.")
    # Definitions of Grades    
    else:
        if total_grade == 45:
            letter_grade = 'A'
        elif 40 <= total_grade < 45:
            letter_grade = 'A-'
        elif 35 <= total_grade < 40:
            letter_grade = 'B+'
        elif 30 <= total_grade < 35:
            letter_grade = 'B'
        elif 25 <= total_grade < 30:
            letter_grade = 'B-'
        elif 20 <= total_grade < 25:
            letter_grade = 'C+'
        elif 15 <= total_grade < 20:
            letter_grade = 'C'
        elif 10 <= total_grade < 15:
            letter_grade = 'C-'
        elif 5 <= total_grade < 10:
            letter_grade = 'D+'
        elif 0 <= total_grade < 5:
            letter_grade = 'D'
        else:
            print("Your grade does not translate to a letter grade.")
        print(f'Your grade is {letter_grade}.')
```

The example below will give a grade of a C.

```{code-cell} ipython3
# Example 1
compute_grade(9, 6, 2)
```

The example below will give a grade of a B.

```{code-cell} ipython3
# Example 2
compute_grade(15, 16, 2)
```

The example below will give a grade of an A-.

```{code-cell} ipython3
# Example 3
compute_grade(15, 14, 12)
```
