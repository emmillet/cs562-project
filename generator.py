import subprocess
import re

class phi:
    def __init__(self):
        # select attributes
        self.S = []
        # number of grouping variables
        self.n = 0 
        # grouping attributes
        self.V = []
        # F vector
        self.F = []
        # select condition vector
        self.sigma = []
        # having condition
        self.G = None


def get_input():
    # if command line flag, prompt user to enter the phi values through command line
    cmnd = input("Please enter f to read from a file or u to read from user input: ").strip().lower()
    if cmnd == 'u':
        S = input("Please enter the select attribute(s) (S): ").strip().lower()
        n = input("Please enter the number of grouping variable(s) (n): ")
        V = input("Please enter the grouping attribute(s), seperated by a comma (V): ").strip().lower().split(",")
        F = input("Please enter the aggregate function(s) seperated by a comma [F]: ").strip().lower().split(",")
        sigma = input("Please enter the grouping variable predicate(s), seperated by a comma [sigma]: ").strip().lower()
        G = input("Please enter the predicate(s) for the having clause: ").strip().lower()
        return [S], int(n), V, F, [sigma], G
    elif cmnd == 'f':
        file = input("Please enter the name of the file to read: ").strip()
        return parser(file)
    else:
        get_input()

import re


def parser(filename: str):
    """
    Parses an Extended SQL (ESQL) query from a file and outputs the corresponding PHI operands.
    
    Arguments:
    filename -- path to the file containing the ESQL query
    """

    # Read and clean up the query from the file
    with open(filename, 'r') as file:
        esql_query = file.read()
    lines = esql_query.strip().split('\n')

    # EXTRACT SELECT
    select_line = next(line for line in lines if line.lower().startswith("select"))
    select_attrs = select_line.replace("select", "", 1).strip()
    projected_attrs = [attr.strip() for attr in select_attrs.split(",")]

    # EXTRACT GROUP BY
    group_by_line = next(line for line in lines if line.lower().startswith("group by"))
    group_by_body = group_by_line.replace("group by", "", 1).strip()
    group_attr, group_vars_raw = group_by_body.split(":", 1)
    group_attr = group_attr.strip()
    group_vars = [var.strip() for var in group_vars_raw.split(",")]
    n = len(group_vars)

    # EXTRACT SIGMA

    # Join all lines into a single string to use with re.findall
    lines_string = "\n".join(lines)
    # Now use re.findall to capture the specific attribute and its value
    sigma_preds = re.findall(r"\w+\.(\w+) = '(.*?)'", lines_string)
    # Create the formatted list in one line using both the attribute and its value
    sigma = [f"{attribute}='{value}'" for attribute, value in sigma_preds]

    # EXTRACT HAVING
    having_line = next((line for line in lines if line.lower().startswith("having")), "")
    having_pred = having_line.replace("having", "", 1).strip()
    if having_pred.endswith(";"):
        having_pred = having_pred[:-1].strip()

    # Regex for matching aggregate functions
    agg_pattern = re.compile(r"(sum|avg)\((\w)\.([\w_]+)\)", re.IGNORECASE)

    # Step 1: Find ALL aggregate functions across SELECT and HAVING
    f_list = []
    def get_label(func, var, field):
        """
        Generates a label for an aggregate function based on the grouping variable.
        
        Arguments:
        func -- aggregate function (sum or avg)
        var -- the grouping variable (e.g., 'x', 'y', 'z')
        field -- the field being aggregated (e.g., 'quant')
        """
        index = group_vars.index(var) + 1
        return f"{index}_{func.lower()}_{field}"

    # Collect labels for all aggregate functions in SELECT and HAVING
    for text in projected_attrs + [having_pred]:
        for match in agg_pattern.finditer(text):
            func, var, field = match.groups()
            label = get_label(func, var, field)
            if label not in f_list:
                f_list.append(label)

    # Step 2: Replace aggregate functions in the SELECT and HAVING clauses with labels
    def replace_agg(match):
        func, var, field = match.groups()
        return get_label(func, var, field)

    s_transformed = [agg_pattern.sub(replace_agg, attr) for attr in projected_attrs]
    g_transformed = agg_pattern.sub(replace_agg, having_pred)

    return s_transformed, n, group_attr, sorted(f_list), sigma, g_transformed


def main():
    """
    This is the generator code. It should take in the MF structure and generate the code
    needed to run the query. That generated code should be saved to a 
    file (e.g. _generated.py) and then run.
    """

    body = """
    for row in cur:
        if row['quant'] > 10:
            _global.append(row)
    """

    # Note: The f allows formatting with variables.
    #       Also, note the indentation is preserved.
    tmp = f"""
import os
import psycopg2
import psycopg2.extras
import tabulate
from dotenv import load_dotenv

# DO NOT EDIT THIS FILE, IT IS GENERATED BY generator.py

def query():
    load_dotenv()

    user = os.getenv('USER')
    password = os.getenv('PASSWORD')
    dbname = os.getenv('DBNAME')

    conn = psycopg2.connect("dbname="+dbname+" user="+user+" password="+password,
                            cursor_factory=psycopg2.extras.DictCursor)
    cur = conn.cursor()
    cur.execute("SELECT * FROM sales")
    
    _global = []
    {body}
    
    return tabulate.tabulate(_global,
                        headers="keys", tablefmt="psql")

def main():
    print(query())
    
if "__main__" == __name__:
    main()
    """

    # Write the generated code to a file
    open("_generated.py", "w").write(tmp)
    # Execute the generated code
    subprocess.run(["python", "_generated.py"])


if "__main__" == __name__:
    main()
