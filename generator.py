""" This software is a Query Processing Engine (QPE) built to reduce computational load with SQL queries by modifying the
    structure of a query to extend the group by & having clauses and create a such that clause, ultimatley building a MF
    structure to allow for simplified, scalable, and efficient processing. 
    The code below implements MF queries only, and is unfortunately unable to handle: 
    - EMF queries
    - Arithmetic in "such that" clause
    - .* selections (i.e. z.* or count(z.*))
    
    Zachary Emmanuel Altuna -- CWID: 20015297
    Emma Millet -- CWID: 20014914
    """

import os
import subprocess
import re
from datetime import datetime, date
from dotenv import load_dotenv
import psycopg2
import tabulate


"""
Global class utilized to parse, organize, and process ESQL MF queries. 
"""
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

def process_user_input(S, n, V, F, sigma, G, where):
    """Helper function to clean user input"""
    F_clean = [f.strip() for f in F if f.strip()]
    F_clean = sorted(set(F_clean), key=lambda x: (int(x[0]) if x[0].isdigit() else 0, x))

    sigma_groups = []
    sigma_raw = ','.join(sigma).strip().split(',')
    for pred in sigma_raw:
        pred = pred.strip()
        if pred:
            sigma_groups.append(pred)
    
    S_groups = []
    S_raw = ','.join(S).strip().split(',')
    for s in S_raw:
        s = s.strip()
        if s:
            S_groups.append(s)

    return S_groups, n, V, F_clean, sigma_groups, G, where

def get_input():
    """
    Prompts user for input, either to read from a file or input Phi directly through the command line. 
    Regardless of input, it returns phi and the where clause of the query.
    """
    cmnd = input("Please enter f to read from a file or u to read from user input: ").strip().lower()
    if cmnd == 'u':
        S = input("Please enter the select attribute(s) (S): ").strip().lower()
        n = input("Please enter the number of grouping variable(s) (n): ")
        V = input("Please enter the grouping attribute(s), seperated by a comma (V): ").strip().lower().split(",")
        F = input("Please enter the aggregate function(s) seperated by a comma [F]: ").strip().lower().split(",")
        sigma = input("Please enter the grouping variable predicate(s), seperated by a comma [sigma]: ").strip()
        G = input("Please enter the predicate(s) for the having clause: ").strip().lower()
        where = input("Please enter any 'WHERE' clause for the query, NOT including the word WHERE: ").strip().lower()
        return process_user_input([S], int(n), V, F, [sigma], G, where)
    # if command line flag, prompt user to enter the phi values through command line
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

    # EXTRACT WHERE

    where_line = next((line for line in lines if line.lower().startswith("where")), "")
    where = ""
    if where_line:
        where = where_line.replace("where", "", 1).strip()

    # EXTRACT GROUP BY
    group_by_line = next(line for line in lines if line.lower().startswith("group by"))
    group_by_body = group_by_line.replace("group by", "", 1).strip()
    group_attr = []
    group_vars_raw = []
    if ':' in group_by_body:
        group_attr, group_vars_raw = group_by_body.split(":", 1)
    else:
        group_attr = group_by_body
        group_vars_raw = []
    group_attr = [ga.strip() for ga in group_attr.split(',')]
    group_vars = [var.strip() for var in group_vars_raw.split(",")] if len(group_vars_raw) != 0 else [] 
    n = len(group_vars)

    # EXTRACT SIGMA

    # Join all lines into a single string to use with re.findall
    lines_string = "\n".join(lines)
    
    sigma = []
    preds_per_gv = []
    if 'such that' in lines_string:
        preds_per_gv = lines_string.split('such that')[1].split('having')[0]
    if preds_per_gv:
        if ',' in preds_per_gv: 
            preds_per_gv = preds_per_gv.split(',')
            for preds in preds_per_gv:
                preds = preds.strip().split('and')
                for pred in preds:
                    pred = pred.strip()
                sigma.append(preds)
        else: 
            preds_per_gv = preds_per_gv.split('and')
            for preds in preds_per_gv:
                for pred in preds:
                    pred = pred.strip()
                sigma.append(preds)

    # EXTRACT HAVING
    having_line = next((line for line in lines if line.lower().startswith("having")), "")
    having_pred = having_line.replace("having", "", 1).strip()
    if having_pred.endswith(";"):
        having_pred = having_pred[:-1].strip()

    # Regex for matching aggregate functions in both formats
    agg_pattern = re.compile(r"(?:(\w+)\.(sum|avg|count|max|min)\(([\w_]+)\)|(sum|avg|count|max|min)\((\w+)\.([\w_]+)\))", re.IGNORECASE)

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
        return f"f{index}_{func.lower()}_{field}"

    # Collect labels for all aggregate functions in SELECT and HAVING
    for text in projected_attrs + [having_pred]:
        for match in agg_pattern.finditer(text):
            # Handle both formats: x.count(prod) and count(x.prod)
            var1, func1, field1, func2, var2, field2 = match.groups()
            if var1 is not None:  # First format: x.count(prod)
                label = get_label(func1, var1, field1)
            else:  # Second format: count(x.prod)
                label = get_label(func2, var2, field2)
            if label not in f_list:
                f_list.append(label)

    # Step 2: Replace aggregate functions in the SELECT and HAVING clauses with labels
    def replace_agg(match):
        var1, func1, field1, func2, var2, field2 = match.groups()
        if var1 is not None:  # First format: x.count(prod)
            return get_label(func1, var1, field1)
        else:  # Second format: count(x.prod)
            return get_label(func2, var2, field2)

    s_transformed = [agg_pattern.sub(replace_agg, attr) for attr in projected_attrs]
    g_transformed = agg_pattern.sub(replace_agg, having_pred)

    return s_transformed, n, group_attr, sorted(f_list), sigma, g_transformed, where

def execute_query(query, where_clause=None):
    """
    Connects to the database, executes a given query with an optional WHERE clause, 
    and returns the query result.

    Parameters:
    - query (str): The SQL query to execute.
    - where_clause (str): Optional WHERE clause to filter the query. Default is None.
    
    Returns:
    - list: A list of rows returned by the query.
    """
    load_dotenv()  # Load environment variables for DB credentials

    user = os.getenv('USER')
    password = os.getenv('PASSWORD')
    dbname = os.getenv('DBNAME')

    # Connect to the PostgreSQL database
    try:
        conn = psycopg2.connect(
            f"dbname={dbname} user={user} password={password}",
            cursor_factory=psycopg2.extras.DictCursor
        )
        cur = conn.cursor()

        query = "SELECT * FROM sales"

        # Add WHERE clause if provided
        if where_clause:
            query += f" WHERE {where_clause}"
        
        print("Executing query:", query)
        cur.execute(query)

        # Fetch all results
        results = cur.fetchall()
        cur.close()
        conn.close()

        return results
    # if anything goes wrong
    except Exception as e:
        print(f"Error executing query: {e}")
        return []

def mf_add(curr_row, group_fields, agg_fields, use_sql=False, gv_fields=None):
    """
    General-purpose function to add a new row to the global mf_structure.

    Parameters:
    - curr_row (dict): The current row from the sales table.
    - group_fields (list): Grouping attributes (e.g., ['cust', 'prod']).
    - agg_fields (list): Aggregate field names (e.g., ['avg_quant', 'sum_quant']).
    - use_sql (bool): If True, only initialize attributes in 'group_fields' and 'agg_fields'.
                      If False, also initialize grouping variable fields using 'gv_fields'.
    - gv_fields (list of dict): Required if use_sql=False. Used to initialize fields like f1.sum_quant.

    Returns:
    - The newly added row (dict) in mf_struct.
    """

    row_struct = {}

    # Normalize group fields to list
    group_attrs = [group_fields] if isinstance(group_fields, str) else group_fields
    for ga in group_attrs:
        row_struct[ga] = curr_row.get(ga, 0.0)

    # Handle grouping variable fields (if not using SQL-style)
    if not use_sql and gv_fields:
        gv_idx = 1
        for gv in gv_fields:
            for key in gv:
                for val in gv[key]:
                    row_struct[f"f{gv_idx}.{val}"] = ""
            gv_idx += 1

    # Initialize aggregate fields
    for agg_field in agg_fields:
        if 'sum' in agg_field:
            row_struct[agg_field] = 0.0
        elif 'avg' in agg_field:
            row_struct[agg_field] = 0.0
            row_struct[agg_field.replace('avg', 'sum')] = 0.0
            row_struct[agg_field.replace('avg', 'count')] = 0
        elif 'count' in agg_field:
            row_struct[agg_field] = 0
        elif 'max' in agg_field:
            row_struct[agg_field] = -float('inf')
        elif 'min' in agg_field:
            row_struct[agg_field] = float('inf')

    mf_struct.append(row_struct)
    return mf_struct[-1]


def mf_lookup(curr_row, v):
    """
    Look up the index of an entry in the mf_struct based on group-by attributes.

    Parameters:
    - curr_row (dict): The current row containing attribute-value pairs.
    - v (list): A list of group-by attribute keys to compare against the current row.

    Returns:
    - int: The index of the matching entry in `mf_struct` if found, otherwise -1.
    """
    for i in range(len(mf_struct)):
        check = True  # Assume match is found unless proven otherwise
        
        # Compare each group-by attribute in curr_row with the corresponding entry in mf_struct
        for ga in v:
            if curr_row[ga] != mf_struct[i][ga]:  # If a mismatch is found, mark check as False
                check = False
                break
        
        # If all group-by attributes match, return the index of the matching entry
        if check:
            return i
    
    # If no match is found, return -1 to indicate failure
    return -1

    
def gv_cmp(sigma_pred, gv_row, table_row):
    """
    Compare two values (from either table_row or gv_row) based on the given predicate.

    Parameters:
    - sigma_pred (str): The comparison predicate to evaluate, e.g., "column1 = column2".
    - gv_row (dict): The row of group-by values
    - table_row (dict): The row of table values

    Returns:
    - bool: True if the comparison is equal, False if unequal
    """
    # Split on any comparison operator (=, <, >, <=, >=, !=, <>)
    parts = re.split(r'([=<>!]=?|<>|<=|>=)', sigma_pred.strip(), 1)
    if len(parts) < 3:
        return False  # invalid predicate format
    
    lcol = parts[0].strip()
    op = parts[1].strip()
    rcol = parts[2].strip()

    # Determine if columns reference rows or are literals
    l_has_gv = '.' in lcol
    r_has_gv = '.' in rcol

    # # Determine if columns invole equations, unsupported
    # l_has_op = '/' or '*' or '+' or '-' in lcol
    # r_has_op = '/' or '*' or '+' or '-' in rcol 

    if l_has_gv:
        lcol = lcol.split('.')[1].strip()
    if r_has_gv:
        rcol = rcol.split('.')[1].strip()

    lval = 0
    rval = 0
    # Get values
    try:
        lval = table_row[lcol] if l_has_gv else gv_row[lcol]
        rval = table_row[rcol] if r_has_gv else gv_row[rcol]
    except:
        lval = table_row[lcol] if l_has_gv else lcol
        rval = table_row[rcol] if r_has_gv else rcol

    # Try to convert values to appropriate types
    try:
        # Handle date comparisons
        if isinstance(lval, str) and re.match(r'\d{4}-\d{2}-\d{2}', lval):
            lval = datetime.strptime(lval, '%Y-%m-%d').date()
        if isinstance(rval, str) and re.match(r'\d{4}-\d{2}-\d{2}', rval):
            rval = datetime.strptime(rval, '%Y-%m-%d').date()
        
        # Handle numeric comparisons
        if not isinstance(lval, (date, datetime)):
            lval = float(lval) if isinstance(lval, str) and lval.replace('.','',1).isdigit() else lval
        if not isinstance(rval, (date, datetime)):
            rval = float(rval) if isinstance(rval, str) and rval.replace('.','',1).isdigit() else rval
    except (ValueError, AttributeError):
        pass

    # Perform comparison
    try:
        if op == '=':
            return lval == rval
        elif op == '<':
            return lval < rval
        elif op == '>':
            return lval > rval
        elif op == '<=':
            return lval <= rval
        elif op == '>=':
            return lval >= rval
        elif op in ('!=', '<>'):
            return lval != rval
    except TypeError:
        return False  # Incomparable types
    
    return False
    

def mf_populate(v=None, f=None, where=None, gv_fields=None, s=None, use_sql=False):
    """
    General-purpose function to populate the global in-memory mf_structure with base values.

    This function queries the 'sales' table using an optional WHERE clause and populates the
    mf_structure depending on the selected mode (standard or SQL-style).

    Parameters:
    - where (str): SQL WHERE clause to filter base table rows.
    - use_sql (bool): If True, uses mf_add_sql(). If False, uses mf_lookup + mf_add().
    - s (list): Required if use_sql=True. SQL-style structure for grouping and aggregation.
    - v (int): Required if use_sql=False.
    - f_vect (list): Required if use_sql=False. List of aggregate functions per grouping variable.
    - gv_fields (list): Required if use_sql=False. List of attributes defining grouping uniqueness.
    """

    # Build and execute the SELECT query
    res = execute_query(where)

    # Process each row based on mode, sql or esql
    for row in res:
        if use_sql:
            if s is None:
                raise ValueError("Parameter 's' must be provided when use_sql=True.")
            mf_add(row, s, s, use_sql=True)
        else:
            if v is None or f is None or gv_fields is None:
                raise ValueError("Parameters 'v', 'f_vect', and 'gv_fields' must be provided when use_sql=False.")
            pos = mf_lookup(row, v)
            if pos == -1:
                mf_add(row, v, f, use_sql=False, gv_fields=gv_fields)


def print_sample_groups(groups, max_print=3):
    """
    Helper function to assist in debugging by printing out a few sample groups/tuples from the current mf_struct
    """

    print("Sample groups:")
    for i, group in enumerate(groups[:max_print]):
        print(f"Group {i}: {group}")
    if len(groups) > max_print:
        print(f"... plus {len(groups)-max_print} more groups")

# Function to do passes for H table algorithm as detailed in research paper 
def H_table(phi, where):
    """
    Executes the H-Table multi-pass algorithm to populate the global mf_struct with grouped and aggregated results.

    This function implements logic for both standard SQL-style and extended multi-variable grouping queries,
    as defined by a PHI structure.

    Parameters:
    - phi: Global class containing information on query
    - where (str): SQL WHERE clause to apply on the base query.
    
    Returns:
    - NF Struct containing a list of dictionaries representing the outputted queried data. 
    """
    global mf_struct
    mf_struct = []
    if isinstance(phi.V, str):
        phi.V = [phi.V]
    print(f"PHI parameters received:")
    print(f"S: {phi.S}")
    print(f"n: {phi.n}")
    print(f"V: {phi.V}") 
    print(f"F: {phi.F}")
    print(f"sigma: {phi.sigma}")
    print(f"G: {phi.G}")
    
    gv_fields = [] # for things in the select clause like x.prod x.date etc
    for s in phi.S:
        if '.' in s and '(' not in s:
            gv, field = s.split('.') 
            gv = gv.strip()
            field = field.strip()
            gv_check = True
            for pos in range(len(gv_fields)):
                if gv == list(gv_fields[pos].keys())[0]:
                    gv_fields[pos][gv].append(field)
                    gv_check = False
                    break
            if gv_check:
                new_gv = {}
                new_gv[gv] = [field]
                gv_fields.append(new_gv)
            

    # SCAN 1: Populate initial groups
    print("\nSCAN 1: Populating initial groups")
    if (phi.n > 0):
        mf_populate(v = phi.V, f = phi.F, where = where, gv_fields = gv_fields)
    else: 
        mf_populate(s = phi.S, where = where, use_sql=True)
    print(f"Initial groups created: {len(mf_struct)} entries")
    print_sample_groups(mf_struct)

    if (phi.n == 0):
        data = execute_query(where)

        # Extract the relevant fields for grouping (cust, prod)
        group_by_columns = phi.V  # List of columns for GROUP BY (e.g., ['cust', 'prod'])
        aggregates = [agg for agg in phi.S if '(' in agg]  # Extract aggregates like avg(quant), max(quant)

        # Create a structure to hold the groups
        grouped_data = {}

        # Iterate over rows to group by 'cust', 'prod'
        for row in data:
            # Extract the values for the group by columns (cust, prod)
            group_key = tuple(row[col] for col in group_by_columns)
            
            if group_key not in grouped_data:
                grouped_data[group_key] = {agg: [] for agg in aggregates}

            # For each aggregate, collect values (e.g., for avg(quant), sum(quant), etc.)
            for agg in aggregates:
                column = agg.split('(')[1][:-1]  # Extract 'quant' from avg(quant)
                grouped_data[group_key][agg].append(row[column])

        # Process the aggregates and store them in mf_struct
        mf_struct = []
        
        for group_key, group_values in grouped_data.items():
            # Create the corresponding entry in mf_struct
            group_entry = dict(zip(group_by_columns, group_key))
            
            for agg, values in group_values.items():
                if 'avg' in agg:
                    group_entry[agg] = sum(values) / len(values) if values else None
                elif 'sum' in agg:
                    group_entry[agg] = sum(values)
                elif 'count' in agg:
                    group_entry[agg] = len(values)
                elif 'max' in agg:
                    group_entry[agg] = max(values) if values else None
                elif 'min' in agg:
                    group_entry[agg] = min(values) if values else None

            # Append the group entry to mf_struct
            mf_struct.append(group_entry)

    else:
        # Process each grouping variable in sequence
        for gv_idx in range(phi.n):
            # Determine which aggregate fields belong to this grouping variable
            gv_prefix = f"f{gv_idx+1}_"
            target_fields = [f for f in phi.F if f.startswith(gv_prefix)]

            preds = phi.sigma[gv_idx]

            print(f"\nSCAN {gv_idx+2}: Processing grouping variable {gv_idx+1}")
            print(f"Predicates: {preds}")
            print(f"Target fields: {target_fields}")
            # Honestly not sure why, but calling the helper here causes issues
            load_dotenv()
            conn = psycopg2.connect(
                f"dbname={os.getenv('DBNAME')} user={os.getenv('USER')} password={os.getenv('PASSWORD')}",
                cursor_factory=psycopg2.extras.DictCursor
            )
            cur = conn.cursor()
            query = "SELECT * FROM sales"
            if where.strip():  # if where clause is non-empty, alter query
                query += f" WHERE {where}"
            cur.execute(query)
            
            rows_processed = 0
            rows_matched = 0
            
            for row in cur:
                rows_processed += 1
                pos = mf_lookup(row, phi.V)
                if pos == -1:
                    continue

                # Handle predicates (now supports all comparison operators)
                if not isinstance(preds, list):
                    preds = [preds]  # Convert single predicate to list for uniform handling

                match_all_preds = True
                for pred in preds:
                    if pred and not gv_cmp(pred, mf_struct[pos], row):
                        match_all_preds = False
                        break
                
                if not match_all_preds:
                    continue
                
                for field in list(mf_struct[pos].keys()):
                    if f"{gv_prefix.split('_')[0]}." in field:
                        _gv, col = field.split('.')
                        mf_struct[pos][field] = row[col]

                rows_matched += 1
                # Update all target fields for this grouping variable
                for field in target_fields:
                    _gv, agg, col = field.split('_')
                    if agg == 'sum':
                        mf_struct[pos][field] += float(row[col])
                    elif agg == 'avg':
                        sum_field = field.replace('avg', 'sum')
                        if (not 'sum' in ' '.join(target_fields)):
                            mf_struct[pos][sum_field] += float(row[col])
                        count_field = field.replace('avg', 'count')
                        if (not 'count' in ' '.join(target_fields)):
                            mf_struct[pos][count_field] += 1
                        if mf_struct[pos][count_field] > 0:
                            mf_struct[pos][field] = mf_struct[pos][sum_field] / mf_struct[pos][count_field]
                    elif agg == 'count':
                        mf_struct[pos][field] += 1
                    elif agg == 'max':
                        mf_struct[pos][field] = max(mf_struct[pos][field], float(row[col]))
                    elif agg == 'min':
                        mf_struct[pos][field] = min(mf_struct[pos][field], float(row[col]))
            
            print(f"Processed {rows_processed} rows, matched {rows_matched} rows")
            print(f"Updated {len(mf_struct)} groups")
            print_sample_groups(mf_struct)
    # check if phi.G is defined
    if phi.G:
        print("\nApplying HAVING clause:", phi.G)
        initial_count = len(mf_struct)

        if mf_struct:
            print("Debug:")
            print({k: v for k, v in mf_struct[0].items()})
        
        having_clause = phi.G.replace('<>', '!=')
        # Evaluate HAVING clause
        filtered = []
        for entry in mf_struct:
            if eval(having_clause, {}, entry):
                filtered.append(entry)
        
        mf_struct = filtered
        print(f"Filtered from {initial_count} to {len(mf_struct)} entries")
    # Check if phi.S is defined and mf_struct is not empty
    if phi.S and mf_struct:
        columns_to_keep = phi.S
        filtered_mf_struct = []  # Store filtered rows

        # Iterate through each row in the mf_struct
        for row in mf_struct:
            filtered_row = {}  # Store filtered values for the current row

            # Process each column to keep
            for col in columns_to_keep:
                # Handle operations for columns involving division, multiplication, addition, or subtraction
                if '/' in col:
                    dividend, divisor = col.split('/')
                    filtered_row[col] = row[dividend.strip()] / row[divisor.strip()]
                elif '*' in col:
                    num1, num2 = col.split('*')
                    filtered_row[col] = row[num1.strip()] * row[num2.strip()]
                elif '+' in col:
                    num1, num2 = col.split('+')
                    filtered_row[col] = row[num1.strip()] + row[num2.strip()]
                elif '-' in col:
                    num1, num2 = col.split('-')
                    filtered_row[col] = row[num1.strip()] - row[num2.strip()]

                # Handle group-by variables (e.g., f1.prod)
                elif '.' in col and '(' not in col:
                    gv, new_col = col.strip().split('.')
                    gv_idx = 1
                    for pos in range(len(gv_fields))[1:]:
                        if gv == list(gv_fields[pos].keys())[0]:
                            gv_idx = pos
                            break
                    mf_field = f"f{gv_idx}.{new_col}"
                    filtered_row[col] = row[mf_field]
                
                # Otherwise, just add the column value directly
                else:
                    filtered_row[col] = row[col]

            # Append the filtered row to the result
            filtered_mf_struct.append(filtered_row)

        return filtered_mf_struct
    
    # Return the original mf_struct if no filtering is needed
    return mf_struct



def main():
    """
    This is the generator code. It should take in the MF structure and generate the code
    needed to run the query. That generated code should be saved to a 
    file (e.g. _generated.py) and then run.
    """
    # Body gets input from user, then parses into phi. This is then given to H_table to do the processing
    body = """
    S, n, V, F, sigma, G, where = get_input()
    class mf_struct:
        def __init__(self):
            self.S = [] # projected values
            self.n = 0 # number of grouping variables
            self.V = [] # group by attributes
            self.F = [] # list of aggregates
            self.sigma = [] # grouping variables predicates 
            self.G = None # having clause
    mf_struct = mf_struct() 
    mf_struct.S = S
    mf_struct.n = n
    mf_struct.V = V
    mf_struct.F = F
    mf_struct.sigma = sigma
    mf_struct.G = G
    H = H_table(mf_struct, where)
    """

    # Note: The f allows formatting with variables.
    #       Also, note the indentation is preserved.
    tmp = f"""
import os
import psycopg2
import psycopg2.extras
import tabulate
from dotenv import load_dotenv
from generator import *

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
    
    return tabulate.tabulate(H,
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
