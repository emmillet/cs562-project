import os
import subprocess
import re

from dotenv import load_dotenv
import psycopg2
import tabulate

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
    group_attr, group_vars_raw = group_by_body.split(";", 1)
    group_attr = group_attr.strip()
    group_vars = [var.strip() for var in group_vars_raw.split(",")]
    n = len(group_vars)

    # EXTRACT SIGMA

    # Join all lines into a single string to use with re.findall
    lines_string = "\n".join(lines)
    # Now use re.findall to capture the specific attribute and its value
    sigma_preds = re.findall(r"(\w+\.\w+\s*[=<>]+\s*['\"]?\w+['\"]?)", lines_string)
    # Create the formatted list in one line using both the attribute and its value
    sigma = [pred.strip().replace(' ', '') for pred in sigma_preds] 

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
        return f"f{index}_{func.lower()}_{field}"

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

# PLEASE NOTE: 
# these defintions are just here to see how we might want the logic to look like 
# and then well put this junk into the actual _generated.py

# not too sure calling this would work because of like the possibility of not being able to close the connection and stuff when finished but just a thought 
# thus its logic is kinda hardcoded everywhere
# def connect():
#     def connect():
#     load_dotenv()

#     user = os.getenv('USER')
#     password = os.getenv('PASSWORD')
#     dbname = os.getenv('DBNAME')

#     conn = psycopg2.connect("dbname="+dbname+" user="+user+" password="+password,
#                             cursor_factory=psycopg2.extras.DictCursor)
#     cur = conn.cursor()
#     return cur

def schema():
    load_dotenv()

    user = os.getenv('USER')
    password = os.getenv('PASSWORD')
    dbname = os.getenv('DBNAME')

    conn = psycopg2.connect("dbname="+dbname+" user="+user+" password="+password,
                            cursor_factory=psycopg2.extras.DictCursor)
    cur = conn.cursor()
    cur.execute("SELECT column_name, data_type FROM information_schema.columns")

    schema = []
    for row in cur:
        schema.append(row) # of the form {'column_name': name (obv), 'data_type': type (also obv)}
    
    return schema

def mf_construct(v, f_vect): # prob not the final format for mf_struct but just to have smth down to work on the other things
    mf_struct = {}
    for ga in v:
        mf_struct[ga] = []
    for gv in f_vect:
        mf_struct[f"f{gv}"] = []

    return mf_struct # makes sense for the specific nature of the columns in the mf structure table of the paper but implementation-wise not quite

mf_struct = [] # populate with mf_add() using dictionaries of structure {"grp_attr1": grp_attr1, "grp_attr2": grp_attr2, ..., "grp_v1": grp_v1, "grp_v2": grp_v2}
           
def mf_add(curr_row, v, f_vect):
    row_struct = {}
    # Handle grouping attributes
    group_attrs = [v] if isinstance(v, str) else v
    for ga in group_attrs:
        row_struct[ga] = curr_row[ga]
    
    # Initialize all aggregate fields
    for agg_field in f_vect:
        if 'sum' in agg_field:
            row_struct[agg_field] = 0.0
        elif 'avg' in agg_field:
            row_struct[agg_field] = 0.0  # Will calculate actual avg later
            sum_field = agg_field.replace('avg', 'sum')
            count_field = agg_field.replace('avg', 'count')
            row_struct[sum_field] = 0.0
            row_struct[count_field] = 0
        elif 'count' in agg_field:
            row_struct[agg_field] = 0
        elif 'max' in agg_field:
            row_struct[agg_field] = -float('inf')
        elif 'min' in agg_field:
            row_struct[agg_field] = float('inf')
    
    mf_struct.append(row_struct)
    return mf_struct[-1]

def mf_lookup(curr_row, v): # look up mf structure entry by group by attributes
    for i in range(len(mf_struct)): # using indexes so that we can return -1 to indicate failure
        check = True
        for ga in v:
            if curr_row[ga] != mf_struct[i][ga]:
                check = False
                break
        if (check):
            return i 
    return -1
            
def output(): # output mf_struct as a table
    # also move this into generated.py
    return tabulate.tabulate(mf_struct, headers="keys", tablefmt="psql") # change as needed ofc

def sigma_decouple(sigma_pred): # removes the "="
    # this is assuming im correctly understanding how we're storing sigma in phi
    dec_sigma = sigma_pred.split('=')
    return dec_sigma

def gv_cmp(curr_row, sigma_pred): # takes an mf_structure row, the entire phi f_vect, a single sigma_pred
    dec_sigma = sigma_decouple(sigma_pred)
    cmp_col = dec_sigma[0].split('.')[1]
    if (curr_row[cmp_col] == dec_sigma[1]): # expand to account for predicates that rely on pre established predicates 
        return True
    else:
        return False
    

# refer to diagrams in the project guide pdf and second research paper 
def mf_populate(v, f_vect): # populates the in-memory global mf_structure w the base values before predicate compares
    # also should prob have phi set up globally so we dont have to keep passing in phi.v and phi.f_vect
    # can prob do that query() function in tmp here before the loop

    load_dotenv() # again, can prob make a function for this repetitive code (i.e. the query() in tmp)

    user = os.getenv('USER')
    password = os.getenv('PASSWORD')
    dbname = os.getenv('DBNAME')

    conn = psycopg2.connect("dbname="+dbname+" user="+user+" password="+password,
                            cursor_factory=psycopg2.extras.DictCursor)
    cur = conn.cursor()
    cur.execute("SELECT * FROM sales")
    for row in cur:
        pos = mf_lookup(row, v)
        if (pos == -1):
            mf_add(row, v, f_vect)

# populate mf structure columns according to predicates and stuff
# input: 
# gv -- grouping variable by which this populaates the mf_structure
# v -- phi's v field
# sigma -- sigma field from phi 
def mf_populate_pred(gv, v, sigma): 
    load_dotenv() # again, can prob make a function for this repetitive code (i.e. the query() in tmp)

    user = os.getenv('USER')
    password = os.getenv('PASSWORD')
    dbname = os.getenv('DBNAME')

    conn = psycopg2.connect("dbname="+dbname+" user="+user+" password="+password,
                            cursor_factory=psycopg2.extras.DictCursor)
    cur = conn.cursor()
    cur.execute("SELECT * FROM sales")
    gv_parsed = gv.strip().lower().split('_') # gets the name of the grouping variable so as to compare that with the predicates in sigma
    preds = []
    for pred in sigma:
        if gv_parsed[1] in sigma:
            preds.append(sigma)
    for row in cur:
        check = True
        for pred in preds:
            if (gv_cmp(row, pred) == False):
                check = False # stop processing current row 
                break
        if (check):
            # grouping variable aggregate functions and attributes (i.e. count, quant from count_1_quant and stuff)
            gv_aggfn, gv_attr = gv_parsed[0], gv_parsed[2]
            if gv_aggfn == ["count"]:
                pos = mf_lookup(row, v)
                mf_struct[pos][gv_aggfn] += 1
            else:
                break
# helper function for print debugging in H_table 
def print_sample_groups(groups, max_print=3):
    """Print sample groups for debugging"""
    print("Sample groups:")
    for i, group in enumerate(groups[:max_print]):
        print(f"Group {i}: {group}")
    if len(groups) > max_print:
        print(f"... plus {len(groups)-max_print} more groups")

# Function to do passes for H table algorithm as detailed in research paper 
def H_table(phi):
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
    # SCAN 1: Populate initial groups
    print("\nSCAN 1: Populating initial groups")
    mf_populate(phi.V, phi.F)
    print(f"Initial groups created: {len(mf_struct)} entries")
    print_sample_groups(mf_struct)

    # Process each grouping variable in sequence
    for gv_idx in range(phi.n):
        # Get the corresponding predicate from sigma
        pred = phi.sigma[gv_idx] if gv_idx < len(phi.sigma) else ""
        
        # Determine which aggregate fields belong to this grouping variable
        gv_prefix = f"f{gv_idx+1}_"
        target_fields = [f for f in phi.F if f.startswith(gv_prefix)]
        
        print(f"\nSCAN {gv_idx+2}: Processing grouping variable {gv_idx+1}")
        print(f"Predicate: {pred}")
        print(f"Target fields: {target_fields}")
        
        load_dotenv()
        conn = psycopg2.connect(
            f"dbname={os.getenv('DBNAME')} user={os.getenv('USER')} password={os.getenv('PASSWORD')}",
            cursor_factory=psycopg2.extras.DictCursor
        )
        cur = conn.cursor()
        cur.execute("SELECT * FROM sales")
        
        rows_processed = 0
        rows_matched = 0
        
        for row in cur:
            rows_processed += 1
            # Check predicate if exists
            if pred and not gv_cmp(row, pred):
                continue
                
            pos = mf_lookup(row, phi.V)
            if pos == -1:
                continue
                
            rows_matched += 1
            # Update all target fields for this grouping variable
            for field in target_fields:
                if 'sum' in field:
                    mf_struct[pos][field] += float(row['quant'])
                elif 'avg' in field:
                    sum_field = field.replace('avg', 'sum')
                    count_field = field.replace('avg', 'count')
                    mf_struct[pos][sum_field] += float(row['quant'])
                    mf_struct[pos][count_field] += 1
                    if mf_struct[pos][count_field] > 0:
                        mf_struct[pos][field] = mf_struct[pos][sum_field] / mf_struct[pos][count_field]
                elif 'count' in field:
                    mf_struct[pos][field] += 1
                elif 'max' in field:
                    mf_struct[pos][field] = max(mf_struct[pos][field], float(row['quant']))
                elif 'min' in field:
                    mf_struct[pos][field] = min(mf_struct[pos][field], float(row['quant']))
        
        print(f"Processed {rows_processed} rows, matched {rows_matched} rows")
        print(f"Updated {len(mf_struct)} groups")
        print_sample_groups(mf_struct)

    # Apply HAVING clause
    if phi.G:
        print("\nApplying HAVING clause:", phi.G)
        initial_count = len(mf_struct)
        
        try:
            # Print for debug
            if mf_struct:
                print("Debug:")
                print({k: v for k, v in mf_struct[0].items()})
            
            # Evaluate HAVING clause
            filtered = []
            for entry in mf_struct:
                try:
                    if eval(phi.G, {}, entry):
                        filtered.append(entry)
                except Exception as e:
                    print(f"Error evaluating HAVING on entry: {e}")
                    continue
            
            mf_struct = filtered
            print(f"Filtered from {initial_count} to {len(mf_struct)} entries")
            
        except Exception as e:
            print(f"Error evaluating HAVING clause: {e}")
            print("Available fields:", mf_struct[0].keys() if mf_struct else "empty")
            return []
        
    # might not work for all queries but it's to filter the mf to only the S attributes for output

    if phi.S and mf_struct:
        # Get the list of columns to keep (both grouping attributes and selected aggregates)
        columns_to_keep = []
        
        # First add grouping attributes (V)
        columns_to_keep.extend(phi.V)
        
        # Then add any aggregate functions mentioned in S
        for s in phi.S:
            # Check if this is an aggregate function (starts with f)
            if s.startswith('f') and s in mf_struct[0]:
                columns_to_keep.append(s)
            # Also keep any grouping attributes mentioned in S
            elif s in phi.V and s not in columns_to_keep:
                columns_to_keep.append(s)
        
        # Filter each row to only keep the requested columns
        filtered_mf_struct = []
        for row in mf_struct:
            filtered_row = {col: row[col] for col in columns_to_keep if col in row}
            filtered_mf_struct.append(filtered_row)
        
        return filtered_mf_struct
    
    return mf_struct


def main():
    """
    This is the generator code. It should take in the MF structure and generate the code
    needed to run the query. That generated code should be saved to a 
    file (e.g. _generated.py) and then run.
    """
    # Body gets input from user, then parses into phi. This is then given to H_table to do the processing
    body = """
    S, n, V, F, sigma, G = get_input()
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
    H = H_table(mf_struct)
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
