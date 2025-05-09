from decimal import Decimal
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from generator import parser, phi, H_table
import os
import psycopg2
from dotenv import load_dotenv
from generator import parser, phi, H_table

load_dotenv()

def clean_value(v):
    """Convert values to consistent string format with controlled precision"""
    if isinstance(v, (float, Decimal)):
        # Round to 6 decimal places to handle floating point precision issues
        rounded = round(float(v), 6)
        if rounded.is_integer():
            return str(int(rounded))
        return str(rounded)
    return str(v)

def compare_results(mf_results, sql_results):
    # Compare each set of tuples
    mf_tuples = {tuple(clean_value(v) for v in row.values()) for row in mf_results}
    sql_tuples = {tuple(clean_value(v) for v in row) for row in sql_results}
    return mf_tuples == sql_tuples

def run_test(mf_file, sql_query, test_num):
    print(f"\n=== Test {test_num}: {mf_file} ===")
    
    try:
        # Run MF query
        S, n, V, F, sigma, G, where = parser(mf_file)
        phi_obj = phi()
        phi_obj.S = S
        phi_obj.n = n
        phi_obj.V = V
        phi_obj.F = F
        phi_obj.sigma = sigma
        phi_obj.G = G
        mf_results = H_table(phi_obj, where)
        
        # Run SQL query
        conn = psycopg2.connect(
            dbname=os.getenv('DBNAME'),
            user=os.getenv('USER'),
            password=os.getenv('PASSWORD'),
            host=os.getenv('HOST', 'localhost')
        )
        with conn.cursor() as cur:
            cur.execute(sql_query)
            sql_results = [list(row) for row in cur.fetchall()]
            
        # Compare results
        print(f"MF rows: {len(mf_results)} | SQL rows: {len(sql_results)}")
        
        if not mf_results and not sql_results:
            print("Both empty")
            return True
            
        is_match = compare_results(mf_results, sql_results)
        # Output results
        if is_match:
            print("PASSED")
        else:
            print("FAILED")
            print("\nMF Results:")
            for row in mf_results:
                print(row)
            print("\nSQL Results:")
            for row in sql_results:
                print(row)
                
        return is_match
    
    # Handle any exceptions, just in case
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    # Any new queries can be added into the test cases below. 
    test_cases = [
        {
            "mf_file": "mf_input1.txt",
            "sql_query": """
                SELECT cust, 
                       SUM(CASE WHEN state = 'NY' THEN quant ELSE 0 END),
                       SUM(CASE WHEN state = 'NJ' THEN quant ELSE 0 END),
                       SUM(CASE WHEN state = 'CT' THEN quant ELSE 0 END)
                FROM sales
                GROUP BY cust
                HAVING SUM(CASE WHEN state = 'NY' THEN quant ELSE 0 END) > 
                       2 * SUM(CASE WHEN state = 'NJ' THEN quant ELSE 0 END)
                       OR
                       AVG(CASE WHEN state = 'NY' THEN quant ELSE NULL END) > 
                       AVG(CASE WHEN state = 'CT' THEN quant ELSE NULL END)
            """
        },
        {
            "mf_file": "mf_input2.txt",
            "sql_query": """
                SELECT x.cust, AVG(x.quant), AVG(y.quant)
                FROM (SELECT cust, quant FROM sales WHERE date > '2018-12-31') x
                JOIN (SELECT cust, quant FROM sales WHERE date < '2019-01-01') y
                ON x.cust = y.cust
                GROUP BY x.cust
                HAVING AVG(y.quant) > AVG(x.quant)
            """
        },
        {
            "mf_file": "mf_input3.txt",
            "sql_query": """
                SELECT cust,
                       SUM(CASE WHEN day <= 15 THEN quant END),
                       SUM(CASE WHEN day > 15 THEN quant END)
                FROM sales WHERE year = 2020
                GROUP BY cust
                HAVING SUM(CASE WHEN day <= 15 THEN quant END) >
                       SUM(CASE WHEN day > 15 THEN quant END)
            """
        },
        {
            "mf_file": "mf_input4.txt",
            "sql_query": """
                SELECT prod, 
                       AVG(CASE WHEN year = 2019 THEN quant END),
                       AVG(CASE WHEN year = 2020 THEN quant END)
                FROM sales WHERE year IN (2019, 2020)
                GROUP BY prod
                HAVING AVG(CASE WHEN year = 2019 THEN quant END) >
                       AVG(CASE WHEN year = 2020 THEN quant END)
            """
        },
        {
            "mf_file": "mf_input5.txt",
            "sql_query": """
                SELECT year, 
                       SUM(CASE WHEN state = 'NJ' THEN quant END),
                       SUM(CASE WHEN state = 'NY' THEN quant END)
                FROM sales WHERE state IN ('NJ', 'NY')
                GROUP BY year
                HAVING SUM(CASE WHEN state = 'NJ' THEN quant END) >
                       SUM(CASE WHEN state = 'NY' THEN quant END)
            """
        }
    ]

    passed = 0
    failed = 0
    
    # Run tests
    for i, test in enumerate(test_cases, 1):
        if run_test(test["mf_file"], test["sql_query"], i):
            passed += 1
        else:
            failed += 1

    # Print final results
    print("\n------------------------------------------------------------ \nTest Summary:")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"TOTAL:  {len(test_cases)}")
    print(f"Success Rate: {passed/len(test_cases)*100:.1f}%")

if __name__ == "__main__":
    main()