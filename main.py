import numpy as np
import pandas as pd

def extract_parameters(data):
    date = data.loc[44576:44864, 'region']
    demand_max = [data.iloc[44576:44864, ['AF',' AG', 'AI', 'AK']].sum(axis=1),  # House 1
             data.iloc[44576:44864, ['AL',' AM', 'AN', 'AP']].sum(axis=1),  # House 2
             data.iloc[44576:44864, ['AQ',' AR', 'AS', 'AW', 'AX']].sum(axis=1),  # House 3
             data.iloc[44576:44864, ['AY',' AZ', 'BA', 'BD', 'BF', 'BG']].sum(axis=1),  # House 4
             data.iloc[44576:44864, ['BH', 'BJ', 'BK']].sum(axis=1),  # House 5
             data.iloc[44576:44864, ['BL', 'BM', 'BN', 'BR']].sum(axis=1)]  # House 6
    g_pv = [data.loc[44576:44864, 'AJ'],    # House 1
            np.zeros(44864-44576),          # House 2
            data.loc[44576:44864, ['AV', 'BE']],    # House 3, 4
            np.zeros(44864-44576),      # House 5
            data.loc[44576:44864, 'AJ']]      # House 6

    # xi_ch
    # R_ch
    # xi_dis
    # R_dis
    return {
        'date': date,
        'demand_max': demand_max,
        'g_pv': g_pv
    }
    

def qp_matrices(data):
    """
    Extract the necessary matrices and vectors for the quadratic programming problem.
    
    Parameters:
    data (pandas.DataFrame): DataFrame containing the problem parameters
    
    Returns:
    dict: A dictionary containing the Q, p, G, h, A, b matrices/vectors
    """
    # Extract problem parameters from the data
    D_max = data['D_max'].iloc[0]
    D_min = data['D_min'].iloc[0]
    g_pv_i = data['g_pv_i'].iloc[0]
    xi_ch = data['xi_ch'].iloc[0]
    R_ch = data['R_ch'].iloc[0]
    xi_dis = data['xi_dis'].iloc[0]
    R_dis = data['R_dis'].iloc[0]
    A = data['A'].iloc[0]
    B = data['B'].iloc[0]
    C = data['C'].iloc[0]
    D = data['D'].iloc[0]
    E = data['E'].iloc[0]

    # Create Q matrix
    Q = np.zeros((5, 5))
    Q[0, 0] = 2 * A

    # Create p vector
    p = np.array([B, C, -C, D, D])

    # Create G matrix and h vector
    G = np.array([
        [1, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0],
        [-1, 0, 1, 1, 0],
        [1, 0, -1, -1, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [0, -1, 0, 0, -1],
        [0, 0, -1, 0, 0]
    ])

    h = np.array([
        D_max,
        -D_min,
        g_pv_i,
        -g_pv_i,
        xi_ch * R_ch,
        xi_dis * R_dis,
        0,
        0
    ])

    # No equality constraints
    A = np.zeros((0, 5))
    b = np.array([])

    return {
        'Q': Q,
        'p': p,
        'G': G,
        'h': h,
        'A': A,
        'b': b
    }

def solve_qp(Q, p, G, h, A, b):
    """
    Solve the quadratic programming problem.
    
    Parameters:
    Q (numpy.ndarray): Q matrix
    p (numpy.ndarray): p vector
    G (numpy.ndarray): G matrix
    h (numpy.ndarray): h vector
    A (numpy.ndarray): A matrix
    b (numpy.ndarray): b vector
    
    Returns:
    numpy.ndarray: Optimal solution vector
    """
    # return quadprog.solve_qp(Q, p, G, h, A, b)[0]

if __name__ == "__main__":
    # Load data from Excel file
    data = pd.read_excel('household_data.xlsx')

    # Extract matrices and vectors
    # param = extract_parameters(data)
    # print(param['date'][0])
    column_names = list(data.columns)
    print(column_names)
