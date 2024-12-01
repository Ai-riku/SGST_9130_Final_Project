# columns we want to use
relevant_cols = ['cet_cest_timestamp',
                 
                 'DE_KN_residential1_dishwasher', 'DE_KN_residential1_freezer', 'DE_KN_residential1_heat_pump',
                 'DE_KN_residential1_pv','DE_KN_residential1_washing_machine',
                 
                 'DE_KN_residential3_circulation_pump', 'DE_KN_residential3_dishwasher', 'DE_KN_residential3_freezer',
                 'DE_KN_residential3_pv', 'DE_KN_residential3_refrigerator', 'DE_KN_residential3_washing_machine',
                 
                 'DE_KN_residential4_dishwasher', 'DE_KN_residential4_ev', 'DE_KN_residential4_freezer',
                 'DE_KN_residential4_heat_pump', 'DE_KN_residential4_pv', 'DE_KN_residential4_refrigerator',
                 'DE_KN_residential4_washing_machine',
                 
                 'DE_KN_residential6_circulation_pump', 'DE_KN_residential6_dishwasher', 'DE_KN_residential6_freezer',
                 'DE_KN_residential6_grid_import', 'DE_KN_residential6_pv', 'DE_KN_residential6_washing_machine']

# extract the columns we want to use and drop any rows containing NaN cells
raw_household_data = pd.read_csv('opsd-household_data-2020-04-15/household_data_60min_singleindex.csv', usecols=relevant_cols).dropna()

# calculate discrete difference to get consumption for time slot (data is cumulative)
raw_household_data[relevant_cols[1:]] = raw_household_data[relevant_cols[1:]].diff()

# drop initial NaN row that results from the discrete difference
raw_household_data = raw_household_data.dropna()

# reset the index so it starts at 0
raw_household_data = raw_household_data.reset_index(drop=True)


# main grid pricing
prices = {'p_max': 0.1597,
          'p_normal': 0.1097,
          'p_min': 0.0597}
def price(current_timestamp):
    # BC Hydro's time-of-day pricing scheme w/o daily surcharge
    # current_timestamp is a string with the following format
    # YYYY-MM-DDTHH:MM:SS+ZZZZ
    # e.g. 2014-12-11T18:00:00+0100
    ts = datetime.strptime(current_timestamp, '%Y-%m-%dT%H:%M:%S%z')
    if ((ts.hour >= 7) and (ts.hour < 16)) or ((ts.hour >= 21) and (ts.hour < 23)):
        return prices['p_normal'] # off-peak: [7,16) u [21,23)
    elif (ts.hour >= 16) and (ts.hour < 21):
        return prices['p_max'] # on-peak: [16,21)
    else:
        return prices['p_min'] # overnight

# generate price column
raw_household_data['price'] = raw_household_data['cet_cest_timestamp'].apply(price)


# clean up raw data to create household data ready for algo use
def form_household_data(dataframe, hh_i_orig, hh_i_new):
    # compute D_max and D_min for household i
    # delete household i's appliance columns
    # rename household i's pv column to g_pv_i
    # generate alpha column for household i
    # add None columns for g_l, g_s, g_ch, g_dis, and D
    D_max_cols = [] # columns to sum for D_max (all appliances)
    D_min_cols = [] # columns to sum for D_min (critical appliances)
    for label in dataframe.columns:
        if str(hh_i_orig) in label and 'pv' not in label:
            D_max_cols.append(label)
            if 'freezer' in label or 'refrigerator' in label:
                D_min_cols.append(label)
    dataframe[f'D_max_{hh_i_new}'] = dataframe[D_max_cols].sum(axis=1) # compute D_max
    dataframe[f'D_min_{hh_i_new}'] = dataframe[D_min_cols].sum(axis=1) # compute D_min
    dataframe = dataframe.drop(columns=D_max_cols) # delete appliance columns
    dataframe = dataframe.rename(columns={f'DE_KN_residential{hh_i_orig}_pv': f'g_pv_{hh_i_new}'})
    dataframe[f'alpha_{hh_i_new}'] = np.random.default_rng(10*hh_i_new).integers(
        low=15, high=35, size=len(dataframe), endpoint=True).astype('f') / 10
    dataframe[[f'g_l_{hh_i_new}',f'g_s_{hh_i_new}',f'g_ch_{hh_i_new}',f'g_dis_{hh_i_new}',f'D_{hh_i_new}']] = None
    return dataframe

# create new dataframe for algo use
household_data = form_household_data(raw_household_data, 1, 1)
household_data = form_household_data(household_data, 3, 2)
household_data = form_household_data(household_data, 4, 3)
household_data = form_household_data(household_data, 6, 4)

# shorten the timestamp column label
household_data = household_data.rename(columns={'cet_cest_timestamp': 'timestamp'})

# delete the original dataframe
del raw_household_data
gc.collect()

# dict to store battery parameters (using numbers from paper)
battery = {'S_cap': 80.0,
           'S_max': 80.0,
           'S_min': 0.2*80.0,
           'eta_ch': 0.8,
           'eta_dis': 1.25,
           'R_ch': 0.15*80.0,
           'R_dis': 0.15*80.0,
           's_0': 80.0}

# dict to store broadcast values
broadcast = {'t': 0,
             'k': None,
             'V': None,
             'K_b': None,
             'reoptimize': False}

# coordinator class
class Coordinator:
    def __init__(self, num_households):
        self.V = battery['eta_ch'] \
                * (battery['S_max'] \
                   -battery['S_min'] \
                   -battery['eta_ch']*battery['R_ch'] \
                   -battery['eta_dis']*battery['R_dis']) \
                / (prices['p_max']-prices['p_min'])
        self.K_b = battery['s_0']-battery['S_min']-battery['eta_dis']*battery['R_dis']-self.V*prices['p_max']
        self.Con = [1/num_households for _ in range(num_households)]

    def begin_optimization(self):
        broadcast['k'] = 1
        broadcast['V'] = self.V
        broadcast['K_b'] = self.K_b

    def evaluate_solutions(self):
        # sum up all g_ch_i and g_s_i
        # sum up all g_dis_i
        # if both sums are leq R_ch and R_dis, respectively, broadcast['reoptimize'] = False
        # else broadcast['reoptimize'] = True
            # broadcast['k'] += 1
            # compute household_i.xi_ch
            # compute household_i.xi_dis
                # if self.Con[i-1] < 0: household_i.xi_dis = 0
        pass

    def end_optimization(self):
        # update self.K_b
        # update self.Con
        pass

    def _update_Con(self, hh_index, hh_rec):
        self.Con[hh_index] = self.Con[hh_index] \
                                   + battery['eta_ch']*(hh_rec['g_ch']+hh_rec['g_s']) \
                                   - battery['eta_dis']*hh_rec['g_dis']

# household class
class Household:
    def __init__(self, hh_index, beta):
        self.index = hh_index # household identifier
        self.beta = beta # maximum load-shedding ratio
        self.alpha = f'alpha_{hh_index}' # dataframe column label for sensitivity to load shedding
        self.D_max = f'D_max_{hh_index}' # dataframe column label for desired load
        self.D_min = f'D_min_{hh_index}' # dataframe column label for non-sheddable load
        self.g_pv = f'g_pv_{hh_index}' # dataframe column label for generated PV energy
        self.g_l = f'g_l_{hh_index}' # dataframe column label for optimized energy from grid
        self.g_s = f'g_s_{hh_index}' # dataframe column label for optimized grid energy to battery
        self.g_ch = f'g_ch_{hh_index}' # dataframe column label for optimized PV energy to battery
        self.g_dis = f'g_dis_{hh_index}' # dataframe column label for optimized energy from battery
        self.D = f'D_{hh_index}' # dataframe column label for optimized load
        self.rec = None # dict of data for current time slot
        self.H_l = 0 # load queue
        self.xi_ch = None # portion of R_ch taken
        self.xi_dis = None # portion of R_dis taken
        self.x = None # optimization solution vector

    def begin_optimization(self):
        # extract row with this household's data for time slot t
        self.rec = household_data.loc[[broadcast['t']],
                                      ['timestamp','price',self.alpha,self.D_max,self.D_min,self.g_pv,
                                       self.g_l,self.g_s,self.g_ch,self.g_dis,self.D]]
        # rename column labels to be generic
        self.rec = self.rec.rename(columns={self.alpha:'alpha',self.D_max:'D_max',self.D_min:'D_min',self.g_pv:'g_pv',
                                            self.g_l:'g_l',self.g_s:'g_s',self.g_ch:'g_ch',self.g_dis:'g_dis',self.D:'D'})
        # save row as dict
        self.rec = self.rec.to_dict(orient='records')[0]
        # initialize xi
        self.xi_ch = 1
        self.xi_dis = 1

    def compute_solution(self):
        if self.rec['g_pv'] >= self.rec['D_max']:
            self.x = self._solve_a()
        else:
            self.x = self._solve_b()

    def end_optimization(self):
        self._update_H_l(self)

    def _solve_a(self):
        Q = matrix([[2*broadcast['V']*self.rec['alpha'],0.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0]])
        p = matrix([-(self.H_l/(self.rec['D_max']-self.rec['D_min']))-2*broadcast['V']*self.rec['alpha']*self.rec['D_max'],
                    broadcast['K_b'],
                    broadcast['V']*self.rec['price'],
                    broadcast['V']*self.rec['price']])
        G = matrix([[1.0,-1.0,1.0,0.0,0.0],
                    [0.0,0.0,1.0,1.0,-1.0],
                    [0.0,0.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,1.0,-1.0]])
        h = matrix([self.rec['D_max'],
                    -self.rec['D_min'],
                    self.rec['g_pv'],
                    self.xi_ch*battery['R_ch'],
                    0.0])
        sol = solvers.qp(Q, p, G, h)
        return sol['x']

    def _solve_b(self):
        Q = matrix([[2*broadcast['V']*self.rec['alpha'],0.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0]])
        p = matrix([-(self.H_l/(self.rec['D_max']-self.rec['D_min']))-2*broadcast['V']*self.rec['alpha']*self.rec['D_max'],
                    -broadcast['K_b'],
                    broadcast['V']*self.rec['price'],
                    broadcast['V']*self.rec['price']])
        G = matrix([[1.0,-1.0,0.0,0.0],
                    [0.0,0.0,1.0,1.0],
                    [0.0,0.0,0.0,0.0],
                    [0.0,0.0,0.0,0.0]])
        h = matrix([self.rec['D_max'],
                    -self.rec['D_min'],
                    self.xi_dis*battery['R_dis'],
                    0.0])
        A = matrix([1.0,-1.0,-1.0,0.0], (1,4))
        b = matrix(self.rec['g_pv'])
        sol = solvers.qp(Q, p, G, h, A, b)
        return sol['x']
	
    def _update_H_l(self):
        self.H_l = max(self.H_l-self.beta,0) + \
                   ((self.rec['D_max']-self.rec['D'])/(self.rec['D_max']-self.rec['D_min']))
        
# create coordinator object
coordinator = Coordinator(4)

# create household objects
rng = np.random.default_rng(seed=42) # seed for beta values randomly generated in [0.5,0.7]
betas = rng.integers(low=50, high=70, size=4, endpoint=True) / 100
household_1 = Household(1, betas[0].item())
household_2 = Household(2, betas[1].item())
household_3 = Household(3, betas[2].item())
household_4 = Household(4, betas[3].item())

# run algo
coordinator.begin_optimization()
household_1.begin_optimization()
household_2.begin_optimization()
household_3.begin_optimization()
household_4.begin_optimization()
household_1.compute_solution()
household_2.compute_solution()
household_3.compute_solution()
household_4.compute_solution()