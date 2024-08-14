import numpy as np
from scipy import linalg
import sequence_jacobian as sj
from sequence_jacobian.blocks.support.het_support import CombinedTransition

"""Part 1: hetblock with some extra generality"""

@sj.het(exogenous='Pi', policy='a', backward='Va', backward_init=sj.hetblocks.hh_sim.hh_init)
def hh(Va_p, a_grid, y, rsub, rinc, r, beta, eis, min_a_enforced, to_savers, to_debtors):
    # allow distinction between substitution (rsub) and income (rinc) effects
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + rinc) * a_grid + y[:, np.newaxis]
    
    # add transfers to savers / debtors, to easily calculate their weighted iMPCs
    if to_savers != 0:
        coh += to_savers*a_grid*(a_grid > 0)
    if to_debtors != 0:
        coh += to_debtors*a_grid*(a_grid < 0)

    a = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    sj.misc.setmin(a, min_a_enforced) # will move around with real interest rate
    c = coh - a
    Va = (1 + rsub) * c ** (-1 / eis)
    return Va, a, c

def income(w, e_grid, Tr):
    y = w * e_grid + Tr
    return y

hh = hh.add_hetinputs([income, sj.hetblocks.hh_sim.make_grids])

@sj.simple
def borrowing_constraint(Bbar, rinc):
    min_a_enforced = Bbar /(1 + rinc(1))
    return min_a_enforced

hh_all = sj.combine([hh, borrowing_constraint])


"""Part 2: calibrations and hetblock steady states for HANK-I and HANK-II"""

r = 0.005
calib_all = dict(rsub=r, rinc=r, r=r, eis=1, w=1, rho_e=0.966, sd_e=0.5, n_e=11,
                 n_a=500, Tr=0, to_savers=0, to_debtors=0)

calib1 = dict(Bbar=-36.33, min_a=-36.33/(1+r), max_a=300)
calib2 = dict(Bbar=-2, min_a=-2/(1+r), max_a=50)

ss1 = hh_all.solve_steady_state(calib_all | calib1, unknowns={'beta':(0.98, 0.994)}, targets=['A'])
ss2 = hh_all.solve_steady_state(calib_all | calib2, unknowns={'beta':(0.98, 0.994)}, targets=['A'])


"""Part 3: extract key distributional statistics"""

def saver_debtor(ss, T):
    imp = hh.impulse_linear(ss, inputs={'to_savers': (np.arange(T)==0)}, outputs=['C', 'A'])
    saver_assets = (imp['C'] + imp['A'])[0]
    saver_impc = imp['C'] / saver_assets

    imp = hh.impulse_linear(ss, inputs={'to_debtors': (np.arange(T)==0)}, outputs=['C', 'A'])
    debtor_assets = (imp['C'] + imp['A'])[0]
    debtor_impc = imp['C'] / debtor_assets

    assert np.abs(saver_assets + debtor_assets) < 1E-5

    return saver_assets, saver_impc, debtor_impc

def worker_average(ss, T):
    imp = hh.impulse_linear(ss, inputs={'w': (np.arange(T)==0)}, outputs=['C', 'A'])
    wage_impc = imp['C']

    imp = hh.impulse_linear(ss, inputs={'Tr': (np.arange(11)==0)}, outputs=['C', 'A'])
    ave_impc = imp['C']

    return wage_impc, ave_impc


"""Part 4: add general equilibrium blocks and calibration"""

@sj.simple
def income_split(Y, G, vartheta, sigma, varphi, Tax):
    # everything here only works to 1st order, starting from Y=N=Z=1 baseline normalization
    wactual = (Y-G)**sigma * Y**varphi # assume Y=1, G=0 baseline
    profit = 1 - wactual
    w = wactual*Y + (1-vartheta)*Y*profit
    Tr = vartheta*Y*profit - Tax # paper uses lump-sum taxation
    return w, Tr

@sj.simple
def govbudget(B, r, Tax):
    # only works to 1st order around B=0, otherwise need to set r0 = rss
    G = Tax - (1+r)*B(-1) + B
    return G

@sj.simple
def market_clearing(C, Y, A, B, G):
    goods_mkt = C + G - Y
    asset_mkt = A - B
    return goods_mkt, asset_mkt

model = sj.combine([hh, income_split, govbudget, market_clearing, borrowing_constraint])
params = dict(Y=1, sigma=1, varphi=1, vartheta=0.5, B=0, Tax=0)

def get_ss_ge(ss, **kwargs):
    """Return general equilibrium steady state with optional changes"""
    calib_ge = {**ss, **params, **kwargs}
    return model.steady_state(calib_ge)


"""Part 5: Jacobians with respect to rsub, rinc, r, Z"""

def get_jacobian(ss, T):
    R = 1 + ss['r'] # scale by this to get vs. percentage change in R
    G = model.solve_jacobian(ss, unknowns=['Y'], targets=['asset_mkt'],
        inputs=['rsub', 'rinc', 'Tax', 'B'], outputs=['Y'], T=T+1)
    # r has ex-post timing, so we don't want response to r_0, only r_1 onward
    G_sub = R*G['Y', 'rsub'][:-1, 1:]
    G_inc = R*G['Y', 'rinc'][:-1, 1:]
    G_r = G_sub + G_inc
    return dict(rsub=G_sub, rinc=G_inc, r=G_r,
                T=G['Y', 'Tax'][:-1, :-1], B=G['Y', 'B'][:-1, :-1])


"""Part 6: representative-agent Jacobian with respect to r"""

def get_RA_jacobian(T, eis=1):
    return np.triu(-eis*np.ones((T,T)))


"""Part 7: tax plan formulas"""

def Bplan(dG, rho_B):
    """Determine financing for any desired time path of spending dG_t,
    solving dB_t = rho_B (dB_{t-1} + dG_t)"""
    T = len(dG)
    dB = np.empty(T)
    for t in range(T):
        dB_lag = dB[t-1] if t>0 else 0
        dB[t] = rho_B * (dB_lag + dG[t])
    return dB

def Tplan(dG, dB, r):
    """Get tax plan dT_t coresponding to any path dG_t, dB_t"""
    # not using this directly anymore
    dB_lag = np.concatenate(([0], dB[:-1]))
    dT = dG + (1+r)*dB_lag - dB
    return dT


"""Part 8: helper code for steady-state MPCs"""

def get_mpcs(c, a, a_grid, r, forward=False):
    mpcs = np.empty_like(c)
        
    # symmetric differences away from boundaries
    mpcs[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (a_grid[2:] - a_grid[:-2]) / (1+r)

    # asymmetric first differences at boundaries
    mpcs[:, 0]  = (c[:, 1] - c[:, 0]) / (a_grid[1] - a_grid[0]) / (1+r)
    mpcs[:, -1] = (c[:, -1] - c[:, -2]) / (a_grid[-1] - a_grid[-2]) / (1+r)

    # special case of exactly constrained today: should have MPC=1 today
    # and MPC=0 for all future dates
    if forward:
        mpcs[a == a_grid[0]] = 0
    else:
        mpcs[a == a_grid[0]] = 1
    return mpcs


"""Part 9: helper code for expectation functions"""

def expectation_functions(ss, o, T):
    # use same (undocumented) functions as in HetBlock._jacobian
    ss = hh.extract_ss_dict(ss)
    exog = hh.make_exog_law_of_motion(ss)
    endog = hh.make_endog_law_of_motion(ss)
    law_of_motion = CombinedTransition([exog, endog]).forward_shockable(ss['Dbeg'])
    return hh.expectation_vectors(ss[o], T, law_of_motion)


"""Part 10: Werning neutral version of model"""

mu = 1.010128 # solved for this value to achieve same fraction of constrained agents
w = 1/mu
A = (1-1/mu)/r

calib_neutral = calib_all | dict(min_a_enforced=0, min_a=0, max_a=50, w=w)
ss_neutral = hh.solve_steady_state(calib_neutral, unknowns={'beta':(0.98, 0.994)}, targets={'A': A})

@sj.solved(unknowns={'Pi': (0., 4.)}, targets=['val'])
def income_split(Y, Tax, mu, r_ante, Pi):
    w = 1/mu * (Y-Tax) # this goes to labor, entering hh block as 'w'
    pi = (1-1/mu) * (Y-Tax) # this goes to profits
    val = (Pi(1) + pi(1)) / (1+r_ante) - Pi # valuation condition for firm
    return w, pi, val

@sj.simple
def returns(pi, Pi):
    r = (pi + Pi)/Pi(-1) - 1
    rsub = r
    rinc = r
    return rsub, rinc

@sj.simple
def govbudget_neutral(B, G, r_ante):
    Tax = G + (1+r_ante(-1))*B(-1) - B
    return Tax

@sj.simple
def market_clearing_neutral(C, Y, A, Pi, B, G):
    goods_mkt = C + G - Y
    asset_mkt = A - B - Pi
    return goods_mkt, asset_mkt

model_neutral = sj.combine([hh, income_split, govbudget_neutral, market_clearing_neutral, returns])
params_neutral = dict(Y=1, mu=mu, B=0, G=0, r_ante=r)
ss_neutral = model_neutral.steady_state({**ss_neutral, **params_neutral})
