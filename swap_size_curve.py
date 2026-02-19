import pandas as pd 
import numpy as np

def tick_to_sqrtP(tick: int) -> float:
    return 1.0001 ** (tick / 2)


def build_L_per_tick_interval(snapshot: pd.DataFrame) -> pd.Series:
    """
    Returns a Series indexed by tickLower giving L for interval [tickLower, tickLower+1)
    If your intervals are wider than 1 tick, this expands them.
    """
    rows = []
    for tl, tu, L in snapshot[['tickLower','tickUpper','active_liquidity_L']].itertuples(index=False):
        # expand interval into 1-tick pieces
        for t in range(int(tl), int(tu)):
            rows.append((t, L))
    s = pd.Series(dict(rows))
    # if multiple entries per tick (shouldn't happen), sum them
    return s.groupby(level=0).sum().sort_index()

def swap_size_curve(dip: pd.DataFrame, k_max=50, tick_step=1, timestamp = None):
    """
    Returns a dataframe with required amount_in to move +/-k ticks from current poolTick.
    fee: pool fee as fraction (e.g. 0.0001, 0.0005, 0.003). Applied as amount_in / (1-fee).
    """
    snap = dip[dip['timestamp'] == timestamp].copy()
    ts = timestamp
    cur_tick = int(snap['poolTick'].iloc[0])

    # liquidity per 1-tick interval
    L_by_tick = build_L_per_tick_interval(snap)

    # helper: get L for interval [t, t+1)
    def L_interval(t):
        return float(L_by_tick.loc[t])  # raises if missing

    out = []
    amt_up_token1 = 0.0   # token1 in (price up)
    amt_down_token0 = 0.0 # token0 in (price down)

    for k in range(1, k_max + 1):
        # ---- move UP by tick_step (repeat k times)
        t_up_from = cur_tick + (k-1)*tick_step
        t_up_to   = cur_tick + k*tick_step

        # sum per 1-tick substep if tick_step > 1
        delta1 = 0.0
        for t in range(t_up_from, t_up_to):
            L = L_interval(t)
            sa = tick_to_sqrtP(t)
            sb = tick_to_sqrtP(t+1)
            delta1 += L * (sb - sa)
        amt_up_token1 += delta1

        # ---- move DOWN by tick_step (repeat k times)
        # crossing intervals [cur_tick-1,cur_tick), [cur_tick-2,cur_tick-1), ...
        t_dn_from = cur_tick - (k-1)*tick_step
        t_dn_to   = cur_tick - k*tick_step

        delta0 = 0.0
        for t in range(t_dn_to, t_dn_from):  # e.g. t = cur_tick-k ... cur_tick-1
            L = L_interval(t)  # interval [t, t+1) is active when moving down across it
            sa = tick_to_sqrtP(t+1)  # start at higher tick boundary
            sb = tick_to_sqrtP(t)    # end at lower tick boundary
            delta0 += L * (1.0/sb - 1.0/sa)
        amt_down_token0 += delta0

        # apply fee to amount_in (simplified: divides by (1-fee))
        up_in  = amt_up_token1
        dn_in  = amt_down_token0

        out.append({
            "k_ticks": k*tick_step,
            "token1_in_to_move_up": up_in,
            "token0_in_to_move_down": dn_in,
            "tick_up": cur_tick + k*tick_step,
            "tick_down": cur_tick - k*tick_step,
        })

    return pd.DataFrame(out), cur_tick, ts

def add_swap_size_metrics():
    liq_price = pd.read_parquet('./data/Uniswap/hourly_liquidity_pricecentered_full.parquet')
    integral = []; Ts = [] ; tangent_down = [] ; tangent_up = []
    for k in liq_price.timestamp.unique():
        curve, _, Tt = swap_size_curve(liq_price, k_max=50, tick_step=1, timestamp=k) 
        integral.append(np.sum(curve['token0_in_to_move_down'] - curve['token1_in_to_move_up'])/ np.sum(curve['token0_in_to_move_down'] + curve['token1_in_to_move_up']))
        tangent_down.append(np.sum(curve['token0_in_to_move_down'][:3]) / 6e6)
        tangent_up.append(np.sum(curve['token1_in_to_move_up'][:3]) / 6e6)
        Ts.append(Tt)
    integral = np.array(integral)
    tangent_down = np.array(tangent_down)
    tangent_up = np.array(tangent_up)
    Ts = np.array(Ts)
    df_integral = pd.DataFrame({'swap_size_imbalance' : integral, 'tangent_down': tangent_down, 'tangent_up': tangent_up, 'hour': pd.to_datetime(Ts, unit = 's', utc = True)})
    df_integral.index = df_integral.hour
    df_integral = df_integral.drop(columns = ['hour'])
    return df_integral

if __name__ == "__main__":
    try:
        df_integral = add_swap_size_metrics()
        df_integral.to_parquet('./data/Uniswap/swap_size_metrics.parquet')
        print('---- saved swap size metrics to parquet ----')
    except Exception as e:
        print(f"Error when computing swap size curves: {e}")