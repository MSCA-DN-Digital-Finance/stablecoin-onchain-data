import numpy as np
import pandas as pd
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import plotly.graph_objects as go

def plot_uniswap_liquidity_last24h_plotly(df, hour_col="hour"):
    """
    Build an interactive Plotly figure with a slider to browse the last 24 hours.

    Expects columns:
      - hour_col (default 'hour'): datetime-like or sortable
      - 'priceMid'
      - 'usd_token0'
      - 'usd_token1'
    """
    d = df.copy()
    d[hour_col] = pd.to_datetime(d[hour_col])

    hours = np.array(sorted(d[hour_col].unique()))
    if len(hours) == 0:
        raise ValueError(f"No hours found in df['{hour_col}'].")

    hours = hours[-24*7:]  # last 24 available hours (or fewer)
    d = d[d[hour_col].isin(hours)]

    fig = go.Figure()
    trace_groups = []  # list of lists of trace indices per hour

    for h in hours:
        df_h = d[d[hour_col] == h].sort_values("priceMid")
        if df_h.empty:
            continue

        x = df_h["priceMid"].to_numpy()
        if len(x) >= 2:
            w = float(np.min(np.diff(x)) * 0.8)
        else:
            w = 0.01  # fallback

        y0 = df_h["usd_token0"].to_numpy()
        y1 = df_h["usd_token1"].to_numpy()
        y_max = float(np.nanmax(y0 + y1)) if len(y0) else 1.0
        y_top = max(1.0, y_max * 1.05)

        # Bars
        i0 = len(fig.data)
        fig.add_trace(go.Bar(x=x, y=y0, name="USDC", marker_color="cornflowerblue",
                             width=w, visible=False, hovertemplate="Price: %{x:.6f}<br>USDC: $%{y:.2~s}<extra></extra>"))
        i1 = len(fig.data)
        fig.add_trace(go.Bar(x=x, y=y1, name="USDT", marker_color="mediumaquamarine",
                             width=w, visible=False, hovertemplate="Price: %{x:.6f}<br>USDT: $%{y:.2~s}<extra></extra>"))

        # Vertical lines as scatter traces (so slider can toggle them)
        i2 = len(fig.data)
        fig.add_trace(go.Scatter(x=[1, 1], y=[0, y_top], mode="lines",
                                 name="price=1", line=dict(color="black", width=2),
                                 visible=False, showlegend=False))

        active = df_h.loc[df_h["usd_token0"] > 0, "priceMid"]
        x_active = float(active.iloc[0]) if len(active) else None
        i3 = len(fig.data)
        if x_active is not None:
            fig.add_trace(go.Scatter(x=[x_active, x_active], y=[0, y_top], mode="lines",
                                     name="active tick", line=dict(color="grey", width=2, dash="dash"),
                                     visible=False, showlegend=True))
        else:
            # placeholder (keeps group size consistent)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                     name="active tick", visible=False, showlegend=True))

        trace_groups.append([i0, i1, i2, i3])

    if not trace_groups:
        raise ValueError("No data available for the last 24 hours after filtering.")

    # Make the most recent hour visible initially
    for idx in trace_groups[-1]:
        fig.data[idx].visible = True

    steps = []
    n_traces = len(fig.data)
    for k, h in enumerate(hours[-len(trace_groups):]):  # align with any skipped empty hours
        visible = [False] * n_traces
        for idx in trace_groups[k]:
            visible[idx] = True

        steps.append(dict(
            method="update",
            args=[{"visible": visible},
                  {"title": f""}],
            label=pd.Timestamp(h).strftime("%Y-%m-%d %H:%M")
        ))

    fig.update_layout(
        barmode="overlay",  # matches your matplotlib look (bars drawn on top of each other)
        title=f"",
        xaxis_title="USDC price",
        yaxis_title="tick liquidity ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        sliders=[dict(
            active=len(steps) - 1,
            currentvalue=dict(prefix="Hour: "),
            pad=dict(t=50),
            steps=steps
        )],
        margin=dict(b=110),
        width=1200,
        height=800,

    )
    fig.update_layout(
        plot_bgcolor='white',
        
    )

    fig.write_html("./site/uni_liquidity.html")

if __name__ == '__main__':
    df = pd.read_parquet('./data/Uniswap/hourly_liquidity_full.parquet')
    plot_uniswap_liquidity_last24h_plotly(df)