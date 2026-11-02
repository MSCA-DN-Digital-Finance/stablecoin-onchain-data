from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_uniswap_liquidity_dashboard(
    df,
    hour_col="hour",
    hours=168,
    output_path="./site/uni_liquidity.html",
):
    """Responsive liquidity chart with a compact hourly scrubber."""
    columns = [hour_col, "priceMid", "usd_token0", "usd_token1"]
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if hours < 1:
        raise ValueError("hours must be positive")

    d = df[columns].copy()
    d[hour_col] = pd.to_datetime(d[hour_col], utc=True, errors="coerce")

    for column in columns[1:]:
        d[column] = pd.to_numeric(d[column], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    times = sorted(d[hour_col].unique())[-hours:]

    if not times:
        raise ValueError("No valid hourly liquidity observations")

    d = d[d[hour_col].isin(times)]
    snapshots, labels = [], []

    grid = np.sort(d.priceMid.unique())
    gaps = np.diff(grid)
    width = (
        float(np.median(gaps[gaps > 0]) * 0.88)
        if len(gaps)
        else 0.0001
    )

    for hour, group in d.groupby(hour_col, sort=True):
        group = (
            group.groupby("priceMid", as_index=False)[
                ["usd_token0", "usd_token1"]
            ]
            .sum()
            .sort_values("priceMid")
        )

        x = group.priceMid.to_numpy()
        y0 = group.usd_token0.to_numpy()
        y1 = group.usd_token1.to_numpy()
        top = max(1.0, float(np.max(y0 + y1)) * 1.08)

        active = group.loc[group.usd_token0 > 0, "priceMid"]
        active_price = float(active.iloc[0]) if len(active) else None

        traces = [
            go.Bar(
                x=x, y=y0, width=width,
                name="USDC", marker_color="#054476",
                hovertemplate=(
                    "Price: %{x:.6f}<br>"
                    "USDC: %{y:$,.0f}<extra></extra>"
                ),
            ),
            go.Bar(
                x=x, y=y1, width=width,
                name="USDT", marker_color="#DB3549",
                hovertemplate=(
                    "Price: %{x:.6f}<br>"
                    "USDT: %{y:$,.0f}<extra></extra>"
                ),
            ),
            go.Scatter(
                x=[1, 1], y=[0, top],
                mode="lines", name="Peg",
                line=dict(color="#17384f", width=1.3),
                showlegend=False, hoverinfo="skip",
            ),
            go.Scatter(
                x=[active_price, active_price], y=[0, top],
                mode="lines", name="Active bin",
                line=dict(color="#8094a3", width=1.3, dash="dash"),
                hoverinfo="skip",
            ),
        ]

        labels.append(
            pd.Timestamp(hour).strftime("%d %b %Y · %H:%M UTC")
        )
        snapshots.append(
            dict(
                x=x.tolist(),
                y0=y0.tolist(),
                y1=y1.tolist(),
                active=active_price,
                top=top,
            )
        )

    # Initially show the latest hour.
    fig = go.Figure(data=traces)
    fig.update_layout(
        autosize=True,
        barmode="stack",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="Arial, sans-serif",
            size=11,
            color="#17384f",
        ),
        margin=dict(l=58, r=12, t=40, b=38),
        xaxis=dict(
            title="USDT per USDC",
            tickformat=".4f",
            gridcolor="#edf2f6",
        ),
        yaxis=dict(
            title="Liquidity ($)",
            tickformat="~s",
            gridcolor="#edf2f6",
            range=[0, top],
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.02,
            yanchor="bottom",
            traceorder="normal",
        ),
        hovermode="x unified",
        bargap=0,
    )

    chart = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id="liquidity-chart",
        default_width="100%",
        default_height="100%",
        config=dict(
            responsive=True,
            displaylogo=False,
            scrollZoom=False,
            modeBarButtonsToRemove=["select2d", "lasso2d"],
        ),
    )

    controls = """
    <div data-liquidity-controls>
      <div class="meta">
        <span id="hour-label"></span>
        <span id="hour-count"></span>
      </div>
      <div class="scrubber">
        <button id="previous" aria-label="Previous hour">‹</button>
        <input id="hour-slider" type="range"
               min="0" step="1" aria-label="Liquidity hour">
        <button id="next" aria-label="Next hour">›</button>
      </div>
    </div>
    """

    script = """
    <script>
    const labels = __LABELS__;
    const hours = __HOURS__;
    const plot = document.getElementById('liquidity-chart');
    const slider = document.getElementById('hour-slider');

    slider.max = labels.length - 1;
    slider.value = slider.max;

    function showHour(index) {
      index = Math.max(0, Math.min(labels.length - 1, index));
      slider.value = index;

      document.getElementById('hour-label').textContent = labels[index];
      document.getElementById('hour-count').textContent =
        `${index + 1} / ${labels.length}`;
      document.getElementById('previous').disabled = index === 0;
      document.getElementById('next').disabled =
        index === labels.length - 1;

      const d = hours[index];

      return Plotly.update(plot, {
        x: [d.x, d.x, [1, 1], [d.active, d.active]],
        y: [d.y0, d.y1, [0, d.top], [0, d.top]]
      }, {
        'yaxis.range': [0, d.top]
      });
    }

    slider.addEventListener('input', () =>
      showHour(Number(slider.value))
    );

    document.getElementById('previous').onclick = () =>
      showHour(Number(slider.value) - 1);

    document.getElementById('next').onclick = () =>
      showHour(Number(slider.value) + 1);

    document.getElementById('hour-label').textContent = labels.at(-1);
    document.getElementById('hour-count').textContent =
      `${labels.length} / ${labels.length}`;
    document.getElementById('next').disabled = true;
    </script>
    """
    script = script.replace("__LABELS__", json.dumps(labels))
    script = script.replace("__HOURS__", json.dumps(snapshots))

    html = """<!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport"
            content="width=device-width, initial-scale=1">
      <title>Uniswap liquidity — USDC / USDT</title>
      <style>
        * { box-sizing: border-box; }
        html, body {
          height: 100%;
          width: 100%;
          margin: 0;
          overflow: hidden;
        }
        body {
          display: grid;
          grid-template-rows: minmax(0, 1fr) 56px;
          background: white;
          font-family: Arial, sans-serif;
          color: #17384f;
        }
        #plot-wrap {
          min-height: 0;
          min-width: 0;
          overflow: hidden;
        }
        #plot-wrap > div {
          height: 100%;
          width: 100%;
        }
        [data-liquidity-controls] {
          padding: 5px 12px 7px;
          border-top: 1px solid #dce6ed;
        }
        .meta {
          display: flex;
          justify-content: space-between;
          font-size: 10px;
          color: #61798b;
        }
        .scrubber {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-top: 3px;
        }
        input {
          flex: 1;
          min-width: 0;
          accent-color: #054476;
          cursor: pointer;
        }
        button {
          width: 24px;
          height: 23px;
          border: 1px solid #dce6ed;
          border-radius: 4px;
          background: white;
          color: #054476;
          cursor: pointer;
          font-size: 17px;
        }
        button:disabled {
          opacity: .3;
          cursor: default;
        }
      </style>
    </head>
    <body>
      <div id="plot-wrap">__CHART__</div>
      __CONTROLS__
      __SCRIPT__
    </body>
    </html>
    """

    html = (
        html.replace("__CHART__", chart)
        .replace("__CONTROLS__", controls)
        .replace("__SCRIPT__", script)
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return fig


if __name__ == "__main__":
    df = pd.read_parquet("./data/Uniswap/hourly_liquidity_full.parquet")
    plot_uniswap_liquidity_dashboard(df)