import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import openpyxl

# ── 1. Read REM expected monthly IPC (D7:D13 = Jan–Jul 2026) ─────────────
wb = openpyxl.load_workbook(
    "datos/tablas-relevamiento-expectativas-mercado-ene-2026.xlsx", data_only=True
)
ws = wb["Cuadros de resultados"]
rem_ipc_monthly = [ws.cell(row=r, column=4).value for r in range(7, 14)]  # D7..D13
# These are % values for Jan 2026 – Jul 2026
rem_months = pd.date_range("2026-01-01", periods=7, freq="MS")
rem = pd.DataFrame({"fecha": rem_months, "ipc_mensual": rem_ipc_monthly})
print("REM monthly IPC (%):")
print(rem.to_string(index=False))

# ── 2. Project IPC Aug–Dec 2026 by continuing the trend ──────────────────
# Fit a linear trend to the 7 REM points and extrapolate
from numpy.polynomial import polynomial as P
x = np.arange(len(rem))
coeffs = P.polyfit(x, rem["ipc_mensual"].values, 1)  # linear fit
x_proj = np.arange(len(rem), len(rem) + 5)  # Aug–Dec
ipc_proj = P.polyval(x_proj, coeffs)
# Ensure projected IPC doesn't go below a reasonable floor
ipc_proj = np.maximum(ipc_proj, 0.5)

proj_months = pd.date_range("2026-08-01", periods=5, freq="MS")
proj = pd.DataFrame({"fecha": proj_months, "ipc_mensual": ipc_proj})

ipc_2026 = pd.concat([rem, proj], ignore_index=True)
print("\nFull projected monthly IPC for 2026:")
print(ipc_2026.to_string(index=False))

# ── 3. Read crudo_profasis and IPC data ───────────────────────────────────
crudo = pd.read_csv("datos/crudo_profasis.csv", parse_dates=["fecha"])
ipc = pd.read_csv("datos/ipc_nuevo.csv", parse_dates=["fecha"])

# Keep only data from Nov 2023 onward (to match the adjusted plot)
crudo = crudo[crudo["fecha"] >= "2023-11-01"].copy().reset_index(drop=True)

# ── 4. Build IPC index projection for 2026 ───────────────────────────────
# Get the last known IPC index value and extend it with REM projections
last_ipc_row = ipc[ipc["fecha"] <= "2026-02-01"].iloc[-1]
last_ipc_fecha = last_ipc_row["fecha"]
last_ipc_value = last_ipc_row["indice"]

# Build monthly IPC index from the last known month forward
ipc_index_proj = [last_ipc_value]
ipc_dates_proj = [last_ipc_fecha]
for _, row in ipc_2026[ipc_2026["fecha"] > last_ipc_fecha].iterrows():
    new_val = ipc_index_proj[-1] * (1 + row["ipc_mensual"] / 100)
    ipc_index_proj.append(new_val)
    ipc_dates_proj.append(row["fecha"])

ipc_proj_df = pd.DataFrame({"fecha": ipc_dates_proj, "indice": ipc_index_proj})

# Merge with existing IPC (keep existing, add new months)
ipc_full = pd.concat([
    ipc[ipc["fecha"] < ipc_proj_df["fecha"].min()],
    ipc_proj_df
], ignore_index=True)

# ── 5. Build the three scenarios ──────────────────────────────────────────
# Last known salary value and its date
last_known_idx = crudo[crudo["fecha"] <= "2026-02-01"].index[-1]
last_known_salary = crudo.loc[last_known_idx, "salario"]
last_known_date = crudo.loc[last_known_idx, "fecha"]

# Extend crudo dates to Dec 2026
future_dates = pd.date_range(
    start=last_known_date + pd.DateOffset(months=1),
    end="2026-12-01",
    freq="MS"
)

# --- Scenario 1: Monthly increases = projected IPC ---
scenario1 = crudo[["fecha", "salario"]].copy()
scenario1 = scenario1.rename(columns={"salario": "escenario_ipc"})

prev_salary = last_known_salary
for d in future_dates:
    ipc_row = ipc_2026[ipc_2026["fecha"] == d]
    if len(ipc_row) > 0:
        monthly_pct = ipc_row["ipc_mensual"].values[0]
    else:
        monthly_pct = ipc_2026["ipc_mensual"].iloc[-1]  # fallback
    new_salary = round(prev_salary * (1 + monthly_pct / 100))
    scenario1 = pd.concat([scenario1, pd.DataFrame({
        "fecha": [d], "escenario_ipc": [new_salary]
    })], ignore_index=True)
    prev_salary = new_salary

# --- Scenario 2: Monthly IPC + 4.1% discrete bumps in Mar, Jul, Sep 2026 ---
dec_2025_salary = crudo.loc[crudo["fecha"] == "2025-12-01", "salario"].values[0]
bump = dec_2025_salary * 0.041  # 4.1% of Dec-2025 salary

scenario2 = crudo[["fecha", "salario"]].copy()
scenario2 = scenario2.rename(columns={"salario": "escenario_41"})

# Build salary path: monthly IPC increase + discrete bumps
s2_salary = last_known_salary
for d in future_dates:
    # First apply monthly IPC
    ipc_row = ipc_2026[ipc_2026["fecha"] == d]
    if len(ipc_row) > 0:
        monthly_pct = ipc_row["ipc_mensual"].values[0]
    else:
        monthly_pct = ipc_2026["ipc_mensual"].iloc[-1]
    s2_salary = s2_salary * (1 + monthly_pct / 100)
    # Then add discrete bump if applicable
    if d.month in (3, 7, 9) and d.year == 2026:
        s2_salary = s2_salary + bump
    s2_salary = round(s2_salary)
    scenario2 = pd.concat([scenario2, pd.DataFrame({
        "fecha": [d], "escenario_41": [s2_salary]
    })], ignore_index=True)

# --- Scenario 4: Pessimistic – salary increases IPC minus shortfall, 12% annual loss ---
# Jan-Feb are known (salary flat = 0% vs IPC ~4.5%, already ~4.5 pp lost)
# Distribute remaining ~7.5 pp loss across Mar-Dec (0.2 to ~1.3 pp shortfall)
shortfalls = np.linspace(0.2, 1.3, len(future_dates))  # 10 months, sum ≈ 7.5

scenario4 = crudo[["fecha", "salario"]].copy()
scenario4 = scenario4.rename(columns={"salario": "escenario_pesimista"})

s4_salary = last_known_salary
for i, d in enumerate(future_dates):
    ipc_row = ipc_2026[ipc_2026["fecha"] == d]
    if len(ipc_row) > 0:
        monthly_pct = ipc_row["ipc_mensual"].values[0]
    else:
        monthly_pct = ipc_2026["ipc_mensual"].iloc[-1]
    effective_pct = max(monthly_pct - shortfalls[i], 0)
    s4_salary = round(s4_salary * (1 + effective_pct / 100))
    scenario4 = pd.concat([scenario4, pd.DataFrame({
        "fecha": [d], "escenario_pesimista": [s4_salary]
    })], ignore_index=True)

# --- Scenario 3: IPC-adjusted salary from Nov-2023 ---
# This is the "ajustado" line: salary from Nov 2023 maintained at constant real value
salario_nov2023 = crudo.loc[crudo["fecha"] == "2023-11-01", "salario"].values[0]
indice_nov2023 = ipc.loc[ipc["fecha"] == "2023-11-01", "indice"].values[0]

# Build the full adjusted series using the full IPC (including projections)
all_dates = pd.concat([
    crudo[["fecha"]],
    pd.DataFrame({"fecha": future_dates})
], ignore_index=True).drop_duplicates().sort_values("fecha").reset_index(drop=True)

scenario3 = all_dates.merge(ipc_full[["fecha", "indice"]], on="fecha", how="left")
# Forward-fill any missing IPC
scenario3["indice"] = scenario3["indice"].ffill()
scenario3["ajustado"] = salario_nov2023 * (scenario3["indice"] / indice_nov2023)

# ── 6. Merge everything ──────────────────────────────────────────────────
merged = scenario3[["fecha", "ajustado"]].copy()
merged = merged.merge(scenario1, on="fecha", how="left")
merged = merged.merge(scenario2, on="fecha", how="left")
merged = merged.merge(scenario4, on="fecha", how="left")

print("\nProjection data:")
print(merged[merged["fecha"] >= "2025-10-01"].to_string(index=False))

# ── 7. Plot (2026 only) ───────────────────────────────────────────────────
current_date = pd.Timestamp.now().strftime("%d/%m/%Y")

# Filter to 2026 only
plot_data = merged[merged["fecha"] >= "2026-01-01"].copy().reset_index(drop=True)

fig, ax = plt.subplots(figsize=(3840/300, 2700/300), dpi=300)

# Plot the four scenarios
ax.plot(plot_data["fecha"], plot_data["escenario_ipc"], color="blue", linewidth=4,
        marker="o", markersize=6, label="Escenario optimista: aumento = IPC (REM)", zorder=2.7)
ax.plot(plot_data["fecha"], plot_data["escenario_pesimista"], color="steelblue", linewidth=4,
        marker="D", markersize=6, label="Escenario pesimista: aumento IPC−12% (similar a lo perdido en 2025)", zorder=2.65)
ax.plot(plot_data["fecha"], plot_data["escenario_41"], color="darkviolet", linewidth=4,
        marker="s", markersize=6, label="Escenario optimista + 12,3% aumento (modificación de la ley propuesta por LLA)", zorder=2.6)
ax.plot(plot_data["fecha"], plot_data["ajustado"], color="red", linewidth=4,
        marker="o", markersize=6, label="Ley de Financiamiento Universitario", zorder=2.5)

# Annotations for final values
for col, color, ha_off in [
    ("escenario_ipc", "blue", -25),
    ("escenario_pesimista", "steelblue", -40),
    ("escenario_41", "darkviolet", 30),
    ("ajustado", "red", 15),
]:
    final_val = plot_data[col].iloc[-1]
    final_date = plot_data["fecha"].iloc[-1]
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec=color, alpha=1, zorder=3)
    ax.annotate(
        f"${final_val:,.0f}",
        xy=(final_date, final_val),
        xycoords="data",
        xytext=(20, ha_off),
        textcoords="offset points",
        color=color,
        fontsize=13,
        fontweight="bold",
        bbox=bbox_props,
        arrowprops=dict(arrowstyle="->", color=color, lw=2, shrinkB=4, zorder=2.5),
        zorder=3,
    )

# Also annotate starting values
for col, color, y_off in [("escenario_ipc", "blue", 75), ("ajustado", "red", 45)]:
    first_val = plot_data[col].dropna().iloc[0]
    first_date = plot_data["fecha"].iloc[0]
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec=color, alpha=1, zorder=3)
    ax.annotate(
        f"${first_val:,.0f}",
        xy=(first_date, first_val),
        xycoords="data",
        xytext=(15, y_off),
        textcoords="offset points",
        color=color,
        fontsize=13,
        fontweight="bold",
        bbox=bbox_props,
        arrowprops=dict(arrowstyle="->", color=color, lw=2, shrinkB=4, zorder=2.5),
        zorder=3,
    )

# Y axis
min_value = min(plot_data["escenario_ipc"].min(), plot_data["escenario_pesimista"].min(),
                plot_data["escenario_41"].min(), plot_data["ajustado"].min())
max_value = max(plot_data["escenario_ipc"].max(), plot_data["escenario_pesimista"].max(),
                plot_data["escenario_41"].max(), plot_data["ajustado"].max())
ylim_min = np.floor(0.9 * min_value / 100000) * 100000
ylim_max = np.ceil(1.15 * max_value / 100000) * 100000

ax.set_ylim(ylim_min, ylim_max)
yticks = np.arange(ylim_min + 100000, ylim_max, 100000)
ax.set_yticks(yticks)

for y in yticks:
    ax.axhline(y=y, color="gray", linestyle="--", linewidth=0.5)

def millones_formatter(x, p):
    return f"{x/1000000:,.1f}M".replace(".", ",")

ax.yaxis.set_major_formatter(plt.FuncFormatter(millones_formatter))

# Secondary Y axis
ax2 = ax.twinx()
ax2.set_ylim(ylim_min, ylim_max)
ax2.set_yticks(yticks)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(millones_formatter))

# X axis
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.set_xlim(plot_data["fecha"].min() - pd.DateOffset(days=15),
            plot_data["fecha"].max() + pd.DateOffset(months=2))
ax.tick_params(axis="x", rotation=90, labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax2.tick_params(axis="y", labelsize=12)

# Title and labels
ax.set_title(
    "Proyección salarial - Profesor Asistente Dedicación Exclusiva \nEscenarios de salario de bolsillo hasta diciembre de 2026",
    fontsize=28,
)
ax.set_xlabel("Fecha", fontsize=20)
ax.set_ylabel("Salario de bolsillo", fontsize=20)
ax2.set_ylabel("Salario de bolsillo", fontsize=20)
handles, labels = ax.get_legend_handles_labels()
# Reorder: red, purple, blue, steelblue
order = [3, 2, 0, 1]  # ajustado, escenario_41, escenario_ipc, escenario_pesimista
ax.legend([handles[i] for i in order], [labels[i] for i in order],
          fontsize=13, loc="upper left")

footnote = (
    f"IPC proyectado según REM-BCRA (enero 2026), extrapolación lineal para ago-dic.\n"
    f"Propuesta LLA: IPC mensual + tres aumentos de 4,1% (sobre dic-2025) en mar, jul y sep 2026.\n"
    f"Gráfico generado el {current_date}."
)
plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=12, style="italic")

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig("plots/proyeccion_profasis_2026.png")
plt.close()

print("\nPlot saved to plots/proyeccion_profasis_2026.png")

