import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time 
from scipy.stats import kstest 
import sys
from scipy import stats
# Read data from the CSV file
file_path = sys.argv[1]
data = pd.read_csv(file_path, header=None)

colors = [
    'rgb(55, 126, 184)',   # Plotly Blue
    'rgb(228, 26, 28)',    # Plotly Red
    'rgb(77, 175, 74)',    # Plotly Green
    'rgb(152, 78, 163)',   # Plotly Purple
    'rgb(255, 127, 0)',    # Plotly Orange
    'rgb(0, 139, 139)' ,     # Dark Cyan
    '#e377c2',  # Plotly Pink
]

fig = px.box(data)
fig.update_traces(boxpoints='outliers', marker_color='rgb(55, 126, 184)')

# Update the layout based on the provided settings
fig.update_layout(
    yaxis=dict(title=r'p value', showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True),
    xaxis=dict(title='Henze-Zirkler', showline=True, linewidth=1, linecolor='black', ticks='outside', mirror=True, automargin=True),
    paper_bgcolor='white',  # Set background color to white
    plot_bgcolor='white',   # Set plot area color to white
    font_family="Serif", 
    font_size=14, 
    margin=dict(l=5, r=5, t=5, b=5),
    width=500, 
    height=350,
    barmode='overlay'  # This will make the histograms overlap
)

fig.write_image("pvals_box.pdf")
time.sleep(1)
fig.write_image("pvals_box.pdf")

ks_statistic, ks_p_value = kstest(data, stats.uniform.cdf)
print(f"ks statistic: {ks_statistic}")
print(f"ks pval: {ks_p_value}")

