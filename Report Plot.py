import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.ndimage import gaussian_filter1d

# format settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'axes.linewidth': 0.8,
    'legend.frameon': False, 
    'figure.figsize': (6,4.5), 
    'figure.dpi': 300
})

# Load Data
data = []
with open('runs/20260129_115824/q_table_train_log.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# Calculate Rolling Averages
# Note: we log every 10 episodes, window 50 = 500 episodes
window = 50
df['return_mean'] = df['return'].rolling(window=window, min_periods=1).mean()
df['return_std'] = df['return'].rolling(window=window, min_periods=1).std()

# Plot
plt.figure()
# Plot Mean Line
plt.plot(df['ep'], df['return_mean'], color='#004488', linewidth=1.5, label='Rolling Average over 500 episodes')
# Plot Standard Deviation (Shaded)
plt.fill_between(
    df['ep'], 
    df['return_mean'] - df['return_std'], 
    df['return_mean'] + df['return_std'], 
    color='#004488', 
    alpha=0.2,
    linewidth=0
)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.xlim(left=0)

# Add Vertical Lines for Curriculum Stages
stage_changes = df.drop_duplicates(subset=['stage'])

for idx, row in stage_changes.iterrows():
    ep = row['ep']
    stage = int(row['stage'])
    
    # Vertical dotted line (thinner, purely structural)
    plt.axvline(x=ep, color='black', linestyle=':', linewidth=1.0)
        
    # Arrow Annotation
    y_val = df.loc[df['ep'] == ep, 'return_mean'].iloc[0]
        
    plt.annotate(
        f'Stage {stage}', 
        xy=(ep, y_val), 
        xytext=(ep + 50, y_val - 300),
        fontfamily='serif'
    )

plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('Evaluation/Sarsa_fast_learning_curve.png')      
plt.show()


