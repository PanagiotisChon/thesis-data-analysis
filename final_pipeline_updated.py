
import pandas as pd, numpy as np, time, json, warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, silhouette_score)

plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
np.random.seed(42)

def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100

def metrics(y_true, y_pred):
    return {
        'R2':   r2_score(y_true, y_pred),
        'MSE':  mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
    }

# =========================================================
# ΦΟΡΤΩΣΗ ΚΑΙ ΑΡΧΙΚΗ ΕΠΙΣΚΟΠΗΣΗ
# =========================================================
df = pd.read_csv('greecelistings.csv')

print(f"Μέγεθος dataset: {df.shape[0]} γραμμές × {df.shape[1]} στήλες\n")
df.info()
print('\n')

print("Πλήθος ανά location_region:")
print(df["location_region"].value_counts(), '\n')

print("Ποσοστό ανά location_region:")
print(df["location_region"].value_counts(normalize=True) * 100, '\n')

counts = df["res_type"].value_counts()
percent = df["res_type"].value_counts(normalize=True) * 100
median = df.groupby("res_type")["res_price"].median()
result = pd.DataFrame({
    'Πλήθος': counts,
    'Ποσοστό (%)': percent,
    'Διάμεση τιμή (€)': median
})
print("Ανάλυση ανά τύπο κατοικίας:")
print(result.sort_values("Πλήθος", ascending=False), '\n')

# Έλεγχος missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Πλήθος ελλειπόντων': missing,
    'Ποσοστό (%)': missing_pct
})
missing_df = missing_df[missing_df['Πλήθος ελλειπόντων'] > 0]
print("ΕΛΛΕΙΠΟΥΣΕΣ ΤΙΜΕΣ:")
print(missing_df.sort_values('Πλήθος ελλειπόντων', ascending=False), '\n')

# Στατιστική περιγραφή αριθμητικών
numeric_cols = ['res_price', 'res_price_sqr', 'res_sqr',
                'construction_year', 'bedrooms', 'bathrooms']
stats = pd.DataFrame({
    'Mean':     df[numeric_cols].mean().round(2),
    'Median':   df[numeric_cols].median().round(2),
    'Std':      df[numeric_cols].std().round(2),
    'Min':      df[numeric_cols].min().round(2),
    'Max':      df[numeric_cols].max().round(2),
    'Variance': df[numeric_cols].var().round(2)
})
print("ΣΤΑΤΙΣΤΙΚΗ ΠΕΡΙΓΡΑΦΗ:")
print(stats, '\n')

# Ποσοτικός εντοπισμός outliers με κανόνα IQR
def count_outliers_iqr(s):
    s = s.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    return ((s < low) | (s > high)).sum(), low, high

print("OUTLIERS (μέθοδος IQR):")
for col in numeric_cols:
    n, low, high = count_outliers_iqr(df[col])
    print(f"  {col:20s}: {n:5d} outliers  | "
          f"όρια: [{low:.2f}, {high:.2f}]")
print()

# =========================================================
# ΓΡΑΦΗΜΑΤΑ ΚΕΦΑΛΑΙΟΥ 3 (πριν τον καθαρισμό)
# =========================================================
print("Παραγωγή γραφημάτων κεφαλαίου 3...")
titles_gr = ['Τιμή πώλησης (€)','Τιμή ανά τ.μ. (€/τ.μ.)','Εμβαδό (τ.μ.)',
             'Έτος κατασκευής','Υπνοδωμάτια','Μπάνια']

# Σχήμα 3.1 — Boxplots raw
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for ax, col, t in zip(axes, numeric_cols, titles_gr):
    sns.boxplot(y=df[col].dropna(), ax=ax, color='steelblue')
    ax.set_title(t, fontsize=11, fontweight='bold'); ax.set_ylabel('')
plt.suptitle('Boxplots βασικών αριθμητικών χαρακτηριστικών',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('boxplots_raw.png', dpi=120, bbox_inches='tight')
plt.close()
print("  [OK] boxplots_raw.png")

# =========================================================
# ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ (με location_name)
# =========================================================
df = df.drop(columns=['res_date','res_address','deleted','deleted_at'])  # ΚΡΑΤΑΜΕ location_name
df = df[(df['res_price']>=20000)&(df['res_price']<=3_000_000)]
df = df[(df['res_price_sqr']>=300)&(df['res_price_sqr']<=12000)]
df = df[(df['res_sqr']>=20)&(df['res_sqr']<=1000)]
df = df[(df['bedrooms'].isna())|((df['bedrooms']>=1)&(df['bedrooms']<=8))]
df = df[(df['bathrooms'].isna())|((df['bathrooms']>=1)&(df['bathrooms']<=6))]
df = df[(df['construction_year'].isna())|((df['construction_year']>=1900)&(df['construction_year']<=2025))]

df['has_parking'] = df['parking'].notna().astype(int)
df = df.drop(columns=['parking'])
df['energyclass'] = df['energyclass'].fillna('Άγνωστο')
df['status'] = df['status'].fillna('Άγνωστο')
for c in ['bedrooms','bathrooms','construction_year']:
    df[c] = df.groupby('res_type')[c].transform(lambda s: s.fillna(s.median()))
    df[c] = df[c].fillna(df[c].median())
df['levels'] = df['levels'].fillna('Άγνωστο').astype(str).str.split(',').str[0].replace('nan','Άγνωστο')
df['location_name'] = df['location_name'].fillna('Άγνωστο')

n_unique_loc = df['location_name'].nunique()
print(f"Dataset: {df.shape}, unique location_name: {n_unique_loc}")

# =========================================================
# ΓΡΑΦΗΜΑΤΑ ΚΕΦΑΛΑΙΟΥ 4 (μετά τον καθαρισμό)
# =========================================================
print("Παραγωγή γραφημάτων κεφαλαίου 4...")

# Σχήμα 4.1 — Boxplots clean
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for ax, col, t in zip(axes, numeric_cols, titles_gr):
    sns.boxplot(y=df[col], ax=ax, color='seagreen')
    ax.set_title(t, fontsize=11, fontweight='bold'); ax.set_ylabel('')
plt.suptitle('Boxplots μετά τον καθαρισμό outliers',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('boxplots_clean.png', dpi=120, bbox_inches='tight')
plt.close()
print("  [OK] boxplots_clean.png")

# Σχήμα 4.2 — Ιστογράμματα
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for ax, col, t in zip(axes, numeric_cols, titles_gr):
    ax.hist(df[col].dropna(), bins=40, color='steelblue', edgecolor='white')
    ax.set_title(t, fontsize=11, fontweight='bold')
    ax.set_ylabel('Συχνότητα')
plt.suptitle('Ιστογράμματα κατανομής αριθμητικών χαρακτηριστικών',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('histograms.png', dpi=120, bbox_inches='tight')
plt.close()
print("  [OK] histograms.png")

# Σχήμα 4.3 — Heatmap συσχετίσεων
num_for_corr = ['res_price','res_price_sqr','res_sqr','construction_year',
                'bedrooms','bathrooms','auto_heating','solar','cooling',
                'safe_door','gas','fireplace','furniture','student','has_parking']
corr = df[num_for_corr].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'label': 'Συσχέτιση Pearson'})
plt.title('Heatmap συσχετίσεων αριθμητικών χαρακτηριστικών',
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
print("  [OK] heatmap.png")

# Σχήμα 4.4 — Pairplot
pair_cols = ['res_price','res_sqr','bedrooms','bathrooms']
sample = df[pair_cols].sample(n=min(2000, len(df)), random_state=42)
g = sns.pairplot(sample, diag_kind='hist',
                 plot_kws={'alpha': 0.4, 's': 12, 'color': 'steelblue'},
                 diag_kws={'color': 'steelblue', 'edgecolor': 'white'})
g.fig.suptitle('Pairplot τιμής, εμβαδού, υπνοδωματίων & μπάνιων',
               fontsize=12, fontweight='bold', y=1.01)
plt.savefig('pairplot.png', dpi=110, bbox_inches='tight')
plt.close()
print("  [OK] pairplot.png\n")

y_full = df['res_price'].reset_index(drop=True)
X_base = df.drop(columns=['res_price','res_price_sqr']).reset_index(drop=True)

X_tr, X_te, y_tr, y_te = train_test_split(X_base, y_full, test_size=0.20, random_state=42)
print(f"Train: {X_tr.shape[0]}, Test: {X_te.shape[0]}")

# ---------- Target encoding για location_name (από train ΜΟΝΟ) ----------
loc_mean = y_tr.groupby(X_tr['location_name']).mean()
loc_count = X_tr['location_name'].value_counts()
global_mean = y_tr.mean()
k_smooth = 20

def t_enc(s):
    cnt = s.map(loc_count).fillna(0)
    mn = s.map(loc_mean).fillna(global_mean)
    return (cnt*mn + k_smooth*global_mean)/(cnt+k_smooth)

X_tr = X_tr.copy(); X_te = X_te.copy()
X_tr['loc_target_enc'] = t_enc(X_tr['location_name']).values
X_te['loc_target_enc'] = t_enc(X_te['location_name']).values
X_tr['loc_frequency']  = X_tr['location_name'].map(loc_count).fillna(0).values
X_te['loc_frequency']  = X_te['location_name'].map(loc_count).fillna(0).values
X_tr = X_tr.drop(columns=['location_name'])
X_te = X_te.drop(columns=['location_name'])

# ---------- One-hot εναπομεινάντων κατηγορικών ----------
cat = ['location_region','res_type','levels','status','energyclass']
X_tr = pd.get_dummies(X_tr, columns=cat, drop_first=True)
X_te = pd.get_dummies(X_te, columns=cat, drop_first=True)
X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
for c in X_tr.select_dtypes('bool').columns:
    X_tr[c] = X_tr[c].astype(int); X_te[c] = X_te[c].astype(int)

# ---------- Scaling αριθμητικών ----------
num_feats = ['res_sqr','construction_year','bedrooms','bathrooms',
             'loc_target_enc','loc_frequency']
scaler = StandardScaler()
X_tr[num_feats] = scaler.fit_transform(X_tr[num_feats])
X_te[num_feats] = scaler.transform(X_te[num_feats])

print(f"X_train ΜΕ location features: {X_tr.shape}\n")

# ---------- Log-transform του target ----------
y_tr_log = np.log1p(y_tr)
y_te_log = np.log1p(y_te)

# =========================================================
# REGRESSION MODELS
# =========================================================
print("="*80)
print("REGRESSION (στόχος = log(price))")
print("="*80)

models_results = {}
predictions = {}

# Linear Regression — ΧΩΡΙΣ log (το γραμμικό αποδίδει χειρότερα σε log-space)
t = time.time()
lr = LinearRegression()
lr.fit(X_tr, y_tr)
pred = lr.predict(X_te)
m = metrics(y_te, pred)
m['time'] = time.time() - t
models_results['Linear Regression'] = m
predictions['Linear Regression'] = pred
print(f"Linear Regression       R²={m['R2']:.4f}  RMSE={m['RMSE']:>9,.0f}  MAE={m['MAE']:>9,.0f}  MAPE={m['MAPE']:5.2f}%  ({m['time']:.2f}s)")

# Decision Tree
t = time.time()
dt = DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42)
dt.fit(X_tr, y_tr_log)
pred = np.expm1(dt.predict(X_te))
m = metrics(y_te, pred)
m['time'] = time.time() - t
models_results['Decision Tree'] = m
predictions['Decision Tree'] = pred
print(f"Decision Tree           R²={m['R2']:.4f}  RMSE={m['RMSE']:>9,.0f}  MAE={m['MAE']:>9,.0f}  MAPE={m['MAPE']:5.2f}%  ({m['time']:.2f}s)")

# Random Forest
t = time.time()
rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=2,
                            n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr_log)
pred = np.expm1(rf.predict(X_te))
m = metrics(y_te, pred)
m['time'] = time.time() - t
models_results['Random Forest'] = m
predictions['Random Forest'] = pred
print(f"Random Forest           R²={m['R2']:.4f}  RMSE={m['RMSE']:>9,.0f}  MAE={m['MAE']:>9,.0f}  MAPE={m['MAPE']:5.2f}%  ({m['time']:.2f}s)")

# HistGradientBoosting
t = time.time()
hgb = HistGradientBoostingRegressor(max_iter=500, max_depth=8,
                                     learning_rate=0.05, min_samples_leaf=20,
                                     random_state=42)
hgb.fit(X_tr, y_tr_log)
pred = np.expm1(hgb.predict(X_te))
m = metrics(y_te, pred)
m['time'] = time.time() - t
models_results['HistGradientBoosting'] = m
predictions['HistGradientBoosting'] = pred
print(f"HistGradientBoosting    R²={m['R2']:.4f}  RMSE={m['RMSE']:>9,.0f}  MAE={m['MAE']:>9,.0f}  MAPE={m['MAPE']:5.2f}%  ({m['time']:.2f}s)")

# =========================================================
# FEATURE IMPORTANCE (από Random Forest)
# =========================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*80)
fi = pd.DataFrame({
    'feature': X_tr.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(fi.head(15).to_string(index=False))
fi.to_csv('feature_importance_final.csv', index=False, encoding='utf-8-sig')

# =========================================================
# ΓΡΑΦΗΜΑΤΑ
# =========================================================

# 1. Predicted vs Actual (4 μοντέλα)
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
names_order = ['Linear Regression','Decision Tree','Random Forest','HistGradientBoosting']
for ax, name in zip(axes, names_order):
    pred = predictions[name]
    ax.scatter(y_te, pred, alpha=0.3, s=10, color='steelblue', edgecolor='none')
    lims = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    ax.plot(lims, lims, 'r--', lw=2)
    r2 = models_results[name]['R2']
    ax.set_xlabel('Πραγματική τιμή (€)', fontsize=10)
    ax.set_ylabel('Προβλεπόμενη τιμή (€)', fontsize=10)
    ax.set_title(f'{name}\nR² = {r2:.4f}', fontsize=11, fontweight='bold')
    ax.ticklabel_format(style='plain', axis='both')
plt.suptitle('Προβλεπόμενες vs Πραγματικές τιμές (test set)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('predicted_vs_actual_v2.png', dpi=120, bbox_inches='tight')
plt.close()

# 2. Error distribution (4 μοντέλα)
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, name in zip(axes, names_order):
    pred = predictions[name]
    err = y_te - pred
    ax.hist(err, bins=50, color='coral', edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Σφάλμα (€)', fontsize=10)
    ax.set_ylabel('Συχνότητα', fontsize=10)
    mae_v = models_results[name]['MAE']
    ax.set_title(f'{name}\nMAE = {mae_v:,.0f}€', fontsize=11, fontweight='bold')
plt.suptitle('Κατανομή σφαλμάτων (test set)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('error_distribution_v2.png', dpi=120, bbox_inches='tight')
plt.close()

# 3. Comparison bars (R² και RMSE)
df_res = pd.DataFrame([
    {'name': n, **models_results[n]} for n in names_order
])
df_res_sorted = df_res.sort_values('R2', ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
colors = ['#3498db','#e74c3c','#2ecc71','#9b59b6']
axes[0].barh(df_res_sorted['name'], df_res_sorted['R2'], color=colors)
axes[0].set_xlabel('R²', fontsize=11)
axes[0].set_title('R² ανά μοντέλο', fontsize=12, fontweight='bold')
axes[0].set_xlim(0, 1)
for i, v in enumerate(df_res_sorted['R2']):
    axes[0].text(v+0.01, i, f'{v:.4f}', va='center', fontsize=10)
axes[1].barh(df_res_sorted['name'], df_res_sorted['RMSE']/1000, color=colors)
axes[1].set_xlabel('RMSE (×1.000 €)', fontsize=11)
axes[1].set_title('RMSE ανά μοντέλο', fontsize=12, fontweight='bold')
for i, v in enumerate(df_res_sorted['RMSE']/1000):
    axes[1].text(v+1, i, f'{v:.1f}k', va='center', fontsize=10)
plt.suptitle('Συγκριτική αξιολόγηση μοντέλων Regression',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('model_comparison_v2.png', dpi=120, bbox_inches='tight')
plt.close()

# 4. Feature importance bar chart
fig, ax = plt.subplots(figsize=(10, 7))
top15 = fi.head(15).iloc[::-1]
ax.barh(top15['feature'], top15['importance'], color='steelblue')
ax.set_xlabel('Importance', fontsize=11)
ax.set_title('TOP-15 σημαντικότερα χαρακτηριστικά (Random Forest)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance_v2.png', dpi=120, bbox_inches='tight')
plt.close()

print("\n[OK] Γραφήματα regression: predicted_vs_actual_v2, error_distribution_v2,")
print("     model_comparison_v2, feature_importance_v2")

# =========================================================
# CLUSTERING
# =========================================================
print("\n" + "="*80)
print("CLUSTERING")
print("="*80)

X_all = pd.concat([X_tr, X_te], axis=0).reset_index(drop=True)
y_all = pd.concat([y_tr, y_te], axis=0).reset_index(drop=True)

clu_features = ['res_sqr','construction_year','bedrooms','bathrooms',
                'has_parking','fireplace','solar','cooling','loc_target_enc']
X_clu = X_all[clu_features].copy()
# Επανατυποποίηση για ίδια κλίμακα σε όλο το block
scaler_c = StandardScaler()
X_clu_s = scaler_c.fit_transform(X_clu)
print(f"Clustering matrix: {X_clu_s.shape}")

# Elbow + silhouette για k=2..8
ks = range(2, 9)
inertias, sils = [], []
sample_idx = np.random.choice(len(X_clu_s), size=min(5000, len(X_clu_s)), replace=False)
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_clu_s)
    inertias.append(km.inertia_)
    s = silhouette_score(X_clu_s[sample_idx], km.predict(X_clu_s[sample_idx]))
    sils.append(s)
    print(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={s:.4f}")

best_k = list(ks)[int(np.argmax(sils))]
print(f"\nΒέλτιστο k = {best_k}, silhouette = {max(sils):.4f}")

# Selection plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(ks), inertias, 'o-', color='steelblue', lw=2)
axes[0].set_xlabel('Αριθμός clusters (k)', fontsize=11)
axes[0].set_ylabel('Inertia', fontsize=11)
axes[0].set_title('Elbow method', fontsize=12, fontweight='bold')
axes[1].plot(list(ks), sils, 'o-', color='seagreen', lw=2)
axes[1].set_xlabel('Αριθμός clusters (k)', fontsize=11)
axes[1].set_ylabel('Silhouette score', fontsize=11)
axes[1].set_title('Silhouette score', fontsize=12, fontweight='bold')
axes[1].axvline(best_k, color='red', linestyle='--', alpha=0.5, label=f'best k = {best_k}')
axes[1].legend()
plt.suptitle('Επιλογή αριθμού clusters για K-Means',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('kmeans_selection_v2.png', dpi=120, bbox_inches='tight')
plt.close()

# Final K-Means
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
km_labels = kmeans.fit_predict(X_clu_s)
sil_km = silhouette_score(X_clu_s[sample_idx], kmeans.predict(X_clu_s[sample_idx]))

# Profile σε real units (αναπαράγουμε τα πραγματικά χαρακτηριστικά)
df_full = df.reset_index(drop=True)
# Αναμίξουμε το profile με τις original τιμές που έχουμε στο df_full
# Σημείωση: X_all/X_clu_s είναι σε ανακατεμένη σειρά train→test. Πρέπει να συγχρονίσουμε.
# Καλύτερα: ξανατρέχουμε το profile χρησιμοποιώντας τα clu_features όπως υπήρχαν στο df πριν το scaling.

# Πάμε ξανά πιο σωστά: clustering πάνω σε original-feature space για το profile
df_clu_orig = pd.DataFrame({
    'res_sqr': df['res_sqr'].values,
    'construction_year': df['construction_year'].values,
    'bedrooms': df['bedrooms'].values,
    'bathrooms': df['bathrooms'].values,
    'has_parking': df['has_parking'].values,
    'fireplace': df['fireplace'].values,
    'solar': df['solar'].values,
    'cooling': df['cooling'].values,
}).reset_index(drop=True)
df_clu_orig['price'] = df['res_price'].values
df_clu_orig['res_type'] = df['res_type'].values
df_clu_orig['region'] = df['location_region'].values
df_clu_orig['location'] = df['location_name'].values

# Μετράμε mean price ανά γειτονιά (παντού-διαθέσιμη πληροφορία ως feature)
loc_mean_full = df.groupby('location_name')['res_price'].mean()
df_clu_orig['loc_mean_price'] = df_clu_orig['location'].map(loc_mean_full).fillna(df['res_price'].mean())

# Standardize για K-Means
clu_cols = ['res_sqr','construction_year','bedrooms','bathrooms',
            'has_parking','fireplace','solar','cooling','loc_mean_price']
X_real_clu = StandardScaler().fit_transform(df_clu_orig[clu_cols])

# ΞΑΝΑ τρέξιμο: silhouette για k=2..8 στο σωστό feature space
print("\nΕπαλήθευση clustering στο feature space που χρησιμοποιείται για το profile:")
ks2 = range(2, 9)
inertias2, sils2 = [], []
sample_idx2 = np.random.choice(len(X_real_clu), size=min(5000, len(X_real_clu)), replace=False)
for k in ks2:
    km_k = KMeans(n_clusters=k, n_init=10, random_state=42)
    km_k.fit(X_real_clu)
    inertias2.append(km_k.inertia_)
    s2 = silhouette_score(X_real_clu[sample_idx2], km_k.predict(X_real_clu[sample_idx2]))
    sils2.append(s2)
    print(f"  k={k}: inertia={km_k.inertia_:.0f}, silhouette={s2:.4f}")

best_k2 = list(ks2)[int(np.argmax(sils2))]
print(f"\nΕπιλεγμένο k = {best_k2} (silhouette = {max(sils2):.4f})")

# Final K-Means στο σωστό feature space
kmeans = KMeans(n_clusters=best_k2, n_init=10, random_state=42)
km_labels = kmeans.fit_predict(X_real_clu)
sil_km = silhouette_score(X_real_clu[sample_idx2], kmeans.predict(X_real_clu[sample_idx2]))
print(f"Final K-Means silhouette: {sil_km:.4f}")

# Selection plot (διορθωμένο)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(ks2), inertias2, 'o-', color='steelblue', lw=2)
axes[0].set_xlabel('Αριθμός clusters (k)', fontsize=11)
axes[0].set_ylabel('Inertia', fontsize=11)
axes[0].set_title('Elbow method', fontsize=12, fontweight='bold')
axes[1].plot(list(ks2), sils2, 'o-', color='seagreen', lw=2)
axes[1].set_xlabel('Αριθμός clusters (k)', fontsize=11)
axes[1].set_ylabel('Silhouette score', fontsize=11)
axes[1].set_title('Silhouette score', fontsize=12, fontweight='bold')
axes[1].axvline(best_k2, color='red', linestyle='--', alpha=0.5, label=f'best k = {best_k2}')
axes[1].legend()
plt.suptitle('Επιλογή αριθμού clusters για K-Means',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('kmeans_selection_v2.png', dpi=120, bbox_inches='tight')
plt.close()

# Cluster profile
df_clu_orig['cluster'] = km_labels
profile = df_clu_orig.groupby('cluster').agg(
    n=('price','size'),
    mean_price=('price','mean'),
    median_price=('price','median'),
    mean_sqr=('res_sqr','mean'),
    mean_bedrooms=('bedrooms','mean'),
    mean_bathrooms=('bathrooms','mean'),
    mean_year=('construction_year','mean'),
    pct_parking=('has_parking','mean'),
    pct_fireplace=('fireplace','mean'),
).round(2)
profile['pct_parking']   = (profile['pct_parking']*100).round(1)
profile['pct_fireplace'] = (profile['pct_fireplace']*100).round(1)
profile = profile.sort_values('mean_price', ascending=False)
print("\nProfile clusters (πραγματικές μονάδες, ταξινόμηση κατά τιμή):")
print(profile.to_string())
profile.to_csv('kmeans_profile_v2.csv', encoding='utf-8-sig')

# Επικρατούσα περιοχή και τύπος ανά cluster
print("\nΕπικρατούσες περιοχές και τύποι ανά cluster:")
for c in sorted(df_clu_orig['cluster'].unique()):
    sub = df_clu_orig[df_clu_orig['cluster']==c]
    top_region = sub['region'].value_counts(normalize=True).head(1)
    top_type = sub['res_type'].value_counts(normalize=True).head(2)
    top_loc = sub['location'].value_counts().head(3)
    print(f"  Cluster {c} (n={len(sub)}):")
    print(f"    Region:   {dict(top_region.round(2))}")
    print(f"    Type:     {dict(top_type.round(2))}")
    print(f"    Top locs: {dict(top_loc)}")

# DBSCAN
print("\nDBSCAN(eps=0.8, min_samples=20):")
dbs = DBSCAN(eps=0.8, min_samples=20, n_jobs=-1)
db_labels = dbs.fit_predict(X_real_clu)
n_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
print(f"  Clusters: {n_db}, Noise: {n_noise} ({n_noise/len(db_labels)*100:.1f}%)")

mask = db_labels != -1
if n_db >= 2 and mask.sum() > 100:
    sub_idx = np.random.choice(np.where(mask)[0], size=min(5000, mask.sum()), replace=False)
    sil_db = silhouette_score(X_real_clu[sub_idx], db_labels[sub_idx])
    print(f"  Silhouette: {sil_db:.4f}")
else:
    sil_db = float('nan')

# Clusters PCA visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_real_clu)
print(f"\nPCA: explained variance = {pca.explained_variance_ratio_.sum()*100:.1f}%")

viz_idx = np.random.choice(len(X_pca), size=min(5000, len(X_pca)), replace=False)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sc1 = axes[0].scatter(X_pca[viz_idx,0], X_pca[viz_idx,1], c=km_labels[viz_idx],
                       cmap='tab10', alpha=0.5, s=15)
cent_pca = pca.transform(kmeans.cluster_centers_)
axes[0].scatter(cent_pca[:,0], cent_pca[:,1], marker='X', s=250, c='black',
                edgecolor='white', linewidth=2, label='centroids')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
axes[0].set_title(f'K-Means (k={best_k2}, silhouette={sil_km:.3f})',
                  fontsize=12, fontweight='bold')
axes[0].legend()
plt.colorbar(sc1, ax=axes[0], label='Cluster')

sc2 = axes[1].scatter(X_pca[viz_idx,0], X_pca[viz_idx,1], c=db_labels[viz_idx],
                       cmap='tab10', alpha=0.5, s=15)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
sil_str = f"{sil_db:.3f}" if not np.isnan(sil_db) else "—"
axes[1].set_title(f'DBSCAN ({n_db} clusters, noise={n_noise}, sil={sil_str})',
                  fontsize=12, fontweight='bold')
plt.colorbar(sc2, ax=axes[1], label='Cluster')
plt.suptitle('Οπτικοποίηση clusters σε χώρο PCA (2D)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('clusters_pca_v2.png', dpi=120, bbox_inches='tight')
plt.close()

# Final summary save
final = {
    'n_unique_loc': int(n_unique_loc),
    'n_features': int(X_tr.shape[1]),
    'regression': {n: {k: float(v) if isinstance(v,(int,float,np.floating)) else v
                       for k,v in m.items()}
                   for n, m in models_results.items()},
    'clustering': {
        'kmeans': {
            'k': int(best_k2),
            'silhouette': float(sil_km),
            'inertias': [float(x) for x in inertias2],
            'silhouettes': [float(x) for x in sils2],
        },
        'dbscan': {
            'eps': 0.8, 'min_samples': 20,
            'n_clusters': int(n_db),
            'n_noise': int(n_noise),
            'silhouette': None if np.isnan(sil_db) else float(sil_db),
        },
        'pca_explained': float(pca.explained_variance_ratio_.sum()),
        'pc1': float(pca.explained_variance_ratio_[0]),
        'pc2': float(pca.explained_variance_ratio_[1]),
    },
}
with open('final_summary.json','w',encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent=2)
print("\n[OK] final_summary.json saved")
