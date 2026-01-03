import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import pandas as pd
import optuna
import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None) #dataframe doesn't get cut off
pd.set_option('display.width', None)

#IMPORTING DATA AND CREATING DATAFRAMES
df = pd.read_csv('Collected Data  - Data Points.csv', header=0)
print(df.info())

input_columns = ['Fly Ash (kg/m3)',
                 'Slag (kg/m3)',
                 'SiO2:Al2O3',
                 'CaO:SiO2',
                 'CaO:Al2O3',
                 'Fe2O3:Al2O3',
                 'Na2SiO3:NaOH',
                 'NaOH (M)',
                 'Activator:Binder',
                 'Extra Water (kg/m3)',
                 'Liquid:Binder',
                 'Fine Aggregate :Total Aggregate',
                 'Total Aggregate :Binder',
                 'Curing Temp (C)',
                 'Curing Time (hr)',
                 'Age before Exposure (Days)',
                 'H2SO4 (M)',
                 'Days Submerged']
X = df[input_columns]

output_columns = ['Mass Change (%)', 'Compressive Strength (MPa)']
y = df[output_columns]

dfxy = pd.concat([X, y], axis=1)

#univariate analysis (frequency graphs and boxplots)
print(X.describe()) #outputs count, mean, std, min, 25th, median, 75th, max
print(y.describe())

#frequency graphs
fig1, axs = plt.subplots(5, 4, figsize = (12, 8))

axs[0,0].hist(df['Fly Ash (kg/m3)'], bins = 'auto')
axs[0,0].set_title('Fly Ash (kg/m3)')
axs[0,0].grid(True)

axs[0,1].hist(df['Slag (kg/m3)'], bins = 'auto')
axs[0,1].set_title('Slag (kg/m3)')
axs[0,1].grid(True)

axs[0,2].hist(df['SiO2:Al2O3'], bins = 'auto')
axs[0,2].set_title('SiO2:Al2O3')
axs[0,2].grid(True)

axs[0,3].hist(df['CaO:SiO2'], bins = 'auto')
axs[0,3].set_title('CaO:SiO2')
axs[0,3].grid(True)

axs[1,0].hist(df['CaO:Al2O3'], bins = 'auto')
axs[1,0].set_title('CaO:Al2O3')
axs[1,0].grid(True)

axs[1,1].hist(df['Fe2O3:Al2O3'], bins = 'auto')
axs[1,1].set_title('Fe2O3:Al2O3')
axs[1,1].grid(True)

axs[1,2].hist(df['Na2SiO3:NaOH'], bins = 'auto')
axs[1,2].set_title('Na2SiO3:NaOH')
axs[1,2].grid(True)

axs[1,3].hist(df['NaOH (M)'], bins = 'auto')
axs[1,3].set_title('NaOH (M)')
axs[1,3].grid(True)

axs[2,0].hist(df['Activator:Binder'], bins = 'auto')
axs[2,0].set_title('Activator:Binder')
axs[2,0].grid(True)

axs[2,1].hist(df['Extra Water (kg/m3)'], bins = 'auto')
axs[2,1].set_title('Extra Water (kg/m3)')
axs[2,1].grid(True)

axs[2,2].hist(df['Liquid:Binder'], bins = 'auto')
axs[2,2].set_title('Liquid:Binder')
axs[2,2].grid(True)

axs[2,3].hist(df['Fine Aggregate :Total Aggregate'], bins = 'auto')
axs[2,3].set_title('Total Aggregate :Total Binder')
axs[2,3].grid(True)

axs[3,0].hist(df['Total Aggregate :Binder'], bins = 'auto')
axs[3,0].set_title('Total Aggregate :Binder')
axs[3,0].grid(True)

axs[3,1].hist(df['Curing Temp (C)'], bins = 'auto')
axs[3,1].set_title('Curing Temp (C)')
axs[3,1].grid(True)

axs[3,2].hist(df['Curing Time (hr)'], bins = 'auto')
axs[3,2].set_title('Curing Time (hr)')
axs[3,2].grid(True)

axs[3,3].hist(df['Age before Exposure (Days)'], bins = 'auto')
axs[3,3].set_title('Age before Exposure (Days)')
axs[3,3].grid(True)

axs[4,0].hist(df['H2SO4 (M)'], bins = 'auto')
axs[4,0].set_title('H2SO4 (M)')
axs[4,0].grid(True)

axs[4,1].hist(df['Days Submerged'], bins = 'auto')
axs[4,1].set_title('Days Submerged')
axs[4,1].grid(True)

axs[4,2].hist(df['Mass Change (%)'], bins = 'auto')
axs[4,2].set_title('Mass Change (%)')
axs[4,2].grid(True)

axs[4,3].hist(df['Compressive Strength (MPa)'], bins = 'auto')
axs[4,3].set_title('Compressive Strength (MPa)')
axs[4,3].grid(True)

plt.tight_layout()
plt.show()

#boxplots
fig2, axs = plt.subplots(5, 4, figsize = (12, 8))

axs[0,0].boxplot(df['Fly Ash (kg/m3)'], vert=False)
axs[0,0].set_title('Fly Ash (kg/m3)')
axs[0,0].grid(True)

axs[0,1].boxplot(df['Slag (kg/m3)'], vert=False)
axs[0,1].set_title('Slag (kg/m3)')
axs[0,1].grid(True)

axs[0,2].boxplot(df['SiO2:Al2O3'], vert=False)
axs[0,2].set_title('SiO2:Al2O3')
axs[0,2].grid(True)

axs[0,3].boxplot(df['CaO:SiO2'], vert=False)
axs[0,3].set_title('CaO:SiO2')
axs[0,3].grid(True)

axs[1,0].boxplot(df['CaO:Al2O3'], vert=False)
axs[1,0].set_title('CaO:Al2O3')
axs[1,0].grid(True)

axs[1,1].boxplot(df['Fe2O3:Al2O3'], vert=False)
axs[1,1].set_title('Fe2O3:Al2O3')
axs[1,1].grid(True)

axs[1,2].boxplot(df['Na2SiO3:NaOH'], vert=False)
axs[1,2].set_title('Na2SiO3:NaOH')
axs[1,2].grid(True)

axs[1,3].boxplot(df['NaOH (M)'], vert=False)
axs[1,3].set_title('NaOH (M)')
axs[1,3].grid(True)

axs[2,0].boxplot(df['Activator:Binder'], vert=False)
axs[2,0].set_title('Activator:Binder')
axs[2,0].grid(True)

axs[2,1].boxplot(df['Extra Water (kg/m3)'], vert=False)
axs[2,1].set_title('Extra Water (kg/m3)')
axs[2,1].grid(True)

axs[2,2].boxplot(df['Liquid:Binder'], vert=False)
axs[2,2].set_title('Liquid:Binder')
axs[2,2].grid(True)

axs[2,3].boxplot(df['Fine Aggregate :Total Aggregate'], vert=False)
axs[2,3].set_title('Total Aggregate :Total Binder')
axs[2,3].grid(True)

axs[3,0].boxplot(df['Total Aggregate :Binder'], vert=False)
axs[3,0].set_title('Total Aggregate :Binder')
axs[3,0].grid(True)

axs[3,1].boxplot(df['Curing Temp (C)'], vert=False)
axs[3,1].set_title('Curing Temp (C)')
axs[3,1].grid(True)

axs[3,2].boxplot(df['Curing Time (hr)'], vert=False)
axs[3,2].set_title('Curing Time (hr)')
axs[3,2].grid(True)

axs[3,3].boxplot(df['Age before Exposure (Days)'], vert=False)
axs[3,3].set_title('Age before Exposure (Days)')
axs[3,3].grid(True)

axs[4,0].boxplot(df['H2SO4 (M)'], vert=False)
axs[4,0].set_title('H2SO4 (M)')
axs[4,0].grid(True)

axs[4,1].boxplot(df['Days Submerged'], vert=False)
axs[4,1].set_title('Days Submerged')
axs[4,1].grid(True)

axs[4,2].boxplot(df['Mass Change (%)'], vert=False)
axs[4,2].set_title('Mass Change (%)')
axs[4,2].grid(True)

axs[4,3].boxplot(df['Compressive Strength (MPa)'], vert=False)
axs[4,3].set_title('Compressive Strength (MPa)')
axs[4,3].grid(True)

plt.tight_layout()
plt.show()

#bivariate analysis (analyzing correlations and multicollinearity through heatmap matrix/Variance Inflation Factor)
matrix = dfxy.corr()
print(matrix)
plt.figure(figsize=(20,15))
sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt='.2g', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

#check for missing values
#removing mass change because too many NaN (might add back later)
y_new = y.drop('Mass Change (%)', axis=1)
dfxy_new = pd.concat([X, y_new], axis=1)

#remove outliers (using 1.5x IQR rule)

#eliminate duplicate data points

#SPLITTING DATA INTO TRAIN, TEST, VAL
seed=7 #makes sure we have same split of data each time
test_size=0.20 #20% of data
val_size=0.1875 #18.75% of 80% = 15% of total data
X_main, X_test, y_main, y_test = train_test_split(X, y_new, test_size=test_size, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=val_size, random_state=seed)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

#optuna - using KFold, Score=R2 + setting parameter guidelines
def objective(trial):
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 1e-8, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1,10),
        'eta' : trial.suggest_float('eta', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
    }
    model = xgb.XGBRegressor(**param, seed=seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
    mean_scores = scores.mean()
    print(f"R2 scores for each fold: {scores}")
    print(f"Mean R2 score: {mean_scores}")
    return mean_scores

#optimzing hyperparameters with optuna
study = optuna.create_study(study_name='XGBoost', direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True) #maximizing r2
best_params = study.best_params
print(f"Best parameters: {best_params}")

#FINAL TRAIN OF MODEL
best_model = xgb.XGBRegressor(**best_params, seed=seed, early_stopping_rounds=10)
best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

print(f"Best iteration: {best_model.best_iteration}")
print(f"Best validation error: {best_model.best_score}")

#FINAL EVALUATION OF MODEL
y_pred = best_model.predict(X_val)

rmse = root_mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2: {r2}")

#GUI OUTPUT
def get_output():
    user_input1 = entry1.get()
    float_input1 = np.float64(user_input1)
    user_input2 = entry2.get()
    float_input2 = np.float64(user_input2)
    user_input3 = entry3.get()
    float_input3 = np.float64(user_input3)
    user_input4 = entry4.get()
    float_input4 = np.float64(user_input4)
    user_input_si_al = float_input3/float_input4
    user_input5 = entry5.get()
    float_input5 = np.float64(user_input5)
    user_input_ca_si = float_input5/float_input3
    user_input_ca_al = float_input5/float_input4
    user_input6 = entry6.get()
    float_input6 = np.float64(user_input6)
    user_input_fe_al = float_input6/float_input4
    user_input7 = entry7.get()
    float_input7 = np.float64(user_input7)
    user_input8 = entry8.get()
    float_input8 = np.float64(user_input8)
    user_input_sio3_oh = float_input7/float_input8
    user_input9 = entry9.get()
    float_input9 = np.float64(user_input9)
    user_input10 = entry10.get()
    float_input10 = np.float64(user_input10)
    user_input11 = entry11.get()
    float_input11 = np.float64(user_input11)
    user_input12 = entry12.get()
    float_input12 = np.float64(user_input12)
    user_input13 = entry13.get()
    float_input13 = np.float64(user_input13)
    user_input14 = entry14.get()
    float_input14 = np.float64(user_input14)
    user_input15 = entry15.get()
    float_input15 = np.float64(user_input15)
    user_input16 = entry16.get()
    float_input16 = np.float64(user_input16)
    user_input17 = entry17.get()
    float_input17 = np.float64(user_input17)
    user_input_activator_binder = (float_input7+float_input8)/(float_input1+float_input2)
    user_input_liquid_binder = (float_input7+float_input8+float_input10)/(float_input1+float_input2)
    user_input_fa_ta = float_input12/(float_input12+float_input11)
    user_input_ta_binder = (float_input11+float_input12)/(float_input1+float_input2)
    user_input_df = pd.DataFrame({
        'Fly Ash (kg/m3)': [float_input1],
        'Slag (kg/m3)': [float_input2],
        'SiO2:Al2O3': [user_input_si_al],
        'CaO:SiO2': [user_input_ca_si],
        'CaO:Al2O3': [user_input_ca_al],
        'Fe2O3:Al2O3': [user_input_fe_al],
        'Na2SiO3:NaOH': [user_input_sio3_oh],
        'NaOH (M)': [float_input9],
        'Activator:Binder': [user_input_activator_binder],
        'Extra Water (kg/m3)': [float_input10],
        'Liquid:Binder': [user_input_liquid_binder],
        'Fine Aggregate :Total Aggregate': [user_input_fa_ta],
        'Total Aggregate :Binder': [user_input_ta_binder],
        'Curing Temp (C)': [float_input13],
        'Curing Time (hr)': [float_input14],
        'Age before Exposure (Days)': [float_input15],
        'H2SO4 (M)': [float_input16],
        'Days Submerged': [float_input17],
    })
    print(user_input_df)
    output = best_model.predict(user_input_df)
    print(output)
    result_label.config(text=f"Predicted Compressive Strength: {output}")
    return output

#creating gui
root = tk.Tk() #create root window
root.title("XGBoost") #setting window title

label1 = tk.Label(root, text="Fly Ash (kg/m3):")
entry1 = tk.Entry(root, width=40)

label2 = tk.Label(root, text="Slag (kg/m3):")
entry2 = tk.Entry(root, width=40)

label3 = tk.Label(root, text="SiO2 (weight %):")
entry3 = tk.Entry(root, width=40)

label4 = tk.Label(root, text="Al2O3 (weight %):")
entry4 = tk.Entry(root, width=40)

label5 = tk.Label(root, text="CaO (weight %):")
entry5 = tk.Entry(root, width=40)

label6 = tk.Label(root, text="Fe2O3 (weight %):")
entry6 = tk.Entry(root, width=40)

label7 = tk.Label(root, text="Na2SiO3 (kg/m3):")
entry7 = tk.Entry(root, width=40)

label8 = tk.Label(root, text="NaOH (kg/m3):")
entry8 = tk.Entry(root, width=40)

label9 = tk.Label(root, text="NaOH (M):")
entry9 = tk.Entry(root, width=40)

label10 = tk.Label(root, text="Extra Water (kg/m3):")
entry10 = tk.Entry(root, width=40)

label11 = tk.Label(root, text="Course Aggregate (kg/m3):")
entry11 = tk.Entry(root, width=40)

label12 = tk.Label(root, text="Fine Aggregate (kg/m3):")
entry12 = tk.Entry(root, width=40)

label13 = tk.Label(root, text="Curing Temp (C):")
entry13 = tk.Entry(root, width=40)

label14 = tk.Label(root, text="Curing Time (hr)")
entry14 = tk.Entry(root, width=40)

label15 = tk.Label(root, text="Age before Exposure (Days):")
entry15 = tk.Entry(root, width=40)

label16 = tk.Label(root, text="H2SO4 (M):")
entry16 = tk.Entry(root, width=40)

label17 = tk.Label(root, text="Days Submerged in H2SO4:")
entry17 = tk.Entry(root, width=40)

button = tk.Button(root, text="Get Compressive Strength (MPa)",command=get_output)
result_label = tk.Label(root, text="")

label1.grid(row=0, column=0, sticky='e', padx=5, pady=5)
entry1.grid(row=0, column=1, sticky='w', padx=5, pady=5)

label2.grid(row=1, column=0, sticky='e', padx=5, pady=5)
entry2.grid(row=1, column=1, sticky='w', padx=5, pady=5)

label3.grid(row=2, column=0, sticky='e', padx=5, pady=5)
entry3.grid(row=2, column=1, sticky='w', padx=5, pady=5)

label4.grid(row=3, column=0, sticky='e', padx=5, pady=5)
entry4.grid(row=3, column=1, sticky='w', padx=5, pady=5)

label5.grid(row=4, column=0, sticky='e', padx=5, pady=5)
entry5.grid(row=4, column=1, sticky='w', padx=5, pady=5)

label6.grid(row=5, column=0, sticky='e', padx=5, pady=5)
entry6.grid(row=5, column=1, sticky='w', padx=5, pady=5)

label7.grid(row=6, column=0, sticky='e', padx=5, pady=5)
entry7.grid(row=6, column=1, sticky='w', padx=5, pady=5)

label8.grid(row=7, column=0, sticky='e', padx=5, pady=5)
entry8.grid(row=7, column=1, sticky='w', padx=5, pady=5)

label9.grid(row=8, column=0, sticky='e', padx=5, pady=5)
entry9.grid(row=8, column=1, sticky='w', padx=5, pady=5)

label10.grid(row=9, column=0, sticky='e', padx=5, pady=5)
entry10.grid(row=9, column=1, sticky='w', padx=5, pady=5)

label11.grid(row=10, column=0, sticky='e', padx=5, pady=5)
entry11.grid(row=10, column=1, sticky='w', padx=5, pady=5)

label12.grid(row=11, column=0, sticky='e', padx=5, pady=5)
entry12.grid(row=11, column=1, sticky='w', padx=5, pady=5)

label13.grid(row=12, column=0, sticky='e', padx=5, pady=5)
entry13.grid(row=12, column=1, sticky='w', padx=5, pady=5)

label14.grid(row=13, column=0, sticky='e', padx=5, pady=5)
entry14.grid(row=13, column=1, sticky='w', padx=5, pady=5)

label15.grid(row=14, column=0, sticky='e', padx=5, pady=5)
entry15.grid(row=14, column=1, sticky='w', padx=5, pady=5)

label16.grid(row=15, column=0, sticky='e', padx=5, pady=5)
entry16.grid(row=15, column=1, sticky='w', padx=5, pady=5)

label17.grid(row=16, column=0, sticky='e', padx=5, pady=5)
entry17.grid(row=16, column=1, sticky='w', padx=5, pady=5)

button.grid(row=17, column=0, padx=5, pady=5)
result_label.grid(row=17, column=1, padx=5, pady=5)

root.mainloop() #continuously monitoring for interactions

#POST-MODEL ANALYSIS