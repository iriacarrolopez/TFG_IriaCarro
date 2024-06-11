import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
#-----------------
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# REGRESIÓN LOGÍSTICA
from sklearn.linear_model import LogisticRegression
# REGRESIÓN LINEAL
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# DECISION TREE REGRESSOR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
# RANDOM FOREST REGRESSOR
from sklearn.ensemble import RandomForestRegressor
# GRID SEARCH CV
from sklearn.model_selection import GridSearchCV
#-----------------
csv_file_path = "/source/housing.csv"

df = pd.read_csv(csv_file_path)

def preprocess_data(csv_file_path):
    # Leer el archivo CSV
    df = pd.read_csv(csv_file_path)
    
    # Eliminar filas con valores nulos
    cleaned_df = df.dropna().copy()

    # Eliminar la columna 'ocean_proximity' si existe
    if 'ocean_proximity' in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns='ocean_proximity')

    # Añadir la columna 'income_category'
    cleaned_df['income_category'] = pd.cut(cleaned_df['median_income'], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])

    # Dividir los datos en conjuntos de entrenamiento y prueba
    train, test = train_test_split(cleaned_df, test_size=0.2, stratify=cleaned_df['income_category'], random_state=42)

    # Crear copia de los datos de entrenamiento
    copy_train = train.copy()

    # Separar características y etiquetas de entrenamiento
    X_train = copy_train.drop(columns=['median_house_value', 'income_category'])
    y_train = copy_train['median_house_value'].copy()

    return X_train, y_train, test

# Pipeline de preprocesamiento
def preprocess_pipeline(X_train):
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
        def __init__(self, add_bedrooms_per_room=True, add_rooms_per_household=True, add_population_per_household=True):
            self.add_bedrooms_per_room = add_bedrooms_per_room
            self.add_rooms_per_household = add_rooms_per_household
            self.add_population_per_household = add_population_per_household
        
        def fit(self, X, y=None):
            return self
        
        def transform(self, X, y=None):
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                bedrooms_per_room = np.log(bedrooms_per_room + 1)
                X = np.c_[X, bedrooms_per_room]
                
            if self.add_rooms_per_household:
                rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
                rooms_per_household = np.log(rooms_per_household + 1)
                X = np.c_[X, rooms_per_household]
                
            if self.add_population_per_household:
                population_per_household = X[:, population_ix] / X[:, households_ix]
                population_per_household = np.log(population_per_household + 1)
                X = np.c_[X, population_per_household]
            
            # Log transform selected features
            X[:, rooms_ix] = np.log(X[:, rooms_ix] + 1)
            X[:, bedrooms_ix] = np.log(X[:, bedrooms_ix] + 1)
            X[:, population_ix] = np.log(X[:, population_ix] + 1)
            X[:, households_ix] = np.log(X[:, households_ix] + 1)
                
            return X

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attrib_adder', CombinedAttributeAdder()),
        ('std_scaler', StandardScaler()),
    ])

    num_attribs = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs)
    ])

    return full_pipeline.fit_transform(X_train)

# REGRESIÓN LINEAL
def linear_regression(fit_intercept, copy_X, n_jobs, positive):
    reg = LinearRegression(fit_intercept, copy_X, n_jobs, positive)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_train)
    error = np.mean((y_pred - y_train) ** 2)
    return reg, error

X_train, y_train, test = preprocess_data(csv_file_path)
processed_X_train = preprocess_pipeline(X_train)

# REGRESIÓN LOGISTICA
def logistic_regression(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_inter, multi_class, verbose, warm_start, l1_ratio):
    # Entrenamiento del modelo de regresión logística
    log_reg = LogisticRegression(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_inter, multi_class, verbose, warm_start, l1_ratio)
    log_reg.fit(X_train, y_train)
    
    return log_reg

# DECISION TREE REGRESSOR
def decission_tree_regressor(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease, ccp_alpha, monotonic_cst):
    # Entrenamiento del modelo DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease, ccp_alpha, monotonic_cst)
    tree_reg.fit(X_train, y_train)
    
    # Predicciones del árbol de decisiones
    tree_predictions = tree_reg.predict(X_train)
    tree_error = mean_squared_error(y_train, tree_predictions)
    
    # Validación cruzada para DecisionTreeRegressor
    tree_scores = cross_val_score(tree_reg, X_train, y_train, scoring='neg_mean_squared_error')
    tree_rmse_scores = np.sqrt(-tree_scores)
    tree_rmse = tree_rmse_scores.mean()

    return tree_error, tree_rmse
    

# RANDOM FOREST REGRESSOR
def random_forest_regressor(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, ccp_alpha, max_samples, monotonic_cst):
    # Entrenamiento del modelo de Random Forest Regressor
    forest_reg = RandomForestRegressor(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, ccp_alpha, max_samples, monotonic_cst)
    forest_reg.fit(X_train, y_train)
    
    # Validación cruzada para calcular los puntajes de error
    forest_reg_scores = cross_val_score(forest_reg, X_train, y_train, scoring='neg_mean_squared_error')
    
    # Cálculo del error RMSE promedio
    forest_rmse = np.sqrt(-forest_reg_scores.mean())
    
    return forest_rmse


# GRID SEARCH CV
def train_model_with_grid_search(X_train, y_train, test_size=0.2):
    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
    
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features':[2, 4, 6, 8]},
        {'bootstrap':[False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]}
    ]
    
    grid_search = GridSearchCV(
        RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    print("Mejores parámetros encontrados:", grid_search.best_params_)
    print("Mejor estimador encontrado:", grid_search.best_estimator_)

    results = grid_search.cv_results_
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print("RMSE:", np.sqrt(-mean_score), "Parámetros:", params)

    final_model = grid_search.best_estimator_
    final_predictions = final_model.predict(X_test)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    
    print("RMSE final en el conjunto de prueba:", final_rmse)
    
    return final_model, final_rmse

