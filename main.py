# IMPORTS
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, BayesianRidge, PoissonRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance

# Function for positioning home image
def set_image_position():
    st.write("""
    <style>
        .stApp {
            padding-top: 4rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Data loading
df_og = pd.read_csv('covid_dataset.csv')
df_processed = pd.read_csv('covid_dataset_processed.csv')
# Translation
df_og.rename(columns={
    'data_rap_zakazenia': 'report_date',
    'wiek': 'age',
    'producent': 'manufacturer',
    'teryt_woj': 'voivodeship',
    'teryt_pow': 'county',
    'kat_wiek': 'age_group',
    'plec': 'gender',
    'dawka_ost': 'dose',
    'numer_zarazenia': 'infection_number',
    'liczba_zaraportowanych_zakazonych': 'number_of_reported_cases'
}, inplace=True)

dose_mapping = {"pelna_dawka": "full", "przypominajaca": "booster", "jedna_dawka": "single",
                "uzupe³niaj¹ca": "additional"}
gender_mapping = {"K": "female", "M": "male"}

df_og['gender'] = df_og['gender'].replace(gender_mapping)
df_og['dose'] = df_og['dose'].replace(dose_mapping)

selected = None

# Sidebar config
with st.sidebar:
    selected = option_menu(
        menu_title='Navigation',
        options=['Home', 'Data Scraping', 'Original Dataset', 'Processed Dataset', 'Modeling'],
        icons=['house', 'cloud-download', 'database', 'database-check', 'diagram-3'],
        menu_icon='map',
        default_index=0
    )

# Content on the pages
if selected == 'Home':
    st.title('COVID-19: Prediction of reported cases based on vaccination and demographic details')
    set_image_position()
    with st.columns(3)[1]:
        st.image('Covid19.png', width=240)
    st.header('Summary:')
    goal_text = '''
    The goal of this project was to predict the number of reported COVID-19 cases based on vaccination details and demographic information (e.g. age group) with the use of machine learning techniques. 
   
    The exploratory analysis and visualization of data allowed for a more in-depth understanding of the topic, to which the original data was processed accordingly.
    
    A comparative analysis of multiple regression models revealed the strengths and weaknesses of the chosen approach, with implications for further development that could ultimately benefit research in the medical field.
    '''
    st.write(goal_text)
    st.header('Authors:')
    authors_list_md = '''
    * Hien Anh Nguyen, s22192  
    * Filip Schulz, s22455
    '''
    st.markdown(authors_list_md)
    st.header('Libraries Used:')
    libraries_list_md = '''
    * Pandas
    * GeoPandas
    * MatPlotLib
    * Seaborn
    * Streamlit + streamlit_option_menu
    * Selenium
    * Statsmodels
    * Numpy
    * Scikit-learn
    * Pickle
    '''
    st.markdown(libraries_list_md)

if selected == 'Original Dataset':
    st.title('Original Dataset')
    st.write("This page shows the initial dataset including exploratory data analysis.")
    st.markdown('__Source:__ https://tinyurl.com/pad-covid-dataset')
    st.dataframe(df_og)
    st.header('Translation:')
    st.write('Headers and values were translated from Polish to English.')
    translation_code = '''
    df.rename(columns={
        'data_rap_zakazenia': 'report_date',
        'wiek': 'age',
        'producent': 'manufacturer',
        'teryt_woj': 'voivodeship',
        'teryt_pow': 'county',
        'kat_wiek': 'age_group',
        'plec': 'gender',
        'dawka_ost': 'dose',
        'numer_zarazenia': 'infection_number',
        'liczba_zaraportowanych_zakazonych': 'number_of_reported_cases'
    }, inplace=True)

    dose_mapping = {"pelna_dawka": "full", "przypominajaca": "booster", "jedna_dawka": "single", "uzupe³niaj¹ca": "additional"}
    gender_mapping = {"K": "female", "M": "male"}

    df['gender'] = df['gender'].replace(gender_mapping)
    df['dose'] = df['dose'].replace(dose_mapping)
    '''
    st.code(translation_code, language='python')
    st.header('Exploratory data analysis:')
    st.markdown('### Basic data distribution:')

    # Age group distribution plot
    age_group_cases = df_og.groupby('age_group')['number_of_reported_cases'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=age_group_cases, x='age_group', y='number_of_reported_cases', hue='age_group', ax=ax)
    ax.set_title('Distribution of Reported Cases by Age Group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Number of Reported Cases')
    st.pyplot(fig)

    # Gender Distribution
    gender_cases = df_og.groupby('gender')['number_of_reported_cases'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=gender_cases, x='gender', y='number_of_reported_cases', hue='gender', ax=ax)
    ax.set_title('Distribution of Reported Cases by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Number of Reported Cases')
    st.pyplot(fig)

    # Infection number
    df_aggregated = df_og.groupby('infection_number')['number_of_reported_cases'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df_aggregated, x='infection_number', y='number_of_reported_cases', hue='infection_number', ax=ax)
    ax.set_title('Total Number of Reported Cases by Infection Number')
    ax.set_xlabel('Infection number')
    ax.set_ylabel('Total Number of Reported Cases')
    ax.legend([], frameon=False)
    st.pyplot(fig)

    st.markdown('### Geographical distribution:')
    # Map of total number of cases by voivodeship
    gdf = gpd.read_file("regions_medium.geojson")
    df_aggregated = df_og.groupby('voivodeship')['number_of_reported_cases'].sum().reset_index()
    index_column_name = 'id'
    merged = gdf.set_index(index_column_name).join(df_aggregated.set_index('voivodeship'))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot the outlines of all regions
    gdf.plot(ax=ax, edgecolor='lightgrey', facecolor='none')

    # Plot the filled polygons with color representing the number of reported cases
    merged.plot(column='number_of_reported_cases', ax=ax, legend=True, cmap='OrRd',
                legend_kwds={'label': "Number of Reported Cases",
                             'orientation': "horizontal"})

    # Annotate each region with its ID
    for idx, row in merged.iterrows():
        ax.annotate(text=str(idx), xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                    color='black', fontsize=8, ha='center')

    plt.title('Total Number of Reported Cases by Voivodeship')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    st.pyplot(fig)

    st.markdown('### Vaccine manufacturers distribution:')
    df_aggregated = df_og.groupby('manufacturer')['number_of_reported_cases'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_aggregated, x='manufacturer', y='number_of_reported_cases', hue='manufacturer', ax=ax)
    ax.set_title('Total Number of Reported Cases by Vaccine Manufacturer')
    ax.set_xlabel('Vaccine Manufacturer')
    ax.set_ylabel('Total Number of Reported Cases')
    st.pyplot(fig)

if selected == 'Processed Dataset':
    st.title('Processed Dataset')
    st.write('This page shows the step-by-step process of data preparation. It includes the visualization of the outcome.')
    st.dataframe(df_processed)
    processed_data_page_option = st.radio('', ('Data exploration + processing', 'Visualization'), horizontal=True)
    if processed_data_page_option == 'Data exploration + processing':
        st.header('Basic Data Exploration + Processing')
        st.markdown('### Sum of missing values for each column:')
        #st.write(df_og.isnull().sum(), width=100)
        missing_values_df = df_og.isna().sum().rename_axis('Column').reset_index()
        st.table(missing_values_df)
        st.header('Replacing/dropping rows with unknown values and removal of unnecessary columns')
        st.markdown('__Unique values of age_group:__ [0-18, 25-34, 35-44, 45-54, 55-64, 65-74, 19-24, 75-84, 85-94, BD, 95+]')
        st.markdown('__Unique values of gender:__  [female, male, nieznana]')
        st.markdown('__Unique values of manufacturer:__ [nan, Pfizer, Johnson&Johnson, Astra Zeneca, Moderna, brak danych]')
        replacing_code = '''
        df = df[df['county'].notna()]
        df['county'] = df['county'].astype(int)
        df = df.drop(columns=['Unnamed: 0', 'report_date', 'age'])
        df = df[df.age_group != 'BD']
        df = df[df.gender != 'nieznana']
        df = df[df.manufacturer != 'brak danych']'''
        st.code(replacing_code, language='python')
        st.header('Changing numerical value of number of infections into a boolean variable indicating whether it is a first infection or not:')
        infections_code = '''
        df['multiple_infections'] = df['infection_number'] > 1
        df['multiple_infections'] = df['multiple_infections'].astype(bool)
        df = df.drop(columns=['infection_number'])'''
        st.code(infections_code, language='python')
        st.header('Encoding and missing data handling')
        encoding_code = '''
        df_encoded = pd.get_dummies(df, columns=['gender'], drop_first=True)
        df_encoded = pd.get_dummies(df_encoded, columns=['age_group'], drop_first=False)
        df_encoded = pd.get_dummies(df, columns=['manufacturer'], drop_first=False)
        df_encoded['dose'].fillna('unvaccinated', inplace=True)
        df_encoded = pd.get_dummies(df_encoded, columns=['dose'], drop_first=True)'''
        st.code(encoding_code, language='python')
        st.write('Rows without value of dawka_ost also do not have any value for vaccine manufacturer meaning that '
                'the reported group was not vaccinated, thus the missing values were filled with appropriate label.')
        st.header('Grouping rows with same values into a single row with number_of_reported_cases equal to the sum of individual cases in the repeating rows')
        grouping_code = '''
        df_encoded_grouped = df_encoded.groupby([
            'voivodeship', 'county', 'multiple_infections', 'gender_male',
            'age_group_0-18', 'age_group_19-24', 'age_group_25-34', 'age_group_35-44', 
            'age_group_45-54', 'age_group_55-64', 'age_group_65-74', 'age_group_75-84', 
            'age_group_85-94', 'age_group_95+', 'manufacturer_Astra Zeneca', 
            'manufacturer_Johnson&Johnson', 'manufacturer_Moderna', 'manufacturer_Pfizer',
            'dose_booster', 'dose_full', 'dose_single', 'dose_unvaccinated'],
        as_index=False)['number_of_reported_cases'].sum()'''
        st.code(grouping_code, language='python')
        st.header('Processing summary: ')
        grouping_summary_md = '''
        __Max value of number_of_reported_cases before grouping:__ 12\n
        __Max value of number_of_reported_cases after grouping:__ 57
        * The number of rows went down from 9764 to 6504.
        * At the same time the max value of number_of_reported_cases in a single report went up from 12 to 57 indicating a
        large number of cases that shared the same attributes but were reported individually.
        * Despite removing some of the columns, the number of them increased due to additional columns added with encoding.'''
        st.markdown(grouping_summary_md)
    else:
        st.header('Processed data visualization')
        # Age distribution
        age_group_map = {
            (True, False, False, False, False, False, False, False, False, False): '0-18',
            (False, True, False, False, False, False, False, False, False, False): '19-24',
            (False, False, True, False, False, False, False, False, False, False): '25-34',
            (False, False, False, True, False, False, False, False, False, False): '35-44',
            (False, False, False, False, True, False, False, False, False, False): '45-54',
            (False, False, False, False, False, True, False, False, False, False): '55-64',
            (False, False, False, False, False, False, True, False, False, False): '65-74',
            (False, False, False, False, False, False, False, True, False, False): '75-84',
            (False, False, False, False, False, False, False, False, True, False): '85-94',
            (False, False, False, False, False, False, False, False, False, True): '95+'
        }
        df_aggregated = df_processed.groupby([
            'age_group_0-18', 'age_group_19-24', 'age_group_25-34', 'age_group_35-44',
            'age_group_45-54', 'age_group_55-64', 'age_group_65-74', 'age_group_75-84',
            'age_group_85-94', 'age_group_95+'])['number_of_reported_cases'].sum()
        df_aggregated.index = df_aggregated.index.map(age_group_map)
        fig, ax = plt.subplots(figsize=(10, 6))
        df_aggregated.plot(kind='bar', ax=ax)
        ax.set_title('Distribution of Reported Cases by Age Group')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Number of Reported Cases')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)
        # Gender distribution
        gender_map = {
            False: 'Female',
            True: 'Male',
        }
        df_aggregated = df_processed.groupby(['gender_male'])['number_of_reported_cases'].sum()
        df_aggregated.index = df_aggregated.index.map(gender_map)
        fig, ax = plt.subplots(figsize=(10, 6))
        df_aggregated.plot(kind='bar', ax=ax)
        ax.set_title('Distribution of Reported Cases by Gender')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Number of Reported Cases')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)
        # Multiple infections plot
        df_aggregated = df_processed.groupby('multiple_infections')['number_of_reported_cases'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df_aggregated, x='multiple_infections', y='number_of_reported_cases',
                    hue='multiple_infections', ax=ax)
        ax.set_title('Total Number of Reported Cases by Multiple Infections')
        ax.set_xlabel('Had COVID Before?')
        ax.set_ylabel('Total Number of Reported Cases')
        ax.legend([], frameon=False)
        st.pyplot(fig)
        # Manufacturers plot
        # Mapping boolean values to vaccine manufacturer names
        manufacturer_map = {
            (False, False, False, False): 'None',
            (True, False, False, False): 'AstraZeneca',
            (False, True, False, False): 'Johnson&Johnson',
            (False, False, True, False): 'Moderna',
            (False, False, False, True): 'Pfizer'
        }
        df_aggregated = df_processed.groupby(['AstraZeneca', 'Johnson&Johnson', 'Moderna', 'Pfizer'])[
            'number_of_reported_cases'].sum()
        df_aggregated.index = df_aggregated.index.map(manufacturer_map)
        fig, ax = plt.subplots(figsize=(8, 4))
        df_aggregated.plot(kind='bar', ax=ax)
        ax.set_title('Sum of Reported Cases by Vaccine Manufacturer')
        ax.set_xlabel('Vaccine Manufacturer')
        ax.set_ylabel('Number of Reported Cases')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right')
        st.pyplot(fig)
        
if selected == 'Data Scraping':
    st.title('Data Scraping')
    scraping_intro_md = '''
    This page shows the source code of the webscraper utilized to obtain the data.
    * __Source:__ https://tinyurl.com/pad-covid-dataset
    * __Library Used:__ Selenium'''
    st.markdown(scraping_intro_md)
    scraping_sourcecode = '''
    from selenium import webdriver
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.common.by import By
    from selenium.webdriver import ActionChains
    from selenium.common.exceptions import StaleElementReferenceException
    from selenium.webdriver.support import expected_conditions as EC
    import pandas as pd
    import time

    driver = webdriver.Chrome()
    driver.get("https://dane.gov.pl/pl/dataset/2582,statystyki-zakazen-i-zgonow-z-powodu-covid-19-z-uwzglednieniem-zaszczepienia-przeciw-covid-19/resource/36897/table?page=1&per_page=20&q=&sort=")

    # Define a function to get all elements on the page
    def get_all_elements():
        return driver.find_elements(By.XPATH, "//*")

    # Display information about each element found
    def display_element_info(element):
        print("Tag Name:", element.tag_name)
        print("Text:", element.text)
        print("Attribute 'id':", element.get_attribute("id"))
        print("Attribute 'class':", element.get_attribute("class"))
        print("------------")

    # Find all elements on the page and store them in a variable
    all_elements = get_all_elements()

    # Display information about each element found
    for element in all_elements:
        try:
            display_element_info(element)
        except StaleElementReferenceException:
            # If the element is stale, refind it and display its information
            all_elements = get_all_elements()
            for refreshed_element in all_elements:
                if refreshed_element == element:
                    display_element_info(refreshed_element)

    time.sleep(5)

    try:
        cookie_popup = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "modal-content"))
        )
        # Once the cookie consent popup is found, click the accept button
        close_button = cookie_popup.find_element(By.ID, "footer-close")
        close_button.click()

    except:
        # If the cookie popup doesn't appear, continue without accepting cookies
        print("No cookie consent popup found or it took too long to appear.")

    time.sleep(5)

    data = []  # Scraped data goes here
    wait = WebDriverWait(driver, 10)
    table_page_number = 7  # Init index of the "next page button". It increases up to 11

    for page_number in range(1, 498):  # Feel free to lower the upper bound for testing
        table = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "datagrid")))  # Wait for table to appear on page

        if page_number == 1:  # Get columns names
            header_row = table.find_element(By.XPATH,
                        "/html/body/app-root/app-main-layout/main/app-dataset-parent/div/app-dataset-resource/section/div[4]/div[2]/app-resource-table-no-filters/div/div[3]/div/table/thead/tr")
            columns_names = [th.text for th in header_row.find_elements(By.CLASS_NAME, "datagrid__heading")]
        rows = table.find_elements(By.XPATH, ".//tbody/tr")
        for row in rows:
            cells = row.find_elements(By.XPATH, "./td")
            row_data = [cell.text for cell in cells]  # Get text from cell
            data.append(row_data)

        # Wait until the "next button" (it's an arrow tho) is present on the page
        next_button = wait.until(EC.presence_of_element_located((By.XPATH,
                    f"/html/body/app-root/app-main-layout/main/app-dataset-parent/div/app-dataset-resource/section/div[4]/div[2]/app-resource-table-no-filters/div/div[4]/div/app-pagination/nav/div/ul/li[{table_page_number}]/a")))
        if table_page_number < 11:
            table_page_number += 1

        actions = ActionChains(
            driver)  # ActionChains instead of regular CLICK because for some reason CLICK turned out to be a bit more buggy
        actions.move_to_element(next_button).click().perform()
        # Let the page load after clicking
        time.sleep(5)

    driver.quit()

    df.to_csv('covid_dataset.csv')'''
    st.code(scraping_sourcecode, language='python')

if selected == 'Modeling':
    st.title('Modeling')
    st.write('This page shows the machine learning modeling and evaluation process.')
    overview_text = '''
    This analysis compares the performance of various regression models for predicting the number of reported Covid cases based on vaccination and demographic details in the Polish e-Health Centre (Centrum e-Zdrowia) data.
    '''
    data_prep_text = '''
    **Defining feature set and target variable**
    ```python
    X = df_processed.drop(columns=['number_of_reported_cases'])
    y = df_processed.number_of_reported_cases
    ```
    **Train and test set data splitting**
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
    '''
    metrics_text = '''
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Cross-Validation Mean Squared Error (CV-MSE)
    - Deviance (for Poisson Regression)
    '''
    consid_text = '''
    **Balance between Accuracy and Generalizability**: Random Forest and Decision Tree with Bootstrap Aggregation appear to be good initial choices due to their balance between MAE and CV-MSE.
    
    **Prioritize Generalizability**: If generalizability to unseen data is crucial, Gradient Boosting might be a good option despite its slightly higher MAE.
    
    **Overfitting Concerns**: Ridge Regressors can be considered if overfitting is a major concern, but be aware of the potential sacrifice in capturing the true relationships.
    
    **Count Data Specific Model**: Poisson Regression offers a statistically sound approach for count data and provides interpretable results, making it a valuable model to consider.
    '''
    further_dev_text = '''
    Explore feature engineering techniques like combining similar categories to potentially improve model performance for all models.
    
    Consider hyperparameter tuning for promising models (Random Forest, Gradient Boosting) to see if accuracy can be further improved.
    
    Validate the chosen model(s) on a separate hold-out dataset to ensure their generalizability to unseen data.
    
    For Poisson Regression, explore the estimated coefficients to understand how vaccination details influence the predicted number of cases.
    '''
    with st.expander("Overview"):
        st.write(overview_text)
    with st.expander("Data Preparation"):
        st.write(data_prep_text)
    with st.expander("Evaluation Metrics"):
        st.write(metrics_text)
    with st.expander("Considerations"):
        st.write(consid_text)
    with st.expander("Further Development"):
        st.write(further_dev_text)
    st.markdown("""<hr style="height:4px;border:none;color:#7C7C7C;background-color:#7C7C7C;" /> """, unsafe_allow_html=True)
    
    modeling_page_option = st.radio('', ('Decision Tree-Based Models', 'Gradient Boosting Regression', 'Ridge Regression', 'Poisson Regression'), horizontal=True)
    if modeling_page_option == 'Decision Tree-Based Models':
        st.header('Decision Tree-Based Models')
        summary_text = '''
        These models (Base Decision Tree, Decision Tree with Bootstrap Aggregation, Decision Tree with Pruning, Random Forest) are effective for handling categorical data like vaccine manufacturer and dose.
        Their performance (MAE around 0.75) suggests they capture the general trend of cases within age groups. However, some of them can be prone to overfitting.
        '''
        st.write(summary_text)
        # Base DT
        st.markdown('### Base Decision Tree Regressor')
        st.markdown('#### Modeling')
        base_regr_code = '''
        # Initialize the Decision Tree Regressor
        base_regr = DecisionTreeRegressor(random_state=42)

        # Train the model
        base_regr.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred_base = base_regr.predict(X_test)
        '''
        st.code(base_regr_code, language='python')

        st.markdown('#### Evaluation')
        # Model
        X = df_processed.drop(columns=['number_of_reported_cases'])
        y = df_processed.number_of_reported_cases
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        base_regr = DecisionTreeRegressor(random_state=42)
        base_regr.fit(X_train, y_train)
        y_pred_base = base_regr.predict(X_test)
        # Evaluation
        mae_base = mean_absolute_error(y_test, y_pred_base)
        mse_base = mean_squared_error(y_test, y_pred_base)
        cv_scores = cross_val_score(base_regr, X, y, cv=5, scoring='neg_mean_squared_error')
        mean_cv_score = -np.mean(cv_scores)
        base_regr_eval = pd.DataFrame(
            [
                {"Metric": "MAE", "Value": mae_base},
                {"Metric": "MSE", "Value": mse_base},
                {"Metric": "CV-MSE", "Value": mean_cv_score},
            ]
        )
        st.dataframe(base_regr_eval, use_container_width=True)

        # Bagging DT
        st.markdown('### Decision Tree Regressor with Bootstrap Aggregation (Bagging)')
        st.markdown('#### Modeling')
        bagging_regr_code = '''
        # Initialize the Bagging Regressor with the base Decision Tree Regressor
        bagging_regr = BaggingRegressor(estimator=base_regr, n_estimators=100, random_state=42)

        # Train the bagging model on the training data
        bagging_regr.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred_bagging = bagging_regr.predict(X_test)
        '''
        st.code(bagging_regr_code, language='python')

        st.markdown('#### Evaluation')
        # Model
        bagging_regr = BaggingRegressor(estimator=base_regr, n_estimators=100, random_state=42)
        bagging_regr.fit(X_train, y_train)
        y_pred_bagging = bagging_regr.predict(X_test)
        # Evaluation
        mae_bagging = mean_absolute_error(y_test, y_pred_bagging)
        mse_bagging = mean_squared_error(y_test, y_pred_bagging)
        cv_scores = cross_val_score(bagging_regr, X, y, cv=5, scoring='neg_mean_squared_error')
        mean_cv_score = -np.mean(cv_scores)
        bagging_regr_eval = pd.DataFrame(
            [
                {"Metric": "MAE", "Value": mae_bagging},
                {"Metric": "MSE", "Value": mse_bagging},
                {"Metric": "CV-MSE", "Value": mean_cv_score},
            ]
        )
        st.dataframe(bagging_regr_eval, use_container_width=True)

        # Pruned DT
        st.markdown('### Decision Tree Regressor with Pruning')
        st.markdown('#### Modeling')
        pruned_regr_code = '''
        # Initialize the Decision Tree Regressor with pruning parameters
        pruned_regr = DecisionTreeRegressor(
            max_depth=5,            # Maximum depth of the tree
            min_samples_split=10,   # Minimum number of samples required to split an internal node
            min_samples_leaf=5,     # Minimum number of samples required to be at a leaf node
            max_leaf_nodes=20,      # Maximum number of leaf nodes
            random_state=42
        )

        # Train the model on the training data
        pruned_regr.fit(X_train, y_train)

        # Make predictions on the testing data
        y_pred_pruned = pruned_regr.predict(X_test)
        '''
        st.code(pruned_regr_code, language='python')

        st.markdown('#### Evaluation')
        # Model
        pruned_regr = DecisionTreeRegressor(
            max_depth=5,               # Maximum depth of the tree
            min_samples_split=10,      # Minimum number of samples required to split an internal node
            min_samples_leaf=5,        # Minimum number of samples required to be at a leaf node
            max_leaf_nodes=20,         # Maximum number of leaf nodes
            random_state=42
        )
        pruned_regr.fit(X_train, y_train)
        y_pred_pruned = pruned_regr.predict(X_test)
        # Evaluation
        mae_pruned = mean_absolute_error(y_test, y_pred_pruned)
        mse_pruned = mean_squared_error(y_test, y_pred_pruned)
        cv_scores = cross_val_score(pruned_regr, X, y, cv=5, scoring='neg_mean_squared_error')
        mean_cv_score = -np.mean(cv_scores)
        pruned_regr_eval = pd.DataFrame(
            [
                {"Metric": "MAE", "Value": mae_pruned},
                {"Metric": "MSE", "Value": mse_pruned},
                {"Metric": "CV-MSE", "Value": mean_cv_score},
            ]
        )
        st.dataframe(pruned_regr_eval, use_container_width=True)

        st.markdown('#### Visualization')
        fig, ax = plt.subplots(figsize=(12, 6))  # Adjust width and height as needed
        plot_tree(pruned_regr, ax=ax, filled=True, feature_names=X.columns, rounded=True)
        st.pyplot(fig)

        # Random Forest
        st.markdown('### Random Forest Regressor')
        st.markdown('#### Modeling')
        rf_code = '''
        # Fit Random Forest regression model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Predict on the testing set
        y_pred_rf = rf.predict(X_test)
        '''
        st.code(rf_code, language='python')
        
        st.markdown('#### Evaluation')
        # Model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        # Evaluation
        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        cv_mse = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=5)
        mean_cv_score = -np.mean(cv_scores)
        rf_eval = pd.DataFrame(
            [
                {"Metric": "MAE", "Value": mae_rf},
                {"Metric": "MSE", "Value": mse_rf},
                {"Metric": "CV-MSE", "Value": mean_cv_score},
            ]
        )
        st.dataframe(rf_eval, use_container_width=True)

    if modeling_page_option == 'Gradient Boosting Regression':
        st.header('Gradient Boosting Regression')
        # Model
        X = df_processed.drop(columns=['number_of_reported_cases'])
        y = df_processed.number_of_reported_cases
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gbr.fit(X_train, y_train)
        y_pred_gbr = gbr.predict(X_test)
        # Evaluation
        mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
        mse_gbr = mean_squared_error(y_test, y_pred_gbr)
        cv_scores = cross_val_score(gbr, X, y, cv=5, scoring='neg_mean_squared_error')
        mean_cv_score = -np.mean(cv_scores)
        summary_text = f'''
        This model can achieve good performance on complex relationships like those between vaccination details and cases.
        
        Its MAE (around {mae_gbr:.2f}) is slightly higher than some decision tree models, but its lower CV-MSE suggests it might generalize better to unseen data.
        '''
        st.write(summary_text)
        st.markdown('### Gradient Boosting Regressor')
        st.markdown('#### Modeling')
        gbr_code = '''
        # Fit Gradient Boosting regression model
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gbr.fit(X_train, y_train)

        # Predict on the testing set
        y_pred_gbr = gbr.predict(X_test)
        '''
        st.code(gbr_code, language='python')
        st.markdown('#### Evaluation')       
        gbr_eval = pd.DataFrame(
            [
                {"Metric": "MAE", "Value": mae_gbr},
                {"Metric": "MSE", "Value": mse_gbr},
                {"Metric": "CV-MSE", "Value": mean_cv_score},
            ]
        )
        st.dataframe(gbr_eval, use_container_width=True)

    if modeling_page_option == 'Ridge Regression':
        st.header('Ridge Regression')
        summary_text = '''
        These models focus on reducing model complexity and overfitting.

        While they have lower CV-MSE (potentially less overfitting), their higher MAE suggests they might under-capture the intricacies of the data.
        '''
        st.write(summary_text)
        # Ridge
        st.markdown('### Ridge Regressor')
        st.markdown('#### Modeling')
        ridge_code = '''
        # Fit Ridge regression model
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)

        # Predict on the testing set
        y_pred_ridge = ridge.predict(X_test)
        '''
        st.code(ridge_code, language='python')
        st.markdown('#### Evaluation')
        # Model
        X = df_processed.drop(columns=['number_of_reported_cases'])
        y = df_processed.number_of_reported_cases
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        # Evaluation
        mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        cv_scores = cross_val_score(ridge, X, y, cv=5, scoring='neg_mean_squared_error')
        mean_cv_score = -np.mean(cv_scores)
        ridge_eval = pd.DataFrame(
            [
                {"Metric": "MAE", "Value": mae_ridge},
                {"Metric": "MSE", "Value": mse_ridge},
                {"Metric": "CV-MSE", "Value": mean_cv_score},
            ]
        )
        st.dataframe(ridge_eval, use_container_width=True)

        # Bayesian Ridge
        st.markdown('### Bayesian Ridge Regressor')
        st.markdown('#### Modeling')
        bayesian_ridge_code = '''
        # Fit Bayesian Ridge regression model
        bayesian_ridge = BayesianRidge()
        bayesian_ridge.fit(X_train, y_train)

        # Predict on the testing set
        y_pred_bayesian_ridge = bayesian_ridge.predict(X_test)
        '''
        st.code(bayesian_ridge_code, language='python')
        st.markdown('#### Evaluation')
        # Model
        bayesian_ridge = BayesianRidge()
        bayesian_ridge.fit(X_train, y_train)
        y_pred_bayesian_ridge = bayesian_ridge.predict(X_test)
        # Evaluation
        mae_bayesian_ridge = mean_absolute_error(y_test, y_pred_bayesian_ridge)
        mse_bayesian_ridge = mean_squared_error(y_test, y_pred_bayesian_ridge)
        cv_scores = cross_val_score(bayesian_ridge, X, y, cv=5, scoring='neg_mean_squared_error')
        mean_cv_score = -np.mean(cv_scores)
        bayesian_ridge_eval = pd.DataFrame(
            [
                {"Metric": "MAE", "Value": mae_bayesian_ridge},
                {"Metric": "MSE", "Value": mse_bayesian_ridge},
                {"Metric": "CV-MSE", "Value": mean_cv_score},
            ]
        )
        st.dataframe(bayesian_ridge_eval, use_container_width=True)

    if modeling_page_option == 'Poisson Regression':
        st.header('Poisson Regression')
        # Model
        X = df_processed.drop(columns=['number_of_reported_cases'])
        y = df_processed.number_of_reported_cases
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        poisson = PoissonRegressor(max_iter=500)
        poisson.fit(X_train, y_train)
        y_pred_poisson = poisson.predict(X_test)
        # Evaluation
        mae_poisson = mean_absolute_error(y_test, y_pred_poisson)
        mse_poisson = mean_squared_error(y_test, y_pred_poisson)
        dev_poisson = mean_poisson_deviance(y_test, y_pred_poisson)
        summary_text = f'''
        This is a statistical model specifically designed for count data like the number of Covid cases.
        
        Its MAE ({mae_poisson:.2f}) is higher than some tree-based models, but its Deviance ({dev_poisson:.2f}) is a relevant metric for evaluating goodness-of-fit in Poisson Regression. It directly models the probability distribution of case occurrences.
        '''
        st.write(summary_text)
        st.markdown('### Poisson Regressor')
        st.markdown('#### Modeling')
        poisson_code = '''
        # Fit Poisson Regression Model
        poisson = PoissonRegressor(max_iter=500)
        poisson.fit(X_train, y_train)

        # Predict on the testing set
        y_pred_poisson = poisson.predict(X_test)
        '''
        st.code(poisson_code, language='python')
        st.markdown('#### Evaluation')

        poisson_eval = pd.DataFrame(
            [
                {"Metric": "MAE", "Value": mae_poisson},
                {"Metric": "MSE", "Value": mse_poisson},
                {"Metric": "Deviance", "Value": dev_poisson},
            ]
        )
        st.dataframe(poisson_eval, use_container_width=True)
