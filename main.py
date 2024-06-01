# IMPORTS
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

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
        options=['Home', 'Original Dataset', 'Processed Dataset', 'Data Scrapping', 'Models'],
        icons=['house', 'database', 'database-check', 'cloud-download', 'diagram-3'],
        menu_icon='map',
        default_index=0
    )

# Content on the pages
if selected == 'Home':
    st.title('TUTAJ NAZWA PROJEKTU')
    with st.columns(3)[1]:
        st.image('Covid19.png', width=240)
    st.header('Goal of the Project:')
    st.write('Olu proszę napisz tutaj ładny goal projektu bo jesteś lepsza w pisaniu.')
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
    ax.set_xlabel('Had COVID Before?')
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
    st.write('This page show the step-by-step process of data preparation. It includes the visualization of the outcome.')
    st.dataframe(df_processed)
    processed_data_page_option = st.radio('', ('Data exploration + processing', 'Visualization'), horizontal=True)
    if processed_data_page_option == 'Data exploration + processing':
        st.header('Basic Data Exploration + Processing')
        st.markdown('### Sum of missing values for each column:')
        st.write(df_og.isnull().sum())
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
        df_encoded = pd.get_dummies(df, columns=['manufacturer'], drop_first=False)
        df_encoded['dose'].fillna('unvaccinated', inplace=True)'''
        st.code(encoding_code, language='python')
        st.write('Rows without value of dawka_ost also do not have any value for vaccine manufacturer meaning that '
                'the reported group was not vaccinated, thus the missing values were filled with appropriate label.')
        st.header('Grouping rows with same values into a single row with number_of_reported_cases equal to the sum of individual cases in the repeating rows')
        grouping_code = '''df_encoded_grouped = df_encoded.groupby([
        'voivodeship', 'county', 'age_group', 'gender', 'dose',
        'multiple_infections', 'manufacturer_Astra Zeneca', 'manufacturer_Johnson&Johnson',
        'manufacturer_Moderna', 'manufacturer_Pfizer'],
        as_index=False)['number_of_reported_cases'].sum()'''
        st.code(grouping_code, language='python')
        st.header('Processing summary: ')
        grouping_summary_md = '''
        __Max value of number_of_reported_cases before grouping:__ 12\n
        __Max value of number_of_reported_cases after grouping:__ 57
        * The number of rows went down from 9764 to 6504.
        * At the same time the max value of number_of_reported_cases in a single report went up from 12 to 57 indicating a
        large number of cases that shared the same attributes but were reported individually.
        * Despite removing some of the columns, the number of them did not change due to additional columns added with encoding.'''
        st.markdown(grouping_summary_md)
    else:
        st.header('Processed data visualization')
        # Age distribution
        age_group_cases = df_processed.groupby('age_group')['number_of_reported_cases'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=age_group_cases, x='age_group', y='number_of_reported_cases', hue='age_group', ax=ax)
        ax.set_title('Distribution of Reported Cases by Age Group')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Number of Reported Cases')
        st.pyplot(fig)
        # Gender distribution
        gender_cases = df_processed.groupby('gender')['number_of_reported_cases'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=gender_cases, x='gender', y='number_of_reported_cases', hue='gender', ax=ax)
        ax.set_title('Distribution of Reported Cases by Gender')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Number of Reported Cases')
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
if selected == 'Data Scrapping':
    st.title('Data Scrapping')
    scrapping_intro_md = '''
    This page show the source code of the webscraper utilized to obtain the data.
    * __Source:__ https://tinyurl.com/pad-covid-dataset
    * __Library Used:__ Selenium'''
    st.markdown(scrapping_intro_md)
    scrapping_sourcecode = '''
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

    data = []  # Scraped data goes here
    wait = WebDriverWait(driver, 10)
    table_page_number = 7  # Init index of the "next page button". It increases up to 11

    for page_number in range(1, 495):  # Feel free to lower the upper bound for testing
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
        time.sleep(4)

    driver.quit()
    df.to_csv('covid_dataset.csv')'''
    st.code(scrapping_sourcecode, language='python')
if selected == 'Models':
    st.title('Models')
    st.title('!!!!!!!!!!!!!!!TODO!!!!!!!!!!!!!!!!')
