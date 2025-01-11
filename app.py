import streamlit as st
import pandas as pd
import os
import plotly.express as px
from sqlalchemy import create_engine
import toml
from fpdf import FPDF
from pathlib import Path
import os
import torch
from sklearn.preprocessing import LabelEncoder
import requests
import zipfile
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Database connection details
DB_HOST = "junction.proxy.rlwy.net"
DB_USER = "root"
DB_PASSWORD = "GKesHFOMJkurJYvpaVNuqRgTEGYOgFQN"
DB_NAME = "railway"
DB_PORT = "27554"

# Define the database connection
@st.cache_resource
def get_engine():
    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    return engine

engine = get_engine()

# Load data from weapon_data1 and join with dbo_images
# Load data from weapon_data1
@st.cache_data
def load_data():
    query = """
    SELECT * 
    FROM dbo_final_text1
    WHERE Weapon_Name IN (
        SELECT DISTINCT Weapon_Name
        FROM dbo_final_text1
    );
    """
    return pd.read_sql(query, engine)

data = load_data()

# Resolve the directory path
current_dir = Path(__file__).resolve().parent  # Use resolve() to get the absolute path
os.chdir(current_dir)  # Change the current working directory

# Load the .toml configuration
try:
    pages_config = toml.load(current_dir / ".streamlit/pages.toml")
except Exception as e:
    st.error(f"Error loading pages.toml: {e}")
    st.stop()



# Validate 'pages' key in the configuration
if "pages" not in pages_config:
    st.error("'pages' key not found in the pages.toml file.")
    st.stop()



# Handle Page Navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Sidebar Navigation
st.sidebar.markdown("### Navigation")
page_names = ["Home"] + [page["name"] for page in pages_config["pages"]]
selected_page = st.sidebar.selectbox("Go to", page_names, key="page_selector")

if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    st.experimental_set_query_params(page=selected_page)

# Main Content Rendering Based on Selected Page
if st.session_state.current_page == "Home":
    # Dashboard Page
    st.title("Weapon Insights Dashboard")
    st.write("Explore weapon specifications, search, and visualize data interactively.")

    # Display filtered data
    st.write("### Filtered Data Table")
    st.dataframe(data)

   # Filter the data to exclude origins starting with "source: "
    filtered_data = data[~data['Origin'].str.startswith('Source:', na=False)].drop(columns=['Source'], errors='ignore')

    # Threat Distribution by Origin
    st.write("### Threat Distribution by Origin")
    if not filtered_data.empty:
        fig = px.bar(
            filtered_data,
            x="Origin",
            y="Weapon_Name",
            color="Type",
            title="Threat Distribution by Origin",
            labels={"Weapon_Name": "Weapon Name", "Origin": "Country of Origin"},
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for visualization.")

    
    # Load Top 5 Countries Data
    @st.cache_data
    def load_top_countries():
        query = """
        SELECT Origin, COUNT(*) as Weapon_Count
        FROM dbo_final_text1
        GROUP BY Origin
        ORDER BY Weapon_Count DESC
        LIMIT 5;
        """
        return pd.read_sql(query, engine)

    # Display Top 5 Countries Map
    top_countries_data = load_top_countries()
    st.title("Top 5 Countries by Weapon Production")
    st.write("This map shows the top 5 countries that produce the highest number of weapons.")

    if not top_countries_data.empty:
        # Display the map
        st.write("### Top 5 Countries Map")
        fig = px.choropleth(
            top_countries_data,
            locations="Origin",
            locationmode="country names",
            color="Weapon_Count",
            hover_name="Origin",
            title="Top 5 Countries by Weapon Production",
            color_continuous_scale=px.colors.sequential.Plasma,
        )
        fig.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"))
        st.plotly_chart(fig)

        # Display the data in a table
        st.write("### Top 5 Countries Data")
        st.dataframe(top_countries_data)
    else:
        st.warning("No data available to display.")

    

    # Weapon Categories by Origin (New Graph)
    top_countries_data = (
        filtered_data.groupby("Origin").size().reset_index(name="Weapon_Count")
    )

    if not top_countries_data.empty:
        st.write("### Weapon Categories by Origin")
        weapon_origin_distribution = (
            filtered_data.groupby(["Origin", "Type"])
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(
            weapon_origin_distribution,
            x="Origin",
            y="Count",
            color="Type",
            title="Distribution of Weapon Categories by Origin",
            labels={"Origin": "Country of Origin", "Count": "Number of Weapons"},
            barmode="stack",
            color_discrete_sequence=px.colors.sequential.Viridis,
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for visualization.")

    


   # 1. Weapon Production Over Time
    st.write("### Weapon Production Over Time")
    if not data.empty:
        fig = px.line(
            data,
            x="Development",
            y="Weapon_Name",
            color="Type",
            title="Weapon Production Over Time",
            labels={"Weapon_Name": "Number of Weapons", "Development": "Production Year Range"},
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for visualization.")

  
    # Define a function to clean and extract numeric weights
    def clean_weight_column(weight):
        """Extract the first numeric value from the weight string, if available."""
        import re
        if isinstance(weight, str):
            match = re.search(r"\d+(\.\d+)?", weight)  # Match numbers with optional decimals
            if match:
                return float(match.group())
        return None  # Return None if no valid number is found


    # Display Categories with Representative Images
   # Display Categories with Representative Images
    st.write("### Weapon Categories")
    IMAGE_FOLDER = "normalized_images"
    placeholder_image_path = os.path.join(IMAGE_FOLDER, "placeholder.jpeg")
    categories = sorted(data["Type"].dropna().unique())

    cols_per_row = 3
    rows = [categories[i:i + cols_per_row] for i in range(0, len(categories), cols_per_row)]

    # Function to normalize the category name for the redirection URL
    def normalize_name_for_url(name):
        """Normalize category names for URL redirection."""
        return name.replace(" ", "+").replace("_", "+").title()

    # Function to clean up the category name for button labels
    def clean_category_name(name):
        parts = name.split("_")
        cleaned_name = " ".join(parts[1:]).title()
        return cleaned_name
   

    for row in rows:
        cols = st.columns(len(row))
        for col, category in zip(cols, row):
            # Image logic remains the same
            category_dir = os.path.join(IMAGE_FOLDER, category.replace(" ", "_"))
            category_image = None

            if os.path.exists(category_dir) and os.path.isdir(category_dir):
                for file_name in os.listdir(category_dir):
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        category_image = os.path.join(category_dir, file_name)
                        break

            # Display image or placeholder
            if category_image and os.path.exists(category_image):
                col.image(category_image, caption=category, use_container_width=True)
                with col:
                    with open(category_image, "rb") as file:
                      col.download_button(
                        label="Download as PNG",
                        data=file,
                        file_name=os.path.basename(category_image),
                        mime="image/png"
                    )
                # Add navigation button
                cleaned_name =(category)
                col.button(f"Go to {cleaned_name} Category through navigation bar")
                    
                
                
            elif os.path.exists(placeholder_image_path):
                col.image(placeholder_image_path, caption=f"{category} (Placeholder)", use_container_width=True)
            else:
                col.error(f"No image available for {category}")

        
    st.write("### News Section")

    # Prepare the data for the news
    news_data = data[["Weapon_Name", "Type", "Development", "Weight", "Downloaded_Image_Name"]].dropna().reset_index(
        drop=True
    )
    total_news_items = len(news_data)

    # State to keep track of the current news index
    if "news_index" not in st.session_state:
        st.session_state.news_index = 0
    
    # Function to move to the next news item
    def next_news():
        st.session_state.news_index = (st.session_state.news_index + 1) % total_news_items

    # Function to move to the previous news item
    def prev_news():
        st.session_state.news_index = (st.session_state.news_index - 1) % total_news_items


    def generate_single_news_pdf(news_item, image_path):
     pdf = FPDF()
     pdf.set_auto_page_break(auto=True, margin=15)
     pdf.add_page()

     # Add the title
     pdf.set_font("Arial", size=16, style="B")
     pdf.cell(0, 10, f"News: {news_item['Weapon_Name']}", ln=True, align="C")
     pdf.ln(10)

     # Add the image
     if image_path and os.path.exists(image_path):
         pdf.image(image_path, x=10, y=pdf.get_y(), w=100)
         pdf.ln(50)

     # Add the details
     pdf.set_font("Arial", size=12)
     for key in ["Weapon_Name", "Weapon_Category", "Development", "Weight", "Status"]:
         pdf.cell(0, 10, f"{key.replace('_', ' ')}: {news_item[key]}", ln=True)

     # Save the PDF
     pdf_output_path = f"{news_item['Weapon_Name']}_news.pdf"
     pdf.output(pdf_output_path)
     return pdf_output_path
       

    # Display the current news item
    current_news = news_data.iloc[st.session_state.news_index]

    # Get the image for the current news item
    image_path = None
    if pd.notnull(current_news["Downloaded_Image_Name"]):
        image_name = current_news["Downloaded_Image_Name"]
        weapon_category = current_news["Type"].replace(" ", "_")  # Use Weapon_Category from the data

        # Construct the folder path (include the subfolder structure)
        category_folder = os.path.join(IMAGE_FOLDER, weapon_category)  # Only use Weapon_Category for folder

        # Normalize filenames for better matching
        def normalize_name(name):
            # Remove leading underscores, lowercase, replace "_", and strip extensions
            return (
                name.lower()
                .strip()
                .replace("_", " ")
                .replace(".jpg", "")
                .replace(".jpeg", "")
            )

        # Normalized image name (retain numbers)
        normalized_image_name = normalize_name(image_name)
        if os.path.exists(category_folder) and os.path.isdir(category_folder):
            available_files = [normalize_name(f) for f in os.listdir(category_folder)]

            # Match normalized names
            matching_file = next((f for f in os.listdir(category_folder) if normalize_name(f) == normalized_image_name), None)
            if matching_file:
                image_path = os.path.join(category_folder, matching_file)
            else:
                st.write(f"Image {image_name} not found in {category_folder}. Using placeholder.")
        else:
            st.write(f"Category folder does not exist: {category_folder}")

    # Use placeholder if image path is not found
    if not image_path or not os.path.exists(image_path):
        image_path = placeholder_image_path

    # Display the news image
    st.image(
        image_path,
        caption=f"Image for {current_news['Weapon_Name']}",
        use_container_width=True,
    )

    
    # Display the news description
    st.write(
        f"**Here is {current_news['Weapon_Name']}**, developed in **{current_news['Development']}**"
    )

    # Navigation buttons for the news
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Previous"):
            prev_news()
    with col3:
        if st.button("➡️ Next"):
            next_news()
       # Button to download the current news item
    with col2:
        if st.button("Download Current News as PDF"):
            pdf_path = generate_single_news_pdf(current_news, image_path)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download Current News",
                    data=f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf"
                )
            os.remove(pdf_path)  # Clean up temporary file

# Dynamically Created Pages Based on .toml
else:
    import os
    import pandas as pd
    from sqlalchemy import create_engine
    from fpdf import FPDF
    import streamlit as st

    # Dynamically get the current page
    current_page = st.session_state.current_page

    # Define the list of categories for matching
    main_categories = [
        "Vehicles", "Infantry_Weapons", "Firearms", "Aviation Subsystems",
        "Artillery", "Ammunitions", "Aircraft"
    ]

    # Extract the category from the current page's normalized name
    def extract_main_category(name):
        """Extract the main category based on the naming convention."""
        for category in main_categories:
            if category.lower() in name.lower():
                return category
        return "Unknown"

    category_name = extract_main_category(current_page)

    # Display the Page Title
    st.title(f"{current_page}")

    # Display the Category Heading
    st.header(f"Category: {category_name}")

    # Add description or further dynamic content
    st.write(f"This is the dynamically created page for **{current_page}** under the category **{category_name}**.")

    # Normalize the category name
    def normalize_name(name):
        """Normalize category names by removing prefixes and handling spaces or special characters."""
        unwanted_prefixes = ["Vehicles", "Infantry_Weapons", "Firearms", "Aviation Subsystems",
                             "Artillery", "Ammunitions", "Aircraft"]

        for prefix in unwanted_prefixes:
            if name.lower().startswith(prefix.lower()):
                name = name[len(prefix):].strip("_")

        normalized_name = name.lower().replace("+", " ").replace("_", " ").replace("-", " ").strip()
        return normalized_name

    # Function to find image files matching the category
    def find_images_for_category(base_folder, category_name):
        """Find all images in a folder matching the normalized category name."""
        normalized_category = normalize_name(category_name)
        images = []
        for root, _, files in os.walk(base_folder):
            if normalized_category in normalize_name(root):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        images.append((os.path.join(root, file), file))  # Return full path and file name
        return images

    # Function to load details for the images from the database
    def load_image_details(file_name):
        """Load additional details for a given image from the database table."""
        try:
            query = f"""
            SELECT Weapon_Name AS 'Weapon Name', Development AS 'Development Era', Production AS 'Production Era', Origin, 
                   Weapon_Category AS 'Weapon Type', Status, Designations AS 'Designation', Caliber
            FROM weapon_data1
            WHERE Downloaded_Image_Name = '{file_name}'
            """
            result = pd.read_sql(query, engine)
            if not result.empty:
                details = result.iloc[0].dropna().to_dict()  # Drop any columns with NaN values
                return {key: value for key, value in details.items() if value != "Unknown"}
        except Exception as e:
            print(f"Error loading details for {file_name}: {e}")
            engine.rollback()
        return {}

    # Function to create a PDF with image details
    def create_pdf(images_with_details, output_file):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        for image_path, details in images_with_details:
            pdf.add_page()

            # Add the image
            if os.path.exists(image_path):
                pdf.image(image_path, x=10, y=10, w=100)

            # Add the details
            pdf.set_font("Arial", size=12)
            pdf.ln(110)  # Move below the image
            for key, value in details.items():
                safe_value = str(value).encode('latin-1', 'ignore').decode('latin-1')  # Handle unsupported characters
                pdf.cell(0, 10, f"{key}: {safe_value}", ln=True)

        pdf.output(output_file)

    # Base image directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_FOLDER = os.path.join(BASE_DIR, "weapon_images_final1")
    placeholder_image_path = os.path.join(IMAGE_FOLDER, "placeholder.jpeg")
    print(f"IMAGE_FOLDER: {IMAGE_FOLDER}")
    print(f"Files in IMAGE_FOLDER: {os.listdir(IMAGE_FOLDER)}")

    for root, dirs, files in os.walk(IMAGE_FOLDER):
        print(f"Root: {root}, Dirs: {dirs}, Files: {files}")

    # Normalize category name
    normalized_category_name = normalize_name(current_page)

    # Find images for the category
    images = find_images_for_category(IMAGE_FOLDER, current_page)

    # Ensure valid data exists for the current category
    data["Normalized_Weapon_Category"] = data["Weapon_Category"].apply(normalize_name)  # Normalize weapon categories
    normalized_current_page = normalize_name(current_page)

    category_filter = data[
        data["Normalized_Weapon_Category"] == normalized_current_page
    ]

    if not category_filter.empty:
        available_years = ["All"] + sorted(category_filter["Development"].dropna().unique())
        available_origins = ["All"] + sorted(category_filter["Origin"].dropna().unique())

        st.write("### Filter Options")
        col1, col2 = st.columns(2)
        with col1:
            selected_year = st.selectbox("Filter by Year", options=available_years)
        with col2:
            if selected_year != "All":
                filtered_by_year = category_filter[category_filter["Development"] == selected_year]
                available_origins = ["All"] + sorted(filtered_by_year["Origin"].dropna().unique())
            selected_origin = st.selectbox("Filter by Origin", options=available_origins)

        if selected_origin != "All":
            filtered_by_origin = category_filter[category_filter["Origin"] == selected_origin]
            available_years = ["All"] + sorted(filtered_by_origin["Development"].dropna().unique())
            selected_year = st.selectbox("Filter by Year", options=available_years, index=available_years.index(selected_year) if selected_year in available_years else 0)
    else:
        available_years = ["All"]
        available_origins = ["All"]
        selected_year = st.selectbox("Filter by Year", options=available_years)
        selected_origin = st.selectbox("Filter by Origin", options=available_origins)

    # Apply filters to the images
    filtered_images = []
    for image_path, file_name in images:
        details = load_image_details(file_name)
        if details and (selected_year == "All" or details.get("Development Era") == selected_year) and \
           (selected_origin == "All" or details.get("Origin") == selected_origin):
            filtered_images.append((image_path, file_name, details))

    # Display images and their details
    if filtered_images:
        st.write("### Weapon Images")
        cols_per_row = 4
        rows = [filtered_images[i:i + cols_per_row] for i in range(0, len(filtered_images), cols_per_row)]

        for row in rows:
            cols = st.columns(len(row))
            for col, (image_path, file_name, details) in zip(cols, row):
                if os.path.exists(image_path):
                    col.image(image_path, caption=file_name, use_container_width=True)
                else:
                    col.image(placeholder_image_path, caption="Image Not Available", use_container_width=True)

                if col.button(f"Details: {file_name}", key=f"details_button_{file_name}"):
                    with st.expander(f"Details of {file_name}", expanded=True):
                        for key, value in details.items():
                            st.write(f"**{key}:** {value}")

        pdf_file = os.path.join(BASE_DIR, "weapon_images_details.pdf")
        create_pdf([(img[0], img[2]) for img in filtered_images], pdf_file)
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="weapon_images_details.pdf",
                mime="application/pdf",
            )
    else:
        st.warning("No images found for the selected filters.")
