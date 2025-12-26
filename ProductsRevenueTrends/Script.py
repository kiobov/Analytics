import pandas as pd
import numpy as np
import random


class OrderDataCleaner:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        # Track cleaning actions for documentation
        self.cleaning_log = []

    # ---------- helpers ----------
    @staticmethod
    def normalize(val):
        if pd.isna(val):
            return ""
        return str(val).strip()

    @staticmethod
    def generate_id(existing, digits):
        if pd.isna(existing) or str(existing).strip() == "":
            return random.randint(10**(digits-1), 10**digits - 1)
        return existing

    def log_action(self, action, count, details=""):
        """Log cleaning actions for documentation"""
        self.cleaning_log.append({
            'action': action,
            'records_affected': count,
            'details': details
        })

    # ---------- country / region ----------
    def clean_country_and_region(self):
        country_map = {
            "US": "United States",
            "USA": "United States",
            "CL": "Chile",
            "CA": "Canada",
            "UK": "United Kingdom",
            "IL": "Israel",
            "BR": "Brazil",
            "AU": "Australia",
            "AG": "Argentina",
            "SA": "Saudi Arabia",
            "MX": "Mexico",
            "JP": "Japan",
            "IT": "Italy",
            "FR": "France",
            "AR": "Argentina",
            "DE": "Germany",
            "KR": "South Korea",
            "SG": "Singapore"
        }

        self.df["Country_Code_clean"] = (
            self.df["Country Code"]
            .apply(self.normalize)
            .replace(country_map)
        )

        # Log missing regions
        missing_regions = self.df["Region"].isna().sum()
        self.log_action("Handle missing regions", missing_regions, "Replaced with 'Unknown'")

        self.df["Region_clean"] = (
            self.df["Region"]
            .apply(self.normalize)
            .replace("", "Unknown")
        )

        self.df["AccountCreationMethod_clean"] = (
            self.df["Account Creation Method"]
            .apply(self.normalize)
            .replace("", "Unknown")
        )

    # ---------- currency ----------
    def clean_currency(self):
        currency_map = {
            "US": "USD",
            "usd": "USD",
            "USD": "USD"
        }

        self.df["Currency_Clean"] = (
            self.df["Currency"]
            .apply(self.normalize)
            .replace(currency_map)
            .replace("", "USD")  # Default to USD if missing
        )

    # ---------- product ----------
    def clean_product_name(self):
        product_map = {
            "1TB STORAGE": "External Storage 1TB",
            "EXTERNAL STORAGE": "External Storage 1TB",
            "EXTERNAL STORAGE 1TB": "External Storage 1TB",

            "4K GAMING MONITOR": "27in 4K Gaming Monitor",
            '27" 4K MONITOR': "27in 4K Gaming Monitor",
            "27IN 4K GAMING MONITOR": "27in 4K Gaming Monitor",

            "DELL LAPTOP": "Dell Gaming Laptop",
            "DELL GAMING LAPTOP": "Dell Gaming Laptop",

            "DELL MOUSE": "Dell Gaming Mouse",
            "DELL GAMING MOUSE": "Dell Gaming Mouse",
            "GAMING MOUSE": "Dell Gaming Mouse",

            "LOGITECH HEADSET": "Logitech Gaming Headset",
            "LOGITECH GAMING HEADSET": "Logitech Gaming Headset",
            "GAMING HEADSET": "Logitech Gaming Headset",

            "XBOX SERIES X": "Microsoft Xbox Series X",
            "XBOXSERIESX": "Microsoft Xbox Series X",
            "XBOX SERIESX": "Microsoft Xbox Series X",
            "MICROSOFT XBOX SERIES X": "Microsoft Xbox Series X",

            "NINTENDO SWITCH": "Nintendo Switch",
            "NINTENDOSWITCH": "Nintendo Switch",
            "SWITCH": "Nintendo Switch",

            "PLAYSTATION 5": "PlayStation 5",
            "PLAY STATION 5": "PlayStation 5",
            "PS5": "PlayStation 5",
            "PLAYSTATION5": "PlayStation 5",
        }

        self.df["Product_Name_Clean"] = (
            self.df["Product Name"]
            .apply(self.normalize)
            .str.upper()
            .replace(product_map)
        )

    # ---------- marketing channels & platforms ----------
    def clean_channels_and_platforms(self):
        """Standardize marketing channels and purchase platforms"""

        channel_map = {
            "ONLINE": "Online",
            "online": "Online",
            "IN-STORE": "In-Store",
            "in-store": "In-Store",
            "MOBILE APP": "Mobile App",
            "mobile app": "Mobile App",
            "PARTNER": "Partner",
            "partner": "Partner"
        }

        platform_map = {
            "AMAZON": "Amazon",
            "amazon": "Amazon",
            "SHOPIFY": "Shopify",
            "shopify": "Shopify",
            "EBAY": "eBay",
            "ebay": "eBay",
            "DIRECT": "Direct",
            "direct": "Direct",
            "WALMART": "Walmart",
            "walmart": "Walmart"
        }

        self.df["Marketing_Channel_clean"] = (
            self.df["Marketing Channel"]
            .apply(self.normalize)
            .replace(channel_map)
        )

        self.df["Purchase_Platform_clean"] = (
            self.df["Purchase Platform"]
            .apply(self.normalize)
            .replace(platform_map)
        )

    # ---------- ids ----------
    def clean_ids(self):
        missing_order_ids = self.df["Order Id"].isna().sum()
        missing_product_ids = self.df["Product Id"].isna().sum()
        missing_user_ids = self.df["User Id"].isna().sum()

        self.log_action("Generate missing Order IDs", missing_order_ids)
        self.log_action("Generate missing Product IDs", missing_product_ids)
        self.log_action("Generate missing User IDs", missing_user_ids)

        self.df["Order_Id_clean"] = self.df["Order Id"].apply(
            lambda x: self.generate_id(x, 10)
        )
        self.df["Product_Id_clean"] = self.df["Product Id"].apply(
            lambda x: self.generate_id(x, 8)
        )
        self.df["User_Id_clean"] = self.df["User Id"].apply(
            lambda x: self.generate_id(x, 9)
        )

    # ---------- prices (HANDLE NEGATIVE & OUTLIERS) ----------
    def clean_prices(self):
        """
        Clean Local Price column:
        1. Handle missing values
        2. Fix negative prices (data entry errors - make absolute)
        3. Handle outliers (prices < $1 or > $5000)
        4. Handle zero prices
        """

        # Convert to numeric
        self.df['Local Price'] = pd.to_numeric(self.df['Local Price'], errors='coerce')

        # Log missing prices
        missing_prices = self.df['Local Price'].isna().sum()
        self.log_action("Missing prices", missing_prices, "Will be handled based on product")

        # Handle NEGATIVE prices (data entry error - make absolute)
        negative_prices = (self.df['Local Price'] < 0).sum()
        if negative_prices > 0:
            self.log_action("Negative prices", negative_prices, "Converted to absolute value")
            self.df['Local Price'] = self.df['Local Price'].abs()

        # Handle ZERO prices (likely errors)
        zero_prices = (self.df['Local Price'] == 0).sum()
        if zero_prices > 0:
            self.log_action("Zero prices", zero_prices, "Will be imputed based on product median")

        # Handle EXTREME OUTLIERS (< $1 or > $5000)
        low_outliers = ((self.df['Local Price'] > 0) & (self.df['Local Price'] < 1)).sum()
        high_outliers = (self.df['Local Price'] > 5000).sum()

        if low_outliers > 0:
            self.log_action("Unrealistically low prices (<$1)", low_outliers, "Set to NaN for imputation")
            self.df.loc[(self.df['Local Price'] > 0) & (self.df['Local Price'] < 1), 'Local Price'] = np.nan

        if high_outliers > 0:
            self.log_action("Unrealistically high prices (>$5000)", high_outliers, "Set to NaN for imputation")
            self.df.loc[self.df['Local Price'] > 5000, 'Local Price'] = np.nan

        # IMPUTE missing/invalid prices based on product median
        self.impute_prices_by_product()

        # Create clean price column
        self.df['USD_Price_clean'] = self.df['Local Price'].round(2)

    def impute_prices_by_product(self):
        """Replace missing/invalid prices with product median price"""

        # Calculate median price per product
        product_medians = self.df.groupby('Product_Name_Clean')['Local Price'].median()

        # Fill missing prices with product median
        for product, median_price in product_medians.items():
            mask = (self.df['Product_Name_Clean'] == product) & (self.df['Local Price'].isna())
            imputed_count = mask.sum()

            if imputed_count > 0:
                self.df.loc[mask, 'Local Price'] = median_price
                self.log_action(f"Imputed prices for {product}", imputed_count, f"Used median: ${median_price:.2f}")

    # ---------- remove duplicates ----------
    def remove_duplicates(self):
        """Remove duplicate records"""
        initial_count = len(self.df)

        # Remove exact duplicates
        self.df = self.df.drop_duplicates(keep='first')

        duplicates_removed = initial_count - len(self.df)
        if duplicates_removed > 0:
            self.log_action("Duplicate records removed", duplicates_removed, "Kept first occurrence")

    # ---------- timestamps / dates ----------
    def clean_timestamps(self):
        """
        Clean timestamp columns by converting all formats to M/D/YYYY:
        - Keep existing M/D/YYYY format (7/14/2023)
        - Convert YYYY-MM-DD to M/D/YYYY (2023-07-21 → 7/21/2023)
        - Convert ISO timestamps to M/D/YYYY (2023-07-12T00:27:32.306Z → 7/12/2023)
        - Handle NaN/invalid values
        """
        
        date_columns = ["Purchase Ts", "Created On", "Ship Ts", "Delivery Ts", "Refund Ts"]
        
        for col in date_columns:
            if col not in self.df.columns:
                continue
            
            # Convert to string and clean
            series = self.df[col].fillna('').astype(str).str.strip()
            
            # Track invalid dates
            invalid_count = 0
            cleaned_dates = []
            
            for val in series:
                # Handle empty/invalid values
                if val in ['', 'nan', 'None', 'NaT', '-']:
                    cleaned_dates.append(None)
                    invalid_count += 1
                    continue
                
                try:
                    # Try to parse the date (handles all formats)
                    parsed_date = pd.to_datetime(val, errors='coerce')
                    
                    if pd.isna(parsed_date):
                        cleaned_dates.append(None)
                        invalid_count += 1
                    else:
                        # Convert to M/D/YYYY format
                        formatted_date = parsed_date.strftime('%-m/%-d/%Y')
                        cleaned_dates.append(formatted_date)
                
                except:
                    cleaned_dates.append(None)
                    invalid_count += 1
            
            # Replace column with cleaned dates
            self.df[col] = cleaned_dates
            
            # Log failures
            if invalid_count > 0:
                self.log_action(
                    f"Invalid dates in {col}",
                    invalid_count,
                    "Converted to NaN"
                )

    # ---------- finalize columns ----------
    def finalize_columns(self):
        """
        Replace all original messy columns with cleaned versions
        and remove the _clean/_Clean suffix columns
        """
        
        # Map of clean column -> original column
        column_replacements = {
            'Country_Code_clean': 'Country Code',
            'Region_clean': 'Region',
            'AccountCreationMethod_clean': 'Account Creation Method',
            'Currency_Clean': 'Currency',
            'Product_Name_Clean': 'Product Name',
            'Marketing_Channel_clean': 'Marketing Channel',
            'Purchase_Platform_clean': 'Purchase Platform',
            'Order_Id_clean': 'Order Id',
            'Product_Id_clean': 'Product Id',
            'User_Id_clean': 'User Id',
            'USD_Price_clean': 'Local Price'
        }
        
        # Replace original columns with cleaned versions
        for clean_col, original_col in column_replacements.items():
            if clean_col in self.df.columns:
                self.df[original_col] = self.df[clean_col]
                self.df.drop(columns=[clean_col], inplace=True)
        
        print(" Replaced all original columns with cleaned data")

    # ---------- generate cleaning report ----------
    def generate_cleaning_report(self, output_path="data_cleaning_report.txt"):
        """Generate a report of all cleaning actions taken"""

        total_records = len(self.df)

        report = []
        report.append("=" * 80)
        report.append("PIXELFORGE GAMING CO. - DATA CLEANING REPORT")
        report.append("=" * 80)
        report.append(f"\nTotal Records: {total_records:,}")
        report.append(f"\nCleaning Actions Performed:\n")

        for i, log_entry in enumerate(self.cleaning_log, 1):
            report.append(f"{i}. {log_entry['action']}")
            report.append(f"   Records Affected: {log_entry['records_affected']:,}")
            if log_entry['details']:
                report.append(f"   Details: {log_entry['details']}")
            report.append("")

        report.append("=" * 80)
        report.append("DATA QUALITY SUMMARY")
        report.append("=" * 80)

        # Summary statistics
        report.append(f"\nPrice Statistics:")
        report.append(f"  Min Price: ${self.df['Local Price'].min():.2f}")
        report.append(f"  Max Price: ${self.df['Local Price'].max():.2f}")
        report.append(f"  Median Price: ${self.df['Local Price'].median():.2f}")
        report.append(f"  Mean Price: ${self.df['Local Price'].mean():.2f}")

        report.append(f"\nProduct Distribution:")
        for product, count in self.df['Product Name'].value_counts().items():
            report.append(f"  {product}: {count:,} orders")

        report_text = "\n".join(report)

        # Save to file
        with open(output_path, 'w') as f:
            f.write(report_text)

        # Also print to console
        print(report_text)

        return report_text

    # ---------- run all ----------
    def run_all(self, output_path="Cleaned_PixelForge_Orders.csv",
                report_path="Data_Cleaning_Report.txt"):
        """Run all cleaning steps and generate outputs"""

        print("Starting data cleaning process...\n")

        print("1. Cleaning countries and regions...")
        self.clean_country_and_region()

        print("2. Cleaning currency...")
        self.clean_currency()

        print("3. Cleaning product names...")
        self.clean_product_name()

        print("4. Cleaning marketing channels and platforms...")
        self.clean_channels_and_platforms()

        print("5. Cleaning IDs...")
        self.clean_ids()

        print("6. Cleaning prices (handling negatives, zeros, outliers)...")
        self.clean_prices()

        print("7. Removing duplicates...")
        self.remove_duplicates()

        print("8. Cleaning timestamps (converting to M/D/YYYY format)...")
        self.clean_timestamps()

        print("9. Finalizing columns (replacing originals with cleaned data)...")
        self.finalize_columns()

        print("10. Generating cleaning report...")
        self.generate_cleaning_report(report_path)

        print(f"\n11. Saving cleaned data to {output_path}...")
        self.df.to_csv(output_path, index=False)

        print("\n Data cleaning complete!")
        print(f" Cleaned dataset: {output_path}")
        print(f" Cleaning report: {report_path}")

        return self.df


# ============================================================================
# FEATURE ENGINEERING CLASS (Separate from cleaning)
# ============================================================================

class FeatureEngineer:
    def __init__(self, df):
        """
        Initialize with a cleaned dataframe
        
        Args:
            df: Cleaned pandas DataFrame (output from OrderDataCleaner)
        """
        self.df = df.copy()
        self.engineering_log = []
    
    def log_action(self, action, count, details=""):
        """Log feature engineering actions"""
        self.engineering_log.append({
            'action': action,
            'records_affected': count,
            'details': details
        })
    
    def create_time_features(self):
        """
        Create time-based features from Purchase Ts:
        - Purchase_Year
        - Purchase_Month
        - Purchase_Quarter
        - Year_Month (for time series)
        """
        
        if 'Purchase Ts' not in self.df.columns:
            print(" 'Purchase Ts' column not found. Skipping time features.")
            return
        
        # Convert to datetime
        self.df['Purchase_Ts_dt'] = pd.to_datetime(self.df['Purchase Ts'], errors='coerce')
        
        # Extract components
        self.df['Purchase_Year'] = self.df['Purchase_Ts_dt'].dt.year
        self.df['Purchase_Month'] = self.df['Purchase_Ts_dt'].dt.month
        self.df['Purchase_Quarter'] = self.df['Purchase_Ts_dt'].dt.quarter
        self.df['Year_Month'] = self.df['Purchase_Ts_dt'].dt.to_period('M').astype(str)
        
        # Drop temporary datetime column
        self.df.drop(columns=['Purchase_Ts_dt'], inplace=True)
        
        valid_count = self.df['Purchase_Year'].notna().sum()
        self.log_action("Created time-based features", valid_count,
                       "Purchase_Year, Purchase_Month, Purchase_Quarter, Year_Month")
        
        print(f" Created time features for {valid_count:,} records")
    
    def create_delivery_days(self):
        """
        Calculate Delivery_Days (time between Ship and Delivery)
        """
        
        if 'Ship Ts' not in self.df.columns or 'Delivery Ts' not in self.df.columns:
            print(" 'Ship Ts' or 'Delivery Ts' not found. Skipping delivery days.")
            return
        
        # Convert to datetime
        ship_dt = pd.to_datetime(self.df['Ship Ts'], errors='coerce')
        delivery_dt = pd.to_datetime(self.df['Delivery Ts'], errors='coerce')
        
        # Calculate days
        self.df['Delivery_Days'] = (delivery_dt - ship_dt).dt.days
        
        valid_count = self.df['Delivery_Days'].notna().sum()
        avg_days = self.df['Delivery_Days'].mean()
        
        self.log_action("Calculated Delivery_Days", valid_count,
                       f"Average: {avg_days:.1f} days")
        
        print(f" Calculated delivery days for {valid_count:,} records (avg: {avg_days:.1f} days)")
    
    def create_refund_flag(self):
        """
        Create Is_Refunded binary flag (1 = refunded, 0 = not refunded)
        """
        
        if 'Refund Ts' not in self.df.columns:
            print(" 'Refund Ts' not found. Skipping refund flag.")
            return
        
        # Convert to datetime and create flag
        refund_dt = pd.to_datetime(self.df['Refund Ts'], errors='coerce')
        self.df['Is_Refunded'] = refund_dt.notna().astype(int)
        
        refunded_count = self.df['Is_Refunded'].sum()
        refund_rate = (refunded_count / len(self.df)) * 100
        
        self.log_action("Created Is_Refunded flag", refunded_count,
                       f"Refund rate: {refund_rate:.2f}%")
        
        print(f" Created refund flag: {refunded_count:,} refunded orders ({refund_rate:.2f}%)")
    
    def create_all_features(self):
        """Run all feature engineering steps"""
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING - Creating Analysis Columns")
        print("="*60 + "\n")
        
        self.create_time_features()
        self.create_delivery_days()
        self.create_refund_flag()
        
        print("\n Feature engineering complete!")
        
        return self.df
    
    def save_and_report(self, csv_path="Cleaned_PixelForge_Orders.csv", 
                       report_path="Data_Cleaning_Report.txt"):
        """
        Save the updated dataframe with new features to the SAME CSV
        and append feature engineering report to the SAME report file
        """
        
        # Save updated dataframe (overwrites the cleaned CSV)
        self.df.to_csv(csv_path, index=False)
        print(f" Updated {csv_path} with new feature columns")
        
        # Prepare feature engineering report
        report = []
        report.append("\n\n" + "=" * 80)
        report.append("FEATURE ENGINEERING REPORT")
        report.append("=" * 80)
        report.append("\nNew Columns Created for Analysis:\n")
        
        for i, log_entry in enumerate(self.engineering_log, 1):
            report.append(f"{i}. {log_entry['action']}")
            report.append(f"   Records Affected: {log_entry['records_affected']:,}")
            if log_entry['details']:
                report.append(f"   Details: {log_entry['details']}")
            report.append("")
        
        # Add column summary
        new_columns = ['Purchase_Year', 'Purchase_Month', 'Purchase_Quarter', 
                      'Year_Month', 'Delivery_Days', 'Is_Refunded']
        existing_columns = [col for col in new_columns if col in self.df.columns]
        
        report.append("\nNew Columns Added:")
        for col in existing_columns:
            report.append(f"  - {col}")
        
        report_text = "\n".join(report)
        
        # Append to existing report
        with open(report_path, 'a') as f:
            f.write(report_text)
        
        print(f" Feature engineering report appended to {report_path}")
        
        return report_text


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # STEP 1: CLEAN THE DATA
    print("STEP 1: DATA CLEANING")
    print("="*60)
    cleaner = OrderDataCleaner("OrdersDataset.csv")
    cleaned_df = cleaner.run_all(
        output_path="Cleaned_PixelForge_Orders.csv",
        report_path="Data_Cleaning_Report.txt"
    )
    
    # STEP 2: FEATURE ENGINEERING (Adds columns to the same CSV)
    print("\n\nSTEP 2: FEATURE ENGINEERING")
    print("="*60)
    engineer = FeatureEngineer(cleaned_df)
    analysis_df = engineer.create_all_features()
    
    # Save back to the SAME files (updates CSV and appends to report)
    engineer.save_and_report(
        csv_path="Cleaned_PixelForge_Orders.csv",
        report_path="Data_Cleaning_Report.txt"
    )
    
    print(f"\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f" Final dataset: Cleaned_PixelForge_Orders.csv")
    print(f" Complete report: Data_Cleaning_Report.txt")
    print(f"\nFinal dataset shape: {analysis_df.shape}")
    print(f"New columns added: Purchase_Year, Purchase_Month, Purchase_Quarter,")
    print(f"                   Year_Month, Delivery_Days, Is_Refunded")
    print(f"\nAll columns: {list(analysis_df.columns)}")