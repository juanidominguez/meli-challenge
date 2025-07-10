"""
Feature Engineering Pipeline for MercadoLibre Product Classification
==================================================================

This module provides a comprehensive feature engineering pipeline for transforming
raw product data into features suitable for machine learning models.

The pipeline includes:
- Data normalization and cleaning
- Feature extraction from complex fields (timestamps, text, lists)
- Categorical encoding and numerical transformations
- Seller and product metadata enrichment

Usage:
    from feature_engineering_pipeline import make_full_pipeline
    
    pipeline = make_full_pipeline(target_name='condition')
    X_processed = pipeline.fit_transform(X_train, y_train)
"""

from __future__ import annotations
import re
import ast
import math
import unicodedata
import json
import logging
from typing import List, Tuple, Union, Optional, Any, cast
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import category_encoders as ce
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder, MeanEncoder
from feature_engine.selection import DropFeatures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════
# Constants and Configuration
# ════════════════════════════════════════════════════════════════════════

# Tokens that should be treated as null/missing values
NULL_TOKENS = {'', ' ', 'na', 'n/a', 'none', 'null', 'nan', '[]', '{}'}

# Listing type hierarchy mapping
LISTING_TYPE_HIERARCHY = {
    'free': 0,
    'bronze': 1,
    'silver': 2,
    'gold': 3,
    'gold_special': 4,
    'gold_premium': 5,
    'gold_pro': 6
}

# Columns to drop from the dataset
COLUMNS_TO_DROP = [
    'differential_pricing', 'subtitle', 'shipping_dimensions', 'original_price',
    'shipping_methods', 'site_id', 'listing_source', 'coverage_areas',
    'international_delivery_mode', 'seller_address_country_name',
    'seller_address_country_id', 'seller_address_city_name', 'deal_ids',
    'id', 'permalink', 'thumbnail', 'secure_thumbnail', 'base_price',
    'parent_item_id', 'variations', 'title', 'seller_address_state_id',
    'seller_address_city_id', 'sub_status', 'official_store_id',
    'video_id', 'catalog_product_id', 'shipping_tags', 'shipping_free_methods',
    'stop_time', 'last_updated', 'date_created'
]

# Columns for one-hot encoding
ONE_HOT_COLUMNS = ['buying_mode', 'seller_address_state_name', 'shipping_mode', 'status']

# Boolean columns that need special handling
BOOLEAN_COLUMNS = ['automatic_relist', 'shipping_local_pick_up', 'shipping_free_shipping']

# Seller volume thresholds
HIGH_VOLUME_THRESHOLD = 50
HIGH_INVENTORY_THRESHOLD = 10
LOW_PRICE_THRESHOLD = 100

# ════════════════════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════════════════════

def normalize_txt(s: str) -> str:
    """
    Normalize text by removing accents, converting to lowercase, and cleaning whitespace.
    
    Args:
        s: Input string to normalize
        
    Returns:
        Normalized string
    """
    if not isinstance(s, str):
        s = str(s)
    
    # Remove accents and convert to ASCII
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    
    # Convert to lowercase and normalize whitespace
    return re.sub(r'\s+', ' ', s.lower()).strip()


def warranty_to_months(text: Any) -> int:
    """
    Extract warranty duration in months from text.
    
    Args:
        text: Warranty text description
        
    Returns:
        -1: No warranty
         0: Has warranty but duration not specified
        >0: Warranty duration in months
    """
    if pd.isna(text):
        return -1
    
    try:
        txt = normalize_txt(str(text))
        
        if txt in NULL_TOKENS or re.search(r'\b(?:sin|no)\s+garant', txt, re.I):
            return -1
        
        # Regular expressions for time units
        re_units = {
            'year': re.compile(r'(?:un|\d+)\s*a(?:n|ñ)o?s?'),
            'month': re.compile(r'(?:un|\d+)\s*mes(?:es)?'),
            'day': re.compile(r'(?:un|\d+)\s*d[ií]a?s?'),
        }
        
        for unit, rgx in re_units.items():
            match = rgx.search(txt)
            if match:
                # Extract number or default to 1 for "un"
                token = re.search(r'\d+', match.group())
                num = int(token.group()) if token else 1
                
                if unit == 'year':
                    return num * 12
                elif unit == 'day':
                    return max(1, round(num / 30))
                else:  # month
                    return num
        
        # Check for warranty keywords
        if re.search(r'(garant\w*.*\b(?:si|con|fabric|fabr|oficial|total)\b)|'
                    r'\b(?:de|del)\s+fabr(?:ic(?:a|ante)?|)\b', txt, re.I):
            return 0
        
        return -1
        
    except Exception as e:
        logger.warning(f"Error processing warranty text '{text}': {e}")
        return -1


# ════════════════════════════════════════════════════════════════════════
# Transformer Classes
# ════════════════════════════════════════════════════════════════════════

class CatNullNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalize null values in categorical columns.
    
    Converts empty collections, null tokens, and various null representations
    to pandas NA for consistent handling downstream.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CatNullNormalizer':
        """
        Fit the transformer by identifying categorical columns.
        
        Args:
            X: Input DataFrame
            y: Target variable (ignored)
            
        Returns:
            Self for method chaining
        """
        self.cols = X.select_dtypes(include='object').columns.tolist()
        return self
    
    @staticmethod
    def _to_na(x: Any) -> Any:
        """
        Convert various null representations to pandas NA.
        
        Args:
            x: Input value
            
        Returns:
            pandas NA if value represents null, otherwise original value
        """
        try:
            # Handle collections
            if isinstance(x, (list, tuple, set, dict, np.ndarray)):
                return pd.NA if len(x) == 0 else x
            
            # Handle explicit nulls
            if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
                return pd.NA
            
            # Handle string representations of null
            if str(x).strip().lower() in NULL_TOKENS:
                return pd.NA
            
            return x
        except Exception:
            return x
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns by normalizing null values.
        
        Args:
            X: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        out = X.copy()
        for col in self.cols:
            if col in out.columns:
                out[col] = out[col].map(self._to_na)
        return out


class TimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract time-based features from timestamp columns.
    
    Handles multiple timestamp formats including epoch milliseconds,
    epoch seconds, and ISO string formats.
    """
    
    def __init__(self, col: str = 'start_time'):
        """
        Initialize the transformer.
        
        Args:
            col: Name of the timestamp column to process
        """
        self.col = col
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TimeFeatures':
        """Fit method (no-op for this transformer)."""
        return self
    
    def _parse_timestamp(self, s: pd.Series) -> pd.Series:
        """
        Parse timestamps from various formats.
        
        Args:
            s: Series containing timestamps
            
        Returns:
            Series with parsed datetime objects
        """
        try:
            # Try milliseconds first
            ts_ms = pd.to_datetime(s, unit='ms', errors='coerce', utc=True)
            
            # Try seconds for values that failed milliseconds
            ts_s = pd.to_datetime(s, unit='s', errors='coerce', utc=True)
            
            # For numeric values, prefer milliseconds, fallback to seconds
            ts_numeric = np.where(
                s.astype(str).str.isnumeric(),
                ts_ms.fillna(ts_s),
                pd.to_datetime(s, errors='coerce', utc=True)
            )
            
            return pd.Series(ts_numeric).dt.tz_localize(None)
        except Exception as e:
            logger.warning(f"Error parsing timestamps: {e}")
            return pd.Series([pd.NaT] * len(s))
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform timestamp column into time-based features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with time features added and original column removed
        """
        if self.col not in X.columns:
            logger.warning(f"Column '{self.col}' not found in DataFrame")
            return X
        
        out = X.copy()
        ts = self._parse_timestamp(pd.Series(out[self.col]))
        
        out['start_hour'] = ts.dt.hour.astype('Int8')
        out['start_dow'] = ts.dt.weekday.astype('Int8')
        out['start_month'] = ts.dt.month.astype('Int8')
        
        return out.drop(columns=self.col)


class NumericIndicators(BaseEstimator, TransformerMixin):
    """
    Create numeric indicator features and transformations.
    
    Generates binary flags for high inventory and low price,
    and creates log-transformed price feature.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'NumericIndicators':
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform numeric columns into indicator features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with numeric indicators added
        """
        out = X.copy()
        
        # Create indicator features
        if 'initial_quantity' in out.columns:
            out['high_inventory_flag'] = (out['initial_quantity'] > HIGH_INVENTORY_THRESHOLD).astype('int8')
        
        if 'price' in out.columns:
            out['low_price_flag'] = (out['price'] < LOW_PRICE_THRESHOLD).astype('int8')
            out['log_price'] = np.log1p(out['price']).astype('float32')
            out = out.drop(columns='price')
        
        return out


class ListingTypeOrdinal(BaseEstimator, TransformerMixin):
    """
    Convert listing type to ordinal encoding based on hierarchy.
    
    Maps listing types to numerical values reflecting their hierarchy
    in the MercadoLibre ecosystem.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ListingTypeOrdinal':
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform listing type to ordinal values.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with ordinal listing type feature
        """
        out = X.copy()
        
        if 'listing_type_id' in out.columns:
            out['listing_type_rank'] = (
                out['listing_type_id']
                .map(lambda x: LISTING_TYPE_HIERARCHY.get(x, -1))
                .astype('Int8')
            )
            out = out.drop(columns='listing_type_id')
        
        return out


class SellerFeatures(BaseEstimator, TransformerMixin):
    """
    Extract seller-related features.
    
    Creates features based on seller activity volume including
    log-transformed volume and high-volume indicators.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'SellerFeatures':
        """
        Fit the transformer by calculating seller volumes.
        
        Args:
            X: Input DataFrame
            y: Target variable (ignored)
            
        Returns:
            Self for method chaining
        """
        if 'seller_id' in X.columns:
            self.seller_counts = X['seller_id'].astype('str').value_counts()
        else:
            self.seller_counts = pd.Series(dtype='int64')
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform seller information into volume features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with seller features added
        """
        out = X.copy()
        
        if 'seller_id' in out.columns:
            # Calculate seller volume
            volume = out['seller_id'].astype('str').map(lambda x: self.seller_counts.get(x, 0))
            
            out['seller_volume_log'] = np.log1p(volume).astype('float32')
            out['seller_high_volume'] = (volume > HIGH_VOLUME_THRESHOLD).astype('int8')
            
            out = out.drop(columns='seller_id')
        
        return out


class WarrantyTransformer(BaseEstimator, TransformerMixin):
    """
    Transform warranty information into structured features.
    
    Extracts warranty duration and creates indicators for
    warranty presence and duration specification.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'WarrantyTransformer':
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform warranty text into structured features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with warranty features
        """
        out = X.copy()
        
        if 'warranty' in out.columns:
            months = out['warranty'].map(warranty_to_months).astype('int16')
            
            out['warranty_months'] = months
            out['warranty_duration_specified'] = (months >= 0).astype('int8')
            
            out = out.drop(columns='warranty')
        
        return out


class AttrDescFlags(BaseEstimator, TransformerMixin):
    """
    Create flags and counts for attributes and descriptions.
    
    Generates features indicating the presence and count of
    product attributes and descriptions.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AttrDescFlags':
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform attributes and descriptions into count features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with attribute and description features
        """
        out = X.copy()
        
        if 'attributes' in out.columns:
            n_attrs = out['attributes'].apply(
                lambda v: len(v) if isinstance(v, list) else 0
            ).astype('int16')
            
            out['n_attributes'] = n_attrs
            out['has_attributes'] = (n_attrs > 0).astype('int8')
            out = out.drop(columns='attributes')
        
        if 'descriptions' in out.columns:
            out['has_description'] = out['descriptions'].apply(
                lambda v: int(bool(isinstance(v, list) and len(v)))
            ).astype('int8')
            out = out.drop(columns='descriptions')
        
        return out


class PaymentMethodFeatures(BaseEstimator, TransformerMixin):
    """
    Extract features from payment method information.
    
    Analyzes payment methods to create features for card acceptance,
    pickup options, and payment method diversity.
    """
    
    def __init__(self):
        self.card_pattern = re.compile(r'visa|master|american|diners|tarjeta', re.I)
        self.pickup_pattern = re.compile(r'(acordar|reembolso)', re.I)
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PaymentMethodFeatures':
        """Fit method (no-op for this transformer)."""
        return self
    
    def _summarize_payment_methods(self, methods: Any) -> Tuple[int, int, int]:
        """
        Summarize payment method information.
        
        Args:
            methods: Payment methods data
            
        Returns:
            Tuple of (count, accepts_card, pay_on_pickup)
        """
        if not isinstance(methods, list) or not methods:
            return 0, 0, 0
        
        try:
            descriptions = [m.get('description', '') for m in methods if isinstance(m, dict)]
            
            count = len(descriptions)
            accepts_card = int(any(self.card_pattern.search(desc) for desc in descriptions))
            pay_on_pickup = int(any(self.pickup_pattern.search(desc) for desc in descriptions))
            
            return count, accepts_card, pay_on_pickup
        except Exception:
            return 0, 0, 0
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform payment method information into features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with payment method features
        """
        out = X.copy()
        
        if 'non_mercado_pago_payment_methods' in out.columns:
            payment_summary = out['non_mercado_pago_payment_methods'].apply(
                self._summarize_payment_methods
            )
            
            out['n_extra_pay_methods'] = payment_summary.map(lambda t: t[0]).astype('int8')
            out['accepts_card'] = payment_summary.map(lambda t: t[1]).astype('int8')
            out['pay_on_pickup'] = payment_summary.map(lambda t: t[2]).astype('int8')
            
            out = out.drop(columns='non_mercado_pago_payment_methods')
        
        return out


class PictureCount(BaseEstimator, TransformerMixin):
    """
    Count the number of pictures for each product.
    
    Creates a feature indicating the number of product images,
    which can be important for product appeal and completeness.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PictureCount':
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform picture information into count feature.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with picture count feature
        """
        out = X.copy()
        
        if 'pictures' in out.columns:
            out['n_pictures'] = out['pictures'].apply(
                lambda v: len(v) if isinstance(v, list) else 0
            ).astype('int16')
            out = out.drop(columns='pictures')
        
        return out


class TagFeatures(BaseEstimator, TransformerMixin):
    """
    Extract features from product tags.
    
    Creates features for tag presence and count, which can
    indicate product categorization and marketing effort.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TagFeatures':
        """Fit method (no-op for this transformer)."""
        return self
    
    def _parse_tags(self, val: Any) -> List[str]:
        """
        Parse tags from various formats.
        
        Args:
            val: Tag data in various formats
            
        Returns:
            List of parsed tags
        """
        if isinstance(val, list):
            return val
        
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return []
        
        try:
            if isinstance(val, str):
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, list) else []
            return []
        except Exception:
            return []
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform tag information into features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with tag features
        """
        out = X.copy()
        
        if 'tags' in out.columns:
            tag_lists = out['tags'].apply(self._parse_tags)
            
            out['has_tags'] = (tag_lists.str.len() > 0).astype('int8')
            out['n_tags'] = tag_lists.str.len().astype('int16')
            
            out = out.drop(columns='tags')
        
        return out


class CurrencyBinary(BaseEstimator, TransformerMixin):
    """
    Create binary feature for currency type.
    
    Creates an indicator for USD vs other currencies,
    which can be important for pricing and market analysis.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CurrencyBinary':
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform currency information into binary feature.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with currency binary feature
        """
        out = X.copy()
        
        if 'currency_id' in out.columns:
            out['is_USD'] = (out['currency_id'] == 'USD').astype('int8')
            out = out.drop(columns='currency_id')
        
        return out


# ════════════════════════════════════════════════════════════════════════
# Pipeline Factory Function
# ════════════════════════════════════════════════════════════════════════

def make_full_pipeline(target_name: str = 'condition') -> Pipeline:
    """
    Create a complete feature engineering pipeline.
    
    This function builds a comprehensive preprocessing pipeline that transforms
    raw product data into features suitable for machine learning models.
    
    Args:
        target_name: Name of the target column (default: 'condition')
        
    Returns:
        Scikit-learn Pipeline object ready for fitting and transformation
        
    Example:
        >>> pipeline = make_full_pipeline('condition')
        >>> X_processed = pipeline.fit_transform(X_train, y_train)
        >>> X_test_processed = pipeline.transform(X_test)
    """
    logger.info("Creating feature engineering pipeline")
    
    pipeline_steps = [
        ('drop_initial_cols', DropFeatures(features_to_drop=cast(List[Union[str, int]], COLUMNS_TO_DROP))),
        ('normalize_cat_nulls', CatNullNormalizer()),
        ('add_time_features', TimeFeatures()),
        ('add_numeric_indicators', NumericIndicators()),
        ('listing_type_ordinal', ListingTypeOrdinal()),
        ('impute_state_missing', CategoricalImputer(
            variables=cast(List[Union[str, int]], ['seller_address_state_name']), 
            imputation_method='missing'
        )),
        ('rare_labels', RareLabelEncoder(
            variables=cast(List[Union[str, int]], ONE_HOT_COLUMNS), 
            tol=0.01, 
            n_categories=3, 
            replace_with='other'
        )),
        ('add_seller_features', SellerFeatures()),
        ('warranty_features', WarrantyTransformer()),
        ('attribute_description_flags', AttrDescFlags()),
        ('payment_method_features', PaymentMethodFeatures()),
        ('picture_count', PictureCount()),
        ('tag_features', TagFeatures()),
        ('one_hot_nominal', OneHotEncoder(
            variables=cast(List[Union[str, int]], ONE_HOT_COLUMNS), 
            drop_last=False, 
            ignore_format=True
        )),
        ('currency_binary', CurrencyBinary()),
        ('mean_target_encoder', ce.TargetEncoder(
            cols=['category_id'], 
            smoothing=0.3, 
            handle_unknown='return_nan'
        ))
    ]
    
    return Pipeline(steps=pipeline_steps)


# ════════════════════════════════════════════════════════════════════════
# Main Execution (only when run as script)
# ════════════════════════════════════════════════════════════════════════

def main():
    """
    Main function to demonstrate pipeline usage.
    
    This function loads data, creates the pipeline, and processes the data
    to show the pipeline in action.
    """
    logger.info("Starting feature engineering pipeline demonstration")
    
    try:
        # Import here to avoid circular imports
        from new_or_used import build_dataset
        
        # Load raw data
        X_train_raw, y_train, X_test_raw, y_test = build_dataset()
        logger.info(f"Loaded {len(X_train_raw)} training samples and {len(X_test_raw)} test samples")
        
        # Convert to DataFrames
        df_train = pd.json_normalize(X_train_raw, sep='_')
        df_test = pd.json_normalize(X_test_raw, sep='_')
        
        logger.info(f"Training shape: {df_train.shape}, Test shape: {df_test.shape}")
        
        # Create and fit pipeline
        pipeline = make_full_pipeline(target_name='condition')
        
        # Process training data
        X_train_processed = pipeline.fit_transform(df_train.drop(columns='condition'), y_train)
        logger.info(f"Training data processed: {X_train_processed.shape}")
        
        # Process test data
        X_test_processed = pipeline.transform(df_test)
        logger.info(f"Test data processed: {X_test_processed.shape}")
        
        logger.info("Feature engineering pipeline completed successfully")
        
        # Display feature information
        print(f"\n{'='*60}")
        print("FEATURE ENGINEERING PIPELINE RESULTS")
        print(f"{'='*60}")
        print(f"Original training samples: {len(X_train_raw):,}")
        print(f"Original test samples: {len(X_test_raw):,}")
        print(f"Features generated: {X_train_processed.shape[1]}")
        print(f"Training shape after FE: {X_train_processed.shape}")
        print(f"Test shape after FE: {X_test_processed.shape}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}")
        raise


if __name__ == "__main__":
    main()