#!/usr/bin/env python3
"""
Generate CSV metadata for text files
=====================================

This script scans the txtfiles directory and creates a CSV with metadata
for each text file including: publication_name, volume, number, issue_date, page_number, filename, and text content.

Usage:
    python metadatacsv.py
"""

import csv
import os
import re
from pathlib import Path

# Configuration
TXTFILES_DIR = "../HerstoryArchiveTxt"
OUTPUT_CSV = "zine_pagemetadata.csv"


def extract_volume_number(filename):
    """
    Extract volume and issue number from filename.
    Expected format includes: Vol{number} and/or No{number}
    
    Args:
        filename (str): Text filename
    
    Returns:
        tuple: (volume, number) or (None, None) if not found
    """
    volume = None
    number = None
    
    # Look for Vol{number}
    vol_match = re.search(r'Vol_?(\d+)', filename, re.IGNORECASE)
    if vol_match:
        volume = vol_match.group(1)
    
    # Look for No{number}
    no_match = re.search(r'No_?(\d+)', filename, re.IGNORECASE)
    if no_match:
        number = no_match.group(1)
    
    return (volume, number)


def extract_page_number(filename):
    """
    Extract page number from filename.
    Expected format: {base_name}_page{number}.txt
    
    Args:
        filename (str): Text filename
    
    Returns:
        int: Page number, or None if not found
    """
    try:
        # Remove .txt extension
        base = filename.replace('.txt', '')
        # Split by '_page'
        parts = base.split('_page')
        if len(parts) == 2:
            return int(parts[1])
    except:
        pass
    return None


def extract_date_from_text(text_content):
    """
    Extract date from the text content of page 1.
    Looks for common date patterns in the text.
    
    Args:
        text_content (str): Text content from page 1
    
    Returns:
        str: Date string, or empty string if not found
    """
    # Look for common date patterns
    # Pattern 1: Month Year (e.g., "January 1973", "Jan. 1973")
    pattern1 = r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{4}\b'
    
    # Pattern 2: Month Day, Year (e.g., "January 15, 1973")
    pattern2 = r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b'
    
    # Pattern 3: Numeric dates (e.g., "1/15/1973", "1-15-1973")
    pattern3 = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
    
    for pattern in [pattern2, pattern1, pattern3]:
        match = re.search(pattern, text_content, re.IGNORECASE)
        if match:
            return match.group(0)
    
    return ""


def generate_metadata():
    """Generate CSV metadata for all text files."""
    
    if not os.path.exists(TXTFILES_DIR):
        print(f"❌ Directory not found: {TXTFILES_DIR}")
        return
    
    metadata_records = []
    publication_dates = {}  # Store date for each publication base name
    
    print("="*80)
    print("Generating Metadata CSV")
    print("="*80)
    
    # Scan each publication subdirectory
    for publication in TXTFILES_DIR:
        pub_dir = Path(TXTFILES_DIR) / publication
        
        if not pub_dir.exists():
            print(f"⚠️  Skipping {publication} - directory not found")
            continue
        
        # Get all .txt files in this directory
        txt_files = sorted(pub_dir.glob("*.txt"))
        
        print(f"\n📁 Processing {publication}: {len(txt_files)} files")
        
        for txt_file in txt_files:
            # Read file content
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                # Extract metadata
                filename = f"{publication}/{txt_file.name}"
                page_number = extract_page_number(txt_file.name)
                
                # Extract volume and issue number
                volume, issue_number = extract_volume_number(txt_file.name)
                
                # Extract base name (without page number)
                base_name = txt_file.name.replace('.txt', '').split('_page')[0]
                full_base = f"{publication}_{base_name}"
                
                # Extract date from page 1 text content
                if page_number == 1:
                    date = extract_date_from_text(text_content)
                    if date:
                        publication_dates[full_base] = date
                        print(f"  📅 Found date: {date}")
                else:
                    # Use date from page 1 of this document
                    date = publication_dates.get(full_base, "")
                
                # Add to records
                metadata_records.append({
                    'publication_name': publication,
                    'volume': volume if volume else '',
                    'number': issue_number if issue_number else '',
                    'issue_date': date,
                    'page_number': page_number if page_number else '',
                    'filename': filename,
                    'text': text_content
                })
                
                print(f"  ✓ {txt_file.name}")
                
            except Exception as e:
                print(f"  ❌ Error reading {txt_file.name}: {e}")
    
    # Write CSV
    if metadata_records:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['publication_name', 'volume', 'number', 'issue_date', 'page_number', 'filename', 'text']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_records)
        
        print("\n" + "="*80)
        print(f"✅ Metadata CSV generated: {OUTPUT_CSV}")
        print(f"   Total records: {len(metadata_records)}")
        print("="*80)
    else:
        print("\n❌ No text files found")


if __name__ == "__main__":
    generate_metadata()