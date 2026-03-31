import json
import os
import pandas as pd
import re
from collections import Counter
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from datetime import datetime

class DataCleaner:
    def __init__(self):
        self.cleaning_stats = {}
        self.removed_reasons = Counter()
        
    def load_data(self, folder_path):
        """Load all JSON files from generation dataset folder"""
        data = []
        file_count = 0
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        
                        # Handle both list and single dict formats
                        if isinstance(file_data, list):
                            for item in file_data:
                                if isinstance(item, dict) and 'input_text' in item and 'reply_text' in item:
                                    data.append(item)
                        elif isinstance(file_data, dict) and 'input_text' in file_data and 'reply_text' in file_data:
                            data.append(file_data)
                        
                        print(f"✓ Loaded {filename} ({len([x for x in (file_data if isinstance(file_data, list) else [file_data]) if isinstance(x, dict) and 'input_text' in x])} valid entries)")
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"✗ Error loading {filename}: {e}")
        print(f"\n✓ Total files loaded: {file_count}")
        return data

    def clean_text(self, text):
        """Comprehensive text cleaning"""
        # Strip whitespace
        text = text.strip()
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Fix common unicode issues
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200d', '')  # Zero-width joiner
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        return text

    def is_valid_length(self, text, min_length=8, max_length=512):
        """Check if text length is valid"""
        length = len(text.strip())
        return min_length <= length <= max_length

    def has_minimum_quality(self, text):
        """Check if text has minimum quality standards"""
        # Must have at least one letter
        if not any(c.isalpha() for c in text):
            return False
        # Maximum 70% can be punctuation/numbers
        non_alpha = sum(1 for c in text if not c.isalpha() and c != ' ')
        if len(text) > 0 and non_alpha / len(text) > 0.7:
            return False
        return True

    def is_spam(self, text):
        """Detect spam/low-quality text"""
        text_lower = text.lower()
        
        # Spam patterns
        spam_patterns = [
            r'bit\.ly|tinyurl|goo\.gl',  # URL shorteners
            r'viagra|cialis|casino|lottery',  # Common spam
            r'click here|free money|buy now',  # Spam phrases
            r'http[s]?://',  # Too many URLs
            r'xxx|porn',  # Explicit content
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for excessive URL count
        url_count = len(re.findall(r'http[s]?://', text))
        if url_count > 2:
            return True
            
        return False

    def has_high_repetition(self, text, threshold=0.35):
        """Detect texts with too much word repetition"""
        words = text.lower().split()
        if len(words) < 5:
            return False
        
        # Count word frequency
        word_freq = Counter(words)
        most_common_count = word_freq.most_common(1)[0][1]
        
        # If one word appears more than threshold% of the time, it's repetitive
        if most_common_count / len(words) > threshold:
            return True
        return False

    def levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def string_similarity_ratio(self, s1, s2):
        """Calculate similarity ratio using Levenshtein"""
        distance = self.levenshtein_distance(s1.lower(), s2.lower())
        max_length = max(len(s1), len(s2))
        if max_length == 0:
            return 1.0
        return 1 - (distance / max_length)

    def jaccard_similarity(self, s1, s2):
        """Calculate Jaccard similarity of word sets"""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def remove_exact_duplicates(self, data):
        """Remove exact duplicates"""
        print("\n[1] Removing EXACT duplicates...")
        seen = set()
        unique_data = []
        duplicates = 0
        
        for item in data:
            input_clean = item['input_text'].lower().strip()
            reply_clean = item['reply_text'].lower().strip()
            key = (input_clean, reply_clean)
            
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
            else:
                duplicates += 1
        
        print(f"   • Removed: {duplicates} exact duplicates")
        print(f"   • Remaining: {len(unique_data)}")
        self.removed_reasons['exact_duplicates'] = duplicates
        return unique_data

    def remove_short_entries(self, data, min_length=8, max_length=512):
        """Remove entries that are too short or too long"""
        print(f"\n[2] Filtering by length (min: {min_length}, max: {max_length})...")
        filtered = []
        removed = 0
        
        for item in data:
            input_len = len(item['input_text'].strip())
            reply_len = len(item['reply_text'].strip())
            
            if min_length <= input_len <= max_length and min_length <= reply_len <= max_length:
                filtered.append(item)
            else:
                removed += 1
        
        print(f"   • Removed: {removed} short/long entries")
        print(f"   • Remaining: {len(filtered)}")
        self.removed_reasons['length_filter'] = removed
        return filtered

    def remove_low_quality(self, data):
        """Remove low quality entries"""
        print("\n[3] Removing low-quality entries...")
        filtered = []
        removed = 0
        
        for item in data:
            input_text = item['input_text']
            reply_text = item['reply_text']
            
            if (self.has_minimum_quality(input_text) and 
                self.has_minimum_quality(reply_text)):
                filtered.append(item)
            else:
                removed += 1
        
        print(f"   • Removed: {removed} low-quality entries")
        print(f"   • Remaining: {len(filtered)}")
        self.removed_reasons['low_quality'] = removed
        return filtered

    def remove_spam(self, data):
        """Remove spam/malicious entries"""
        print("\n[4] Removing spam entries...")
        filtered = []
        removed = 0
        
        for item in data:
            if not (self.is_spam(item['input_text']) or self.is_spam(item['reply_text'])):
                filtered.append(item)
            else:
                removed += 1
        
        print(f"   • Removed: {removed} spam entries")
        print(f"   • Remaining: {len(filtered)}")
        self.removed_reasons['spam'] = removed
        return filtered

    def remove_repetitive(self, data):
        """Remove entries with excessive repetition"""
        print("\n[5] Removing repetitive entries...")
        filtered = []
        removed = 0
        
        for item in data:
            if not (self.has_high_repetition(item['input_text']) or 
                    self.has_high_repetition(item['reply_text'])):
                filtered.append(item)
            else:
                removed += 1
        
        print(f"   • Removed: {removed} repetitive entries")
        print(f"   • Remaining: {len(filtered)}")
        self.removed_reasons['high_repetition'] = removed
        return filtered

    def remove_near_duplicates(self, data, jaccard_threshold=0.80, levenshtein_threshold=0.88):
        """Remove semantically similar entries (near duplicates)"""
        print(f"\n[6] Removing NEAR duplicates...")
        print(f"   • Jaccard threshold: {jaccard_threshold}")
        print(f"   • Levenshtein threshold: {levenshtein_threshold}")
        
        if len(data) == 0:
            return data
        
        # Use TF-IDF for fast initial screening
        texts = [f"{item['input_text']} {item['reply_text']}" for item in data]
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix)
        
        unique_indices = set(range(len(data)))
        removed = 0
        
        for i in range(len(data)):
            if i not in unique_indices:
                continue
                
            for j in range(i + 1, len(data)):
                if j not in unique_indices:
                    continue
                
                # Use TF-IDF as quick filter
                if similarity[i, j] > 0.85:
                    # Do detailed comparison
                    jac_sim = self.jaccard_similarity(
                        data[i]['input_text'],
                        data[j]['input_text']
                    )
                    lev_sim = self.string_similarity_ratio(
                        data[i]['reply_text'],
                        data[j]['reply_text']
                    )
                    
                    # If both are similar, remove the later one
                    if jac_sim >= jaccard_threshold and lev_sim >= levenshtein_threshold:
                        unique_indices.discard(j)
                        removed += 1
        
        unique_data = [data[idx] for idx in sorted(unique_indices)]
        print(f"   • Removed: {removed} near duplicates")
        print(f"   • Remaining: {len(unique_data)}")
        self.removed_reasons['near_duplicates'] = removed
        return unique_data

    def check_vocabulary_coverage(self, data, min_unique_tokens=0.25):
        """Remove entries with very limited vocabulary"""
        print(f"\n[7] Checking vocabulary coverage (min unique: {min_unique_tokens*100}%)...")
        filtered = []
        removed = 0
        
        for item in data:
            combined_text = (item['input_text'] + ' ' + item['reply_text']).lower().split()
            if len(combined_text) > 0:
                unique_tokens = len(set(combined_text))
                unique_ratio = unique_tokens / len(combined_text)
                
                if unique_ratio >= min_unique_tokens:
                    filtered.append(item)
                else:
                    removed += 1
            else:
                removed += 1
        
        print(f"   • Removed: {removed} low-vocabulary entries")
        print(f"   • Remaining: {len(filtered)}")
        self.removed_reasons['low_vocabulary'] = removed
        return filtered

    def normalize_text_data(self, data):
        """Normalize all text in the data"""
        print("\n[8] Normalizing text...")
        cleaned_data = []
        for item in data:
            try:
                if isinstance(item, dict) and 'input_text' in item and 'reply_text' in item:
                    item['input_text'] = self.clean_text(str(item['input_text']))
                    item['reply_text'] = self.clean_text(str(item['reply_text']))
                    cleaned_data.append(item)
            except Exception as e:
                print(f"   ⚠ Skipped invalid entry: {e}")
                continue
        print(f"   • Normalized {len(cleaned_data)} entries")
        return cleaned_data

    def detect_outliers(self, data):
        """Detect and flag potential outliers"""
        print("\n[9] Detecting statistical outliers...")
        
        input_lengths = [len(item['input_text'].split()) for item in data]
        reply_lengths = [len(item['reply_text'].split()) for item in data]
        
        # Calculate stats
        avg_input = sum(input_lengths) / len(input_lengths) if input_lengths else 0
        avg_reply = sum(reply_lengths) / len(reply_lengths) if reply_lengths else 0
        
        max_input = max(input_lengths) if input_lengths else 0
        max_reply = max(reply_lengths) if reply_lengths else 0
        
        print(f"   • Input: avg {avg_input:.1f} words, max {max_input} words")
        print(f"   • Reply: avg {avg_reply:.1f} words, max {max_reply} words")
        
        return data

    def make_professional(self, reply_text):
        """Convert reply to professional tone (METHOD 1)"""
        reply_clean = reply_text.strip()
        
        # If already starts with professional greeting, don't duplicate
        if reply_clean.lower().startswith('thank you'):
            return reply_clean
        
        # Start with professional greeting
        professional = "Thank you for your message. "
        
        # Capitalize first letter
        if reply_clean and reply_clean[0].islower():
            reply_clean = reply_clean[0].upper() + reply_clean[1:]
        
        # Remove ending period if exists (we'll add our own)
        if reply_clean.endswith('.'):
            reply_clean = reply_clean[:-1]
        
        # Add the reply content
        professional += reply_clean
        
        # Add professional closing
        professional += ". Please let us know if you need any further assistance."
        
        return professional

    def structure_input(self, input_text):
        """Add structure to input (METHOD 2)"""
        return "Customer Query: " + input_text.strip()

    def add_professional_tone(self, data):
        """Apply professional tone enhancements to entire dataset"""
        print("\n[10] Adding professional tone & structure...")
        print("   • METHOD 1: Convert all replies to professional tone")
        print("   • METHOD 2: Add 'Customer Query:' prefix to inputs")
        
        enhanced_data = []
        for item in data:
            try:
                enhanced_item = {
                    'input_text': self.structure_input(item['input_text']),
                    'reply_text': self.make_professional(item['reply_text'])
                }
                enhanced_data.append(enhanced_item)
            except Exception as e:
                print(f"   ⚠ Error processing entry: {e}")
                enhanced_data.append(item)
        
        print(f"   • Enhanced {len(enhanced_data)} entries with professional tone")
        print("\n   📝 Example Transformation:")
        if enhanced_data:
            example = enhanced_data[0]
            print(f"   Input: {example['input_text'][:60]}...")
            print(f"   Reply: {example['reply_text'][:80]}...")
        
        return enhanced_data

    def generate_report(self, original_count, final_count):
        """Generate cleaning report"""
        print("\n" + "="*60)
        print("DATA CLEANING REPORT")
        print("="*60)
        print(f"\nOriginal entries: {original_count}")
        print(f"Final entries: {final_count}")
        print(f"Removed: {original_count - final_count} ({(original_count - final_count) / original_count * 100:.2f}%)")
        print(f"Retained: {final_count} ({final_count / original_count * 100:.2f}%)")
        
        print(f"\nRemoval breakdown:")
        for reason, count in self.removed_reasons.most_common():
            print(f"  • {reason}: {count}")
        
        print("\n" + "="*60)

def load_data(folder_path):
    """Wrapper function for backward compatibility"""
    cleaner = DataCleaner()
    return cleaner.load_data(folder_path)

def remove_exact_duplicates(data):
    """Wrapper function for backward compatibility"""
    cleaner = DataCleaner()
    return cleaner.remove_exact_duplicates(data)

def remove_short_entries(data, min_length=10):
    """Wrapper function for backward compatibility"""
    cleaner = DataCleaner()
    return cleaner.remove_short_entries(data, min_length)

def main():
    """Main function with comprehensive data cleaning pipeline"""
    print("\n" + "="*60)
    print("AI EMAIL REPLY SYSTEM - DATA CLEANING PIPELINE")
    print("="*60)
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Load data
    folder_path = 'generation dataset'
    print(f"\nLoading data from '{folder_path}'...")
    data = cleaner.load_data(folder_path)
    original_count = len(data)
    print(f"\n✓ Total loaded: {original_count} entries")
    
    if len(data) == 0:
        print("✗ No data found! Check your generation dataset folder.")
        return
    
    # Apply cleaning pipeline in order
    print("\n" + "-"*60)
    print("CLEANING PIPELINE")
    print("-"*60)
    
    # Step 1: Normalize text
    data = cleaner.normalize_text_data(data)
    
    # Step 2: Length filtering
    data = cleaner.remove_short_entries(data, min_length=8, max_length=512)
    
    # Step 3: Quality check
    data = cleaner.remove_low_quality(data)
    
    # Step 4: Spam detection
    data = cleaner.remove_spam(data)
    
    # Step 5: Remove repetitive
    data = cleaner.remove_repetitive(data)
    
    # Step 6: Exact duplicates
    data = cleaner.remove_exact_duplicates(data)
    
    # Step 7: Near duplicates
    data = cleaner.remove_near_duplicates(data, jaccard_threshold=0.80, levenshtein_threshold=0.88)
    
    # Step 8: Vocabulary check
    data = cleaner.check_vocabulary_coverage(data, min_unique_tokens=0.25)
    
    # Step 9: Outlier detection
    data = cleaner.detect_outliers(data)
    
    # Step 10: Add professional tone & structure (NEW!)
    data = cleaner.add_professional_tone(data)
    
    # Generate report
    cleaner.generate_report(original_count, len(data))
    
    # Save cleaned data
    output_file = 'cleaned_dataset.json'
    print(f"\nSaving cleaned dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(data)} cleaned entries")
    
    # Save cleaning statistics to CSV
    stats_file = 'cleaning_report.csv'
    report_data = {
        'Metric': ['Original', 'Final', 'Removed', 'Retention Rate (%)'],
        'Count': [
            original_count,
            len(data),
            original_count - len(data),
            (len(data) / original_count * 100) if original_count > 0 else 0
        ]
    }
    
    # Add removal reasons
    for reason, count in cleaner.removed_reasons.items():
        report_data['Metric'].append(reason)
        report_data['Count'].append(count)
    
    df_stats = pd.DataFrame(report_data)
    df_stats.to_csv(stats_file, index=False)
    print(f"✓ Statistics saved to {stats_file}")
    
    print("\n" + "="*60)
    print("✓ DATA CLEANING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nNext steps:")
    print("1. Review cleaned_dataset.json")
    print("2. Review cleaning_report.csv for statistics")
    print("3. Run training: python training_code/2_train_transformer.py")

if __name__ == "__main__":
    main()