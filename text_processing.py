from num2words import num2words
import re

# Process Text Input
def process_text(text):
    """Performs number replacement, cleanup, and normalization in sequence."""
    
    # Replace numbers with words
    def replace_numbers_with_words(text):
        def replace(match):
            return num2words(int(match.group()))
        return re.sub(r'\b\d+\b', replace, text)

    # Normalize text (lowercasing, removing special characters)
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s\']', '', text)  # Remove non-alphanumeric characters
        text = ' '.join(text.split())  # Remove extra spaces
        return text

    # Apply transformations in sequence
    text = replace_numbers_with_words(text)
    text = normalize_text(text)
    
    return text