import re
import pandas as pd

class EntityExtractor:
    def __init__(self):
        self.products = [
            'SmartWatch V2', 'UltraClean Vacuum', 'SoundWave 300', 'PhotoSnap Cam',
            'Vision LED TV', 'RoboChef Blender', 'EcoBreeze AC', 'PowerMax Battery',
            'FitRun Treadmill', 'ProTab X1', 'smartwatch', 'vacuum', 'blender',
            'tv', 'television', 'camera', 'ac', 'air conditioner', 'battery', 'treadmill', 'tablet'
        ]
        
        self.complaint_keywords = [
            'broken', 'error', 'problem', 'issue', 'fault', 'defect',
            'late', 'delay', 'slow', 'wrong', 'incorrect', 'missing',
            'not working', 'malfunction', 'stuck', 'damaged', 'faulty',
            'glitchy', 'cracked', 'lost', 'no response', 'failed',
            'crashed', 'freeze', 'hang', 'bug', 'stopped', 'underbilled',
            'overcharged', 'billing', 'charge', "won't turn on", "doesn't work",
            "not turning on", "dead", "unresponsive"
        ]
        
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b',
            r'\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b(?:yesterday|today|tomorrow)\b',
            r'\b\d{1,2}\s+days?\s+ago\b',
            r'\b\d{1,2}\s+weeks?\s+ago\b',
            r'\bafter\s+\d{1,2}\s+days?\b',
            r'\b\d{1,2}\s+days?\b'
        ]
        
        self.order_patterns = [
            r'#\d{5,6}',
            r'order\s*#?\s*\d{5,6}',
            r'ref\s*#?\s*\d{5,6}',
            r'ticket\s*#?\s*\d{5,6}',
            r'case\s*#?\s*\d{5,6}'
        ]
        
        self.duration_patterns = [
            r'\b\d{1,2}\s+days?\b',
            r'\b\d{1,2}\s+weeks?\b',
            r'\b\d{1,2}\s+months?\b',
            r'\b\d{1,2}\s+years?\b',
            r'\b\d{1,2}\s+hours?\b',
            r'\b\d{1,2}\s+minutes?\b'
        ]
    
    def extract_products(self, text):
        if pd.isna(text):
            return []
        
        text_lower = str(text).lower()
        found_products = []
        
        for product in self.products:
            if product.lower() in text_lower:
                found_products.append(product)
        
        return list(set(found_products))
    
    def extract_dates(self, text):
        if pd.isna(text):
            return []
        
        text_str = str(text)
        found_dates = []
        
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            found_dates.extend(matches)
        
        return list(set(found_dates))
    
    def extract_durations(self, text):
        if pd.isna(text):
            return []
        
        text_str = str(text)
        found_durations = []
        
        for pattern in self.duration_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            found_durations.extend(matches)
        
        return list(set(found_durations))
    
    def extract_complaint_keywords(self, text):
        if pd.isna(text):
            return []
        
        text_lower = str(text).lower()
        found_keywords = []
        
        for keyword in self.complaint_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return list(set(found_keywords))
    
    def extract_order_numbers(self, text):
        if pd.isna(text):
            return []
        
        text_str = str(text)
        found_orders = []
        
        for pattern in self.order_patterns:
            matches = re.findall(pattern, text_str, re.IGNORECASE)
            found_orders.extend(matches)
        
        return list(set(found_orders))
    
    def extract_contact_attempts(self, text):
        if pd.isna(text):
            return False
        
        contact_phrases = [
            'contacted support', 'called support', 'reached out',
            'got no response', 'no response', 'called customer service',
            'emailed support', 'chat support', 'spoke with', 'talked to',
            'previous ticket', 'already contacted', 'second time', 'third time'
        ]
        
        text_lower = str(text).lower()
        for phrase in contact_phrases:
            if phrase in text_lower:
                return True
        
        return False
    
    def extract_urgency_indicators(self, text):
        if pd.isna(text):
            return []
        
        urgency_words = [
            'urgent', 'asap', 'immediately', 'critical', 'emergency',
            'need help', 'urgent help', 'please help', 'right now',
            'as soon as possible', 'very important', 'help!', 'emergency!',
            'at all', 'completely', 'totally'
        ]
        
        text_lower = str(text).lower()
        found_urgency = []
        
        for word in urgency_words:
            if word in text_lower:
                found_urgency.append(word)
        
        return list(set(found_urgency))
    
    def extract_problem_severity(self, text):
        if pd.isna(text):
            return []
        
        severity_indicators = [
            "won't turn on", "doesn't work", "completely broken",
            "not working at all", "totally dead", "completely dead",
            "stopped working", "not responding", "unresponsive"
        ]
        
        text_lower = str(text).lower()
        found_severity = []
        
        for indicator in severity_indicators:
            if indicator in text_lower:
                found_severity.append(indicator)
        
        return found_severity
    
    def extract_all_entities(self, text):
        entities = {
            'products': self.extract_products(text),
            'dates': self.extract_dates(text),
            'durations': self.extract_durations(text),
            'complaint_keywords': self.extract_complaint_keywords(text),
            'order_numbers': self.extract_order_numbers(text),
            'contacted_support': self.extract_contact_attempts(text),
            'urgency_indicators': self.extract_urgency_indicators(text),
            'problem_severity': self.extract_problem_severity(text)
        }
        
        return entities
