import numpy as np
from typing import Dict, List, Any, Optional, Set
import json
import re
from datetime import datetime
import sqlite3
import os

class FieldNode:
    """Represents a field in the dependency graph"""
    
    def __init__(self, field_name: str, value: str = "", confidence: float = 0.0):
        self.field_name = field_name
        self.value = value
        self.confidence = confidence
        self.source_region = ""
        self.ocr_certainty = 0.0
        self.correction_history = []
        self.dependencies = set()  # Fields this depends on
        self.dependents = set()   # Fields that depend on this
        self.last_updated = datetime.now()
        self.reasoning_trace = ""
        
    def add_dependency(self, field_name: str):
        """Add a dependency to this field"""
        self.dependencies.add(field_name)
    
    def add_dependent(self, field_name: str):
        """Add a field that depends on this field"""
        self.dependents.add(field_name)
    
    def update_value(self, new_value: str, confidence: float, reasoning: str = "", 
                    user_correction: bool = False):
        """Update field value and confidence"""
        old_value = self.value
        
        # Store correction history
        if user_correction:
            self.correction_history.append({
                "old_value": old_value,
                "new_value": new_value,
                "timestamp": datetime.now().isoformat(),
                "reasoning": reasoning
            })
        
        self.value = new_value
        self.confidence = confidence
        self.reasoning_trace = reasoning
        self.last_updated = datetime.now()
        
        return {
            "field_name": self.field_name,
            "old_value": old_value,
            "new_value": new_value,
            "confidence": confidence,
            "reasoning": reasoning
        }

class AgentReasoningEngine:
    """
    Layer 4: Agent Reasoning Engine
    Intelligence core with dependency graph connecting fields.
    Implements reactive behavior when fields change.
    """
    
    def __init__(self, db_path: str = "learning_memory.db"):
        self.db_path = db_path
        self.fields = {}  # field_name -> FieldNode
        self.dependency_graph = {}  # field_name -> set of dependencies
        self.correction_memory = {}
        
        # Initialize dependency graph
        self._initialize_dependency_graph()
        
        # Initialize learning database
        self._initialize_database()
        
        # Load learning memory
        self._load_correction_memory()
    
    def _initialize_dependency_graph(self):
    # Dynamic system → no predefined dependencies
        self.fields = {}
        self.dependency_graph = {}

    
    def _initialize_database(self):
        """Initialize SQLite database for learning memory"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create corrections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_name TEXT,
                    original_value TEXT,
                    corrected_value TEXT,
                    timestamp TEXT,
                    reasoning TEXT,
                    context TEXT
                )
            ''')
            
            # Create field patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS field_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_name TEXT,
                    pattern TEXT,
                    confidence_boost REAL,
                    usage_count INTEGER DEFAULT 1
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def _load_correction_memory(self):
        """Load correction history from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT field_name, corrected_value, COUNT(*) as count "
                          "FROM corrections GROUP BY field_name, corrected_value")
            corrections = cursor.fetchall()
            
            for field_name, corrected_value, count in corrections:
                if field_name not in self.correction_memory:
                    self.correction_memory[field_name] = {}
                self.correction_memory[field_name][corrected_value] = count
            
            conn.close()
            
        except Exception as e:
            print(f"Error loading correction memory: {e}")
    
    def initialize_fields(self, extracted_data: Dict):
        try:
            for field_name, field_data in extracted_data.items():

                if field_name not in self.fields:
                    self.fields[field_name] = FieldNode(field_name)

                value = field_data.get("value", "")
                confidence = field_data.get("confidence", 0.0)
                source_region = field_data.get("source_region", "")
                ocr_certainty = field_data.get("confidence", 0.0)

                adjusted_confidence = self._adjust_confidence_with_learning(
                    field_name, value, confidence
                )

                self.fields[field_name].value = value
                self.fields[field_name].confidence = adjusted_confidence
                self.fields[field_name].source_region = source_region
                self.fields[field_name].ocr_certainty = ocr_certainty
                self.fields[field_name].reasoning_trace = f"Initial extraction from {source_region}"

            return {"status": "success", "message": "Fields initialized"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def update_field(self, field_name: str, new_value: str, user_correction: bool = False, 
                    reasoning: str = "") -> Dict:
        """
        Update a field and trigger reasoning cascade
        This is the core reactive behavior
        """
        try:
            if field_name not in self.fields:
                return {"status": "error", "message": f"Field {field_name} not found"}
            
            field = self.fields[field_name]
            
            # Calculate new confidence
            if user_correction:
                new_confidence = 0.95  # High confidence for user corrections
                reasoning = reasoning or f"User corrected {field_name} to '{new_value}'"
            else:
                new_confidence = self._calculate_field_confidence(field_name, new_value)
                reasoning = reasoning or f"Auto-updated {field_name} to '{new_value}'"
            
            # Update the field
            old_value = field.value

            update_result = field.update_value(new_value, new_confidence, reasoning, user_correction)

            if user_correction:
                self._store_correction(field_name, old_value, new_value, reasoning)

            
            # Trigger dependency cascade
            cascade_results = self._trigger_dependency_cascade(field_name)
            
            # Generate reasoning explanation
            explanation = self._generate_reasoning_explanation(field_name, update_result, cascade_results)
            
            return {
                "status": "success",
                "field_update": update_result,
                "cascade_updates": cascade_results,
                "reasoning_explanation": explanation,
                "affected_fields": list(cascade_results.keys())
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _trigger_dependency_cascade(self, updated_field: str) -> Dict:
        """Trigger updates for dependent fields"""
        cascade_results = {}
        
        try:
            field = self.fields[updated_field]
            
            # Process all dependent fields
            for dependent_field in field.dependents:
                if dependent_field in self.fields:
                    dependent = self.fields[dependent_field]
                    
                    # Recalculate dependent field based on new dependency value
                    new_value, new_confidence, reasoning = self._recalculate_dependent_field(
                        dependent_field, updated_field
                    )
                    
                    if new_value and new_value != dependent.value:
                        update_result = dependent.update_value(new_value, new_confidence, reasoning)
                        cascade_results[dependent_field] = update_result
                        
                        # Recursively trigger further dependencies
                        further_cascade = self._trigger_dependency_cascade(dependent_field)
                        cascade_results.update(further_cascade)
            
            return cascade_results
            
        except Exception as e:
            print(f"Error in dependency cascade: {e}")
            return cascade_results
    
    def _recalculate_dependent_field(self, field_name: str, trigger_field: str) -> tuple:
        """Recalculate a dependent field based on trigger field update"""
        try:
            field = self.fields[field_name]
            
            # Different recalculation logic based on field type
            if field_name == "pan" and trigger_field == "investor_name":
                # PAN might be validated against investor name
                new_confidence = min(0.9, field.confidence + 0.1)
                reasoning = f"PAN confidence updated based on investor name consistency"
                return field.value, new_confidence, reasoning
            
            elif field_name == "amount" and trigger_field == "units":
                # Amount might be recalculated based on units
                if field.value and self.fields["units"].value:
                    try:
                        units = float(self.fields["units"].value.replace(',', ''))
                        # Simple heuristic: amount should be reasonable relative to units
                        if units > 0:
                            new_confidence = min(0.95, field.confidence + 0.15)
                            reasoning = f"Amount confidence increased due to consistent units value"
                            return field.value, new_confidence, reasoning
                    except ValueError:
                        pass
            
            elif field_name == "folio_number" and trigger_field == "investor_name":
                # Folio number confidence based on investor name
                new_confidence = min(0.85, field.confidence + 0.1)
                reasoning = f"Folio confidence updated based on investor name"
                return field.value, new_confidence, reasoning
            
            # Default: slight confidence boost for dependency consistency
            new_confidence = min(0.8, field.confidence + 0.05)
            reasoning = f"Confidence updated due to {trigger_field} change"
            return field.value, new_confidence, reasoning
            
        except Exception as e:
            print(f"Error recalculating dependent field: {e}")
            return self.fields[field_name].value, self.fields[field_name].confidence, ""
    
    def _calculate_field_confidence(self, field_name: str, value: str) -> float:
        """Calculate confidence score for a field value"""
        try:
            base_confidence = 0.5
            
            # OCR certainty
            if field_name in self.fields:
                ocr_certainty = self.fields[field_name].ocr_certainty
                base_confidence += ocr_certainty * 0.3
            
            # Cross-field consistency
            consistency_score = self._calculate_cross_field_consistency(field_name, value)
            base_confidence += consistency_score * 0.2
            
            # Pattern validation
            pattern_score = self._validate_field_pattern(field_name, value)
            base_confidence += pattern_score * 0.2
            
            # Learning from corrections
            learning_score = self._get_learning_confidence_boost(field_name, value)
            base_confidence += learning_score * 0.3
            
            return min(1.0, max(0.0, base_confidence))
            
        except Exception as e:
            print(f"Error calculating field confidence: {e}")
            return 0.5
    
    def _calculate_cross_field_consistency(self, field_name: str, value: str) -> float:
        """Calculate consistency score with related fields"""
        try:
            consistency_score = 0.5
            
            if field_name == "pan":
                # PAN should be consistent with investor name format
                if "investor_name" in self.fields and self.fields["investor_name"].value:
                    name_words = len(self.fields["investor_name"].value.split())
                    if name_words >= 2:  # PAN typically associated with full names
                        consistency_score += 0.3
            
            elif field_name == "amount" and "units" in self.fields:
                # Amount should be reasonable relative to units
                if value and self.fields["units"].value:
                    try:
                        amount = float(value.replace(',', ''))
                        units = float(self.fields["units"].value.replace(',', ''))
                        if units > 0 and amount > 0:
                            # Simple heuristic: amount per unit should be reasonable
                            amount_per_unit = amount / units
                            if 10 <= amount_per_unit <= 10000:  # Reasonable NAV range
                                consistency_score += 0.4
                    except ValueError:
                        pass
            
            return min(1.0, consistency_score)
            
        except Exception as e:
            print(f"Error calculating cross-field consistency: {e}")
            return 0.5
    
    def _validate_field_pattern(self, field_name: str, value: str) -> float:
        """Validate field against expected patterns"""
        try:
            if not value:
                return 0.0
            
            pattern_score = 0.5
            
            if field_name == "pan":
                # PAN format: ABCDE1234F
                if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', value.upper()):
                    pattern_score = 1.0
                else:
                    pattern_score = 0.2
            
            elif field_name == "folio_number":
                # Folio format: F followed by numbers
                if re.match(r'^[Ff][0-9]+', value):
                    pattern_score = 1.0
                else:
                    pattern_score = 0.3
            
            elif field_name in ["amount", "units"]:
                # Should be valid numbers
                try:
                    float(value.replace(',', ''))
                    pattern_score = 1.0
                except ValueError:
                    pattern_score = 0.2
            
            elif field_name == "investor_name":
                # Should contain letters and spaces
                if re.match(r'^[A-Za-z\s]+$', value) and len(value) > 3:
                    pattern_score = 0.9
                else:
                    pattern_score = 0.4
            
            return pattern_score
            
        except Exception as e:
            print(f"Error validating field pattern: {e}")
            return 0.5
    
    def _adjust_confidence_with_learning(self, field_name: str, value: str, 
                                       base_confidence: float) -> float:
        """Adjust confidence based on learning from past corrections"""
        try:
            if field_name not in self.correction_memory:
                return base_confidence
            
            field_memory = self.correction_memory[field_name]
            
            # If this value has been corrected before, reduce confidence
            if value in field_memory:
                correction_count = field_memory[value]
                confidence_penalty = min(0.3, correction_count * 0.1)
                return max(0.2, base_confidence - confidence_penalty)
            
            # If similar values have been corrected, slightly reduce confidence
            for corrected_value in field_memory:
                if self._similar_strings(value, corrected_value, 0.8):
                    return max(0.3, base_confidence - 0.1)
            
            return base_confidence
            
        except Exception as e:
            print(f"Error adjusting confidence with learning: {e}")
            return base_confidence
    
    def _get_learning_confidence_boost(self, field_name: str, value: str) -> float:
        """Get confidence boost from learning patterns"""
        try:
            # Check if this value matches learned patterns
            if field_name in self.correction_memory:
                field_memory = self.correction_memory[field_name]
                
                # If this exact value has been confirmed before, boost confidence
                if value in field_memory and field_memory[value] > 2:
                    return 0.2
            
            return 0.0
            
        except Exception as e:
            print(f"Error getting learning confidence boost: {e}")
            return 0.0
    
    def _similar_strings(self, str1: str, str2: str, threshold: float = 0.8) -> bool:
        """Check if two strings are similar using simple character matching"""
        try:
            if not str1 or not str2:
                return False
            
            # Simple similarity based on common characters
            common_chars = set(str1.lower()) & set(str2.lower())
            total_chars = set(str1.lower()) | set(str2.lower())
            
            if not total_chars:
                return False
            
            similarity = len(common_chars) / len(total_chars)
            return similarity >= threshold
            
        except Exception:
            return False
    
    def _store_correction(self, field_name: str, old_value: str, new_value: str, 
                         reasoning: str):
        """Store correction in learning database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO corrections (field_name, original_value, corrected_value, 
                                        timestamp, reasoning, context)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (field_name, old_value, new_value, datetime.now().isoformat(), 
                  reasoning, json.dumps(self.get_field_states())))
            
            conn.commit()
            conn.close()
            
            # Update in-memory correction memory
            if field_name not in self.correction_memory:
                self.correction_memory[field_name] = {}
            if new_value not in self.correction_memory[field_name]:
                self.correction_memory[field_name][new_value] = 0
            self.correction_memory[field_name][new_value] += 1
            
        except Exception as e:
            print(f"Error storing correction: {e}")
    
    def _generate_reasoning_explanation(self, field_name: str, update_result: Dict, 
                                      cascade_results: Dict) -> str:
        """Generate human-readable reasoning explanation"""
        try:
            explanation = f"Updated {field_name} from '{update_result['old_value']}' to '{update_result['new_value']}' "
            explanation += f"with confidence {update_result['confidence']:.2f}. "
            
            if cascade_results:
                affected_fields = list(cascade_results.keys())
                explanation += f"This update affected {len(affected_fields)} related fields: {', '.join(affected_fields)}. "
                
                for affected_field, result in cascade_results.items():
                    explanation += f"{affected_field} confidence updated to {result['confidence']:.2f}. "
            
            # Add learning information
            if field_name in self.correction_memory:
                total_corrections = sum(self.correction_memory[field_name].values())
                explanation += f"Based on {total_corrections} previous corrections for this field. "
            
            return explanation
            
        except Exception as e:
            print(f"Error generating reasoning explanation: {e}")
            return f"Updated {field_name} with reasoning trace."
    
    def get_field_states(self) -> Dict:
        """Get current state of all fields"""
        try:
            states = {}
            for field_name, field in self.fields.items():
                states[field_name] = {
                    "value": field.value,
                    "confidence": field.confidence,
                    "source_region": field.source_region,
                    "ocr_certainty": field.ocr_certainty,
                    "reasoning_trace": field.reasoning_trace,
                    "last_updated": field.last_updated.isoformat(),
                    "dependencies": list(field.dependencies),
                    "dependents": list(field.dependents)
                }
            return states
            
        except Exception as e:
            print(f"Error getting field states: {e}")
            return {}
    
    def get_confidence_color(self, confidence: float) -> str:
        """Get color code based on confidence level"""
        if confidence >= 0.8:
            return "high"  # Green
        elif confidence >= 0.6:
            return "medium"  # Amber
        else:
            return "low"  # Red
    
    def get_field_confidence_details(self, field_name: str) -> Dict:
        """Get detailed confidence breakdown for a field"""
        try:
            if field_name not in self.fields:
                return {"error": f"Field {field_name} not found"}
            
            field = self.fields[field_name]
            
            # Calculate confidence factors
            ocr_certainty = field.ocr_certainty
            cross_field_consistency = self._calculate_cross_field_consistency(field_name, field.value)
            pattern_validation = self._validate_field_pattern(field_name, field.value)
            learning_boost = self._get_learning_confidence_boost(field_name, field.value)
            
            return {
                "field_name": field_name,
                "overall_confidence": field.confidence,
                "factors": {
                    "ocr_certainty": ocr_certainty,
                    "cross_field_consistency": cross_field_consistency,
                    "pattern_validation": pattern_validation,
                    "learning_boost": learning_boost
                },
                "confidence_color": self.get_confidence_color(field.confidence),
                "reasoning_trace": field.reasoning_trace,
                "correction_count": len(field.correction_history)
            }
            
        except Exception as e:
            return {"error": str(e)}
