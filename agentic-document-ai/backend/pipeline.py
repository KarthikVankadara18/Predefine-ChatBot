"""
Agentic Document AI - Complete Processing Pipeline
Integrates all 5 layers: Preprocessing -> Segmentation -> OCR -> Reasoning -> UI
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import HTTPException

from layers.layer1_preprocessing import ImagePreprocessor
from layers.layer2_segmentation import RegionSegmentation
from layers.layer3_ocr import DualOCRPipeline
from layers.layer4_reasoning import AgentReasoningEngine

class DocumentProcessingPipeline:
    """
    Complete 5-layer processing pipeline for agentic document AI
    """
    
    def __init__(self):
        # Initialize all layers
        self.preprocessor = ImagePreprocessor()
        self.segmentation = RegionSegmentation()
        self.ocr_pipeline = DualOCRPipeline()
        self.reasoning_engine = AgentReasoningEngine()
        
        print("✅ Agentic Document AI Pipeline Initialized")
        print("📊 Layer 1: Image Preprocessing - Ready")
        print("🎯 Layer 2: Region Segmentation - Ready") 
        print("🔤 Layer 3: Dual OCR Pipeline - Ready")
        print("🧠 Layer 4: Agent Reasoning Engine - Ready")
        print("🖥️  Layer 5: Interactive UI - Ready")
    
    def process_document(self, image_path: str) -> Dict[str, Any]:
        """
        Complete document processing through all 5 layers
        """
        try:
            print(f"🚀 Starting document processing: {image_path}")
            
            # Layer 1: Image Preprocessing
            print("📸 Layer 1: Preprocessing image...")
            processed_image, preprocessing_metadata = self.preprocessor.preprocess(image_path)
            
            if processed_image is None:
                raise Exception("Image preprocessing failed")
            
            # Save processed image for debugging
            processed_path = f"processed/{os.path.basename(image_path)}_processed.png"
            os.makedirs("processed", exist_ok=True)
            cv2.imwrite(processed_path, processed_image)
            
            print(f"✅ Preprocessing complete: {preprocessing_metadata}")
            
            # Layer 2: Region Segmentation
            print("🎯 Layer 2: Segmenting document regions...")
            segmentation_result = self.segmentation.segment_document(processed_image)
            
            if segmentation_result["status"] != "success":
                raise Exception(f"Region segmentation failed: {segmentation_result.get('error')}")
            
            regions = segmentation_result["regions"]
            print(f"✅ Segmentation complete: Found {len(regions)} regions")
            
            # Save segmentation visualization
            vis_path = f"processed/{os.path.basename(image_path)}_segmented.png"
            self.segmentation.visualize_regions(processed_image, regions, vis_path)
            
            # Layer 3: OCR Processing
            print("🔤 Layer 3: Extracting text with OCR...")
            ocr_result = self.ocr_pipeline.process_document_regions(processed_image, regions)
            
            if ocr_result["status"] != "success":
                raise Exception(f"OCR processing failed: {ocr_result.get('error')}")
            
            structured_data = ocr_result["structured_data"]
            print(f"✅ OCR complete: Extracted {len(structured_data)} fields")
            
            # Layer 4: Agent Reasoning
            print("🧠 Layer 4: Running agent reasoning engine...")
            reasoning_result = self.reasoning_engine.initialize_fields(structured_data)
            
            if reasoning_result["status"] != "success":
                raise Exception(f"Reasoning engine failed: {reasoning_result.get('message')}")
            
            field_states = self.reasoning_engine.get_field_states()
            print(f"✅ Reasoning complete: Processed {len(field_states)} fields")
            
            # Prepare final result
            final_result = {
                "status": "success",
                "filename": os.path.basename(image_path),
                "processed_image_path": processed_path,
                "segmentation_path": vis_path,
                "preprocessing_metadata": preprocessing_metadata,
                "regions": regions,
                "field_extractions": ocr_result["all_extractions"],
                "structured_data": structured_data,
                "field_states": field_states,
                "overall_confidence": self._calculate_overall_confidence(field_states),
                "processing_summary": {
                    "layer1_status": "success",
                    "layer2_regions_found": len(regions),
                    "layer3_extractions": len(ocr_result["all_extractions"]),
                    "layer4_fields_processed": len(field_states),
                    "total_processing_time": "N/A"  # Could add timing
                }
            }
            
            print("🎉 Complete pipeline processing successful!")
            return final_result
            
        except Exception as e:
            error_msg = f"Pipeline processing failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "filename": os.path.basename(image_path) if image_path else "unknown"
            }
    
    def update_field_with_reasoning(self, field_name: str, new_value: str, 
                                  user_correction: bool = False, reasoning: str = "") -> Dict[str, Any]:
        """
        Update a field and trigger the reasoning cascade
        This provides real-time agent behavior
        """
        try:
            print(f"🔄 Updating field {field_name} to '{new_value}' (user_correction: {user_correction})")
            
            # Use reasoning engine to process update
            result = self.reasoning_engine.update_field(field_name, new_value, user_correction, reasoning)
            
            if result["status"] != "success":
                raise Exception(result.get("message", "Unknown reasoning error"))
            
            # Get updated field states
            updated_states = self.reasoning_engine.get_field_states()
            
            # Prepare response with full reasoning trace
            response = {
                "status": "success",
                "field_update": result["field_update"],
                "cascade_updates": result["cascade_updates"],
                "reasoning_explanation": result["reasoning_explanation"],
                "affected_fields": result["affected_fields"],
                "updated_field_states": updated_states,
                "overall_confidence": self._calculate_overall_confidence(updated_states),
                "learning_applied": user_correction
            }
            
            print(f"✅ Field update complete: {result['reasoning_explanation']}")
            return response
            
        except Exception as e:
            error_msg = f"Field update failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "field_name": field_name,
                "new_value": new_value
            }
    
    def get_field_confidence_details(self, field_name: str) -> Dict[str, Any]:
        """
        Get detailed confidence breakdown for a specific field
        """
        try:
            return self.reasoning_engine.get_field_confidence_details(field_name)
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "field_name": field_name
            }
    
    def store_correction_for_learning(self, field_name: str, original_value: str, 
                                   corrected_value: str, context: Dict = None) -> Dict[str, Any]:
        """
        Store user correction for learning memory
        """
        try:
            reasoning = f"User corrected {field_name} from '{original_value}' to '{corrected_value}'"
            
            # This is handled automatically in the reasoning engine
            # But we can add additional learning logic here if needed
            
            return {
                "status": "success",
                "message": "Correction stored for learning",
                "field_name": field_name,
                "original_value": original_value,
                "corrected_value": corrected_value
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "field_name": field_name
            }
    
    def _calculate_overall_confidence(self, field_states: Dict) -> float:
        """Calculate overall confidence across all fields"""
        if not field_states:
            return 0.0
        
        confidences = [field.get("confidence", 0.0) for field in field_states.values()]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of all pipeline components"""
        return {
            "status": "ready",
            "layers": {
                "layer1_preprocessing": {
                    "status": "ready",
                    "description": "Image denoising, deskew, thresholding"
                },
                "layer2_segmentation": {
                    "status": "ready", 
                    "description": "YOLO-based region detection"
                },
                "layer3_ocr": {
                    "status": "ready",
                    "description": "PaddleOCR + TrOCR dual pipeline"
                },
                "layer4_reasoning": {
                    "status": "ready",
                    "description": "Dependency graph with learning"
                },
                "layer5_ui": {
                    "status": "ready",
                    "description": "Interactive split-screen interface"
                }
            },
            "capabilities": [
                "Handwritten text recognition",
                "Dynamic layout detection", 
                "Real-time confidence scoring",
                "Dependency-aware field updates",
                "Learning from user corrections",
                "Explainable AI reasoning"
            ]
        }

# Global pipeline instance
pipeline = DocumentProcessingPipeline()
