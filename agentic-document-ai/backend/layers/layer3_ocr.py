import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from difflib import SequenceMatcher
from PIL import Image
import gc
import re
import matplotlib.pyplot as plt

class DualOCRPipeline:
    """
    Layer 3: Dual OCR Pipeline
    Different OCR for printed vs handwritten text.
    - Printed OCR: PaddleOCR
    - Handwritten OCR: TrOCR (transformer handwriting model)
    """
    
    def __init__(self):
        # Initialize PaddleOCR for printed text
        try:
            self.printed_ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                det_db_box_thresh=0.2,
                det_db_unclip_ratio=2.0,
                rec_batch_num=6,
                device='cpu'
            )
            print("PaddleOCR initialized for printed text")
        except Exception as e:
            print(f"Error initializing PaddleOCR: {e}")
            self.printed_ocr = None
        
        # Initialize TrOCR for handwritten text
        try:
            self.handwritten_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.handwritten_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
            self.handwritten_model = self.handwritten_model.to("cpu")
            self.handwritten_model.eval()   
            print("TrOCR initialized for handwritten text")
        except Exception as e:
            print(f"Error initializing TrOCR: {e}")
            self.handwritten_processor = None
            self.handwritten_model = None
    
    def _improve_ocr_with_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Better preprocessing for document OCR"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Increase contrast
            gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
            
            # Apply CLAHE with smaller tiles for documents
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            
            # Mild sharpening for text
            kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Adaptive thresholding with different method for documents
            binary = cv2.adaptiveThreshold(sharpened, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
            
            # Optional: Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to 3-channel
            result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            return result
            
        except Exception as e:
            print(f"OCR preprocessing error: {e}")
            return image

    def _extract_core_fields_from_rows(self, rows):
        extracted = {}

        for row in rows:
            row_text = self._extract_row_text(row)
            row_text_norm = row_text.lower()

            # ---------- NAME ----------
            if "name" in row_text_norm:
                m = re.search(
                    r'name[^A-Za-z]*([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){0,2})$',
                    row_text.strip()
                )
                if m:
                    extracted["investor_name"] = {
                        "value": self._ocr_char_normalize(m.group(1)).replace("^", "").strip(),
                        "confidence": 0.9
                    }

            # ---------- SCHEME ----------
            if "scheme name" in row_text_norm and "scheme" not in extracted:
                m = re.search(
                    r'scheme\s*name[^A-Za-z]*([A-Za-z0-9\s\-\.]+)',
                    row_text,
                    re.I
                )
                if m:
                    extracted["scheme"] = {
                        "value": self._ocr_char_normalize(m.group(1)),
                        "confidence": 0.9
                    }
            # ---------- AMOUNT ----------
            if "amount" in row_text_norm and re.search(r'\d', row_text):
                amt = self._extract_amount_from_form(row, [], row_text)
                if amt:
                    extracted["amount"] = {
                        "value": self._clean_field_value("amount", amt),
                        "confidence": 0.95
                    }

            # ---------- FOLIO ----------
            if re.search(r'folio|f[o0]l[i1]o|fo[mn]o', row_text_norm):
                nums = re.findall(r'\b\d{6,12}\b', row_text)
                if nums:
                    extracted["folio_number"] = {
                        "value": nums[0],
                        "confidence": 0.9
                    }

        return extracted
    
    def print_final_assignments(self, structured_data: Dict):
        """
        Print clean final extracted fields
        """
        print("\n📌 FINAL EXTRACTED FIELDS\n" + "-" * 35)

        field_order = [
            "amount",
            "investor_name",
            "scheme",
            "folio_number"
        ]

        for field in field_order:
            if field in structured_data:
                print(f"{field:<13}: {structured_data[field]['value']}")

        print("-" * 35 + "\n")
    
    def extract_text_paddleocr(self, image: np.ndarray) -> List[Dict]:
        if self.printed_ocr is None:
            return []

        try:
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            results = self.printed_ocr.ocr(image_rgb)

            extracted_texts = []

            if not results:
                print("⚠ Paddle returned no OCR results")
                return []

            for block in results:
                if not block:
                    continue

                for line in block:
                    if not line or len(line) < 2:
                        continue

                    bbox, (text, confidence) = line

                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]

                    bounding_box = {
                        "x": int(min(x_coords)),
                        "y": int(min(y_coords)),
                        "width": int(max(x_coords) - min(x_coords)),
                        "height": int(max(y_coords) - min(y_coords))
                    }

                    extracted_texts.append({
                        "text": text.strip(),
                        "confidence": float(confidence),
                        "bounding_box": bounding_box,
                        "ocr_engine": "paddleocr",
                        "text_type": "printed"
                    })

            print(f"✅ Paddle extracted {len(extracted_texts)} texts")

            return extracted_texts

        except Exception as e:
            print("🔥 Paddle crashed:", e)
            return []
    
    def _post_process_ocr_results(self, extracted_texts: List[Dict]) -> List[Dict]:
        """Post-process OCR results conservatively"""
        processed_texts = []
        
        for text_info in extracted_texts:
            text = text_info["text"]
            original_text = text
            
            # Only fix obvious OCR errors, don't make assumptions
            obvious_fixes = {
                "|": "/",  # Pipe to slash
                "\\": "/", # Backslash to slash
            }
            
            # Apply fixes only if they make sense in context
            for wrong, correct in obvious_fixes.items():
                if wrong in text and len(text) > 2:  # Don't fix single characters
                    # Check if it's likely an OCR error vs actual text
                    if wrong == "l" and text.isupper():  # Don't fix "l" in uppercase text
                        continue
                    if wrong == "O" and text.islower():  # Don't fix "O" in lowercase text
                        continue
                    text = text.replace(wrong, correct)
            
            # Fix spacing but preserve structure
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Only update if changed AND it looks like an improvement
            if text != original_text:
                # Don't reduce confidence for obvious fixes
                text_info["text"] = text
                text_info["confidence"] = text_info["confidence"]  # Keep original confidence
            
            processed_texts.append(text_info)
        
        return processed_texts
    
    def _ocr_char_normalize(self, text: str) -> str:
        """
        Context-aware OCR cleanup
        - digits inside words → letters
        - symbols inside words → letters
        - preserves real numbers
        """
        chars = list(text)

        for i, c in enumerate(chars):
            prev_c = chars[i - 1] if i > 0 else ""
            next_c = chars[i + 1] if i < len(chars) - 1 else ""

            # digit inside alphabetic word → letter
            if c == "1" and prev_c.isalpha() and next_c.isalpha():
                chars[i] = "l"
            elif c == "0" and prev_c.isalpha() and next_c.isalpha():
                chars[i] = "o"
            elif c == "5" and prev_c.isalpha() and next_c.isalpha():
                chars[i] = "s"

            # symbol inside word
            elif c in "^_`~" and prev_c.isalpha() and next_c.isalpha():
                chars[i] = ""

        return re.sub(r"\s+", " ", "".join(chars)).strip()

    def _normalize_label(self, label: str) -> str:
        label = re.sub(r'[^a-z0-9\s]', '', label.lower())
        label = re.sub(r'\s+', '_', label)

        if 'name' in label:
            return 'investor_name'
        if 'scheme' in label:
            return 'scheme'
        if 'amount' in label:
            return 'amount'
        if 'folio' in label:
            return 'folio_number'

        return label

    def _merge_adjacent_text(self, texts, x_threshold=18, y_threshold=10):
        if not texts:
            return texts

        merged = []

        texts = sorted(
            texts,
            key=lambda t: (t["bounding_box"]["y"], t["bounding_box"]["x"])
        )

        for t in texts:
            if not merged:
                merged.append(t)
                continue

            prev = merged[-1]
            bx = t["bounding_box"]
            pbx = prev["bounding_box"]

            close_x = abs(bx["x"] - (pbx["x"] + pbx["width"])) < x_threshold
            center_y = bx["y"] + bx["height"] // 2
            prev_center_y = pbx["y"] + pbx["height"] // 2
            close_y = abs(center_y - prev_center_y) < y_threshold

            if close_x and close_y:
                prev["text"] += " " + t["text"]
                prev["confidence"] = max(prev["confidence"], t["confidence"])
                prev["bounding_box"]["width"] += bx["width"]
            else:
                merged.append(t)

        return merged

    def _looks_handwritten(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density > 0.18

    def classify_text_type(self, image: np.ndarray, region_type: str = None) -> str:
        """
        Classify if text is printed or handwritten
        Uses region type and image characteristics
        """
        if region_type in ["unknown", None]:
            # Handwritten detection fallback
            if self._looks_handwritten(image):
                return "handwritten"

            else:
                return "printed"
        
        # Fallback: analyze image characteristics
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture characteristics
            # Handwritten text usually has more irregular strokes
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate stroke width variation
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            stroke_widths = []
            
            for contour in contours:
                if len(contour) > 5:
                    rect = cv2.minAreaRect(contour)
                    width = min(rect[1])
                    if width > 0:
                        stroke_widths.append(width)
            
            if stroke_widths:
                width_variation = np.std(stroke_widths) / np.mean(stroke_widths)
            else:
                width_variation = 0
            
            # Heuristic: higher edge density and width variation suggests handwritten
            if edge_density > 0.15 and width_variation > 0.3:
                return "handwritten"
            else:
                return "printed"
                
        except Exception as e:
            print(f"Error in text classification: {e}")
            return "printed"  # Default to printed
    
    def extract_text_trocr(self, image: np.ndarray) -> List[Dict]:
        """Extract text using TrOCR (for handwritten text)"""
        if self.handwritten_processor is None or self.handwritten_model is None:
            return []
        
        try:
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Resize maintaining aspect ratio for better accuracy
            width, height = pil_image.size
            target_size = 384
            if max(width, height) > target_size:
                ratio = target_size / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Preprocess with padding for better results
            pixel_values = self.handwritten_processor(
                images=pil_image, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).pixel_values
            
            # Generate with beam search for better accuracy
            with torch.no_grad():
                generated_ids = self.handwritten_model.generate(
                    pixel_values,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True
                )
                generated_text = self.handwritten_processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]           
                
            # Calculate confidence (TrOCR doesn't provide native confidence, so we estimate)
            confidence = self._estimate_trocr_confidence(image, generated_text)
            
            extracted_text = {
                "text": generated_text.strip(),
                "confidence": confidence,
                "bounding_box": {
                    "x": 0,
                    "y": 0,
                    "width": image.shape[1],
                    "height": image.shape[0]
                },
                "ocr_engine": "trocr",
                "text_type": "handwritten"
            }
            
            return [extracted_text]
            
        except Exception as e:
            print(f"Error in TrOCR extraction: {e}")
            return []
        
    def process_batch(self, images: List[np.ndarray], regions_list: List[List[Dict]]) -> List[Dict]:
        """Process multiple documents with memory management"""
        results = []
        for i, (image, regions) in enumerate(zip(images, regions_list)):
            try:
                result = self.process_document_regions(image, regions)
                results.append(result)
                
                # Clear cache periodically to prevent memory issues
                if i % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"Error processing document {i}: {e}")
                results.append({
                    "error": str(e),
                    "status": "error",
                    "document_index": i
                })
        
        return results
    
    def _estimate_trocr_confidence(self, image: np.ndarray, text: str) -> float:
        """More sophisticated confidence estimation for TrOCR"""
        confidence = 0.6

        if len(text) >= 3:
            confidence += 0.1

        alnum_ratio = sum(c.isalnum() for c in text) / max(len(text), 1)
        confidence += min(0.2, alnum_ratio)

        return max(0.1, min(1.0, confidence))

    
    def extract_text_from_region(self, image: np.ndarray, region: Dict) -> List[Dict]:
        """
        Extract text from a specific region using appropriate OCR
        """
        try:
            # Get region type
            region_type = region.get("region_type", "unknown")
            
            # Classify text type
            text_type = self.classify_text_type(image, region_type)
            
            # Choose appropriate OCR engine
            if text_type == "handwritten":
                extracted_texts = self.extract_text_trocr(image)
            else:
                image = self._improve_ocr_with_preprocessing(image)
                extracted_texts = self.extract_text_paddleocr(image)

            # Add region information to extracted texts
            for text_info in extracted_texts:
                text_info["region_type"] = region_type
                text_info["region_confidence"] = region.get("confidence", 0.0)
            
            return extracted_texts
            
        except Exception as e:
            print(f"Error extracting text from region: {e}")
            return []
    
    def process_document_regions(self, image: np.ndarray, regions: List[Dict]) -> Dict:
        """
        Process all regions in a document
        Returns structured extraction results
        """
        try:
            all_extractions = []
            region_extractions = {}
            
            for region in regions:
                # Extract region image
                region_image = self._extract_region_image(image, region)
                
                if region_image.size == 0:
                    continue
                
                # Extract text from region
                extracted_texts = self.extract_text_from_region(region_image, region)
                
                # Store extractions
                region_type = region.get("region_type", "unknown")
                region_extractions.setdefault(region_type, [])
                region_extractions[region_type].extend(extracted_texts)

                all_extractions.extend(extracted_texts)
            
            # Post-process and structure the results
            # 1. OCR cleanup
            all_extractions = self._post_process_ocr_results(all_extractions)
            all_extractions = self._improve_extraction_accuracy(all_extractions)
            # Filter junk OCR
            all_extractions = [
                t for t in all_extractions
                if t["confidence"] > 0.55 and len(t["text"]) > 2
            ]

            # 2. Merge text blocks
            all_extractions = self._merge_adjacent_text(all_extractions)

            # 3. Structured extraction
            structured_raw = self._structure_extracted_data(all_extractions, region_extractions)

            # 5. Region-aware extraction
            region_data = self._extract_from_regions(region_extractions)
            structured_raw.update(region_data)

            # 6. Clean values
            structured_raw = self._clean_extracted_values(structured_raw)

            # 7. Normalize
            structured_data = self._normalize_fields(structured_raw)


            result = {
                "all_extractions": all_extractions,
                "region_extractions": region_extractions,
                "structured_data": structured_data,
                "total_regions_processed": len(regions),
                "total_text_extractions": len(all_extractions),
                "status": "success"
            }
            self.print_final_assignments(structured_data)
            return result
            
        except Exception as e:
            error_msg = f"Error processing document regions: {e}"
            print(error_msg)
            return {
                "all_extractions": [],
                "region_extractions": {},
                "structured_data": {},
                "error": error_msg,
                "status": "error"
            }
    
    def _soft_name_fallback(self, rows):
        for row in rows:
            words = row.split()
            if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
                return self._ocr_char_normalize(row).title()
        return ""


    def _extract_scheme_by_region(self, row):
        texts = [t["text"] for t in row]
        row_text = " ".join(texts)

        # RULE 1: If label exists, take text after it (structure only)
        m = re.search(r'scheme\s*name\s*[:\-]?\s*(.+)', row_text, re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            return self._ocr_char_normalize(value)

        # RULE 2: Otherwise return the longest text segment in the row
        # (pure layout heuristic)
        longest = max(texts, key=len, default="")
        return self._ocr_char_normalize(longest)

    def _extract_amount_from_form(self, row, all_extractions, row_text):
        # ONLY extract amount if THIS ROW contains digits
        nums = re.findall(r'\d[\d,]*\.\d{2}|\d[\d,]*', row_text)

        for num in nums:
            try:
                val = float(num.replace(',', ''))
                if val > 0:
                    return num
            except:
                continue

        return ""   # ← NO GLOBAL FALLBACK

    def _extract_region_image(self, image: np.ndarray, region: Dict) -> np.ndarray:
        """Extract the image patch for a specific region"""
        try:
            bbox = region["bounding_box"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            
            # Ensure coordinates are within image bounds
            img_height, img_width = image.shape[:2]
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            # Extract region
            region_image = image[y:y+h, x:x+w]
            return region_image
            
        except Exception as e:
            print(f"Error extracting region image: {e}")
            return np.array([])
    
    def debug_ocr_extraction(self, image: np.ndarray):
        """Debug method to see what OCR is actually extracting"""        
        # Get OCR results
        results = self.printed_ocr.ocr(image, cls=True)
        
        # Create a copy to draw on
        debug_image = image.copy()
        
        print(f"\n🔍 DEBUG: Found {len(results[0]) if results else 0} text blocks")
        
        for i, line in enumerate(results[0] if results else []):
            bbox, (text, confidence) = line
            
            # Draw bounding box
            pts = np.array(bbox, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(debug_image, [pts], True, (0, 255, 0), 2)
            
            # Add text label
            x = int(min([p[0] for p in bbox]))
            y = int(min([p[1] for p in bbox]))
            cv2.putText(debug_image, f"{i}: {confidence:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            print(f"{i:3d}. '{text}' (conf: {confidence:.2f})")
        
        # Save debug image
        debug_path = "debug_ocr.png"
        cv2.imwrite(debug_path, debug_image)
        print(f"\n📸 Debug image saved to {debug_path}")
        
        return debug_image

    def _structure_extracted_data(self, all_extractions, region_extractions):
        if not all_extractions:
            return {}

        print("\n🔍 STARTING STRUCTURED EXTRACTION")
        
        # Debug: Print all raw OCR extractions
        print("\n📝 RAW OCR EXTRACTIONS:")
        for i, extraction in enumerate(all_extractions[:20]):  # Show first 20
            print(f"  {i:2d}. '{extraction['text']}' (conf: {extraction['confidence']:.2f}, bbox: {extraction['bounding_box']})")
        if len(all_extractions) > 20:
            print(f"  ... and {len(all_extractions) - 20} more")

        rows = self._group_into_rows(all_extractions)
        
        # Debug: Print row groupings
        print(f"\n📊 GROUPED INTO {len(rows)} ROWS:")
        for i, row in enumerate(rows):
            row_text = self._extract_row_text(row)
            print(f"  Row {i}: '{row_text}'")

        structured = {}

        # ✅ 1. STRUCTURAL label:value extraction (punctuation-based ONLY)
        for row in rows:
            row_text = self._extract_row_text(row)

            if ":" in row_text:
                label, value = row_text.split(":", 1)
                label = label.strip().lower()
                value = value.strip()
                key = self._normalize_label(label)
                if key not in structured or len(value) > len(structured[key]["value"]):
                    structured[key] = {
                        "value": self._ocr_char_normalize(value),
                        "confidence": 0.85,
                        "source": "structural_colon"
                    }

        # ✅ 2. FALLBACK numeric-shape extraction (existing logic)
        for row in rows:
            row_text = self._extract_row_text(row)

            nums = re.findall(r'\b[A-Z0-9]{6,12}\b', row_text)
            if nums:
                key = "field_" + str(len(structured))
                structured[key] = {
                    "value": nums[0],
                    "confidence": 0.75,
                    "source": "numeric_shape"
                }
        # 🔥 Core field extraction from rows (layout-aware)
        core_fields = self._extract_core_fields_from_rows(rows)

        for k, v in core_fields.items():
            if (
                k not in structured
                or len(v["value"]) > len(structured[k]["value"])
            ):
                structured[k] = v

        return structured


    def _group_into_rows(self, texts: List[Dict], y_threshold: int = 12):
        if not texts:
            return []

        sorted_texts = sorted(texts, key=lambda t: t["bounding_box"]["y"])
        rows = []

        current_row = [sorted_texts[0]]
        current_y = sorted_texts[0]["bounding_box"]["y"]

        for text in sorted_texts[1:]:
            y = text["bounding_box"]["y"]

            if abs(y - current_y) <= y_threshold:
                current_row.append(text)
            else:
                current_row.sort(key=lambda t: t["bounding_box"]["x"])
                rows.append(current_row)
                current_row = [text]
                current_y = y

        current_row.sort(key=lambda t: t["bounding_box"]["x"])
        rows.append(current_row)

        return rows

    def _pair_labels_and_values(self, texts):
        """
        Pair label boxes with value boxes using column clustering
        """
        if not texts:
            return []

        # Compute center X of each box
        centers = []
        for t in texts:
            bx = t["bounding_box"]
            cx = bx["x"] + bx["width"] / 2
            centers.append((cx, t))

        # Split into left/right by median X
        xs = [c for c, _ in centers]
        median_x = np.median(xs)

        left = [t for cx, t in centers if cx < median_x]
        right = [t for cx, t in centers if cx >= median_x]

        pairs = []

        for l in left:
            ly = l["bounding_box"]["y"]
            best = None
            best_dist = 9999

            for r in right:
                ry = r["bounding_box"]["y"]
                dy = abs(ly - ry)

                if dy < best_dist:
                    best_dist = dy
                    best = r

            if best and best_dist < 35:  # vertical tolerance
                pairs.append((l, best))

        return pairs

    def _extract_row_text(self, row: List[Dict]) -> str:
        """Extract text from a row of elements"""
        return " ".join([t["text"].strip() for t in row])

    def _extract_from_regions(self, region_extractions: Dict) -> Dict:
        """Extract data from specific region types"""
        extracted = {}
        
        # Extract from investor/personal sections
        for region_type, region_texts in region_extractions.items():
            region_content = " ".join(t["text"] for t in region_texts)
            name_candidate = self._extract_name_from_text(region_content)
        return extracted

    def _extract_name_from_text(self, text: str) -> str:
        """Extract what looks like a person's name from text"""
        # Split into words
        words = text.split()
        
        # Look for patterns that resemble names
        for i in range(len(words) - 1):
            # Check for 2-3 word sequences with proper capitalization
            for length in [2, 3]:
                if i + length <= len(words):
                    candidate = " ".join(words[i:i+length])
                    # Basic name validation: at least 2 words, starts with capital letter
                    if len(candidate.split()) >= 2 and all(c.isalpha() or c.isspace() for c in candidate):
                        return candidate
        return ""

    def _clean_extracted_values(self, structured: Dict) -> Dict:
        """Clean and validate extracted values"""
        cleaned = {}
        
        for field, data in structured.items():
            value = self._clean_field_value(field, data["value"])
            confidence = data["confidence"]
            
            # Clean based on field type
            if field == "investor_name":
                value = re.sub(
                    r'\b(sole[l]?|first|unit|holder|applicant|name|of|the)\b',
                    '',
                    value,
                    flags=re.IGNORECASE
                )

                value = re.sub(r'\s+', ' ', value).strip()

                value = re.sub(r'^[^A-Za-z]+|[^A-Za-z]+$', '', value)
                
            elif field == "amount":
                # Standardize amount format
                value = value.replace('₹', '').replace('Rs.', '').replace('USD', '').strip()
                
            elif field == "pan":
                # Ensure uppercase
                value = value.upper()
                
            elif field == "email":
                # Ensure lowercase
                value = value.lower()
            
            # Only keep if value is meaningful
            if value and len(value.strip()) > 0:
                cleaned[field] = {
                    "value": value,
                    "confidence": confidence
                }
        
        return cleaned

    def _normalize_fields(self, structured):
        normalized = {}
        for k, v in structured.items():
            if k == "scheme":
                normalized[k] = v   
            else:
                normalized[k] = {
                    "value": self._ocr_char_normalize(v["value"]),
                    "confidence": v["confidence"]
                }
        return normalized


    def _clean_field_value(self, field_name: str, value: str) -> str:
        """Clean field value based on field type"""
        if not value:
            return ""
        
        value = str(value).strip()
        
        if field_name in ["scheme", "investor_name"]:
            value = self._ocr_char_normalize(value)
            
        elif field_name == "pan":
            # Remove spaces and ensure uppercase
            value = value.replace(' ', '').upper()
            
        elif field_name == "amount":
            raw = value.strip()

            # Case 1: OCR-broken thousand format like 50,0.00 → 50000
            m = re.match(r'^(\d{1,3}),(\d)\.(\d{2})$', raw)
            if m:
                thousands, hundreds, decimals = m.groups()
                return thousands + hundreds + "00"

            # Case 2: Normal formats
            match = re.search(r'\d[\d,]*(?:\.\d{2})?', raw)
            if match:
                cleaned = match.group(0).replace(',', '')
                try:
                    return str(int(float(cleaned)))
                except:
                    return ""

            return ""

        elif field_name == "folio_number":
            # Remove non-alphanumeric characters
            value = re.sub(r'[^A-Za-z0-9]', '', value)
            if value and value[0].isalpha():
                value = value[0].upper() + value[1:]
                
        elif field_name == "scheme":
            value = self._ocr_char_normalize(value)

            # Remove garbage OCR tokens
            value = re.sub(r'\b(fad|rlq|xy)\b', '', value, flags=re.IGNORECASE)

            # Fix common OCR word damage
            replacements = {
                'seasow': 'season',
                'bords': 'bonds',
            }
            for k, v in replacements.items():
                value = re.sub(rf'\b{k}\b', v, value, flags=re.IGNORECASE)
                value = re.sub(r'\bbandhan\s+al\b', 'bandhan all', value, flags=re.IGNORECASE)
                value = re.sub(r'\s+', ' ', value).strip()

        return value

    def _validate_field_value(self, field_name: str, value: str) -> bool:
        """Validate if a field value is reasonable"""
        if not value:
            return False
        
        if field_name == "pan":
            return len(value) == 10 and re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', value)
        
        elif field_name == "amount":
            return bool(re.search(r'\d', value))

        elif field_name == "folio_number":
            return len(value) >= 6 and any(c.isdigit() for c in value)
        
        elif field_name == "investor_name":
            return len(value) >= 3 and any(c.isalpha() for c in value)
        
        elif field_name == "scheme":
            return len(value) >= 3
        
        return True
    # In your DualOCRPipeline class, add this method:
    def _fallback_extraction(self, image: np.ndarray) -> Dict:
        """Fallback extraction when primary methods fail"""
        try:
            # Use simple PaddleOCR extraction
            extracted_texts = self.extract_text_paddleocr(image)  
            all_text = " ".join([t["text"] for t in extracted_texts])
            
            # Simple structured data
            structured_data = {}
            
            return {
                "all_extractions": extracted_texts,
                "region_extractions": {},
                "structured_data": structured_data,
                "total_regions_processed": 0,
                "total_text_extractions": len(extracted_texts),
                "status": "success_fallback"
            }
            
        except Exception as e:
            return {
                "all_extractions": [],
                "region_extractions": {},
                "structured_data": {},
                "error": str(e),
                "status": "error"
            }
    def _improve_extraction_accuracy(self, extracted_texts: List[Dict]) -> List[Dict]:
        """Apply post-processing to improve OCR accuracy"""
        improved_texts = []
        
        for text_info in extracted_texts:
            text = text_info["text"]
            original_text = text
            
            # Common OCR corrections
            replacements = {
                "|": "/",  # Pipe mistaken for slash in dates
                "\\": "/", # Backslash mistaken for slash
            }
            
            # Apply replacements for numeric contexts
            if re.search(r'\d', text):  # If text contains numbers
                for wrong, correct in replacements.items():
                    text = text.replace(wrong, correct)
            
            # Clean up spaces around punctuation
            text = re.sub(r'\s+([.,:;/])', r'\1', text)
            text = re.sub(r'([.,:;/])\s+', r'\1', text)
            
            # Remove isolated characters that are likely noise
            words = text.split()
            cleaned_words = []
            for word in words:
                if len(word) > 1 or word.isalnum():
                    cleaned_words.append(word)
            
            text = " ".join(cleaned_words)
            
            # Only update if changed significantly
            if text != original_text:
                text_info["text"] = text
                text_info["confidence"] = max(0.1, text_info["confidence"] - 0.05)  # Slightly reduce confidence for corrections
            
            improved_texts.append(text_info)
        
        return improved_texts