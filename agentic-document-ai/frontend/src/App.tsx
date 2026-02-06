import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import DocumentViewer from "./components/DocumentViewer";
import FormEditor from "./components/FormEditor";
import Header from "./components/Header";
import "./App.css";

interface FieldData {
  value: string;
  confidence: number;
  bounding_box?: { x: number; y: number; width: number; height: number };
  source_region?: string;
  reasoning_trace?: string;
}

interface DocumentData {
  fields: Record<string, FieldData>;
  regions: Array<{
    type: string;
    bounding_box: { x: number; y: number; width: number; height: number };
    confidence: number;
  }>;
}

function App() {
  const [documentData, setDocumentData] = useState<DocumentData | null>(null);
  const [imageUrl, setImageUrl] = useState<string>("");
  const [isPdfFile, setIsPdfFile] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState(false);
  const [activeRegion, setActiveRegion] = useState<string | null>(null);

  // ✅ Upload handler
  const handleFileUpload = async (file: File) => {
    setIsLoading(true);

    let previewUrl = "";

    try {
      previewUrl = URL.createObjectURL(file);

      setIsPdfFile(file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf"));

      const formData = new FormData();
      formData.append("file", file);

      const uploadResponse = await fetch("http://localhost:8000/uploadDocument", {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) throw new Error("Upload failed");

      const uploadResult = await uploadResponse.json();

      setDocumentData({
        fields: uploadResult.structured_data || {},
        regions: uploadResult.regions || [],
      });

      // ✅ Use processed image from backend
      if (uploadResult.processed_image_url) {
        const fullUrl = uploadResult.processed_image_url.startsWith("http")
          ? uploadResult.processed_image_url
          : `http://localhost:8000${uploadResult.processed_image_url}`;

        setImageUrl(fullUrl);
        console.log("Using processed image:", fullUrl);
      } else {
        setImageUrl(previewUrl);
        console.log("Fallback to original preview");
      }
    } catch (error) {
      console.error("Error processing document:", error);

      setDocumentData(getMockData());
      setImageUrl(previewUrl);
    } finally {
      setIsLoading(false);
    }
  };

  // ✅ Field update handler
  const handleFieldUpdate = async (fieldName: string, newValue: string) => {
    if (!documentData) return;

    try {
      const response = await fetch("http://localhost:8000/reasonUpdate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          field_name: fieldName,
          value: newValue,
          user_correction: true,
        }),
      });

      if (!response.ok) return;

      const result = await response.json();

      setDocumentData((prev) => {
        if (!prev) return prev;

        const updated = { ...prev };

        updated.fields[fieldName] = {
          ...updated.fields[fieldName],
          value: newValue,
          confidence: 0.95,
          reasoning_trace: result.reasoning_trace,
        };

        return updated;
      });
    } catch (error) {
      console.error("Error updating field:", error);
    }
  };

  const handleRegionClick = (regionType: string) => {
    setActiveRegion(regionType);
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return "high-confidence";
    if (confidence >= 0.6) return "medium-confidence";
    return "low-confidence";
  };

  // ✅ Mock fallback data
  const getMockData = (): DocumentData => ({
    fields: {
      investor_name: { value: "John Doe", confidence: 0.85 },
      folio_number: { value: "F123456", confidence: 0.92 },
      pan: { value: "ABCDE1234F", confidence: 0.78 },
    },
    regions: [
      {
        type: "investor_section",
        bounding_box: { x: 50, y: 150, width: 250, height: 200 },
        confidence: 0.9,
      },
    ],
  });

  return (
    <div className="app">
      <Header />

      <motion.div
        className="main-content"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <div className="split-screen">
          <AnimatePresence>
            <DocumentViewer
              imageUrl={imageUrl}
              regions={documentData?.regions || []}
              activeRegion={activeRegion}
              onRegionClick={handleRegionClick}
              onFileUpload={handleFileUpload}
              isLoading={isLoading}
              isPDF={isPdfFile}
            />

            <FormEditor
              fields={documentData?.fields || {}}
              onFieldUpdate={handleFieldUpdate}
              getConfidenceColor={getConfidenceColor}
            />
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
}

export default App;
