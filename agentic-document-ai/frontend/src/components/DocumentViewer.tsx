import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { RotateCw, Download, Eye, Upload } from "lucide-react";

interface Region {
  type: string;
  bounding_box: { x: number; y: number; width: number; height: number };
  confidence: number;
}

interface DocumentViewerProps {
  imageUrl: string;
  regions: Region[];
  activeRegion: string | null;
  onRegionClick: (regionType: string) => void;
  onFileUpload: (file: File) => void;
  isLoading: boolean;
  isPDF?: boolean;
}

const DocumentViewer: React.FC<DocumentViewerProps> = ({
  imageUrl,
  regions,
  activeRegion,
  onRegionClick,
  onFileUpload,
  isLoading,
  isPDF = false,
}) => {
  const [rotation, setRotation] = useState(0);
  const [showRegions] = useState(true);
  const [imageError, setImageError] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Remove the old isPDF check since we now get it as a prop

  // Reset image states when imageUrl changes
  useEffect(() => {
    if (imageUrl) {
      setImageError(false);
      setImageLoaded(false);
      console.log("DocumentViewer: imageUrl changed to", imageUrl.substring(0, 50) + "...");
    }
  }, [imageUrl]);

  const getRegionColor = (regionType: string): string => {
    const colors: Record<string, string> = {
      investor_section: "rgb(239, 68, 68)", // Red
      transaction_section: "rgb(34, 197, 94)", // Green
      signature: "rgb(59, 130, 246)", // Blue
      metadata_blocks: "rgb(168, 85, 247)", // Purple
      header: "rgb(251, 146, 60)", // Orange
      footer: "rgb(236, 72, 153)", // Pink
      table: "rgb(20, 184, 166)", // Teal
      paragraph: "rgb(156, 163, 175)", // Gray
    };
    return colors[regionType] || "rgb(107, 114, 128)";
  };

  const handleRotate = () => {
    setRotation((prev) => (prev + 90) % 360);
  };

  const handleDownload = () => {
    if (imageUrl) {
      const link = document.createElement("a");
      link.href = imageUrl;
      link.download = "document.png";
      link.click();
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileUpload(file);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleRegionClick = (region: Region, event: React.MouseEvent) => {
    event.stopPropagation();
    onRegionClick(region.type);
  };

  const getRegionStyle = (region: Region): React.CSSProperties => {
    if (!imageRef.current) return {};

    const img = imageRef.current;
    const imgRect = {
      width: img.clientWidth,
      height: img.clientHeight,
    };
    const containerRect = containerRef.current?.getBoundingClientRect();

    if (!containerRect) return {};

    const scaleX = imgRect.width / imageRef.current.naturalWidth;
    const scaleY = imgRect.height / imageRef.current.naturalHeight;

    return {
      position: "absolute",
      left: `${region.bounding_box.x * scaleX}px`,
      top: `${region.bounding_box.y * scaleY}px`,
      width: `${region.bounding_box.width * scaleX}px`,
      height: `${region.bounding_box.height * scaleY}px`,
      border: `2px solid ${getRegionColor(region.type)}`,
      backgroundColor: `${getRegionColor(region.type)}20`,
      cursor: "pointer",
      transition: "all 0.2s ease",
      zIndex: activeRegion === region.type ? 20 : 10,
    };
  };

  return (
    <motion.div
      className="panel"
      initial={{ opacity: 0, x: -50 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      {/* Header */}
      <motion.div
        className="panel-header document-viewer-header"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <div className="panel-title">
          <Eye className="w-5 h-5" />
          <span>Document Viewer</span>
        </div>
        <div className="confidence-display">
          <div className="confidence-label">AI Processing</div>
          <div className="confidence-value">Active</div>
        </div>
      </motion.div>

      {/* Controls */}
      <motion.div
        className="document-controls flex items-center justify-between flex-nowrap px-3"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <div className="control-buttons">
          <AnimatePresence>
            {imageUrl && (
              <motion.div
                className="flex items-center gap-2 whitespace-nowrap"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <motion.button
                  onClick={handleRotate}
                  className="control-button"
                  title="Rotate"
                  whileHover={{ scale: 1.1, rotate: 15 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <RotateCw className="w-4 h-4" />
                </motion.button>

                <motion.button
                  onClick={handleDownload}
                  className="control-button"
                  title="Download"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <Download className="w-4 h-4" />
                </motion.button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <motion.div
          className="text-sm text-gray-500"
          key={regions.length}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {imageUrl ? `${regions.length} regions detected` : ""}
        </motion.div>
      </motion.div>

      {/* Image Container */}
      <motion.div
        ref={containerRef}
        className="document-container h-full overflow-y-auto bg-white"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <AnimatePresence mode="wait">
          {!imageUrl ? (
            <motion.div
              key="upload"
              className="flex items-center justify-center h-full"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.4 }}
            >
              <div className="upload-card">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".png,.jpg,.jpeg,.tiff,.tif,.pdf"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <motion.div
                  animate={{ y: [0, -10, 0] }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }}
                >
                  <Upload className="w-16 h-16 mx-auto mb-4 text-blue-400" />
                </motion.div>
                <p className="upload-title">Upload a document to begin processing</p>
                <motion.button
                  onClick={handleUploadClick}
                  disabled={isLoading}
                  className="upload-button"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Upload className="w-5 h-5" />
                  <span>{isLoading ? "Processing..." : "Choose Document"}</span>
                </motion.button>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="document"
              className="w-full h-full bg-white overflow-auto"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
            >
              <div className="w-full flex justify-center">
                <motion.div
                  className="document-wrapper w-full max-w-[900px] relative"
                  style={{
                    transform: `rotate(${rotation}deg)`,
                    transformOrigin: "center center",
                  }}
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                >
                  {isPDF ? (
                    <iframe
                      src={`${imageUrl}#zoom=page-fit&view=Fit`}
                      title="PDF Preview"
                      className="document-image"
                      style={{
                        width: "100%",
                        minHeight: "100vh",
                        border: "none",
                        display: "block",
                        background: "white",
                      }}
                      onLoad={() => {
                        console.log("PDF loaded successfully");
                        setImageLoaded(true);
                        setImageError(false);
                      }}
                      onError={() => {
                        console.error("PDF failed to load:", imageUrl);
                        setImageError(true);
                        setImageLoaded(false);
                      }}
                    />
                  ) : (
                    <motion.img
                      ref={imageRef}
                      src={imageUrl}
                      alt="Document"
                      className="document-image"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.6 }}
                      onLoad={() => {
                        console.log("Image loaded successfully");
                        setImageLoaded(true);
                        setImageError(false);
                      }}
                      onError={(e) => {
                        console.error("Image failed to load:", e);
                        setImageError(true);
                        setImageLoaded(false);
                      }}
                      style={{
                        width: "100%",
                        height: "auto",
                        display: "block",
                        border: "2px solid #e5e7eb",
                        borderRadius: "0.5rem",
                        backgroundColor: "white",
                      }}
                    />
                  )}
                  {imageError && (
                    <div className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-gray-300 rounded-lg bg-white">
                      <div className="text-center">
                        <div className="w-16 h-16 mx-auto mb-4 bg-gray-200 rounded-full flex items-center justify-center">
                          <Eye className="w-8 h-8 text-gray-400" />
                        </div>
                        <p className="text-gray-500 mb-2">
                          {isPDF ? "PDF preview unavailable" : "Image failed to load"}
                        </p>
                        <p className="text-sm text-gray-400 mb-4">
                          {isPDF
                            ? "The document was processed but PDF preview is not supported in this browser"
                            : "The document was processed but preview is unavailable"}
                        </p>
                        {isPDF && imageUrl && (
                          <a
                            href={imageUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                          >
                            Open PDF in New Tab
                          </a>
                        )}
                        <div className="text-xs text-gray-500 mt-4">
                          <p>Document regions detected: {regions.length}</p>
                          <p>Fields extracted: {Object.keys(regions).length}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Region Overlays */}
                  {showRegions &&
                    imageLoaded &&
                    !isPDF &&
                    regions.map((region, index) => (
                      <motion.div
                        key={`${region.type}-${index}`}
                        style={{
                          ...getRegionStyle(region),
                          border: `2px solid ${getRegionColor(region.type)}`,
                          backgroundColor: `${getRegionColor(region.type)}20`,
                          position: "absolute",
                          zIndex: 10,
                        }}
                        onClick={(e) => handleRegionClick(region, e)}
                        className={`cursor-pointer ${
                          activeRegion === region.type ? "ring-2 ring-white ring-offset-2" : ""
                        }`}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.3, delay: index * 0.1 }}
                        whileHover={{ scale: 1.02 }}
                      >
                        <div
                          className="absolute top-0 left-0 bg-black bg-opacity-75 text-white text-xs px-1 py-0.5 rounded-br"
                          style={{
                            backgroundColor: getRegionColor(region.type),
                            fontSize: "10px",
                          }}
                        >
                          {region.type.replace("_", " ")}
                        </div>
                        <div
                          className="absolute bottom-0 right-0 bg-black bg-opacity-75 text-white text-xs px-1 py-0.5 rounded-tl"
                          style={{ fontSize: "10px" }}
                        >
                          {(region.confidence * 100).toFixed(0)}%
                        </div>
                      </motion.div>
                    ))}
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Region Legend */}
      <AnimatePresence>
        {showRegions && regions.length > 0 && (
          <motion.div
            className="panel-footer"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.4 }}
          >
            <div className="text-xs text-gray-600 mb-2">Detected Regions:</div>
            <div className="flex flex-wrap gap-2">
              {Array.from(new Set(regions.map((r) => r.type))).map((regionType, index) => (
                <motion.div
                  key={regionType}
                  className={`flex items-center space-x-1 px-2 py-1 rounded text-xs cursor-pointer transition-colors ${
                    activeRegion === regionType
                      ? "bg-blue-100 text-blue-700 border border-blue-300"
                      : "bg-white border border-gray-200 hover:bg-gray-50"
                  }`}
                  onClick={() => onRegionClick(regionType)}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: getRegionColor(regionType) }} />
                  <span>{regionType.replace("_", " ")}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default DocumentViewer;
