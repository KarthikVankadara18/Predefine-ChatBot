import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Edit3,
  CheckCircle,
  AlertCircle,
  XCircle,
  Info,
  Save,
  Brain,
} from "lucide-react";

interface FieldData {
  value: string;
  confidence: number;
  bounding_box?: { x: number; y: number; width: number; height: number };
  source_region?: string;
  reasoning_trace?: string;
}

interface FormEditorProps {
  fields: Record<string, FieldData>;
  onFieldUpdate: (fieldName: string, newValue: string) => void;
  getConfidenceColor: (confidence: number) => string;
}

const fieldConfig = {
  investor_name: {
    label: "Investor Name",
    placeholder: "Enter investor name",
    icon: "👤",
    required: true,
  },
  folio_number: {
    label: "Folio Number",
    placeholder: "e.g., F123456",
    icon: "📋",
    required: true,
  },
  pan: {
    label: "PAN",
    placeholder: "e.g., ABCDE1234F",
    icon: "🆔",
    required: true,
  },
  scheme: {
    label: "Scheme",
    placeholder: "Enter scheme name",
    icon: "📊",
    required: true,
  },
  units: {
    label: "Units",
    placeholder: "Enter number of units",
    icon: "🔢",
    required: true,
  },
  amount: {
    label: "Amount",
    placeholder: "Enter amount",
    icon: "💰",
    required: true,
  },
};

const FormEditor: React.FC<FormEditorProps> = ({
  fields,
  onFieldUpdate,
  getConfidenceColor,
}) => {
  const [editingField, setEditingField] = useState<string | null>(null);
  const [tempValues, setTempValues] = useState<Record<string, string>>({});
  const [showReasoning, setShowReasoning] = useState<string | null>(null);

  const getConfidenceIcon = (confidence: number) => {
    if (confidence >= 0.8) {
      return <CheckCircle className="w-4 h-4 text-green-500" />;
    } else if (confidence >= 0.6) {
      return <AlertCircle className="w-4 h-4 text-amber-500" />;
    } else {
      return <XCircle className="w-4 h-4 text-red-500" />;
    }
  };

  const getConfidenceText = (confidence: number) => {
    if (confidence >= 0.8) return "High Confidence";
    if (confidence >= 0.6) return "Medium Confidence";
    return "Low Confidence";
  };

  const handleFieldEdit = (fieldName: string) => {
    setEditingField(fieldName);
    setTempValues({
      ...tempValues,
      [fieldName]: fields[fieldName]?.value || "",
    });
  };

  const handleFieldSave = (fieldName: string) => {
    const newValue = tempValues[fieldName] || "";
    onFieldUpdate(fieldName, newValue);
    setEditingField(null);
  };

  const handleFieldCancel = (fieldName: string) => {
    setEditingField(null);
    setTempValues({
      ...tempValues,
      [fieldName]: fields[fieldName]?.value || "",
    });
  };

  const handleInputChange = (fieldName: string, value: string) => {
    setTempValues({ ...tempValues, [fieldName]: value });
  };

  const getOverallConfidence = () => {
    const fieldValues = Object.values(fields);
    if (fieldValues.length === 0) return 0;
    const totalConfidence = fieldValues.reduce(
      (sum, field) => sum + field.confidence,
      0,
    );
    return totalConfidence / fieldValues.length;
  };

  const overallConfidence = getOverallConfidence();

  // If no fields, show empty state
  if (Object.keys(fields).length === 0) {
    return (
      <motion.div
        className="panel"
        initial={{ opacity: 0, x: 50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        {/* Header */}
        <motion.div
          className="panel-header form-editor-header"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="panel-title">
            <Edit3 className="w-5 h-5" />
            <span>Extracted Fields</span>
          </div>
          <div className="confidence-display">
            <div className="confidence-label">AI Processing</div>
            <div className="confidence-value">Ready</div>
          </div>
        </motion.div>

        {/* Empty State */}
        <motion.div
          className="form-content"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="flex items-center justify-center h-full">
            <motion.div
              className="text-center text-gray-500 p-8"
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <motion.div
                animate={{ rotate: [0, 5, -5, 0] }}
                transition={{
                  duration: 4,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
              >
                <Edit3 className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              </motion.div>
              <p className="text-lg font-medium mb-2">No document processed</p>
              <p className="text-sm">
                Upload a document to see extracted fields
              </p>
            </motion.div>
          </div>
        </motion.div>
      </motion.div>
    );
  }

  return (
    <motion.div
      className="panel"
      initial={{ opacity: 0, x: 50 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      {/* Header */}
      <motion.div
        className="panel-header form-editor-header"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <div className="panel-title">
          <Edit3 className="w-5 h-5" />
          <span>Extracted Fields</span>
        </div>
        <motion.div
          className="confidence-display"
          key={overallConfidence}
          initial={{ scale: 0.8 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 500 }}
        >
          <div className="confidence-label">Overall Confidence</div>
          <div className="confidence-value">
            {(overallConfidence * 100).toFixed(0)}%
          </div>
        </motion.div>
      </motion.div>

      {/* Form Fields */}
      <motion.div
        className="form-content"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <div className="form-fields">
          {Object.entries(fieldConfig).map(([fieldName, config], index) => {
            const field = fields[fieldName];
            const isEditing = editingField === fieldName;
            const confidence = field?.confidence || 0;
            const confidenceColor = getConfidenceColor(confidence);

            return (
              <motion.div
                key={fieldName}
                className={`field-container ${confidenceColor} ${
                  isEditing ? "editing" : ""
                }`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.1 + index * 0.1 }}
                whileHover={{ scale: 1.01 }}
                layout
              >
                <div className="field-header">
                  <div className="field-label">
                    <span className="text-lg">{config.icon}</span>
                    <label>
                      {config.label}
                      {config.required && <span className="required">*</span>}
                    </label>
                  </div>

                  <motion.div
                    className="field-confidence"
                    key={confidence}
                    initial={{ scale: 0.8 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 400 }}
                  >
                    {getConfidenceIcon(confidence)}
                    <span className="confidence-text">
                      {getConfidenceText(confidence)}
                    </span>
                    <span className="text-xs font-medium text-gray-700">
                      ({(confidence * 100).toFixed(0)}%)
                    </span>
                  </motion.div>
                </div>

                <AnimatePresence mode="wait">
                  {isEditing ? (
                    <motion.div
                      key="edit"
                      className="space-y-3"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <motion.input
                        type="text"
                        value={tempValues[fieldName] || ""}
                        onChange={(e) =>
                          handleInputChange(fieldName, e.target.value)
                        }
                        placeholder={config.placeholder}
                        className="field-input"
                        autoFocus
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.2 }}
                      />

                      <div className="field-actions">
                        <div className="action-hint">
                          Press Enter to save, Esc to cancel
                        </div>

                        <div className="action-buttons">
                          <motion.button
                            onClick={() => handleFieldCancel(fieldName)}
                            className="cancel-button"
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                          >
                            Cancel
                          </motion.button>
                          <motion.button
                            onClick={() => handleFieldSave(fieldName)}
                            className="save-button"
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                          >
                            <Save className="w-3 h-3" />
                            <span>Save</span>
                          </motion.button>
                        </div>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="display"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <motion.div
                        className="field-display"
                        onClick={() => handleFieldEdit(fieldName)}
                        whileHover={{ scale: 1.02 }}
                        transition={{
                          type: "spring",
                          stiffness: 400,
                          damping: 10,
                        }}
                      >
                        {field?.value || (
                          <span className="placeholder">
                            Click to edit {config.label.toLowerCase()}
                          </span>
                        )}
                      </motion.div>

                      {field?.source_region && (
                        <motion.div
                          className="field-source"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ duration: 0.3, delay: 0.1 }}
                        >
                          Source: {field.source_region.replace("_", " ")}
                        </motion.div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Reasoning Trace */}
                {field?.reasoning_trace && (
                  <div className="reasoning-section">
                    <motion.button
                      onClick={() =>
                        setShowReasoning(
                          showReasoning === fieldName ? null : fieldName,
                        )
                      }
                      className="reasoning-button"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Brain className="w-3 h-3" />
                      <span>AI Reasoning</span>
                      <Info className="w-3 h-3" />
                    </motion.button>

                    <AnimatePresence>
                      {showReasoning === fieldName && (
                        <motion.div
                          className="reasoning-content"
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          transition={{ duration: 0.3 }}
                        >
                          {field.reasoning_trace}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>
      </motion.div>

      {/* Footer Stats */}
      <motion.div
        className="panel-footer"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <div className="stats-grid">
          <motion.div
            className="stat-item"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400 }}
          >
            <motion.div
              className="stat-value high"
              key={
                Object.values(fields).filter((f) => f.confidence >= 0.8).length
              }
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 500 }}
            >
              {Object.values(fields).filter((f) => f.confidence >= 0.8).length}
            </motion.div>
            <div className="stat-label">High Confidence</div>
          </motion.div>
          <motion.div
            className="stat-item"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400 }}
          >
            <motion.div
              className="stat-value medium"
              key={
                Object.values(fields).filter(
                  (f) => f.confidence >= 0.6 && f.confidence < 0.8,
                ).length
              }
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 500 }}
            >
              {
                Object.values(fields).filter(
                  (f) => f.confidence >= 0.6 && f.confidence < 0.8,
                ).length
              }
            </motion.div>
            <div className="stat-label">Medium Confidence</div>
          </motion.div>
          <motion.div
            className="stat-item"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400 }}
          >
            <motion.div
              className="stat-value low"
              key={
                Object.values(fields).filter((f) => f.confidence < 0.6).length
              }
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 500 }}
            >
              {Object.values(fields).filter((f) => f.confidence < 0.6).length}
            </motion.div>
            <div className="stat-label">Low Confidence</div>
          </motion.div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default FormEditor;
