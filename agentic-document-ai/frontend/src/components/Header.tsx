import React from "react";
import { motion } from "framer-motion";
import { Brain, Zap, Shield } from "lucide-react";

const Header = () => {
  return (
    <motion.header
      className="header"
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <div className="header-content">
        <motion.div
          className="header-left"
          initial={{ x: -50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <motion.div
            className="logo-container"
            whileHover={{ scale: 1.05, rotate: 5 }}
            whileTap={{ scale: 0.95 }}
          >
            <Brain className="w-6 h-6 text-white" />
          </motion.div>
          <div className="header-title">
            <motion.h1
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              Agentic Document AI
            </motion.h1>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              Intelligent Form Processing Assistant
            </motion.p>
          </div>
        </motion.div>

        <motion.div
          className="header-right"
          initial={{ x: 50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <motion.div
            className="header-indicator"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400, damping: 10 }}
          >
            <motion.div
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
            >
              <Zap className="w-4 h-4 text-yellow-500" />
            </motion.div>
            <span>Real-time Processing</span>
          </motion.div>
          <motion.div
            className="header-indicator"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400, damping: 10 }}
          >
            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 2, repeat: Infinity, repeatDelay: 2 }}
            >
              <Shield className="w-4 h-4 text-green-500" />
            </motion.div>
            <span>Confidence Scoring</span>
          </motion.div>
        </motion.div>
      </div>
    </motion.header>
  );
};

export default Header;
